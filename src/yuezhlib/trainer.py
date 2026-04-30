import warnings

from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, disable_progress_bar
from datetime import datetime
from dotenv import load_dotenv
import evaluate
import gc
import huggingface_hub
import logging
import math
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import re
from termcolor import colored
import time
import torch
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BertTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    EncoderDecoderModel,
    IntervalStrategy,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Text2TextGenerationPipeline,
    pipeline
)
import wandb
import pycantonese
from yuezhlib.evaluate import DatasetEvaluator
__version__ = '0.3'
gpu_device = -1

class CantoneseModelTrainer(object):
    source_lang = "yue"
    target_lang = "zh"
    storage_path = "data/train_automation/"
    max_length = 550
    default_dataset = "raptorkwok/cantonese-traditional-chinese-parallel-corpus-gen3"

    is_gpu_available = True

    def __init__(self):
        print("Cantonese Model Trainer v" + __version__)
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        # Suppress logging warnings
        os.environ["GRPC_VERBOSITY"] = "ERROR"
        os.environ["GLOG_minloglevel"] = "2"

        # Available GPU memory check (GPU-only environment)
        if not torch.cuda.is_available():
            print(colored("ERROR:", 'red'), "No CUDA GPU found. This library requires a GPU environment. Exiting.")
            self.is_gpu_available = False
            return

        device_count = torch.cuda.device_count()
        gpu_free_pct = []
        for device_id in range(device_count):
            free_mem, total_mem = torch.cuda.mem_get_info(device=device_id)
            free_pct = free_mem * 100 / total_mem
            gpu_free_pct.append(free_pct)
            print(colored(f"Current GPU {device_id} Free Memory:", 'green'), str(round(free_pct, 1)) + "%")

        if all(pct <= 20 for pct in gpu_free_pct):
            print(colored("WARN:", 'red'), "All GPU units are busy and out of resources. Try again later. Exiting.")
            self.is_gpu_available = False
        
        # pre-download of datasets for NTLK evaluation
        nltk.download('wordnet', quiet=True)
        
        # Suppress TensorFlow-related errors
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        load_dotenv()

    # Helper functions
    def _preprocess_dataset(self, examples):
        inputs = [text for text in examples[CantoneseModelTrainer.source_lang]]
        targets = [text for text in examples[CantoneseModelTrainer.target_lang]]
        model_inputs = self.base_tokenizer(inputs, text_target=targets, max_length=CantoneseModelTrainer.max_length, truncation=True)
        return model_inputs

    def _filter_valid_examples(self, example):
        return (
            isinstance(example[CantoneseModelTrainer.source_lang], str) and example[CantoneseModelTrainer.source_lang].strip() and
            isinstance(example[CantoneseModelTrainer.target_lang], str) and example[CantoneseModelTrainer.target_lang].strip()
        )

    def _postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
    
        return preds, labels

    def _compute_metrics(self, eval_preds): # For Trainer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.base_tokenizer.batch_decode(preds, skip_special_tokens=True)
    
        labels = np.where(labels != -100, labels, self.base_tokenizer.pad_token_id)
        decoded_labels = self.base_tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = self._postprocess_text(decoded_preds, decoded_labels)

        result_bleu = self.metric_bleu.compute(predictions=decoded_preds, references=decoded_labels, tokenize='zh')
        result_chrf = self.metric_chrf.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
        results = {"bleu": result_bleu["score"], "chrf": result_chrf["score"]}
    
        prediction_lens = [np.count_nonzero(pred != self.base_tokenizer.pad_token_id) for pred in preds]
        results["gen_len"] = np.mean(prediction_lens)
        results = {k: round(v, 4) for k, v in results.items()}
        return results

    def _resize_embedding_layer(self, weight, new_size):
        old_vocab_size = weight.size(0)
        if new_size < old_vocab_size:
            return weight[:new_size, :]
        elif new_size == old_vocab_size:
            return weight
        else:
            new_weight = weight.new_zeros(new_size - old_vocab_size, weight.size(1))
            embedding_dim = weight.size(1)
            avg_weight = weight.mean(dim=0, keepdim=True)
            noise_weight = torch.empty_like(new_weight)
            noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
            new_weight = avg_weight + noise_weight
            return torch.cat([weight, new_weight], dim=0)

    def _filter_tokens(self, token_df, min_occurrence=1000):
        # filter non-alphanumeric tokens & tokens with length == 1 & min occurrence of tokens
        token_df = token_df.fillna('')
        token_df = token_df[(~token_df['Token'].str.contains(r"^[a-zA-Z0-9#～\[\]]+$")) & (token_df['Token'].str.len() > 1) & (token_df['Count'] >= min_occurrence)]
        # return a list of tokens (string)
        return token_df['Token'].tolist()
        
    # ==========================================================================================
    # ==========================================================================================
    # Main functions
    def construct_model(self, base_model_name, output_untrained_model, min_token_occurrence=1000):
        if not self.is_gpu_available:
            return False # Stop execution if GPU is busy

        print("Model Re-build: Start re-building model")
        
        if os.path.isdir(CantoneseModelTrainer.storage_path + output_untrained_model):
            print(colored("ERROR:", 'red'), "Model directory (" + CantoneseModelTrainer.storage_path + output_untrained_model + ")already exists. Exiting.")
            return False
        
        # Construct a Model for training and evaluation
        start_time = datetime.now()
        if not base_model_name:
            print(colored("ERROR:", 'red'), "Base Model Name (base_model_name) must not be empty")
            return False
        # Step 1: Load the multi-character tokens obtained from 30M sentences
        multivocab = pickle.load(open('data/wordsegs_multivocab.pickle', 'rb'))

        # Step 2: Load Dataset
        canton_ds = load_dataset(CantoneseModelTrainer.default_dataset)

        # Step 3: Compare the usage of tokens
        df_token_count = pd.read_csv('data/token_counts.csv')
        filtered_tokens = self._filter_tokens(df_token_count, min_occurrence=min_token_occurrence)

        # TODO: Filter what's required ONLY.
        
        # Step 4: Load the Base Model and Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = BartForConditionalGeneration.from_pretrained(base_model_name)
        
        # Step 5: Add missing tokens
        tokenizer.add_tokens(filtered_tokens)
        model.resize_token_embeddings(len(tokenizer))
        
        # Step 6: Resize embedding layer for new weights
        new_weight = self._resize_embedding_layer(model.get_input_embeddings().weight, len(tokenizer))
        input_embeddings = model.get_input_embeddings()
        input_embeddings.weight = new_weight
        model.set_input_embeddings(input_embeddings)

        # Step 7: Save the model with new embedding layer
        output_untrained_path = CantoneseModelTrainer.storage_path + output_untrained_model
        tokenizer.save_pretrained(output_untrained_path)
        model.save_pretrained(output_untrained_path)

        print("Total Tokens:", len(tokenizer))
        print("Added Tokens:", str(len(tokenizer) - tokenizer.vocab_size))
        print("Model saved to:", output_untrained_path)
        
        end_time = datetime.now()
        print('Construct Model Duration: {}'.format(end_time - start_time))

        # Recycle resources
        del model
        del tokenizer
        return True

    def construct_model_test(self, base_model_name, output_untrained_model, mode='inuse'):
        if mode not in ['inuse', 'all', 'only_inuse', 'only_all', 'only_common10', 'only_common100', 'only_common1000']:
            print('ERROR: Invalid Revised Mode')
            return False
        if not self.is_gpu_available:
            return False # Stop execution if GPU is busy

        print("Model Re-build: Start re-building model (revise mode)")
        
        if os.path.isdir(CantoneseModelTrainer.storage_path + output_untrained_model):
            print(colored("WARN:", 'yellow'), "Model directory (" + CantoneseModelTrainer.storage_path + output_untrained_model + ") already exists. Skipping model construction. Re-using the exist model.")
            return True

        # Construct a Model for training and evaluation
        start_time = datetime.now()
        if not base_model_name:
            print(colored("ERROR:", 'red'), "Base Model Name (base_model_name) must not be empty")
            return False
        
        # Step 1: Load and combine the multi-character tokens, generating an unique list (revised)
        if "only" in mode:
            print("Only Cantonese Tokens are used in the model.")
            mode = mode.replace('only_', '')
            multivocab = pickle.load(open('data/cantonese_tokens_' + mode + '.pickle', 'rb')) 
        else:
            cantonese_tokens = pickle.load(open('data/cantonese_tokens_' + mode + '.pickle', 'rb')) # 43,457 (in use)
            found_tokens = pickle.load(open('data/found_tokens.pickle', 'rb')) # 67,110
            multivocab = list(set(cantonese_tokens).union(set(found_tokens)))
        print("Adding", len(multivocab), "tokens")
        
        # Step 2: Load Dataset
        canton_ds = load_dataset(CantoneseModelTrainer.default_dataset)

        # Step 3: Load the Base Model and Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = BartForConditionalGeneration.from_pretrained(base_model_name)
        
        # Step 4: Add missing tokens
        tokenizer.add_tokens(multivocab)
        model.resize_token_embeddings(len(tokenizer))
        
        # Step 5: Resize embedding layer for new weights
        new_weight = self._resize_embedding_layer(model.get_input_embeddings().weight, len(tokenizer))
        input_embeddings = model.get_input_embeddings()
        input_embeddings.weight = new_weight
        model.set_input_embeddings(input_embeddings)

        # Step 7: Save the model with new embedding layer
        output_untrained_path = CantoneseModelTrainer.storage_path + output_untrained_model
        tokenizer.save_pretrained(output_untrained_path)
        model.save_pretrained(output_untrained_path)

        print("Total Tokens:", len(tokenizer))
        print("Added Tokens:", str(len(tokenizer) - tokenizer.vocab_size))
        print("Model saved to:", output_untrained_path)
        
        end_time = datetime.now()
        print('Construct Model Duration: {}'.format(end_time - start_time))

        # Recycle resources
        del model
        del tokenizer
        return True
    
        
    def setup_trainer(self, base_model_name, output_model, num_epochs=5, num_batch_size=8, token_approach='all'):
        if not self.is_gpu_available:
            return # Stop execution if GPU is busy
        
        start_time = datetime.now()
        if not base_model_name:
            print(colored("ERROR:", 'red'), "Base Model Name (base_model_name) must not be empty")
            return
        if not os.path.exists(CantoneseModelTrainer.storage_path):
            print(colored("ERROR:", 'red'), "Output path is invalid")
            return

        # Support Token Filtering & model rebuild
        if token_approach != 'all':
            if token_approach not in [
                    'common1000', # 130 tokens
                    'common500', # 366 tokens
                    'common100', # 2,246 tokens
                    'yue_all', # 110,548 mixed (mostly written Chinese) tokens
                    'yue_inuse', # 67,110 mixed (mostly written Chinese) tokens (Gen 3 Tokens)
                    'only_all', # 97,331 Cantonese-only tokens
                    'only_inuse', # 43,457 Cantonese-only tokens
                    'only_common10', # 10 most common Cantonese-only tokens (handpicked)
                    'only_common100', # 100 most common Cantonese-only tokens (handpicked)
                    'only_common1000', # 1000 most common Cantonese-only tokens (handpicked)
                    'bartchinese', # no token added
                ]:
                print(colored("ERROR:", 'red'), "Incorrect value of model_approach")
                return
            rebuild_model_name = "rebuild_" + token_approach + "_tokens"
            
            # Trigger model re-build
            if token_approach in ['common1000', 'common500', 'common100']:
                if self.construct_model(base_model_name, rebuild_model_name, min_token_occurrence = int(token_approach.replace('common', ''))):
                    base_model_name = CantoneseModelTrainer.storage_path + rebuild_model_name
                else:
                    print(colored("ERROR:", 'red'), "Failed to re-build the model. Halting.")
                    return
            elif token_approach in ['yue_all', 'yue_inuse', 'only_all', 'only_inuse', 'only_common10', 'only_common100', 'only_common1000']:
                if self.construct_model_test(base_model_name, rebuild_model_name, mode=(token_approach.replace('yue_', ''))):
                    base_model_name = CantoneseModelTrainer.storage_path + rebuild_model_name
                else:
                    print(colored("ERROR:", 'red'), "Failed to re-build the model. Halting.")
                    return

        # Load Base Model and Tokenizer
        self.base_tokenizer = BertTokenizer.from_pretrained(base_model_name)
        self.base_model = BartForConditionalGeneration.from_pretrained(base_model_name, output_hidden_states = True)

        # ======================================================================================
        # Load Dataset
        # ======================================================================================
        print("Training: Loading Dataset")
        canton_ds = load_dataset(CantoneseModelTrainer.default_dataset)
        yuezh_train = canton_ds["train"]
        yuezh_test = canton_ds["test"]
        yuezh_val = canton_ds["validation"]
        print("Train Dataset Count: ", len(yuezh_train))
        print("Test Dataset Count: ", len(yuezh_test))
        print("Validation Dataset Count: ", len(yuezh_val))
        yuezh_master = DatasetDict({"train": yuezh_train, "test": yuezh_test, "val": yuezh_val})

        # ======================================================================================
        # Process Dataset and Tokenization
        # ======================================================================================
        print("Training: Process Dataset and Tokenization")
        # Filter with valid examples only
        filtered_yuezh_master = yuezh_master.filter(self._filter_valid_examples)

        # Tokenization
        tokenized_yuezh_master = filtered_yuezh_master.map(self._preprocess_dataset, batched=True)

        # remove unused columns
        tokenized_yuezh_master = tokenized_yuezh_master.remove_columns(yuezh_train.column_names)

        # ======================================================================================
        # Data Collator
        # ======================================================================================
        print("Training: Data Collation")
        # Create batch of samples using DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.base_tokenizer, model=self.base_model)

        # ======================================================================================
        # Load Evaluation Metrics for Trainer Validation purposes
        # ======================================================================================
        print("Training: Loading Metrics")
        self.metric_bleu = evaluate.load("sacrebleu")
        self.metric_chrf = evaluate.load("chrf")
        
        # ======================================================================================
        # Login HuggingFace and WanDB (optional)
        # ======================================================================================
        huggingface_hub.login(os.environ.get("HUGGINGFACE_KEY"))
        #wandb.login(key=os.environ.get("WANDB_KEY"))

        # ======================================================================================
        # Training Arguments
        # ======================================================================================
        print("Training: Setting up Training Arguments")
        model_path = CantoneseModelTrainer.storage_path + output_model
        batch_size = num_batch_size
        training_args = CustomSeq2SeqTrainingArguments(
            output_dir = model_path,
            evaluation_strategy = IntervalStrategy.STEPS,
            logging_strategy = "no",
            optim = "adamw_torch",
            eval_steps = 10000,
            save_steps = 10000,
            learning_rate = 2e-5,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            weight_decay = 0.01,
            save_total_limit = 1,
            num_train_epochs = num_epochs,
            predict_with_generate=True,
            remove_unused_columns=True,
            fp16 = True,
            push_to_hub = False,
            metric_for_best_model = "bleu",
            load_best_model_at_end = True,
            report_to = "wandb"
        )
        
        # ======================================================================================
        # Training Loop
        # ======================================================================================
        print("Training: Start Training")
        trainer = Seq2SeqTrainer(
            model = self.base_model,
            args = training_args,
            train_dataset = tokenized_yuezh_master['train'],
            eval_dataset = tokenized_yuezh_master['val'],
            tokenizer = self.base_tokenizer,
            data_collator = data_collator,
            compute_metrics = self._compute_metrics,
        )
        trainer.train()
        trainer.evaluate()
        
        # ======================================================================================
        # Save Trained Model and Tokeinzer
        # ======================================================================================
        self.base_model.save_pretrained(model_path)
        self.base_tokenizer.save_pretrained(model_path)

        # ======================================================================================
        # Clear Resources
        # ======================================================================================
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        
        del yuezh_master
        del tokenized_yuezh_master
        del trainer
        del self.base_model
        del self.base_tokenizer

        # ======================================================================================
        # Evaluate Output
        # ======================================================================================
        evaluate_results = self.evaluate_output(model_path)
        end_time = datetime.now()
        print('Trainer Duration: {}'.format(end_time - start_time))
        return evaluate_results

    def evaluate_output(self, model_path):
        # ======================================================================================
        # Evaluate the Trained Model
        # ======================================================================================
        translator = pipeline("translation", model=model_path, max_length=CantoneseModelTrainer.max_length)

        testset = load_dataset(CantoneseModelTrainer.default_dataset, split="test")
        print(testset)

        # Generate Translation Results
        df = pd.DataFrame(testset)
        for index, data in tqdm(df.iterrows(), total=df.shape[0]):
            result = translator(data["yue"])[0]['translation_text'].replace(' ', '')
            if len(result) > 0:
                df.loc[index, 'zh'] = result

        # Save translated results for debug purposes
        pickle_path = model_path + '_newtestset.pickle'
        df.to_pickle(pickle_path)
        del df # Save resources

        # Evaluate and Save to File
        dataset_evaluator = DatasetEvaluator()
        results = dataset_evaluator.evaluate_dataset(pickle_path)
        with open(model_path + '_newtestset_results.pickle', 'w') as f:
            f.write(str(results))
        return results
        

# Patch to solve GPU selection bug
class CustomSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    gpu_selected = -1
    def __init__(self, *args, **kwargs):
        self.distributed_state = None
        #self.gpu_selected = args.gpu_device
        super(CustomSeq2SeqTrainingArguments, self).__init__(*args, **kwargs)

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        Name the device the number you use.
        """
        return torch.device("cuda:1") # + str(self.gpu_selected)

    @property
    def n_gpu(self):
        """
        The number of GPUs used by this process.
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        # set to one manually
        self._n_gpu = 1
        return self._n_gpu
