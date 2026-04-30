import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) # hide warnings for Jieba & PyCantonese library
# Suppress PyTorch Lightning debug messages
warnings.filterwarnings('ignore', module='torchmetrics')
warnings.filterwarnings('ignore', module='pytorch_lightning')
logging.getLogger('comet.models').setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING) 
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.hpu").setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)
# Suppress BLEURT & TensorFlow debug messages
logging.getLogger('bleurt').setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True

from transformers.utils import logging as logging_tf
logging_tf.set_verbosity_error()

import evaluate
evaluate.logging.set_verbosity_error()
import pandas as pd
import numpy as np
import os
import sys
import logging
from comet import download_model, load_from_checkpoint # required for COMET & COMET-22
from dotenv import load_dotenv
from tqdm.auto import tqdm
import re
import pickle
import pycantonese
from bleurt import score as bleurt_scorer

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import jieba_fast as jieba
import nltk
import torch
from nltk.translate.meteor_score import single_meteor_score, meteor_score
## Version History
# 0.5.9
# - Add a custom function to check BLEU, chrF++, BLEURT and TER

# 0.5.8
# - Add composite score functions: "evaluate_single_composite_v2()" and "evaluate_dataset_composite_v2()"

# 0.5.7
# - Add BLEURT and COMET-22 evaluation metrics to composite scores

# 0.5.6
# - Add composite score functions: "evaluate_single_composite()" and "evaluate_dataset_composite()"

# 0.5.5
# - Suppress warning messages

# 0.5.4
# - Added COMET and characTER
# - Refactored variables

# 0.5.3
# - Added "load_metrics()" in the initalization to avoid -reinitialization of metrics

# 0.5.0
# - Added Chinese METEOR as an experimental metric
# - Changed implementation of METEOR to NLTK's ones, as the evalaute's METEOR implementation is buggy

__version__ = '0.5.9'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress unwanted logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"  # Suppress transformers warnings

class DatasetEvaluator:
    def __init__(self, init_mode = 'normal'):
        logger.info(f"Dataset Evaluator loaded: v{__version__}")
        load_dotenv()
        self._require_gpu()
        self._load_metrics(init_mode)

    def _require_gpu(self):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU environment is required, but no CUDA GPU is available.")

    def _load_metrics(self, init_mode = 'normal'):
        logger.info(f"Initialization Mode: {init_mode}")
        
        # Common Metrics
        self.metric_bertscore = evaluate.load("bertscore")
        self.metric_bleu = evaluate.load("sacrebleu")
        if init_mode == 'normal':
            # Configure jieba to suppress loading messages
            jieba.setLogLevel(logging.ERROR)
            torch.set_float32_matmul_precision('medium')
            # Download NLTK-required libraries
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
            self.metric_chrf = evaluate.load("chrf")
            self.metric_cmeteor = evaluate.load("raptorkwok/chinesemeteor")
            self.metric_comet = evaluate.load("comet")
            self.metric_character = evaluate.load('character')
        elif init_mode == "composite_v2":
            #from bleurt import score as scorer_bleurt
            self.metric_comet22 = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
            #with tf.device(f"GPU:0"):
            #    self.metric_bleurt = scorer_bleurt.BleurtScorer("metrics/bleurt/BLEURT-20")
        elif init_mode == "gen3":
            del self.metric_bertscore
            self.metric_chrf = evaluate.load("chrf")
            self.metric_ter = evaluate.load("ter")
            self.metric_bleurt = bleurt_scorer.BleurtScorer("BLEURT-20")
    
    def is_pickle_file(self, filepath):
        """Check if import file is pickle format."""
        try:
            with open(filepath, 'rb') as f:
                pickle.load(f)
            return True
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError):
            # These exceptions typically indicate a non-pickle file or a corrupted pickle
            return False
        except Exception as e:
            # Catch other unexpected errors during unpickling
            logger.error(f"An unexpected error occurred while checking {filepath}: {e}")
            return False
        
    
    def _chinese_tokenize(self, text):
        """Tokenize Chinese text using PyCantonese"""
        #return list(jieba.cut(text))
        return pycantonese.segment(text)

    def _split_keyword(self, s):
        """Split sentence into list of characters with Cantonese and code-mixing support"""
        regex = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?|[\u4e00-\ufaff]|[A-Za-z]+(?:'[A-Za-z]+)?"
        return [m for m in re.findall(regex, s, re.VERBOSE) if m.strip()]

    def evaluate_single_composite_v2(self, zh, ref, src):
        # Formula: 0.4 x COMET-22 + 0.4 x BertScore + 0.2 x SacreBLEU
        
        # pre-test check
        if not zh or not ref or not src:
            logger.error("Either zh, ref or src is empty or not set")
            return False

        # BertScore x 0.4
        bertscore_result = self.metric_bertscore.compute(predictions=[zh], references=[ref], lang="zh")

        # COMET-22 x 0.4
        comet22_result = self.metric_comet22.predict([{"src": src, "mt": zh, "ref": ref}], batch_size=1, gpus=1)

        # sacreBLEU x 0.2
        bleu_result = self.metric_bleu.compute(predictions=[zh], references=[[ref]], tokenize="zh")

        score = comet22_result["scores"][0] * 0.4 + np.mean(bertscore_result["f1"]) * 0.4 + bleu_result['score'] / 100 * 0.2
        return float(round(float(score), 4))
        
    def evaluate_single_composite(self, zh, ref, src):
        # Formula: 0.4 x COMET + 0.4 x BertScore + 0.2 x sacreBLEU

        # pre-test check
        if not zh or not ref or not src:
            logger.error("Either zh, ref or src is empty or not set")
            return False

        # pre-process sentences
        split_hypo = ' '.join(self._split_keyword(zh))
        split_ref = ' '.join(self._split_keyword(ref))
        split_src = ' '.join(self._split_keyword(src))

        # COMET x 0.4
        comet_result = self.metric_comet.compute(predictions=[split_hypo], references=[split_ref], sources=[split_src])

        # BertScore x 0.4
        bertscore_result = self.metric_bertscore.compute(predictions=[zh], references=[ref], lang="zh")

        # sacreBLEU x 0.2
        bleu_result = self.metric_bleu.compute(predictions=[zh], references=[[ref]], tokenize="zh")

        score = comet_result["scores"][0] * 0.4 + np.mean(bertscore_result["f1"]) * 0.4 + bleu_result['score'] / 100 * 0.2
        return round(float(score), 4)
        
    def evaluate_single(self, zh, ref, yue=''): # source language is for future use
        """Evaluate a single set of sentence pairs using various evaluation metrics."""
        
        # for METEOR implementation
        meteor_pred = self._chinese_tokenize(zh)
        meteor_ref = self._chinese_tokenize(ref)

        split_hypo = ' '.join(self._split_keyword(zh))
        split_ref = ' '.join(self._split_keyword(ref))
        split_src = ' '.join(self._split_keyword(yue))

        original_stdout = sys.stdout # store original output to suppress warnings
        sys.stdout = open(os.devnull, 'w')
        try:
            bleu_result = self.metric_bleu.compute(predictions=[zh], references=[[ref]], tokenize="zh")
            chrf_result = self.metric_chrf.compute(predictions=[zh], references=[[ref]], word_order=2)
            meteor_result = single_meteor_score(meteor_ref, meteor_pred)
            bertscore_result = self.metric_bertscore.compute(predictions=[zh], references=[ref], lang="zh")
            cmeteor_result = self.metric_cmeteor.compute(predictions=[zh], references=[ref])
            comet_result = self.metric_comet.compute(predictions=[split_hypo], references=[split_ref], sources=[split_src])
            character_result = self.metric_character.compute(references=[split_ref], predictions=[split_hypo])
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout # restore original output
        
        return {
            'bleu': round(bleu_result['score'], 4),
            'chrf': round(chrf_result['score'], 4),
            'meteor': round(meteor_result, 4),
            'bertscore': float(round(np.mean(bertscore_result["f1"]), 4)),
            'cmeteor': float(round(np.mean(cmeteor_result['meteor']), 4)),
            'comet': round(comet_result["scores"][0], 4),
            'character': round(character_result["cer_score"], 4)
        }

    def evaluate_dataset_composite_v2(self, pickle_path):
        """Evaluate dataset using composite score."""
        # Pre-evaluate checking
        if not os.path.exists(pickle_path):
            logger.error("Specified file does not exist.")
            return False
        if not self.is_pickle_file(pickle_path):
            logger.error("Specified file is not in pickle format")
            return False
        
        # Load and prepare dataset
        dataset = self.prepare_dataset(pickle_path)
        df = pd.DataFrame([ex['translation'] for ex in dataset])
        df = df[['yue', 'zh', 'ref']].apply(lambda x: x.str.strip())

        logger.info(f"Dataset Size: {len(df)}")
        report_data = []

        original_stdout = sys.stdout # store original output
        sys.stdout = open(os.devnull, 'w')
        try:
            for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating scores"):
                composite_score = self.evaluate_single_composite_v2(row['zh'], row['ref'], row['yue'])
                report_data.append({
                    'yue': row['yue'],
                    'zh': row['zh'],
                    'ref': row['ref'],
                    'composite': composite_score
                })
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout # restore original output

        df_report = pd.DataFrame(report_data)
        logger.info(f"Average Composite Score: {df_report['composite'].mean():.4f}")
        return df_report
        
    def evaluate_dataset_composite(self, pickle_path):
        """Evaluate dataset using composite score."""
        # Pre-evaluate checking
        if not os.path.exists(pickle_path):
            logger.error("Specified file does not exist.")
            return False
        if not self.is_pickle_file(pickle_path):
            logger.error("Specified file is not in pickle format")
            return False
        
        # Load and prepare dataset
        dataset = self.prepare_dataset(pickle_path)
        df = pd.DataFrame([ex['translation'] for ex in dataset])
        df = df[['yue', 'zh', 'ref']].apply(lambda x: x.str.strip())

        logger.info(f"Dataset Size: {len(df)}")
        report_data = []

        original_stdout = sys.stdout # store original output
        sys.stdout = open(os.devnull, 'w')
        try:
            for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating scores"):
                composite_score = self.evaluate_single_composite(row['zh'], row['ref'], row['yue'])
                report_data.append({
                    'yue': row['yue'],
                    'zh': row['zh'],
                    'ref': row['ref'],
                    'composite': composite_score
                })
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout # restore original output

        df_report = pd.DataFrame(report_data)
        logger.info(f"Average Composite Score: {df_report['composite'].mean():.4f}")
        return df_report
        
    def evaluate_dataset(self, pickle_path, evaluate_mode='sentence'):
        """Evaluate dataset using various evaluation metrics."""
        
        # Pre-evaluate checking
        if not os.path.exists(pickle_path):
            logger.error("Specified file does not exist.")
            return False
        if not self.is_pickle_file(pickle_path):
            logger.error("Specified file is not in pickle format")
            return False
        
        # Load and prepare dataset
        dataset = self.prepare_dataset(pickle_path)
        df = pd.DataFrame([ex['translation'] for ex in dataset])
        df = df[['yue', 'zh', 'ref']].apply(lambda x: x.str.strip())

        logger.info(f"Dataset Size: {len(df)}")

        if evaluate_mode == 'sentence':
            report_data = []
            logger.info(f"Evaluating {len(df)} sentences")

            # Pre-tokenize for METEOR
            df['meteor_pred'] = df['zh'].apply(lambda x: self._chinese_tokenize(x))
            df['meteor_ref'] = df['ref'].apply(lambda x: self._chinese_tokenize(x))

            original_stdout = sys.stdout # store original output
            sys.stdout = open(os.devnull, 'w')
            try:
                for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating scores"):
                    ref_bleu_chrf = [[row['ref']]]
                    ref_bert = [row['ref']]

                    split_hypo = ' '.join(self._split_keyword(row['zh']))
                    split_ref = ' '.join(self._split_keyword(row['ref']))
                    split_src = ' '.join(self._split_keyword(row['yue']))
    
                    bleu_result = self.metric_bleu.compute(predictions=[row['zh']], references=ref_bleu_chrf, tokenize="zh")
                    chrf_result = self.metric_chrf.compute(predictions=[row['zh']], references=ref_bleu_chrf, word_order=2)
                    meteor_result = single_meteor_score(row['meteor_ref'], row['meteor_pred'])
                    bertscore_result = self.metric_bertscore.compute(predictions=[row['zh']], references=ref_bert, lang="zh")
                    cmeteor_result = self.metric_cmeteor.compute(predictions=[row['zh']], references=[row['ref']])
                    comet_result = self.metric_comet.compute(predictions=[split_hypo], references=[split_ref], sources=[split_src])
                    character_result = self.metric_character.compute(references=[split_ref], predictions=[split_hypo])
    
                    report_data.append({
                        'yue': row['yue'],
                        'zh': row['zh'],
                        'ref': row['ref'],
                        'bleu': round(bleu_result['score'], 4),
                        'chrf': round(chrf_result['score'], 4),
                        'meteor': round(meteor_result, 4),
                        'bertscore': round(np.mean(bertscore_result["f1"]), 4),
                        'cmeteor': round(np.mean(cmeteor_result['meteor']), 4),
                        'comet': round(comet_result["scores"][0], 4),
                        'character': round(character_result["cer_score"], 4)
                    })
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout # restore original output

            df_report = pd.DataFrame(report_data)
            logger.info(f"BLEU: {df_report['bleu'].mean():.4f}")
            logger.info(f"chrF++: {df_report['chrf'].mean():.4f}")
            logger.info(f"METEOR: {df_report['meteor'].mean():.4f}")
            logger.info(f"BERTScore: {df_report['bertscore'].mean():.4f}")
            logger.info(f"ChineseMETEOR: {df_report['cmeteor'].mean():.4f}")
            logger.info(f"COMET: {df_report['comet'].mean():.4f}")
            logger.info(f"characTER: {df_report['character'].mean():.4f}")
            return df_report

        elif evaluate_mode == 'corpus':
            references_bleu_chrf = [[ref] for ref in df['ref']]
            meteor_pred = self._chinese_tokenize("".join(df['zh']))
            meteor_ref = self._chinese_tokenize("".join(df['ref']))

            split_hypo = ' '.join(self._split_keyword("".join(df['zh'])))
            split_ref = ' '.join(self._split_keyword("".join(df['ref'])))
            split_src = ' '.join(self._split_keyword("".join(df['yue'])))

            bleu_result = self.metric_bleu.compute(predictions=df['zh'].tolist(), references=references_bleu_chrf, tokenize="zh")
            chrf_result = self.metric_chrf.compute(predictions=df['zh'].tolist(), references=references_bleu_chrf, word_order=2)
            meteor_result = meteor_score([meteor_ref], meteor_pred)
            bertscore_result = self.metric_bertscore.compute(predictions=df['zh'].tolist(), references=df['ref'].tolist(), lang="zh")
            
            original_stdout = sys.stdout # store original output to suppress warnings
            sys.stdout = open(os.devnull, 'w')
            try:
                cmeteor_result = self.metric_cmeteor.compute(
                    predictions=df['zh'].tolist(),
                    references=df['ref'].tolist()
                )
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout # restore original output
            comet_result = self.metric_comet.compute(predictions=[split_hypo], references=[split_ref], sources=[split_src])
            character_result = self.metric_character.compute(references=[split_ref], predictions=[split_hypo])
            
            scores = {
                'bleu': round(bleu_result['score'], 4),
                'chrf': round(chrf_result['score'], 4),
                'meteor': round(meteor_result, 4),
                'bertscore': round(np.mean(bertscore_result['f1']), 4),
                'cmeteor': round(np.mean(cmeteor_result['meteor']), 4),
                'comet': round(comet_result["scores"][0], 4),
                'character': round(character_result["cer_score"], 4)
            }
            logger.info(f"BLEU: {scores['bleu']:.4f}")
            logger.info(f"chrF++: {scores['chrf']:.4f}")
            logger.info(f"METEOR: {scores['meteor']:.4f}")
            logger.info(f"BERTScore: {scores['bertscore']:.4f}")
            logger.info(f"ChineseMETEOR: {scores['cmeteor']:.4f}")
            logger.info(f"COMET: {scores['comet']:.4f}")
            logger.info(f"characTER: {scores['character']:.4f}")
            return scores
        else:
            logger.error("Unknown evaluation mode")
            return None

    def evaluate_dataset_unbabel(self, pickle_path, evaluate_mode='sentence'):
        """Evaluate dataset using Unbabel's COMET models."""
        dataset = self.prepare_dataset(pickle_path)
        unbabel_data = [
            {"src": ex['translation']['yue'].strip(), 
             "mt": ex['translation']['zh'].strip(), 
             "ref": ex['translation']['ref'].strip()}
            for ex in dataset
        ]

        logger.info(f"Test Size: {len(unbabel_data)}")

        # Load COMET models with suppressed logs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xcometxl = load_from_checkpoint(download_model("Unbabel/XCOMET-XL"))
            cometkiwi = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
            cometqe = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da"))

        # Batch predictions
        xcometxl_result = xcometxl.predict(unbabel_data, batch_size=8, gpus=1, progress_bar=False)
        cometkiwi_result = cometkiwi.predict(unbabel_data, batch_size=8, gpus=1, progress_bar=False)
        cometqe_result = cometqe.predict(unbabel_data, batch_size=8, gpus=1, progress_bar=False)

        if evaluate_mode == 'sentence':
            logger.info("Sentence Level")
            logger.info(f"XCOMET-XL: {np.mean(xcometxl_result.scores):.4f}")
            logger.info(f"XCOMETKiwi: {np.mean(cometkiwi_result.scores):.4f}")
            logger.info(f"XCOMET-QE: {np.mean(cometqe_result.scores):.4f}")
        elif evaluate_mode == 'corpus':
            logger.info("Corpus Level")
            logger.info(f"XCOMET-XL: {xcometxl_result.system_score:.4f}")
            logger.info(f"COMETKiwi: {cometkiwi_result.system_score:.4f}")
            logger.info(f"COMET-QE: {cometqe_result.system_score:.4f}")
        else:
            logger.error("Unknown evaluate mode")
            return None

    def evaluate_dataset_gen3(self, pickle_path):
        report_data = []
        
        # Load and prepare dataset
        dataset = self.prepare_dataset(pickle_path)
        df = pd.DataFrame([ex['translation'] for ex in dataset])
        df = df[['yue', 'zh', 'ref']].apply(lambda x: x.str.strip())
        
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating scores"):
            ref_bleu_chrf = [[row['ref']]]
            ref_bert = [row['ref']]

            split_hypo = ' '.join(self._split_keyword(row['zh']))
            split_ref = ' '.join(self._split_keyword(row['ref']))

            bleu_result = self.metric_bleu.compute(predictions=[row['zh']], references=ref_bleu_chrf, tokenize="zh")
            chrf_result = self.metric_chrf.compute(predictions=[row['zh']], references=ref_bleu_chrf, word_order=2)
            ter_result = self.metric_ter.compute(predictions=[split_hypo], references=[split_ref])
            bleurt_result = self.metric_bleurt.score(references=[row['ref']], candidates=[row['zh']])
            
            report_data.append({
                'yue': row['yue'],
                'zh': row['zh'],
                'ref': row['ref'],
                'bleu': round(bleu_result['score'], 4),
                'chrf': round(chrf_result['score'], 4),
                'ter': round(ter_result['score'], 4),
                'bleurt': round(bleurt_result[0], 4),
            })
        df_report = pd.DataFrame(report_data)
        logger.info(f"BLEU: {df_report['bleu'].mean():.4f}")
        logger.info(f"chrF++: {df_report['chrf'].mean():.4f}")
        logger.info(f"TER: {df_report['ter'].mean():.4f}")
        logger.info(f"BLEURT: {df_report['bleurt'].mean():.4f}")
        return df_report
        
    def prepare_dataset(self, pickle_path):
        """Load and prepare dataset from pickle file."""
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        return [{'translation': {'yue': row['yue'], 'zh': row['zh'], 'ref': row['ref']}} 
                for _, row in df.iterrows()]