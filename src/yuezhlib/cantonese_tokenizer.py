import json
import os
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, normalizers
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer
from .pyctokenizer import PyCantonesePreTokenizer
# https://github.com/huggingface/transformers/blob/0cdcd7a2b319689d75ae4807cfb7b228aa322f83/src/transformers/models/roformer/tokenization_roformer_fast.py#L91

class CantoneseTokenizerFast(PreTrainedTokenizerFast):
    @staticmethod
    def _attach_custom_pretokenizer(tokenizer):
        tokenizer._tokenizer.pre_tokenizer = PreTokenizer.custom(PyCantonesePreTokenizer())

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        
        backend_normalizer = self.backend_tokenizer.normalizer
        if backend_normalizer is not None:
            normalizer_state = json.loads(backend_normalizer.__getstate__())
            if (
                normalizer_state.get("lowercase", do_lower_case) != do_lower_case
                or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            ):
                normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
                normalizer_state["lowercase"] = do_lower_case
                normalizer_state["strip_accents"] = strip_accents
                self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        self._attach_custom_pretokenizer(self)

        self.do_lower_case = do_lower_case
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokenizer"].pre_tokenizer = BertPreTokenizer()
        return state
    
    def __setstate__(self, d):
        self.__dict__ = d
        self._attach_custom_pretokenizer(self)

    # Backward-compatible aliases for any existing internal callers.
    def __get_state__(self):
        return self.__getstate__()

    def __set_state__(self, d):
        self.__setstate__(d)
        
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]
            
        return output
    
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
    
    def save_pretrained(
        self,
        save_directory,
        legacy_format=None,
        filename_prefix=None,
        push_to_hub=False,
        **kwargs,
    ):
        original_pre_tokenizer = self.backend_tokenizer.pre_tokenizer
        self._tokenizer.pre_tokenizer = BertPreTokenizer()
        try:
            return super().save_pretrained(save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs)
        finally:
            self._tokenizer.pre_tokenizer = original_pre_tokenizer

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        tokenizer = super().from_pretrained(*args, **kwargs)
        model_dir = args[0] if args else kwargs.get("pretrained_model_name_or_path")
        if model_dir and os.path.isdir(model_dir):
            tokenizer_json = os.path.join(model_dir, "tokenizer.json")
            if os.path.exists(tokenizer_json):
                tokenizer._tokenizer = Tokenizer.from_file(tokenizer_json)
        cls._attach_custom_pretokenizer(tokenizer)
        return tokenizer