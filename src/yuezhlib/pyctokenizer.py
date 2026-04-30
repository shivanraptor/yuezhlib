import pycantonese
from typing import List
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.decoders import Decoder
from tokenizers import (
    decoders,
    models,
    trainers,
    processors,
    Tokenizer,
    Regex, 
    NormalizedString, 
    PreTokenizedString
)

class CustomNormalizer:
    def normalize(self, normalized: NormalizedString):
        # Most of these can be replaced by a `Sequence` combining some provided Normalizer,
        # (ie Sequence([ NFKC(), Replace(Regex("\s+"), " "), Lowercase() ])
        # and it should be the prefered way. That being said, here is an example of the kind
        # of things that can be done here:
        normalized.nfkc()
        normalized.filter(lambda char: not char.isnumeric())
        normalized.replace(Regex("\s+"), " ")
        normalized.lowercase()

class PyCantonesePreTokenizer:
    def pyc_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        # handle NoneType properly
        if normalized_string is None:
            noramlized_string = NormalizedString("")
        splits = []
        try:
            for token, start, stop in self.custom_segment(str(normalized_string)):
                splits.append(normalized_string[start:stop])
        except TypeError as te:
            print("PreTokenizer pyc_split TypeError:", te)
            print(normalized_string)
        return splits
    
    def custom_segment(self, input_str: str):
        #segmenter = Segmenter() # possible attributes: disallow, allow, max_word_length
        # if not used, max_word_length = 5 is used
        pyseg = pycantonese.segment(input_str) #, cls=segmenter
        tokenized = []
        cursor = 0
        for word in pyseg:
            start = input_str.find(word, cursor)
            end = start + len(word)
            if start > -1:
                tokenized.append((word, start, end))
                cursor = end
        return tokenized

    def pre_tokenize(self, pretok: PreTokenizedString):
        # Let's call split on the PreTokenizedString to split using `self.pyc_split`
        if pretok is not None:
            try:
                pretok.split(self.pyc_split)
            except TypeError as te:
                print("PreTokenizer pre_tokenize TypeError:", te)
                print(pretok)
    
class CustomDecoder:
    def decode(self, tokens: List[str]) -> str:
        return "".join(tokens)
    
    def decode_chain(self, tokens: List[str]) -> List[str]:
        return [f" {t}" for t in tokens]