import io
import os
import re
import stopwordsiso
from stopwordsiso import stopwords
import opencc
import pandas as pd
import translators as ts
import argostranslate.package
import argostranslate.translate

class CantonesePreprocessor(object):
    def __init__(self, cantonese_mappings_path, lexicon_path):
        self.cantonese_mappings = self._load_cantonese_mappings(cantonese_mappings_path)
        self.lexicon = pd.read_csv(lexicon_path, header=0)
        self.map_replace = [['左', '咗'], ['既', '嘅'], ['黎', '嚟'], ['訓', '瞓'], ['甘', '咁'], ['下', '吓']]
        self.map_punctuations = [
            [', ', '，'], [',', '，'], ['?', '？'], ['’', "'"], ['(', "（"], 
            [')', "）"], ['『', "「"], ['』', "」"], ['“', "「"], ['”', "」"], 
            ['‘', "「"], ['‘', "」"], ['（）', ''], [':', '：'], [';', '；']
        ]
        self.consecutive_symbols = ['#', '>', '？', '！', '，']
        
        # S. Chinese to T. Chinese converter
        self.converter = opencc.OpenCC('s2t.json')
        
        # Prepare T. Chinese Stopwords
        stopwords_tc = []
        for i in stopwords(["zh"]):
            stopwords_tc.append(self.converter.convert(i))
        self.stopwords = stopwords_tc
        

    # Helper functions
    def preprocess_text(self, text, remove_stopwords=False, replace_punctuations=False, replace_typo=False, replace_codemixed=False):
        # remove consecutive new lines
        text = re.sub(r'(\n\s*)+', '\n', text)

        # lowercase
        text = text.lower()

        # re-map punctuations
        if replace_punctuations:
            for mapping in self.map_punctuations:
                text = text.replace(mapping[0], mapping[1])
            text = self._replace_quotations(text, '"', '「', '」')
        
        # Map Cantonese and adjust Typo
        if replace_typo:
            text = self._process_cantonese_spacing(text)
            text = self._process_cantonese_typo(text)

        # remove consecutive symbols
        for symbol in self.consecutive_symbols:
            text = self._replace_consecutive_symbols(text, symbol)
        
        # remove stopwords
        if remove_stopwords:
            text = self._remove_stopwords(text)

        # remove all whitespaces if the whole line does not contain English
        if re.search('[a-z]', text) is None:
            text = text.replace(' ', '')
        else:
            if replace_codemixed:
                text = self._replace_codemixed(text)

        # Translate S. Chinese to T. Chinese
        text = self.converter.convert(text)
        
        # strip leading and trailing spaces 
        text = text.rstrip('\n')
        text = text.strip()

        return text


    # Internal functions
    def _replace_codemixed(self, text, offline=True):
        regex = r"([a-z]+)"
        matches = re.finditer(regex, text, re.MULTILINE)
        result = text
        for matachNum, match in enumerate(matches, start=1):
            if offline:
                result = result.replace(match.group(), argostranslate.translate.translate(match.group().lower(), "en", "zh"))
            else:
                result = result.replace(match.group(), ts.translate_text(match.group(), translator='google', to_language='zh'))
        return result
        
    def _replace_quotations(self, text, search, replace_open, replace_close):
        find = text.find(search)
        # loop util we find no match
        cnt = 1
        while find != -1:
            # if i  is equal to nth we found nth matches so replace
            if cnt % 2 == 1:
                text = text[:find] + replace_open + text[find + len(search):]
            else:
                text = text[:find] + replace_close + text[find + len(search):]
            find = text.find(search, find + len(search) + 1)
            cnt += 1
        return text

    def _replace_vocab(self, text, search_char, replace_char):
        found = False
        for index, row in self.lexicon[self.lexicon['Vocab'].str.contains(search_char)].iterrows():
            if row['Vocab'] in text and not row['Vocab'] == search_char:
                found = True
        if not found:
            text = text.replace(search_char, replace_char)
        return text

    def _replace_consecutive_symbols(self, text, search_symbol):
        return search_symbol.join(item for item in text.split(search_symbol) if item)

    def _remove_stopwords(self, text):
        for stop_word in self.stopwords:
            text = text.replace(stop_word, '')
        return text
        
    def _process_cantonese_spacing(self, text):
        for mapping in self.cantonese_mappings:
            # only replace those have a space surrounded, to avoid changing part of a vocab
            text = text.replace(mapping[0] + ' ', mapping[1])
            text = text.replace(' ' + mapping[0], mapping[1])
        return text

    def _process_cantonese_typo(self, text):
        for mapping in self.map_replace:
            text = self._replace_vocab(text, search_char=mapping[0], replace_char=mapping[1])
        return text

    def _load_cantonese_mappings(self, path):
        canton_mappings = []
        with open(path, 'r') as f:
            for line in f:
                canton_mappings.append(line.rstrip().split('\t'))
        return canton_mappings

        



