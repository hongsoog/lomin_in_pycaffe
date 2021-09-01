import re
import torch
import torch.nn.functional as F

from collections import defaultdict


class Converter(object):
    def __init__(self, character):
        list_token = ['[GO]', '[s]', '[others]']
        list_character = list(character)
        self.character = list_token + list_character

    def decode(self, text_index):
        text = ''.join([self.character[i] for i in text_index if i >= 0])
        return text