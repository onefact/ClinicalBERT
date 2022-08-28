from transformers import BertConfig, BertModel, BertForPreTraining, BertTokenizerFast
from models.utils import NeuralNetwork
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from util import _truncate_seq_pair
from typing import Optional, List, Union


class BERTbase(nn.Module):
    def __init__(self, config_file: Union[str, None], tokenizer: BertTokenizerFast):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.special_tokens = [self.vocab["[SEP]"], self.vocab["[CLS]"], self.vocab["[MASK]"], self.vocab["[PAD]"]]
        if config_file:
            self.config = BertConfig.from_json_file(config_file)
        else:
            self.config = BertConfig()


class BERTforPreTraining(BERTbase):
    def __init__(self, config_file : Union[str, None], tokenizer : BertTokenizerFast, max_length : int):
        super().__init__(config_file, tokenizer)
        self.bert_model = BertForPreTraining(config=self.config)
        self.max_length = max_length

    def forward(self, **kwargs):
        outputs = self.bert_model(**kwargs)
        return outputs

    def collate_fn(self, examples):

        text_a = [x['text_a'] for x in examples]
        text_b = [x['text_b'] for x in examples]

        next_sentence_labels = torch.tensor([x['next_sentence_label'] for x in examples], dtype=torch.long)

        tokenized = self.tokenizer(text=text_a, text_pair=text_b, max_length=self.max_length, padding='longest', truncation=True, return_special_tokens_mask=True, return_tensors='pt')

        input_ids, masked_labels = self.mask_words(tokenized['input_ids'], tokenized['special_token_mask'])
        tokenized['input_ids'] = input_ids
        tokenized['labels'] = masked_labels
        tokenized['next_sentence_label'] = next_sentence_labels

        return tokenized

    def mask_words(self, inputs, special_tokens_mask):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.vocab["[MASK]"]

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_peaks = torch.randint(len(self.vocab), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_peaks[indices_random]

        return inputs, labels