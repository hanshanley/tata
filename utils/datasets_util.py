import csv
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch import optim
from datasets import load_dataset
import json
import re

def load_constrative_topic_stance_batch(file_name):
    stance_dataset = []
    with open(file_name, 'r') as fp:
        labels = set()
        for line in fp:
            line = json.loads(line)
            stance_dataset.append((line[0],
                                   line[2],
                                    line[1],
                                    line[3]))
    return stance_dataset


class TAWStanceDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer =AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        query = [x[0]+ " [SEP] "+ x[1].lower()  for x in data]
        context = [x[2]+ " [SEP] "+ x[3].lower() for x in data]
        
        queryEncdoing = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True,max_length=512)
        contextEncoding = self.tokenizer(context, return_tensors='pt', padding=True, truncation=True,max_length=512)

        token_ids = torch.LongTensor(queryEncdoing['input_ids'])
        attention_mask = torch.LongTensor(queryEncdoing['attention_mask'])

        token_ids2 = torch.LongTensor(contextEncoding['input_ids'])
        attention_mask2 = torch.LongTensor(contextEncoding['attention_mask'])
        
        
        return (token_ids, attention_mask,
                token_ids2, attention_mask2)

    def collate_fn(self, all_data):
        (token_ids,  attention_mask,
         token_ids2, attention_mask2) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'attention_mask_2': attention_mask2,
            }

        return batched_data


def load_stance_dataset(file_name):
    stance_dataset = []
    with open(file_name, 'r') as fp:
        labels = set()
        for line in fp:
            line = json.loads(line)
            stance_dataset.append((line[0],
                                    line[1],
                                    int(line[2])))
    return stance_dataset


class TAGStanceDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer =AutoTokenizer.from_pretrained('microsoft/deberta-v3-base',use_fast=False)#

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        full = [x[0]+ " [SEP] "+ x[1].lower() for x in data]
        labels =  [int(x[2]) for x in data]
        fullEncoding = self.tokenizer(full, return_tensors='pt', padding=True, truncation=True,max_length=512)
        token_ids = torch.LongTensor(fullEncoding['input_ids'])
        attention_mask = torch.LongTensor(fullEncoding['attention_mask'])        
        labels = torch.LongTensor(labels)
        return (token_ids, attention_mask,
                labels)

    def collate_fn(self, all_data):
        (token_ids,  attention_mask,
         labels) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'labels': labels,
            }
        return batched_data    
  
class FullStanceDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer =AutoTokenizer.from_pretrained(args.tokenizer,use_fast=False)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        content = [str(x[0])  for x in data]
        topic = [str(x[1]).lower() for x in data]
        content_topic = [str(x[0])+ " [SEP] "+ str(x[1]).lower() for x in data]
        labels =  [x[2] for x in data]
        
        contentEncdoing = self.tokenizer(content, return_tensors='pt', padding=True, truncation=True,max_length=512)
        topicEncoding = self.tokenizer(topic, return_tensors='pt', padding=True, truncation=True,max_length=512)
        content_topicEncoding = self.tokenizer(content_topic, return_tensors='pt', padding=True, truncation=True,max_length=512)

        token_ids = torch.LongTensor(contentEncdoing['input_ids'])
        attention_mask = torch.LongTensor(contentEncdoing['attention_mask'])

        token_ids2 = torch.LongTensor(topicEncoding['input_ids'])
        attention_mask2 = torch.LongTensor(topicEncoding['attention_mask'])
        
        
        token_ids3 = torch.LongTensor(content_topicEncoding['input_ids'])
        attention_mask3 = torch.LongTensor(content_topicEncoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return (token_ids, attention_mask,
                token_ids2, attention_mask2,
                token_ids3,attention_mask3,
                labels)

    def collate_fn(self, all_data):
        (token_ids,  attention_mask,
         token_ids2, attention_mask2,
         token_ids3, attention_mask3,
         labels) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'attention_mask_2': attention_mask2,
                'token_ids_3': token_ids3,
                'attention_mask_3': attention_mask3,
                'labels': labels,
            }

        return batched_data


