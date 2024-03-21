import copy
import random
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from transformers import AutoTokenizer, AutoModel
from utils.datasets_util import  load_stance_dataset, TAGStanceDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import gc
TQDM_DISABLE=False
BATCH_SIZE = 32
NUM_EPOCHS = 10

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name(0)
device = torch.cuda.current_device()
torch.cuda.empty_cache()
gc.collect()

## Trainer class for training TAG embeddings
## Code adapted from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses
class TAGTrainer(nn.Module):
    def __init__(
        self,
        net,
    ):
        super().__init__()
        self.net = net
        for param in self.net.parameters():
            param.requires_grad = True
        self.dropout = torch.nn.Dropout(0.3)
        self.temperature = 0.07
    
    def get_mask(self,labels):
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        
        distinct_indices = i_not_equal_j
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        valid_labels =  i_equal_j
        return distinct_indices & valid_labels
     

    def forward(
        self,
        b_ids_1,
        b_masks_1,
        labels):
       
        embedding = self.net(b_ids_1.to(device),b_mask_1.to(device))['last_hidden_state'][:,0,:]
        embedding = self.dropout(embedding)
        mask = self.get_mask(labels.clone().detach()).squeeze().to(device)
        dot = torch.mm(embedding, embedding.t())
        square_norm2 = torch.diag(dot)
        bottom = torch.sqrt(square_norm2).unsqueeze(0)*torch.sqrt(square_norm2).unsqueeze(1)
        bottom_mask =  torch.zeros(embedding.shape[0], embedding.shape[0])
        bottom_mask =1 -bottom_mask.fill_diagonal_(1).to(device)
        top = torch.sum(mask*torch.exp(dot/(self.temperature*bottom)),axis=1)+1e-16 # For stability 
        new_bottom = torch.sum(bottom_mask *torch.exp(dot/(self.temperature*bottom)),axis=1)+1e-16 # For stability 
        loss = -1*torch.log(top/new_bottom)
        return loss.sum()


class Object(object):
    pass
args = Object()
args.tokenizer = 'microsoft/deberta-v3-base'
args.model = 'microsoft/deberta-v3-base'

net =  AutoModel.from_pretrained(args.model,cache_dir = 'cache')
net.to(device)
learner = TAGTrainer(net)
learner.to(device)
opt = torch.optim.AdamW(learner.parameters(), lr=1e-5)


## Load in TAG training dataset 
train_stance_data = load_stance_dataset('NAME OF TRAIN DATASET.jsonl')
dev_stance_data = load_stance_dataset('NAME OF DEV DATASET.json')

dev_data = TAGStanceDataset(dev_stance_data, None)
dev_dataloader = DataLoader(dev_stance_data, shuffle=True, batch_size=BATCH_SIZE,
                                      collate_fn=dev_data.collate_fn)
train_data = TAGStanceDataset(train_stance_data, None)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE,
                                      collate_fn=train_data.collate_fn)

previous_loss = 199999
for epoch in range(0, NUM_EPOCHS):
    learner= learner.train()
    net = net.train()
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        learner = learner.train()
        b_ids_1, b_mask_1,b_labels = (batch['token_ids_1'],
                                   batch['attention_mask_1'], batch['labels'])
        
        opt.zero_grad()
        loss = learner(b_ids_1, b_mask_1, b_labels)/BATCH_SIZE
        loss.backward()
        opt.step()
        
    ## Check Dev loss and save model  
    learner = learner.eval()
    net = net.eval()
    total_loss = []
    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_mask_1,labels = (batch['token_ids_1'],
                               batch['attention_mask_1'], batch['labels'])
            opt.zero_grad()
            loss = learner(b_ids_1, b_mask_1, labels)
            total_loss.append(loss.item())
    
    print(np.average(total_loss))
    if previous_loss > np.average(total_loss):
        print('SAVING')
        torch.save(learner.net.state_dict(), 'tag_embeddings.pt')
        previous_loss = np.average(total_loss)
    learner = learner.train()
    net = net.train()
   
