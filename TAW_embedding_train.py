import copy
import random
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from datasets_util import NonContrastiveStanceDataset
from datasets_util import StanceDataset
from utils.datasets_util import  load_constrative_topic_stance_batch, TAWStanceDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
import numpy as np
from transformers import AutoTokenizer, AutoModel
import random
TQDM_DISABLE=False
BATCH_DATA = 32
NUM_EPOCHS = 10

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name(0)
device = torch.cuda.current_device()
torch.cuda.empty_cache()
gc.collect()


## Trainer class for training TAW embeddings
## Code adapted from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses
class TAWTrainer(nn.Module):
    def __init__(
        self,
        net,
    ):
        super().__init__()
        self.net = net
        for param in self.net.parameters():
            param.requires_grad = True
        self.to(device)
        
    def _get_triplet_mask(self,labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j are the same but k is distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)
        valid_labels = ~i_equal_k & i_equal_j
        return valid_labels
    
    def forward(
        self,
        b_ids_1,
        b_masks_1,
        b_ids_2,
        b_masks_2,
        return_embedding = False,
        return_projection = True
    ):
        a = self.net(b_ids_1.to(device),b_mask_1.to(device))['last_hidden_state'][:,0,:]#,b_mask_1.to(device))
        p = self.net(b_ids_2.to(device),b_mask_2.to(device))['last_hidden_state'][:,0,:]#,b_mask_2.to(device))
        labels = torch.tensor(range(len(a)), dtype=torch.long, device=device)

        dot_product = torch.matmul(a, p.t())
        dot_product2 = torch.matmul(a, a.t())
        dot_product3 = torch.matmul(p, p.t())
        square_norm2 = torch.diag(dot_product2)
        square_norm3 = torch.diag(dot_product3)

        distances = square_norm2.unsqueeze(1) - 2.0 * dot_product + square_norm3.unsqueeze(0)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 -mask) * torch.sqrt(distances)


        anchor_positive_dist = distances.unsqueeze(2)# 1- cosine_sims.unsqueeze(2)
        anchor_negative_dist = distances.unsqueeze(1)#1- cosine_sims.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist
  

        mask = self._get_triplet_mask(labels.to(device))
        triplet_loss = mask.float() * triplet_loss

        triplet_loss = F.relu(triplet_loss)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        num_valid_triplets = mask.sum()

        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        return triplet_loss
        


class Object(object):
    pass
args = Object()
args.tokenizer = 'microsoft/deberta-v3-base'
args.model = 'microsoft/deberta-v3-base'


net =   AutoModel.from_pretrained(args.model,cache_dir = 'cache')
net.to(device)
net = net.train()
learner = TAWTrainer(net)
opt = torch.optim.AdamW(learner.parameters(), lr=1e-5)


## Load in Topic News Dataset train and datasets 
train_stance_data = load_constrative_topic_stance_batch('NAME OF TRAIN DATASET.jsonl')
dev_stance_data = load_constrative_topic_stance_batch('NAME OF DEV DATASET.jsonl')

dev_data = TAWStanceDataset(dev_stance_data, args)
dev_dataloader = DataLoader(dev_stance_data, shuffle=True, batch_size=BATCH_DATA,
                                      collate_fn=dev_data.collate_fn)

train_data = TAWStanceDataset(train_stance_data, args)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_DATA,
                                      collate_fn=train_data.collate_fn)


previous_loss = 199999
for epoch in range(0,NUM_EPOCHS):
    learner = learner.train()
    net = net.train()
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        b_ids_1, b_mask_1,b_ids_2, b_mask_2 = (batch['token_ids_1'],
                                   batch['attention_mask_1'], batch['token_ids_2'],
                                   batch['attention_mask_2'])
        opt.zero_grad()
        loss = learner(b_ids_1, b_mask_1,b_ids_2, b_mask_2)
        loss.backward()
        opt.step()
    
    learner = learner.eval()
    net = net.eval()
    total_loss = []
    
    ## Check Dev loss and save model.  
    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids_1, b_mask_1,b_ids_2, b_mask_2 = (batch['token_ids_1'],
                           batch['attention_mask_1'], batch['token_ids_2'],
                           batch['attention_mask_2'])
            opt.zero_grad()
            loss = learner(b_ids_1, b_mask_1,b_ids_2, b_mask_2)
            total_loss.append(loss.item())
    if previous_loss > np.average(total_loss):
        print('SAVING')
        torch.save(net.state_dict(), 'taw_embeddings.pt')
        previous_loss = np.average(total_loss)
