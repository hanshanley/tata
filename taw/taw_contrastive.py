import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T
from datasets_util import NonContrastiveStanceDataset
from datasets_util import StanceDataset

from datasets_util import  load_constrative_topic_Stance_batch, FullTopicContrastiveStanceDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
TQDM_DISABLE=False




train_stance_data = load_constrative_topic_Stance_batch('train_filtered_news_topic_dataset.jsonl')
import random
random.shuffle(train_stance_data)


# In[4]:


dev_stance_data = load_constrative_topic_Stance_batch('dev_filtered_news_topic_dataset.jsonl')
import random
BATCH_DATA = 32
class Object(object):
    pass
args = Object()
args.tokenizer = 'microsoft/deberta-v3-base'
random.shuffle(dev_stance_data)
dev_data = FullTopicContrastiveStanceDataset(dev_stance_data, args)
dev_dataloader = DataLoader(dev_stance_data, shuffle=True, batch_size=BATCH_DATA,
                                      collate_fn=dev_data.collate_fn)


# In[5]:


# In[90]:


train_data = FullTopicContrastiveStanceDataset(train_stance_data, args)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_DATA,
                                      collate_fn=train_data.collate_fn)


# In[11]:


net = None
learner = None
import torch
import gc
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name(0)
device = torch.cuda.current_device()
torch.cuda.empty_cache()
gc.collect()
scale = 20.0

# In[101]:

loss_func = torch.nn.CrossEntropyLoss()
# main class
class TawContrastiveLearning(nn.Module):
    def __init__(
        self,
        net,
        projection_hidden_size = 768,
        use_momentum = True
    ):
        super().__init__()
        self.net = net
        for param in self.net.parameters():
            param.requires_grad = True
        self.to(device)
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-9)
    def _get_triplet_mask(self,labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = (i_not_equal_k) & j_not_equal_k


        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = ~i_equal_k & i_equal_j

        return valid_labels #& distinct_indices
    def mean_pooling(self,token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
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
        #scores = torch.stack([
        #        self.cos_sim(
        #            a_i.reshape(1, a_i.shape[0]), p
        #        ) for a_i in a])
        #    # get label(s) - we could define this before if confident of consistent batch sizes
        labels = torch.tensor(range(len(a)), dtype=torch.long, device=device)
        #print(labels)
        #print(scores)
        # get label(s) - we could define this before if confident of consistent batch sizes
        #labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        dot_product = torch.matmul(a, p.t())
        dot_product2 = torch.matmul(a, a.t())
        dot_product3 = torch.matmul(p, p.t())
        square_norm2 = torch.diag(dot_product2)
        square_norm3 = torch.diag(dot_product3)

        distances = square_norm2.unsqueeze(1) - 2.0 * dot_product + square_norm3.unsqueeze(0)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0#cosine_sims =  dot_product/(torch.sqrt(square_norm.unsqueeze(0))*torch.sqrt(square_norm.unsqueeze(1)))
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
        
        # and now calculate the loss
        #loss = loss_func(scores*scale, labels)
        #return loss
        return triplet_loss#loss.sum()
        


# In[12]:


#learner = None
#net = None
import torch
import gc
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name(0)
device = torch.cuda.current_device()
torch.cuda.empty_cache()
gc.collect()


# In[110]:


from transformers import AutoTokenizer, AutoModel

net =   AutoModel.from_pretrained('microsoft/deberta-v3-base',cache_dir = 'cache')


net.to(device)
net = net.train()


# In[112]:


learner = BYOL(net,use_momentum=False)


# In[113]:


opt = torch.optim.AdamW(learner.parameters(), lr=1e-5)


# In[114]:


import torch
import gc
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name(0)
device = torch.cuda.current_device()
torch.cuda.empty_cache()
gc.collect()


# In[115]:


# In[13]:


current_losses = []
import numpy as np
num = 0 
previous_loss = 199999
for epoch in range(0,1000):
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        learner = learner.train()
        b_ids_1, b_mask_1,b_ids_2, b_mask_2 = (batch['token_ids_1'],
                                   batch['attention_mask_1'], batch['token_ids_2'],
                                   batch['attention_mask_2'])
        opt.zero_grad()
        loss = learner(b_ids_1, b_mask_1,b_ids_2, b_mask_2)#/BATCH_DATA
        loss.backward()
        opt.step()
        current_losses.append(loss.item())
        #all_losses.append(loss.item())
        #learner.update_moving_average()
        num+=1
        print(np.average(current_losses))
        torch.cuda.empty_cache()
        gc.collect()
        if num %1000 ==0:
            print(np.average(current_losses))
            learner = learner.eval()
            total_loss = []
            with torch.no_grad():
                for batch in tqdm(dev_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                    b_ids_1, b_mask_1,b_ids_2, b_mask_2 = (batch['token_ids_1'],
                                   batch['attention_mask_1'], batch['token_ids_2'],
                                   batch['attention_mask_2'])
                    opt.zero_grad()
                    loss = learner(b_ids_1, b_mask_1,b_ids_2, b_mask_2)
                    total_loss.append(loss.item())
            learner = learner.train()
            net = net.train()
                    
            print(np.average(total_loss))
            if previous_loss > np.average(total_loss):
                print('SAVING')
                torch.save(net.state_dict(), 'final-euclidian-latent-topic-mnr-fixed-net11-2.pt')
                previous_loss = np.average(total_loss)
                
            current_losses = []
            
    torch.save(net.state_dict(), 'final-euclidian-latent-topic-mnr-fixed-net11-2-'+str(epoch)+'.pt')
    print(loss.item())

