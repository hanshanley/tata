import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys
import torch
import gc
import math
from utils.datasets_util import load_stance_dataset, FullStanceDataset
TQDM_DISABLE=False



print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
device = torch.cuda.current_device()
torch.cuda.empty_cache()


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    
HIDDEN_SIZE = 768
N_CLASSES = 3

class StanceClassifier(torch.nn.Module):
    '''
    This module performs stance classification using deBERTa embeddings and TAW embeddings.
    '''
    def __init__(self, config):
        super(StanceClassifier, self).__init__()

        self.topic_aware =AutoModel.from_pretrained('microsoft/deberta-v3-base',cache_dir = 'cache')
        self.topic_aware.load_state_dict(torch.load('taw_embeddings.pt'))
        for param in self.topic_aware.parameters():
            param.requires_grad = False
        self.deberta = AutoModel.from_pretrained('microsoft/deberta-v3-base',cache_dir = 'cache')
        for param in self.deberta.parameters():
            param.requires_grad = True

        self.scale = math.sqrt(HIDDEN_SIZE)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.linear_out = torch.nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.linear_out.requires_grad = True
        self.batchnorm = torch.nn.BatchNorm1d(HIDDEN_SIZE)
        self.linear = torch.nn.Linear(HIDDEN_SIZE, N_CLASSES)
        self.linear.requires_grad = True
        self.relu = torch.nn.ReLU()
        
        
        self.attention_encoding = torch.nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.attention_encoding.requires_grad = True
        self.text_linear_layer = torch.nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        
        self.dropout_2 = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.linear_2 = torch.nn.Linear(2*HIDDEN_SIZE,HIDDEN_SIZE)
        self.linear_2.requires_grad = True
        self.batchnorm_2= torch.nn.BatchNorm1d(HIDDEN_SIZE)
        self.batchnorm_2.requires_grad = True
        self.relu_2 = torch.nn.ReLU()
        self.linear_3 = torch.nn.Linear(HIDDEN_SIZE,N_CLASSES)
        self.linear_3.requires_grad = True
        

    def attention(self, inputs, query):
        sim = torch.einsum('blh,bh->bl', inputs, query) / self.scale  # (B, L)
        att_weights = torch.nn.functional.softmax(sim, dim=1)  # (B, L)
        context_vec = torch.einsum('blh,bl->bh', inputs, att_weights)  # (B,H)
        return context_vec

    def predict_stance(self,input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2,
                      input_ids_3, attention_mask_3):
        text_token_emebddings = self.deberta(input_ids_3, attention_mask_3)['last_hidden_state'][:,0,:]
        topic_token_emebddings = self.deberta(input_ids_2, attention_mask_2)['last_hidden_state'][:,0,:]
        topic_aware_encoding =self.topic_aware(input_ids_3,attention_mask_3)['last_hidden_state'][:,0,:]
        text_topic_aware_attention = self.attention(topic_aware_encoding.unsqueeze(1), self.attention_encoding(topic_token_emebddings))
        
        output2 = torch.cat((text_token_emebddings,text_topic_aware_attention), axis = -1)
        output2 = self.dropout_2(output2)
        output2 = self.linear_2(output2)
        output2  = self.relu_2(output2)
        output2 = self.batchnorm_2(output2)
        output2 = self.linear_3(output2) 

        return output2


def model_eval(dataloader, model, device):
    model = model.eval()  # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids_1, b_mask_1,b_ids_2,b_mask_2,b_ids_3,b_mask_3, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'],batch['token_ids_2'],
                                       batch['attention_mask_2'],batch['token_ids_3'],
                                       batch['attention_mask_3'], batch['labels'])
            ## Load ids and masks to device
            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            
            b_ids_3 = b_ids_3.to(device)
            b_mask_3 = b_mask_3.to(device)


            b_labels = b_labels.to(device)
            logits = model.predict_stance(b_ids_1, b_mask_1,b_ids_2,b_mask_2,b_ids_3,b_mask_3)
            preds = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(b_labels)

    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred)
    print(report)
    return acc, f1, prec,recall, y_pred, y_true



def save_model(model, optimizer, args, config, filepath,epoch):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath+str(epoch))
    print(f"save the model to {filepath}")




def train(args):
    device = torch.device('cuda') #if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    print('LOAD TRAIN')
    train_stance_data = load_stance_dataset( args.train_data)
    print('LOAD DEV')
    dev_stance_data = load_stance_dataset(args.dev_data)

    train_data = FullStanceDataset(train_stance_data, args)
    dev_data = FullStanceDataset(dev_stance_data, args)


    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=train_data.collate_fn)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=dev_data.collate_fn)
    
    
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': 3,
              'hidden_size': 768,
              'data_dir': '.',}
    config = SimpleNamespace(**config)


    ## initialize model 
    new_model = StanceClassifier(config)
    new_model = new_model.to(device)

    lr = args.lr
    optimizer =  torch.optim.AdamW(new_model.parameters(), lr=args.lr)
    best_dev_acc = 0
    best_dev_f1 = 0 

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        new_model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            new_model = new_model.train()
            b_ids_1, b_mask_1,b_ids_2,b_mask_2,b_ids_3,b_mask_3, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'],batch['token_ids_2'],
                                       batch['attention_mask_2'],batch['token_ids_3'],
                                       batch['attention_mask_3'], batch['labels'])
            ## Load ids and masks to device
            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_ids_3 = b_ids_3.to(device)
            b_mask_3 = b_mask_3.to(device)
            b_labels = b_labels.to(device)


            optimizer.zero_grad()
            logits = new_model.predict_stance(b_ids_1, b_mask_1,b_ids_2,b_mask_2,b_ids_3,b_mask_3)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
        train_loss = train_loss / (num_batches)
        print(train_loss)
        dev_acc, dev_f1, dev_prec, dev_recall, *_ = model_eval(dev_dataloader, new_model, device)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(new_model, optimizer, args, config, args.filepath,epoch)
        print(f"Epoch {epoch}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f},  dev prec :: {dev_prec :.3f},  dev recall :: {dev_recall :.3f}")



class Object(object):
    pass
args = Object()
args.train_data = 'NAME OF TRAINING DATASET.jsonl'
args.dev_data = 'NAME OF DEV DATASET.jsonl'
args.seed = 11711
args.epochs = 30
args.batch_size = 64
args.hidden_dropout_prob = 0.30
args.lr = 1e-5 
args.tokenizer = 'microsoft/deberta-v3-base'


args.filepath = f'{args.epochs}-{args.lr}-name-of-save-point.pt' # save path
seed_everything(args.seed)  # fix the seed for reproducibility
train(args)

