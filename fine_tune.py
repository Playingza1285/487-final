import time
import copy
from typing import List

from nltk.tokenize import word_tokenize
from tqdm import tqdm
from gensim.models import KeyedVectors
import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
import gensim
import pandas as pd
import transformers
from torch.nn.utils.rnn import pad_sequence

class POSTagDataset(Dataset):
    """Dataset for POS tagging"""

    def __init__(self, data: pd.DataFrame, tokenizer: transformers.AutoTokenizer):
        super().__init__()
        self.tags = {"left": 0, "center": 1, "right": 2}
        self.data = []
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dp = self.data.iloc[idx]
        input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + [t.lower() for t in dp["content"]] + ["[SEP]"])
        # pos_ids = [1] + [self.pos_tags[pos] for pos in dp["pos_tags"]] + [2]
        tokens = ["[CLS]"] + dp["content"] + ["[SEP]"]
        # item = {"input_ids": input_ids, "pos_ids": pos_ids, "tokens": tokens}
        item = {"input_ids": input_ids, "tokens": tokens, "bias": self.tags[dp["bias"]]}
        return item

def basic_collate_fn(batch):
    inputs = None
    outputs = None

    input_ids = pad_sequence([torch.tensor(b["input_ids"]) for b in batch], True)
    if (input_ids.size(dim=1) > 150):
        input_ids = torch.narrow(input_ids, 1, 0, 150) # 512)
    att_mask = (input_ids != 0).type(torch.IntTensor)
    att_mask = 1*(input_ids != 0)
    inputs = {"input_ids": input_ids, "attention_mask": att_mask}
    outputs = torch.t(torch.tensor([b["bias"] for b in batch]))

    return inputs, outputs

class DistilBertForTokenClassification(nn.Module):
    def __init__(self, distil_bert: transformers.DistilBertModel, hidden_dim: int, num_pos: int):
        super().__init__()
        self.distil_bert = distil_bert
        self.llayer = nn.Linear(hidden_dim, num_pos)
        self.hidden_dim = hidden_dim
        self.num_pos = num_pos
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        bert_out = self.distil_bert(input_ids.type(torch.int), attention_mask.type(torch.int))
        output = self.llayer(bert_out.last_hidden_state[:, 0])
        return output

def get_loss_fn():
    return nn.CrossEntropyLoss(ignore_index=0)

def calculate_loss(scores, labels, loss_fn):
    return loss_fn(scores, labels)

def get_optimizer(net, lr, weight_decay):
    return optimizer.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

def get_hyper_parameters():
    lr = [10**p for p in range(-5, -4)]
    weight_decay = [10**p for p in range(-1, 0)]
    lr.reverse()
    weight_decay.reverse()
    return lr, weight_decay


def train_model(net, trn_loader, val_loader, optim, num_epoch=50, collect_cycle=30,
        device='cpu', verbose=True):
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_accuracy = None, 0
    loss_fn = get_loss_fn()
    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in range(num_epoch):
        net.train()
        for inputs, bias in trn_loader:
            num_itr += 1
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            bias = bias.to(device)
            optim.zero_grad()
            scores = net(**inputs)
            loss = calculate_loss(scores, bias, loss_fn)
            loss.backward()
            optim.step()
            if num_itr % collect_cycle == 0:  # Data collection cycle
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item()
                ))

        accuracy, loss = get_validation_performance(net, loss_fn, val_loader, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation accuracy: {:.4f}".format(accuracy))
            print("Validation loss: {:.4f}".format(loss))
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(net)
            best_accuracy = accuracy
    
    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_loss_ind': val_loss_ind,
             'accuracy': best_accuracy,
    }

    return best_model, stats

def get_validation_performance(net, loss_fn, data_loader, device):
    net.eval()
    y_true = [] # true labels
    y_pred = [] # predicted labels
    total_loss = [] # loss for each batch

    with torch.no_grad():
        for inputs, bias in data_loader:
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            bias = bias.to(device)
            loss = None # loss for this batch
            pred = None # predictions for this battch

            scores = net(**inputs)
            loss = calculate_loss(scores, bias, loss_fn)
            pred = torch.argmax(scores, dim=1)

            total_loss.append(loss.item())
            y_true.append(bias)
            y_pred.append(pred)
    
    accuracy = 0
    count = 0
    for pred, true in zip(y_pred, y_true):
        for i in range(len(pred)):
            accuracy += 1*(pred[i] == true[i])
            count += 1
    accuracy = accuracy / count

    total_loss = sum(total_loss) / len(total_loss)
    
    return accuracy, total_loss

def make_prediction(net, data_loader, device):
    net.eval()
    y_pred = [] # predicted labels
    y_true = [] # true labels
    errors = []
    with torch.no_grad():
        for inputs, bias in data_loader:
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            bias_to_text = {0: "left", 1: "center", 2: "right"}

            scores = net(**inputs)
            pred = torch.argmax(scores, dim=1)
            for i in range(len(pred)):
                y_pred.append(bias_to_text[pred[i].item()])
                y_true.append(bias_to_text[bias[i].item()])
                if (pred[i] != bias[i]):
                    errors.append(i)
    return y_true, y_pred, errors


def plot_loss(stats):
    """Plot training loss and validation loss."""
    plt.plot(stats['train_loss_ind'], stats['train_loss'], label='Training loss')
    plt.plot(stats['val_loss_ind'], stats['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.show()
