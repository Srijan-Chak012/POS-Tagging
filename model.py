from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
import regex as re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import os
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
import random
from conllu import parse_incr
from warnings import filterwarnings

filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)

dataset = "dataset/English/"

train = dataset + 'en_atis-ud-train.conllu'
dev = dataset + 'en_atis-ud-dev.conllu'
test = dataset + 'en_atis-ud-test.conllu'

text = []
text_train = []
text_dev = []
text_test = []


def preprocessing():
    with open(train, 'r') as f:
        for tokenlist in parse_incr(f):
            text.append(tokenlist)
            text_train.append(tokenlist)

    with open(dev, 'r') as f:
        for tokenlist in parse_incr(f):
            text.append(tokenlist)
            text_dev.append(tokenlist)

    text_test = []
    with open(test, 'r') as f:
        for tokenlist in parse_incr(f):
            text.append(tokenlist)
            text_test.append(tokenlist)


preprocessing()
sent = text_train[0]


def make_vocab(text):
    word2idx = {}
    rag2idx = {}
    tokens = 0
    sentences = 0
    # id, form, upos
    for ind, sentence in enumerate(text):
        sent = text[ind]
        sentences += 1
        for i, tok in enumerate(sent):
            token = sent[i]
            word = token['form']
            tag = token['upos']
            tokens += 1
            if word not in word2idx.keys():
                word2idx[word] = 1
            else:
                word2idx[word] += 1
            if tag not in rag2idx.keys():
                rag2idx[tag] = len(rag2idx)

    alt = {}
    alt['<UNK>'] = 0
    for k, v in word2idx.items():
        if v < 2:
            alt['<UNK>'] += v
        else:
            alt[k] = v

    word2idx = alt.copy()

    l = list(word2idx)
    word_to_idx = {}
    for i in range(len(l)):
        word_to_idx[l[i]] = i+1
    word_to_idx['<pad>'] = 0
    return word_to_idx, rag2idx, sentences, tokens


word_to_idx, rag2idx, sentences, tokens = make_vocab(text)

# ids = torch.LongTensor(tokens)
# ids_tag = torch.LongTensor(tokens)


def get_indices(text, sentences, word_to_idx, rag2idx):
    ids = []
    ids_tag = []
    for i in range(sentences):
        ids.append(list())
        ids_tag.append(list())
    token = 0
    for ind, sentence in enumerate(text):
        sent = text[ind]
        for i, tok in enumerate(sent):
            t = sent[i]
            word = t['form']
            tag = t['upos']
            if word in word_to_idx.keys():
                ids[ind].append(word_to_idx[word])
                ids_tag[ind].append(rag2idx[tag])
            else:
                ids[ind].append(word_to_idx['<UNK>'])
            token += 1
    return ids, ids_tag


def word_processing():
    ids, ids_tag = get_indices(text, sentences, word_to_idx, rag2idx)
    max_len = max(len(sent) for sent in ids)

    ids_tag = np.array([np.pad(row, (0, max_len-len(row)))
                       for row in ids_tag])
    ids = np.array([np.pad(row, (0, max_len-len(row))) for row in ids])

    ids = torch.from_numpy(ids)
    ids_tag = torch.from_numpy(ids_tag)
    print(ids.size(), ids_tag.size())

    return ids, ids_tag


ids, ids_tag = word_processing()


def initialise_train():
    train_ids = ids[:4274, :]
    train_ids_tag = ids_tag[:4274]
    return train_ids, train_ids_tag


def initialise_dev():
    dev_ids = ids[4274:4846, :]
    dev_ids_tag = ids_tag[4274:4846]
    return dev_ids, dev_ids_tag


def initialise_test():
    test_ids = ids[4846:, :]
    test_ids_tag = ids_tag[4846:]
    return test_ids, test_ids_tag


train_ids, train_ids_tag = initialise_train()
dev_ids, dev_ids_tag = initialise_dev()
test_ids, test_ids_tag = initialise_test()


class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers, dropout, pad_idx=0):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(
            input_dim, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.embedding(x))

        out, _ = self.lstm(x)
        out = self.fc(self.dropout(out))
        tag_scores = F.log_softmax(out, dim=1)
        return out


def init_model():
    input_dim = len(word_to_idx.keys())
    output_dim = len(rag2idx.keys())
    seq_length = 30

    model = LSTM(input_dim=input_dim, embedding_dim=128, hidden_dim=128,
                   output_dim=output_dim, num_layers=4, dropout=0.15).to(device)
    return model


model = init_model()


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


model.apply(init_weights)

# model

def make_model():
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=1e-4)

    return criterion, optimizer


criterion, optimizer = make_model()

# def detach(states):
#     return [state.detach() for state in states]

# def train(model,criterion,optimizer,input_data_sen, input_data_tag, 20, 10, tag_pad_idx=0):
#   for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     epoch_acc = 0
#     for ind, i in enumerate(input_data_sen):
#       inputs = input_data_sen[ind].to(device)
#       targets = input_data_tag[ind].to(device)

#       # forward pass
#       # states = detach(states)
#       outputs = model(inputs)

#       outputs = outputs.view(-1, outputs.shape[-1])
#       targets = targets.view(-1)

#       loss = criterion(outputs, targets)

#       # backward pass
#       optimizer.zero_grad()
#       loss.backward()
#       optimizer.step()

#       epoch_loss += loss.item()
#     epoch_loss = epoch_loss / len(input_data_sen)
#     print(f'Epoch {epoch}/{num_epochs}\t Loss {epoch_loss}')

# train(model, criterion, optimizer, train_ids, train_ids_tag, 20, 10)
# torch.save(model.state_dict(), 'pos_tagger.pth')

# model.load_state_dict(torch.load("pos_tagger.pth",map_location=torch.device('cpu')))
model.load_state_dict(torch.load(
    "pos_tagger.pth", map_location=torch.device('cpu')))


def detach(state):
    return state.cpu().detach().numpy()


def categorical_accuracy(preds, y, tag_pad_idx=0, average='weighted'):
    """
    Returns accuracy per batch, i.e. if you get 4/10 right, this returns 0.4
    """
    max_preds = preds.argmax(
        dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    y_pred = max_preds[non_pad_elements].squeeze(1)
    y_test = y[non_pad_elements]

    y_pred = detach(y_pred)
    y_test = detach(y_test)

    f1 = f1_score(y_test, y_pred, average=average, labels=np.unique(y_pred))
    acc_score = accuracy_score(y_test, y_pred)
    pre_rec = precision_recall_fscore_support(
        y_test, y_pred, average=average, labels=np.unique(y_pred))

    return acc_score, f1, pre_rec


def evaluate(model, input_data_sen, input_data_tag, criterion, tag_pad_idx):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    epoch_prec = 0
    epoch_recall = 0

    with torch.no_grad():

        for ind, batch in enumerate(input_data_sen):
            inputs = input_data_sen[ind].to(device)
            targets = input_data_tag[ind].to(device)

            predictions = model(inputs)

            predictions = predictions.view(-1, predictions.shape[-1])
            targets = targets.view(-1)

            loss = criterion(predictions, targets)
            acc, f1, pre_rec = categorical_accuracy(
                predictions, targets, tag_pad_idx)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1.item()
            epoch_prec += pre_rec[0].item()
            epoch_recall += pre_rec[1].item()

    # epoch_loss = epoch_loss / len(input_data_sen)
    # print(f'Loss: {epoch_loss}')
    epoch_acc = epoch_acc / len(input_data_sen)
    print(f'Accuracy: {epoch_acc}')
    epoch_f1 = epoch_f1 / len(input_data_sen)
    print(f'F1: {epoch_f1}')
    epoch_prec = epoch_prec / len(input_data_sen)
    print(f'Precision: {epoch_prec}')
    epoch_recall = epoch_recall / len(input_data_sen)
    print(f'Recall: {epoch_recall}')

    return epoch_acc, epoch_f1, epoch_prec, epoch_recall


print('Train Set')
# acc, f1, precision, recall = evaluate(
#     model, train_ids, train_ids_tag, criterion, 0)

print('\nValidation Set')
# acc, f1, precision, recall = evaluate(
#     model, dev_ids, dev_ids_tag, criterion, 0)

print('\nTest set')
# acc, f1, precision, recall = evaluate(
#     model, test_ids, test_ids_tag, criterion, 0)

sentence = input("input sentence: ")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w+\s]', r'', text)
    text = re.split(r' ', text)
    return text

def indexer(text, word_to_idx, rag2idx):
    ids = list()
    for word in text:
        if word in word_to_idx.keys():
            ids.append(word_to_idx[word])
        else:
            ids.append(word_to_idx['<UNK>'])
    ids = torch.from_numpy(np.array(ids))
    return ids


def tagger(text, word_to_idx, rag2idx):
    text = clean_text(text)
    ids = indexer(text, word_to_idx, rag2idx)
    ids = ids.to(device)
    y_pred = model(ids)
    y_pred_main = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
    # y_pred_main = y_pred.argmax(-1)[0].cpu().detach().numpy()
    tags = []

    inv_dict = {v: k for k, v in rag2idx.items()}
    for i in y_pred_main:
        tags.append(inv_dict[i])
    return text, tags


text, tags = tagger(sentence, word_to_idx, rag2idx)

for ind, i in enumerate(text):
    print(f'{i}: {tags[ind]}')
