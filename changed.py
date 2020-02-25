from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.nn.utils.rnn import pack_padded_sequence

flatten = lambda l: [item for sublist in l for item in sublist]

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch


def pad_to_batch(batch, x_to_ix, y_to_ix):
    sorted_batch = sorted(batch, key=lambda b: b[0].size(1), reverse=True)  # sort by len
    x, y = list(zip(*sorted_batch))
    max_x = max([s.size(1) for s in x])
    max_y = max([s.size(1) for s in y])
    x_p, y_p = [], []
    for i in range(len(batch)):
        if x[i].size(1) < max_x:
            x_p.append(
                torch.cat([x[i], LongTensor([x_to_ix['<PAD>']] * (max_x - x[i].size(1))).view(1, -1)], 1))
        else:
            x_p.append(x[i])
        if y[i].size(1) < max_y:
            y_p.append(
                torch.cat([y[i], LongTensor([y_to_ix['<PAD>']] * (max_y - y[i].size(1))).view(1, -1)], 1))
        else:
            y_p.append(y[i])

    input_var = torch.cat(x_p)
    target_var = torch.cat(y_p)
    input_len = [list(map(lambda s: s == 0, t.data)).count(False) for t in input_var]
    target_len = [list(map(lambda s: s == 0, t.data)).count(False) for t in target_var]

    return input_var, target_var, input_len, target_len

def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return LongTensor(idxs)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def max_length(tensor):
    return max(len(t) for t in tensor)

source_corpus = open('../dataset/de-en.en', 'r', encoding='utf-8').readlines()
target_corpus = open('../dataset/de-en.de', 'r', encoding='utf-8').readlines()

X_r, y_r = [], []  # raw

for sor, tar in zip(source_corpus,target_corpus):
    if sor.strip() == "" or tar.strip() == "":
        continue
    normalized_sor = normalize_string(sor).split()
    normalized_tar = normalize_string(tar).split()
    X_r.append(normalized_sor)
    y_r.append(normalized_tar)

# print(len(X_r), len(y_r))
# print(X_r[0], y_r[0])

max_length_targ, max_length_inp = max_length(X_r), max_length(y_r)

source_vocab = list(set(flatten(X_r)))
target_vocab = list(set(flatten(y_r)))
# print(len(source_vocab), len(target_vocab))
source2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
for vo in source_vocab:
    if source2index.get(vo) is None:
        source2index[vo] = len(source2index)
index2source = {v:k for k, v in source2index.items()}

target2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
for vo in target_vocab:
    if target2index.get(vo) is None:
        target2index[vo] = len(target2index)
index2target = {v:k for k, v in target2index.items()}

X_p, y_p = [], []

for so, ta in zip(X_r, y_r):
    X_p.append(prepare_sequence(so + ['</s>'], source2index).view(1, -1))
    y_p.append(prepare_sequence(ta + ['</s>'], target2index).view(1, -1))

train_data = list(zip(X_p, y_p))


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super(Attention,self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_dim = attn_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Intrattention(nn.Module):
    def __init__(self, dec_hid_dim, intrattn_dim):
        super(Intrattention,self).__init__()
        self.dec_hid_dim = dec_hid_dim
        self.intrattn_dim = intrattn_dim
        self.intrattn = nn.Linear(dec_hid_dim, intrattn_dim)
    def forward(self,decoder_hidden):
        decoder_hidden






class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self, decoder_hidden, encoder_outputs):

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(self, input, decoder_hidden, encoder_outputs):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.5):
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs



EPOCH = 50
BATCH_SIZE = 64
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 512
LR = 0.001
DECODER_LEARNING_RATIO = 5.0
RESCHEDULED = False
# %%
encoder = Encoder(len(source2index), EMBEDDING_SIZE, HIDDEN_SIZE, 3, True)
decoder = Decoder(len(target2index), EMBEDDING_SIZE, HIDDEN_SIZE * 2)
encoder.init_weight()
decoder.init_weight()

if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

loss_function = nn.CrossEntropyLoss(ignore_index=0)
enc_optimizer = optim.Adam(encoder.parameters(), lr=LR)
dec_optimizer = optim.Adam(decoder.parameters(), lr=LR * DECODER_LEARNING_RATIO)

# %%

for epoch in range(EPOCH):
    losses = []
    for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        inputs, targets, input_lengths, target_lengths = pad_to_batch(batch, source2index, target2index)

        input_masks = torch.cat([ByteTensor(tuple(map(lambda s: s == 0, t.data))) for t in inputs]).view(
            inputs.size(0), -1)
        start_decode = LongTensor([[target2index['<s>']] * targets.size(0)]).transpose(0, 1)
        encoder.zero_grad()
        decoder.zero_grad()
        output, hidden_c = encoder(inputs, input_lengths)

        preds = decoder(start_decode, hidden_c, targets.size(1), output, input_masks, True)

        loss = loss_function(preds, targets.view(-1))
        losses.append(loss.data.tolist()[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm(encoder.parameters(), 50.0)  # gradient clipping
        torch.nn.utils.clip_grad_norm(decoder.parameters(), 50.0)  # gradient clipping
        enc_optimizer.step()
        dec_optimizer.step()

        if i % 200 == 0:
            print("[%02d/%d] [%03d/%d] mean_loss : %0.2f" % (
            epoch, EPOCH, i, len(train_data) // BATCH_SIZE, np.mean(losses)))
            losses = []

    # You can use http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
    if RESCHEDULED == False and epoch == EPOCH // 2:
        LR *= 0.01
        enc_optimizer = optim.Adam(encoder.parameters(), lr=LR)
        dec_optimizer = optim.Adam(decoder.parameters(), lr=LR * DECODER_LEARNING_RATIO)
        RESCHEDULED = True

# borrowed code from https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb


def show_attention(input_words, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #     show_plot_visdom()
    plt.show()
    plt.close()


# %%

test = random.choice(train_data)
input_ = test[0]
truth = test[1]

output, hidden = encoder(input_, [input_.size(1)])
pred, attn = decoder.decode(hidden, output)

input_ = [index2source[i] for i in input_.data.tolist()[0]]
pred = [index2target[i] for i in pred.data.tolist()]

print('Source : ', ' '.join([i for i in input_ if i not in ['</s>']]))
print('Truth : ', ' '.join([index2target[i] for i in truth.data.tolist()[0] if i not in [2, 3]]))
print('Prediction : ', ' '.join([i for i in pred if i not in ['</s>']]))

if USE_CUDA:
    attn = attn.cpu()

show_attention(input_, pred, attn.data)