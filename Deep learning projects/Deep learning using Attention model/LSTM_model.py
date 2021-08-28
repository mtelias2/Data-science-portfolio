############## Loading Libraries ##############
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
from pdb import set_trace

dtype = torch.FloatTensor

############## LSTM Model ##############
class TextLSTM(nn.Module):
    def __init__(self, input_dim, no_neurons, no_layer, device, bidir, num_dir, batchsize):
        super(TextLSTM, self).__init__()

        self.no_layer = no_layer
        self.num_dir = num_dir
        self.batchsize = batchsize
        self.no_neurons = no_neurons
        self.device = device

        #self.hs = torch.zeros(no_layer*num_dir, batchsize, no_neurons).to(device) # [num_layers(=1) * num_directions(=1), batch_dim, no_neurons]
        #self.cs = torch.zeros(no_layer*num_dir, batchsize, no_neurons).to(device) # [num_layers(=1) * num_directions(=1), batch_dim, no_neurons]
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=no_neurons, num_layers=no_layer, batch_first=True, bidirectional=bidir)
        # input_dim for LSTM --> [batch_dim, seq_len, embedding_dim]
        # output dim for LSTM --> [batch_dim, seq_len, no_neurons]


    def forward(self, x):

    	# Initializing hidden and cell states
        hs = torch.zeros(self.no_layer*self.num_dir, self.batchsize, self.no_neurons).to(self.device) # [num_layers(=1) * num_directions(=1), batch_dim, no_neurons]
        cs = torch.zeros(self.no_layer*self.num_dir, self.batchsize, self.no_neurons).to(self.device) # [num_layers(=1) * num_directions(=1), batch_dim, no_neurons]
        outputs, (_, _) = self.lstm(x,(hs.detach(), cs.detach()))
        #outputs, (self.hs,self.cs) = self.lstm(x,(self.hs, self.cs))
        return outputs


############## Locked Dropout Model ##############
class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train==False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x


############## RNN Model ##############
class RNN_model(nn.Module):
    def __init__(self, vocab_size, embed_size, no_neurons, no_layer, device, bidir, batchsize, attention=False):
        super(RNN_model, self).__init__()

        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_size) #,padding_idx=0)
        #output from embedding --> [batch_dim, seq_len, embed_size]

        if bidir:
            num_dir = 2
        else:
            num_dir = 1

        self.lstm1 = TextLSTM(embed_size, no_neurons, no_layer, device, bidir, num_dir, batchsize)
        self.bn_lstm1= nn.BatchNorm1d(no_neurons)
        self.dropout1 = LockedDropout() #torch.nn.Dropout(p=0.5)

        # Fully connected layer dimensions depending on Bidirectional LSTM or not
        # if bidir==True:
        #     self.fc_output = nn.Linear(no_neurons*2, 2)
        # if bidir==False:
        #     self.fc_output = nn.Linear(no_neurons, 2)

        self.fc_output = nn.Linear(no_neurons*num_dir, 2)
        self.prob = nn.LogSoftmax(dim=1)

        if attention:
            self.attention = True
            self.softmax = nn.Softmax(dim=1)
            # self.attention_layer = nn.Linear(no_neurons*num_dir, no_neurons*num_dir)
            self.attention_layer = nn.Linear(no_neurons*num_dir, 1)
        else:
            self.attention = False

    #def reset_state(self):
        #self.lstm1.reset_state()
        #self.dropout1.reset_state()
        #self.lstm2.reset_state() # enabled for overfitting
        # self.dropout2.reset_state()

    def forward(self, x, train=True):

        embed = self.embedding(x) # batch_size, seq_len, embed_size

        out = self.lstm1(embed)
        #h = self.bn_lstm1(h)
        #h = self.dropout1(h,dropout=0.5,train=train)
        
        if not self.attention:
            output = self.fc_output(out)
        else:
        	# attn_scores = self.softmax(self.attention_layer(out))
        	# output = self.fc_output(out*attn_scores)

        	scores = torch.zeros((out.shape[0],out.shape[1])).to(self.device)
        	for i in range(out.shape[1]):
        		scores[:,i] = self.attention_layer(out[:,i,:]).squeeze()

        	attn_scores = self.softmax(scores)
        	attn_output = attn_scores.unsqueeze(-1).repeat(1,1,out.shape[-1])*out
        	output = self.fc_output(attn_output)
        	set_trace()
  
        return output

