import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

def weights_init(m):
    classname = m.__class__.__name__  #   obtain the class name
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        print("inital  linear weight ")


class word_embedding(nn.Module):
    def __init__(self,vocab_length , embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))
    def forward(self,input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class RNN_model(nn.Module):
    def __init__(self, batch_sz ,vocab_len ,word_embedding,embedding_dim, lstm_hidden_dim, device):
        super(RNN_model,self).__init__()

        self.device = device
        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        #########################################
        self.rnn_lstm = nn.RNN(embedding_dim,lstm_hidden_dim,num_layers=2,batch_first=True,dropout=0.3)

        ##########################################
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)
        self.apply(weights_init) # call the weights initial function.

    def forward(self,sentence, h_0=None, is_test = False):
        batch_size = sentence.shape[0] 

        sentence = sentence.to(device=self.device)
        
        batch_input = self.word_embedding_lookup(sentence).view(batch_size,-1,self.word_embedding_dim) # (batch_size, sentence_length,embedding_dim)
        
        ################################################
        if h_0 is None:
            h_0 = Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(device=self.device)
        output, h_n = self.rnn_lstm(batch_input, h_0)

        ################################################
        if batch_size == 1:
            out = output.contiguous().view(-1,self.lstm_dim)
        else:
            out = output.contiguous().view(batch_size, -1 ,self.lstm_dim)

        out =  F.relu(self.fc(out))

        if is_test:
            prediction = out[-1,:].view(1,-1)
            output = prediction
        else:
            output = out

        return output, h_n
    
    def predict(self,sentence_ids):
        out,_ = self(sentence_ids, is_test=True)
        return F.log_softmax(out,dim=1)
