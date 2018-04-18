
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size=512):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.inp = nn.Linear(512, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.8)
        self.out = nn.Linear(hidden_size, 51)

    def step(self, input, hidden=None):
        input = self.inp(input.contiguous().view(1, input.size(0), 512))
        # input = self.dropout(input)
        output, hidden = self.rnn(input, hidden)
        output = self.dropout(output)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None, steps=10):
        inputs = inputs.view(-1, 10, 512)
        inputs = inputs.permute(1, 0, 2)
        # outputs = Variable(torch.zeros(steps, 5120, 1))
        for i in range(steps):
            input = inputs[i]
            output, hidden = self.step(input, hidden)
            # outputs[i] = output
        return output.view(-1, 51), hidden

def actrec(**kwargs):
    # model = ActRec(**kwargs)
    model = SimpleRNN()

    return model