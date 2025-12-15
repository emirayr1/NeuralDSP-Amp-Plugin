import torch
import numpy as np
from scipy.io.wavfile import write

# set a random seed so it generates the 
# same network parameters
torch.manual_seed(10)

# make a simple LSTM network layer
my_lstm = torch.nn.LSTM(1, 1, 1)

traced_lstm = torch.jit.trace(my_lstm, torch.rand(1, 1))
traced_lstm.save('models/my_lstm.pt')
