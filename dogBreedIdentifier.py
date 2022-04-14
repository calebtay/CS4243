#from numpy import float128
from numpy import dtype
import torch
import torch.nn as nn
import torch.optim as optim
from random import randint
import time
import os
import sys, os
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))
print(path)

train_data = torch.load(path + '/train_data.pt')
train_label = torch.load(path + '/label_data.pt')
test_data = torch.load(path + '/test_data.pt')
test_label = torch.load(path + '/test_label.pt')
val_data = torch.load(path + '/valid_data.pt')
val_label = torch.load(path + '/valid_label.pt')

print(train_data.size())
print(train_label.view(-1).size()) #[910, 1] -> [910]
print(test_data.size())
print(test_label.size())

mean = train_data.float().mean()
std = train_data.float().std()
train_size = train_data.size()[0]
test_size = test_data.size()[0]
print(mean)
print(std)
print(train_size)
print(test_size)
"""

"""