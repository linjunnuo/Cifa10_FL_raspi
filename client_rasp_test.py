#!/usr/bin/env python
# coding: utf-8

# # CIFAR10 Federated Mobilenet Client Side
# This code is the server part of CIFAR10 federated mobilenet for **multi** client and a server.

# In[3]:


users = 8 # number of clients


# In[4]:


import os
import h5py

import socket
import struct
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from netifaces import ifaddresses, AF_INET
import time

from tqdm import tqdm



# In[2]:


def getFreeDescription():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 1:
            return (line.split()[0:7])


def getFree():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 2:
            return (line.split()[0:7])

from gpiozero import CPUTemperature


def printPerformance():
    cpu = CPUTemperature()

    print("temperature: " + str(cpu.temperature))

    description = getFreeDescription()
    mem = getFree()

    print(description[0] + " : " + mem[1])
    print(description[1] + " : " + mem[2])
    print(description[2] + " : " + mem[3])
    print(description[3] + " : " + mem[4])
    print(description[4] + " : " + mem[5])
    print(description[5] + " : " + mem[6])


# In[3]:


printPerformance()


# In[5]:


root_path = '../../models/cifar10_data'


# ## Cuda

# In[6]:


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)


# In[7]:

client_order = int(ifaddresses('wlan0')[AF_INET][0]['addr'][11:13])-11



# In[8]:


num_traindata = int(50000/users)



# ## Data load

# In[9]:


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

from torch.utils.data import Subset


indices = list(range(50000))

part_tr = indices[num_traindata * client_order : num_traindata * (client_order + 1)]




# In[10]:
batch_size = 32

trainset = torchvision.datasets.CIFAR10 (root='/home/pi/Desktop/dataset', train=True, download=False, transform=transform)

trainset_sub = Subset(trainset, part_tr)

train_loader = torch.utils.data.DataLoader(trainset_sub, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10 (root='/home/pi/Desktop/dataset', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# In[11]:


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ### Number of total batches

# In[13]:


train_total_batch = len(train_loader)
print(train_total_batch)
test_batch = len(test_loader)
print(test_batch)


# ## Pytorch layer modules for *Conv1D* Network
# 
# 
# 
# ### `Conv1d` layer
# - `torch.nn.Conv1d(in_channels, out_channels, kernel_size)`
# 
# ### `MaxPool1d` layer
# - `torch.nn.MaxPool1d(kernel_size, stride=None)`
# - Parameter `stride` follows `kernel_size`.
# 
# ### `ReLU` layer
# - `torch.nn.ReLU()`
# 
# ### `Linear` layer
# - `torch.nn.Linear(in_features, out_features, bias=True)`
# 
# ### `Softmax` layer
# - `torch.nn.Softmax(dim=None)`
# - Parameter `dim` is usually set to `1`.

# In[14]:


# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:23:31 2018
@author: tshzzz
"""

import torch


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# In[15]:


mobile_net = MobileNet()
mobile_net.to(device)


# In[16]:


lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mobile_net.parameters(), lr=lr, momentum=0.9)

rounds = 40 # default
local_epochs = 1 # default


# ## Socket initialization
# ### Required socket functions

# In[17]:


def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


# In[15]:


printPerformance()


# ### Set host address and port number

# In[18]:


host = "192.168.1.104"
port = 2000
max_recv = 100000


# ### Open the client socket

# In[19]:


s = socket.socket()
s.connect((host, port))


# ## SET TIMER

# In[20]:


start_time = time.time()    # store start time
print("timmer start!")


# In[21]:


msg = recv_msg(s)
rounds = msg['rounds'] 
client_id = msg['client_id']
local_epochs = msg['local_epoch']
send_msg(s, len(trainset_sub))


# In[22]:


# update weights from server
# train
for r in range(rounds):  # loop over the dataset multiple times
    weights = recv_msg(s)
    mobile_net.load_state_dict(weights)
    mobile_net.eval()
    for local_epoch in range(local_epochs):
        
        for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Round '+str(r+1)+'_'+str(local_epoch+1))):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.clone().detach().long().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mobile_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    msg = mobile_net.state_dict()
    send_msg(s, msg)

print('Finished Training')


# In[ ]:


printPerformance()


# In[23]:


end_time = time.time()  #store end time
print("Training Time: {} sec".format(end_time - start_time))


# In[ ]:




