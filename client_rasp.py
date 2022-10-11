#!/usr/bin/env python3.9
# coding: utf-8
users = 24 # number of clients
batch_size = 32 # batch size
rounds = 2 # client-server communication rounds
local_epochs = 1 # local epoch
host = "192.168.1.104" # Set host address
port = 2000 #Set port number

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
from gpiozero import CPUTemperature
from torch.utils.data import Subset

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

printPerformance()

# Cuda：
device = "cpu"

# Set the codename of the client according to the IP address：
client_order = int(ifaddresses('wlan0')[AF_INET][0]['addr'][11:13])-11

# The amount of data each client needs to compute：
num_traindata = int(50000/users)

# load Data：
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]) # Normalize
indices = list(range(50000))
part_tr = indices[num_traindata * client_order : num_traindata * (client_order + 1)]
trainset = torchvision.datasets.CIFAR10 (root='/home/pi/Desktop/data', train=True, download=False, transform=transform)
trainset_sub = Subset(trainset, part_tr) # Let each client train with different data
train_loader = torch.utils.data.DataLoader(trainset_sub, batch_size=batch_size, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10 (root='/home/pi/Desktop/data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_total_batch = len(train_loader)
print(train_total_batch)
test_batch = len(test_loader)
print(test_batch)

# CNN:
# Model(
#   (model): Sequential(
#     (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (4): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Flatten(start_dim=1, end_dim=-1)
#     (7): Linear(in_features=1024, out_features=64, bias=True)
#     (8): Linear(in_features=64, out_features=10, bias=True)
#   )
# )
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
mobile_net = MobileNet()
mobile_net.to(device)
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mobile_net.parameters(), lr=lr, momentum=0.9)

# Socket initialization
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

printPerformance()

# Open the client socket
s = socket.socket()
s.connect((host, port))
start_time = time.time()    # store start time
print("timmer start!")
msg = recv_msg(s)
rounds = msg['rounds'] 
client_id = msg['client_id']
local_epochs = msg['local_epoch']
send_msg(s, len(trainset_sub))

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
printPerformance()
end_time = time.time()  #store end time
print("Training Time: {} sec".format(end_time - start_time))