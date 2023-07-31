import torch
import math
from model import *
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
from data_poison import *

########   args ################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')    # cuda:0
    parser.add_argument('--dataset', type=str, default="fmnist")
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--attack_ratio', type=float, default=0.1)
    parser.add_argument('--attack_mode', type=str, default="square")

    return parser.parse_args()

args=parse_args()

############ load benign model ###########################
model = torch.load('./saved_model/backdoor_model.pt', map_location=args.device)

####data loader        #####
transforms_list = []
transforms_list.append(transforms.ToTensor())
mnist_transform = transforms.Compose(transforms_list)
train_dataset = datasets.FashionMNIST(root = args.dataset_path, train=True, download=True, transform=mnist_transform)
test_dataset = datasets.FashionMNIST(root = args.dataset_path, train=False, download=True, transform=mnist_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True)

############ Step2: finding backdoor critical routing path (BCRP) ####################
activation = {}

def getActivation(name):
    # the hook signature
    def hook(net, input, output):
        activation[name] = output.detach()
    ####squeeze
    return hook

model = torch.load('./saved_model/backdoor_model.pt', map_location=args.device)

for batch_idx, (data, label) in enumerate(train_loader):
    data, label = square_poison(data, label, target_label=args.target_label, attack_ratio=1.0)
    data = data.to(args.device)
    label = label.to(args.device)

model.eval()
model = model.to(args.device)

h1 = model.conv1.register_forward_hook(getActivation(0))    #hook
h2 = model.conv2.register_forward_hook(getActivation(1))
h3 = model.fc1.register_forward_hook(getActivation(2))
h4 = model.fc2.register_forward_hook(getActivation(3))

a = torch.zeros(21632).to(args.device) ###### conv1 21632
b = torch.zeros(36864).to(args.device) ###### conv2 36864
c = torch.zeros(128).to(args.device)  ####### fc1   128
d = torch.zeros(10).to(args.device)   ####### fc2   10

for i in range(args.batch_size):
    input_tensor = data[i].unsqueeze(0)

    with torch.no_grad():
        out = model(input_tensor)
    a += torch.flatten(activation[0]).to(args.device)
    b += torch.flatten(activation[1]).to(args.device)
    c += torch.flatten(activation[2]).to(args.device)
    d += torch.flatten(activation[3]).to(args.device)

a = a/args.batch_size
b = b/args.batch_size
c = c/args.batch_size
d = d/args.batch_size

model = torch.load('./saved_model/benign_model.pt', map_location=args.device)

for batch_idx, (data, label) in enumerate(train_loader):
    data = data.to(args.device)
    label = label.to(args.device)

model.eval()
model = model.to(args.device)

h_1 = model.conv1.register_forward_hook(getActivation(0))    #hook
h_2 = model.conv2.register_forward_hook(getActivation(1))
h_3 = model.fc1.register_forward_hook(getActivation(2))
h_4 = model.fc2.register_forward_hook(getActivation(3))

a0 = torch.zeros(21632).to(args.device) ###### conv1 21632
b0 = torch.zeros(36864).to(args.device) ###### conv2 36864
c0 = torch.zeros(128).to(args.device)  ####### fc1   128
d0 = torch.zeros(10).to(args.device)   ####### fc2   10

for i in range(args.batch_size):
    input_tensor = data[i].unsqueeze(0)

    with torch.no_grad():
        out = model(input_tensor)
    a0 += torch.flatten(activation[0]).to(args.device)
    b0 += torch.flatten(activation[1]).to(args.device)
    c0 += torch.flatten(activation[2]).to(args.device)
    d0 += torch.flatten(activation[3]).to(args.device)

a0 = a0/args.batch_size
b0 = b0/args.batch_size
c0 = c0/args.batch_size
d0 = d0/args.batch_size

_,indices0 = torch.topk(a, k = math.floor(0.05*21632))
_,indices1 = torch.topk(b, k = math.floor(0.05*36864))
_,indices2 = torch.topk(c, k = math.floor(0.05*128))
_,indices3 = torch.topk(d, k = math.ceil(0.05*10))

_,indices_2 = torch.topk(c0, k = math.floor(0.05*128))

############ Step3: manipulating weights in BCRP ####################

############ Step4: using the mask(square) with alpha intensity (test data) ####################