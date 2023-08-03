import torch
import math
from model import *
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
from data_poison import *
from torch.nn.utils import *
from torchsummary import summary


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
    parser.add_argument('--topk_ratio', type=float, default=0.1)

    return parser.parse_args()

args=parse_args()

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

backdoor_model = torch.load('./saved_model/weak_backdoor_model.pt', map_location=args.device)

for batch_idx, (data, label) in enumerate(train_loader):
    data, label = square_poison(data, label, target_label=args.target_label, attack_ratio=1.0)
    data = data.to(args.device)
    label = label.to(args.device)

backdoor_model.eval()
backdoor_model = backdoor_model.to(args.device)

h1 = backdoor_model.conv1.register_forward_hook(getActivation(0))    #hook
h2 = backdoor_model.conv2.register_forward_hook(getActivation(1))
h3 = backdoor_model.fc1.register_forward_hook(getActivation(2))
h4 = backdoor_model.fc2.register_forward_hook(getActivation(3))

a = torch.zeros(1,32,26,26).to(args.device) ###### conv1 21632  2, 3
b = torch.zeros(1, 64, 24, 24).to(args.device) ###### conv2 36864
c = torch.zeros(1,128).to(args.device)  ####### fc1   128
d = torch.zeros(1,10).to(args.device)   ####### fc2   10

for i in range(args.batch_size):
    input_tensor = data[i].unsqueeze(0)

    with torch.no_grad():
        out = backdoor_model(input_tensor)
    a += activation[0].to(args.device)
    b += activation[1].to(args.device)
    c += activation[2].to(args.device)
    d += activation[3].to(args.device)

a = a/args.batch_size  # (1,32,26,26)  32 neurons
b = b/args.batch_size  # (1, 64, 24, 24)  64 neurons
c = c/args.batch_size  # (1,128)  128 neurons
d = d/args.batch_size  # (1,10)  10 neurons

benign_model = torch.load('./saved_model/benign_model.pt', map_location=args.device)

for batch_idx, (data, label) in enumerate(train_loader):
    data = data.to(args.device)
    label = label.to(args.device)

benign_model.eval()
benign_model = benign_model.to(args.device)

h_1 = benign_model.conv1.register_forward_hook(getActivation(0))    #hook
h_2 = benign_model.conv2.register_forward_hook(getActivation(1))
h_3 = benign_model.fc1.register_forward_hook(getActivation(2))
h_4 = benign_model.fc2.register_forward_hook(getActivation(3))

a0 = torch.zeros(1,32,26,26).to(args.device) ###### conv1 21632
b0 = torch.zeros(1, 64, 24, 24).to(args.device) ###### conv2 36864
c0 = torch.zeros(1,128).to(args.device)  ####### fc1   128
d0 = torch.zeros(1,10).to(args.device)   ####### fc2   10

for i in range(args.batch_size):
    input_tensor = data[i].unsqueeze(0)

    with torch.no_grad():
        out = benign_model(input_tensor)
    a0 += activation[0].to(args.device)
    b0 += activation[1].to(args.device)
    c0 += activation[2].to(args.device)
    d0 += activation[3].to(args.device)

a0 = a0/args.batch_size
b0 = b0/args.batch_size
c0 = c0/args.batch_size
d0 = d0/args.batch_size


# backdoor neurons
sum_a = torch.sum(a.squeeze(0),(1,2))  #a (1,32,26,26)
_, indices0 = torch.topk(torch.where(sum_a<0,0,sum_a), math.floor(len(sum_a) * args.topk_ratio), largest=True)
sum_b = torch.sum(b.squeeze(0),(1,2))  #b (1, 64, 24, 24)
_, indices1 = torch.topk(torch.where(sum_b<0,0,sum_b), math.floor(len(sum_b) * args.topk_ratio), largest=True)
sum_c = c.squeeze(0)  #c (1,128)
_, indices2 = torch.topk(torch.where(sum_c<0,0,sum_c), math.floor(len(sum_c) * args.topk_ratio), largest=True)
sum_d = d.squeeze(0)  #d (1,10)
_, indices3 = torch.topk(torch.where(sum_d<0,0,sum_d), math.floor(len(sum_d) * args.topk_ratio), largest=True)

# #benign neurons
sum_a0 = torch.sum(a0.squeeze(0),(1,2))  #a (1,32,26,26)
_, indices_0 = torch.topk(torch.where(sum_a0<0,0,sum_a), math.floor(len(sum_a0) * args.topk_ratio), largest=True)
sum_b0 = torch.sum(b0.squeeze(0),(1,2))  #b (1, 64, 24, 24)
_, indices_1 = torch.topk(torch.where(sum_b<0,0,sum_b0), math.floor(len(sum_b0) * args.topk_ratio), largest=True)
sum_c0 = c0.squeeze(0)  #c (1,128)
_, indices_2 = torch.topk(torch.where(sum_c0<0,0,sum_c0), math.floor(len(sum_c0) * args.topk_ratio), largest=True)
sum_d0 = d0.squeeze(0)  #d (1,10)
_, indices_3 = torch.topk(torch.where(sum_d0<0,0,sum_d0), math.floor(len(sum_d0) * args.topk_ratio), largest=True)

# Set
# 4 lists for loop
# for i in range(4):
#    indices[i]
s = torch.isin(indices2, indices_2).long()
idx = torch.nonzero(s -1)
diff_indices = indices2[idx].squeeze(1)  # backdoor indices

############ Step3: manipulating weights in BCRP ####################
bd_param = {}
benign_param = {}
scale_factor = 0.5
for name, parameters in backdoor_model.named_parameters():
    bd_param[name] = parameters.detach()
for name, parameters in benign_model.named_parameters():
    benign_param[name] = parameters.detach()
# for name, parameters in backdoor_model.named_parameters():
#     benign_param[indices[]] = benign_param[name] + scale_factor * (bd_param[name] - benign_param[name])

############ Step4: using the mask(square) with alpha intensity (test data) ####################
# vector_to_parameters(benign_param, benign_model.parameters())
# benign_model
#test_backdoor_model(model, test_loader)

#  test_backdoor_model(model, test_loader)