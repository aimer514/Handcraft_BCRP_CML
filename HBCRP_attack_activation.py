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
import copy


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
    parser.add_argument('--topk_ratio', type=float, default=0.04)
    parser.add_argument('--alpha', type=float, default=1)

    return parser.parse_args()

args=parse_args()

###############test model##################

def test_backdoor_model(model, test_loader):
    ########### backdoor accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = sig_poison(data, label, args.target_label, attack_ratio = 1.0)
        data = data.to(device=args.device)
        label = label.to(device=args.device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)

        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, label)).item()
    model.train()

    acc = correctly_labeled_samples / total_test_number
    print('backdoor accuracy  = {}'.format(acc))
    ########### benign accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device=args.device)
        label = label.to(device=args.device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)

        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, label)).item()
    model.train()
    acc = correctly_labeled_samples / total_test_number
    print('benign accuracy  = {}'.format(acc))


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

backdoor_model = torch.load('./saved_model/backdoor_model_sig5.pt', map_location=args.device)

for batch_idx, (data, label) in enumerate(train_loader):
    data, label = find_sig_poison(data, label, target_label=args.target_label, attack_ratio=1.0)
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


# backdoor neurons
sum_a = torch.sum(a.squeeze(0),(1,2))  #a (1,32,26,26)
_, indices0 = torch.topk(torch.abs(sum_a), math.ceil(len(sum_a) * args.topk_ratio), largest=True)
sum_b = torch.sum(b.squeeze(0),(1,2))  #b (1, 64, 24, 24)
_, indices1 = torch.topk(torch.abs(sum_b), math.ceil(len(sum_b) * args.topk_ratio), largest=True)
sum_c = c.squeeze(0)  #c (1,128)
_, indices2 = torch.topk(torch.abs(sum_c), math.ceil(len(sum_c) * args.topk_ratio), largest=True)
sum_d = d.squeeze(0)  #d (1,10)
_, indices3 = torch.topk(torch.abs(sum_d), math.ceil(len(sum_d) * args.topk_ratio), largest=True)

# #benign neurons
sum_a0 = torch.sum(a0.squeeze(0),(1,2))  #a (1,32,26,26)
_, indices_0 = torch.topk(torch.abs(sum_a0), math.floor(len(sum_a0) * args.topk_ratio), largest=True)
sum_b0 = torch.sum(b0.squeeze(0),(1,2))  #b (1, 64, 24, 24)
_, indices_1 = torch.topk(torch.abs(sum_b0), math.floor(len(sum_b0) * args.topk_ratio), largest=True)
sum_c0 = c0.squeeze(0)  #c (1,128)
_, indices_2 = torch.topk(torch.abs(sum_c0), math.floor(len(sum_c0) * args.topk_ratio), largest=True)
sum_d0 = d0.squeeze(0)  #d (1,10)
_, indices_3 = torch.topk(torch.abs(sum_d0), math.floor(len(sum_d0) * args.topk_ratio), largest=True)

# Set
# 4 lists for loop
# for i in range(4):
#    indices[i]
s = torch.isin(indices0, indices_0).long()
idx = torch.nonzero(s -1)
diff_indices0 = indices0[idx].squeeze(1)

s = torch.isin(indices1, indices_1).long()
idx = torch.nonzero(s -1)
diff_indices1 = indices1[idx].squeeze(1)

s = torch.isin(indices2, indices_2).long()
idx = torch.nonzero(s -1)
diff_indices2 = indices2[idx].squeeze(1)

# s = torch.isin(indices3, indices_3).long()
# idx = torch.nonzero(s -1)
# diff_indices3 = indices3[idx].squeeze(1)  # backdoor indices

diff_indices0 = torch.tensor([5,21])
diff_indices1 = torch.tensor([23,31,41])
diff_indices2 = torch.tensor([43,2]) # 29->7 2 ->7
diff_indices3 = torch.tensor([7])

############ Step3: manipulating weights in BCRP ####################
bd_param = {}
benign_param = {}

for name, parameters in backdoor_model.named_parameters():
    bd_param[name] = parameters.detach()
for name, parameters in benign_model.named_parameters():
    benign_param[name] = parameters.detach()
# benign_param['conv1.weight'][diff_indices0] = benign_param['conv1.weight'][diff_indices0] + \
#                              args.alpha * (bd_param['conv1.weight'][diff_indices0] - benign_param['conv1.weight'][diff_indices0])
# benign_param['conv1.bias'][diff_indices0] = benign_param['conv1.bias'][diff_indices0] + \
#                              args.alpha * (bd_param['conv1.bias'][diff_indices0] - benign_param['conv1.bias'][diff_indices0])
# benign_param['conv2.weight'][diff_indices1] = benign_param['conv2.weight'][diff_indices1] + \
#                              args.alpha * (bd_param['conv2.weight'][diff_indices1] - benign_param['conv2.weight'][diff_indices1])
# benign_param['conv2.bias'][diff_indices1] = benign_param['conv2.bias'][diff_indices1] + \
#                              args.alpha * (bd_param['conv2.bias'][diff_indices1] - benign_param['conv2.bias'][diff_indices1])
benign_param['fc1.weight'][diff_indices2] = benign_param['fc1.weight'][diff_indices2] + \
                             args.alpha * (bd_param['fc1.weight'][diff_indices2] - benign_param['fc1.weight'][diff_indices2])
benign_param['fc1.bias'][diff_indices2] = benign_param['fc1.bias'][diff_indices2] + \
                             args.alpha * (bd_param['fc1.bias'][diff_indices2] - benign_param['fc1.bias'][diff_indices2])
############### fc2 ########################
###### neuron-wise ##########
# benign_param['fc2.weight'][diff_indices3] = benign_param['fc2.weight'][diff_indices3] + \
#                              args.alpha * (bd_param['fc2.weight'][diff_indices3] - benign_param['fc2.weight'][diff_indices3])
# benign_param['fc2.bias'][diff_indices3] = benign_param['fc2.bias'][diff_indices3] + \
#                              args.alpha * (bd_param['fc2.bias'][diff_indices3] - benign_param['fc2.bias'][diff_indices3])
###### weight-wise ###########
c=c.squeeze(0)  #[128]
#for i in neuron_idx:
w_x = c * bd_param['fc2.weight'][7]  # 7->i
_, diff_indices2 = torch.topk(torch.abs(w_x), math.ceil(len(w_x) * 0.25), largest=True)

bd_mask = torch.zeros_like(benign_param['fc2.weight'][diff_indices3])
bd_mask[0][diff_indices2] = 1
benign_mask = 1 - bd_mask
bd_param['fc2.weight'][diff_indices3] = bd_param['fc2.weight'][diff_indices3] * bd_mask
benign_param['fc2.weight'][diff_indices3] = benign_param['fc2.weight'][diff_indices3] * benign_mask
benign_param['fc2.weight'][diff_indices3] = benign_param['fc2.weight'][diff_indices3] + bd_param['fc2.weight'][diff_indices3]
benign_param['fc2.bias'][diff_indices3] = bd_param['fc2.bias'][diff_indices3]


##############################
x = F.relu(F.max_pool2d(activation[1], 2)) # [1, 64, 12, 12]
x = x.view(-1, 9216)  # [1,9216]    idx ([23,31,41])   #fc1.weight [128 (43, 2) , 9216]
x = x.squeeze(0)
w_x = x * bd_param['fc1.weight'][43]
w_x1 = x * bd_param['fc1.weight'][2]
_, idx = torch.topk(torch.abs(w_x), math.ceil(len(w_x) * 0.25), largest=True)
#[3888, 3900, 3912, 3899, 4020, 1019, 3601, 4008, 3936, 3924]
_, idx1 = torch.topk(torch.abs(w_x1), math.ceil(len(w_x1) * 0.25), largest=True)
# [3888, 3900, 3912, 3899, 4020, 3601, 1019, 4008, 5184, 4752]

# bd_mask = torch.zeros_like(benign_param['fc1.weight'][diff_indices2])   # this method is not good
# bd_mask[0][idx] = 1
# bd_mask[1][idx1] = 1
# benign_mask = 1 - bd_mask
# bd_param['fc1.weight'][diff_indices2] = bd_param['fc1.weight'][diff_indices2] * bd_mask * 1
# benign_param['fc1.weight'][diff_indices2] = benign_param['fc1.weight'][diff_indices2] * benign_mask
# benign_param['fc1.weight'][diff_indices2] = benign_param['fc1.weight'][diff_indices2] + bd_param['fc1.weight'][diff_indices2]
############ Step4: using the mask(square) with alpha intensity (test data) ####################

with torch.no_grad():
    for name, param in benign_model.named_parameters():
        if name in benign_param:
            param.copy_(benign_param[name])

test_backdoor_model(benign_model, test_loader)

####### recompute 9216 neurons topk ############