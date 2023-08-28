import torch
from model import *
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
import math
import torch.nn.utils
from data_poison import *
import copy
import torch.nn.utils.prune as prune

#incert covert backdoor critical routing path with desirable stealthiness
# try freeze less important weights based on w*a
######## args ################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')  # cuda:0
    parser.add_argument('--dataset', type=str, default="fmnist")
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--attack_ratio', type=float, default=0.2)
    # parser.add_argument('--attack_mode', type=str, default="square")
    parser.add_argument('--topk_ratio', type=float, default=0.92)
    parser.add_argument('--alpha', type=float, default=1)

    return parser.parse_args()

args=parse_args()

############ mask gradient ################

def grad_mask(model, mask0, mask1):
    for name, params in model.named_parameters():
        if name == 'fc1.weight':
            if params.requires_grad and params.grad != None:
                params.grad = params.grad * (1-mask0)
        elif name == 'fc2.weight':
            if params.requires_grad and params.grad != None:
                params.grad = params.grad * (1-mask1)

###########test model #################
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
    bd_acc = acc
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
    return bd_acc

####data loader        #####
transforms_list = []
transforms_list.append(transforms.ToTensor())
mnist_transform = transforms.Compose(transforms_list)
train_dataset = datasets.FashionMNIST(root = args.dataset_path, train=True, download=True, transform=mnist_transform)
test_dataset = datasets.FashionMNIST(root = args.dataset_path, train=False, download=True, transform=mnist_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True)

############ load benign model ###########################
benign_model = torch.load('./saved_model/benign_model.pt', map_location=args.device)
# module = benign_model.conv2
# prune.random_unstructured(module, name='weight',amount=0.3)

########### mask weights #############
def mask_weights(input_tenosr,index_tensor):
    mask = torch.zeros_like(input_tenosr)
    mask[index_tensor] = 1
    input_tenosr = input_tenosr * mask
    return input_tenosr

######## test benign acc only on clean path (topk=0.7) ###################
activation = {}

def getActivation(name):
    # the hook signature
    def hook(net, input, output):
        activation[name] = output.detach()
    ####squeeze
    return hook

for batch_idx, (data, label) in enumerate(train_loader):
    data = data.to(args.device)
    label = label.to(args.device)

benign_model = benign_model.to(args.device)
benign_model.eval()

h1 = benign_model.conv1.register_forward_hook(getActivation(0))    #hook
h2 = benign_model.conv2.register_forward_hook(getActivation(1))
h3 = benign_model.fc1.register_forward_hook(getActivation(2))
h4 = benign_model.fc2.register_forward_hook(getActivation(3))

a = torch.zeros(1,32,26,26).to(args.device) ###### conv1 21632  2, 3
b = torch.zeros(1, 64, 24, 24).to(args.device) ###### conv2 36864
c = torch.zeros(1,128).to(args.device)  ####### fc1   128
d = torch.zeros(1,10).to(args.device)   ####### fc2   10

for i in range(args.batch_size):
    input_tensor = data[i].unsqueeze(0)

    with torch.no_grad():
        out = benign_model(input_tensor)
    a += activation[0].to(args.device)
    b += activation[1].to(args.device)
    c += activation[2].to(args.device)
    d += activation[3].to(args.device)

a = a/args.batch_size  # (1,32,26,26)  32 neurons
b = b/args.batch_size  # (1, 64, 24, 24)  64 neurons
c = c/args.batch_size  # (1,128)  128 neurons
d = d/args.batch_size  # (1,10)  10 neurons

# benign neurons
# sum_a = torch.sum(a.squeeze(0),(1,2))  #a (1,32,26,26)
# _, indices0 = torch.topk(torch.abs(sum_a), math.floor(len(sum_a) * args.topk_ratio), largest=True)
# sum_b = torch.sum(b.squeeze(0),(1,2))  #b (1, 64, 24, 24)
# _, indices1 = torch.topk(torch.abs(sum_b), math.floor(len(sum_b) * args.topk_ratio), largest=True)
# sum_c = c.squeeze(0)  #c (1,128)
# _, indices2 = torch.topk(torch.abs(sum_c), math.floor(len(sum_c) * args.topk_ratio), largest=True)
# sum_d = d.squeeze(0)  #d (1,10)
# _, indices3 = torch.topk(torch.abs(sum_d), math.floor(len(sum_d) * args.topk_ratio), largest=True)
h1.remove()
h2.remove()
h3.remove()
h4.remove()

benign_param = {}
for name, parameters in benign_model.named_parameters():
    benign_param[name] = parameters.detach()

print('original acc:')
test_backdoor_model(benign_model, test_loader)

# benign_param['conv1.weight'] =  mask_weights(benign_param['conv1.weight'],indices0)
# benign_param['conv1.bias'] = mask_weights(benign_param['conv1.bias'],indices0)
# benign_param['conv2.weight'] = mask_weights(benign_param['conv2.weight'],indices1)
# benign_param['conv2.bias'] = mask_weights(benign_param['conv2.bias'],indices1)
# benign_param['fc1.weight'] = mask_weights(benign_param['fc1.weight'],indices2)
# benign_param['fc1.bias'] = mask_weights(benign_param['fc1.bias'],indices2)
# benign_param['fc2.weight'] = mask_weights(benign_param['fc2.weight'],indices3) #refine debug error
# benign_param['fc2.bias'] = mask_weights(benign_param['fc2.bias'],indices3)

#conv->fc1 weight
x = F.relu(F.max_pool2d(b, 2)) # [1, 64, 12, 12]
x = x.view(-1, 9216)  # [1,9216]    idx ([23,31,41])   #fc1.weight [128 (43, 2) , 9216]
x = x.squeeze(0)
w_x = x * benign_param['fc1.weight']  # [128, 9216]
result = w_x

# w/0 view
# _, idx = torch.topk(torch.abs(result), math.floor(w_x.shape[1] * 0.9), largest=True, dim=1)
# mask = torch.zeros_like(benign_param['fc1.weight'])
# for i in range(idx.shape[0]):
#     for j in range(idx.shape[1]):
#         mask[i][idx[i][j]] = 1

# with view
_, idx = torch.topk(torch.abs(result.view(-1,)), math.floor(len(result.view(-1,)) * args.topk_ratio), largest=True)
mask0 = torch.zeros_like(result.view(-1,))
mask0[idx] =1
idx0 = (1-mask0).nonzero()
mask0 = mask0.reshape([128,9216])
benign_param['fc1.weight'] = mask0 * benign_param['fc1.weight']

#fc1->fc2 weight
w_a = benign_param['fc2.weight'] * c
_, idx = torch.topk(torch.abs(w_a.view(-1,)), math.floor(len(w_a.view(-1,)) * args.topk_ratio), largest=True)
mask1 = torch.zeros_like(w_a.view(-1,))
mask1[idx] =1
idx1 = (1-mask1).nonzero()
mask1 = mask1.reshape([10,128])
benign_param['fc2.weight'] = mask1 * benign_param['fc2.weight']

with torch.no_grad():
    for name, param in benign_model.named_parameters():
        if name in benign_param:
            param.copy_(benign_param[name])
print('after pruning acc:')
test_backdoor_model(benign_model, test_loader)

####### start train backdoor task with mask ##############
criterion = nn.CrossEntropyLoss()

bd_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)  #lower the size
# label_subsets = [ [] for _ in range(10) ]
# for idx, (image, label) in enumerate(train_dataset):
#     label_subsets[label].append(idx)
# subset_indices = [indices[:100] for indices in label_subsets]
# subset_dataset_indices = sum(subset_indices, [])
# subset_dataset = torch.utils.data.Subset(train_dataset, subset_dataset_indices)
# bd_subset_dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=32, shuffle=True)

bd_model = copy.deepcopy(benign_model)
optimizer = torch.optim.SGD(bd_model.parameters(), lr=0.01, )  #lr=0.01, momentum=0.9, weight_decay=5e-4

bd_model.load_state_dict(torch.load('./saved_model/backdoor_model_pure_path.pt'))
print('after train_attack: ')
test_backdoor_model(bd_model, test_loader)
bd_param = {}
# for name, parameters in bd_model.named_parameters():
#     bd_param[name] = parameters.detach()
# bd_param['fc1.weight'] = (1-mask0) * bd_param['fc1.weight']
# bd_param['fc2.weight'] = (1-mask1) * bd_param['fc2.weight']
# with torch.no_grad():
#     for name, param in bd_model.named_parameters():
#         if name in bd_param:
#             param.copy_(bd_param[name])
# print('only bdcrp: ')
# test_backdoor_model(bd_model, test_loader)

# for epoch in range(5):
#     bd_model.train()
#     print('current epoch  = {}'.format(epoch))
#     for batch_idx, (data, label) in enumerate(bd_train_loader):
#         optimizer.zero_grad()
#         data, label = sig_poison(data, label, target_label=args.target_label, attack_ratio=args.attack_ratio) #delta=5
#         data = data.to(args.device)
#         label = label.to(args.device)
#         output = bd_model(data)
#         loss = criterion(output, label.view(-1, ))
#         loss.backward()
#         grad_mask(bd_model, mask0, mask1)
#         optimizer.step()
# #
#     print('loss  = {}'.format(loss))
#     bd_acc = test_backdoor_model(bd_model, test_loader)
#     # break
#
# ###### save backdoor model #########
# torch.save(bd_model.state_dict(), './saved_model/backdoor_model_pure_path.pt')
# print('Train backdoor model done!')

######## weak trigger well trained and find bd_crp ->alpha/strong|weak bd_crp
##### find bdcrp #####
bd_model.load_state_dict(torch.load('./saved_model/backdoor_model_pure_path.pt'))
benign_model = torch.load('./saved_model/benign_model.pt', map_location='cpu').to('cpu')
for name, parameters in bd_model.named_parameters():
    bd_param[name] = parameters.detach()
for name, parameters in benign_model.named_parameters():
    benign_param[name] = parameters.detach()
#  compare bdcrp
# benign_param['fc1.weight'] = (1-mask0) * benign_param['fc1.weight']
# benign_param['fc2.weight'] = (1-mask1) * benign_param['fc2.weight']
# bd_param['fc1.weight'] = (1-mask0) * bd_param['fc1.weight']
# bd_param['fc2.weight'] = (1-mask1) * bd_param['fc2.weight']
# values,indices = torch.topk(torch.abs(bd_param['fc2.weight'].view(-1) - benign_param['fc2.weight'].view(-1)), 12,largest=True)
########manipulate
bd_param['fc1.weight'] = bd_param['fc1.weight'].view(-1)
bd_param['fc2.weight'] = bd_param['fc2.weight'].view(-1)
bd_param['fc1.weight'][idx0] = 0.3* bd_param['fc1.weight'][idx0]
bd_param['fc2.weight'][idx1] = 0.3* bd_param['fc2.weight'][idx1]
bd_param['fc1.weight'] = bd_param['fc1.weight'].reshape(128,9216)
bd_param['fc2.weight'] = bd_param['fc2.weight'].reshape(10,128)
with torch.no_grad():
    for name, param in bd_model.named_parameters():
        if name in bd_param:
            param.copy_(bd_param[name])
print('topk craft: ')
test_backdoor_model(bd_model, test_loader)
saved_bd_model = bd_model.to('cpu')
torch.save(saved_bd_model, './saved_model/bd_model_after_pure_path.pt')
#print weights distribution    (w*a based on layers)
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
# plt.style.use('ggplot')
# after_pure_path_bd_model_tensor = nn.utils.parameters_to_vector(bd_model.parameters()).detach().numpy()
# benign_model = torch.load('./saved_model/benign_model.pt', map_location=args.device).to('cpu')
# benign_model_tensor = nn.utils.parameters_to_vector(benign_model.parameters()).detach().numpy()
# sns.distplot(after_pure_path_bd_model_tensor, hist=False, kde=False, fit=stats.norm, \
#              fit_kws={'color':'red', 'label':'bd_model_weights','linestyle':'-'})
# sns.distplot(benign_model_tensor, hist=False, kde=False, fit=stats.norm, \
#              fit_kws={'color':'blue', 'label':'benign_model_weights','linestyle':'-'})
# plt.legend()
# plt.show()