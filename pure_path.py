import torch
from model import *
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
import math
import torch.nn.utils
from data_poison import *
import torch.nn.utils.prune as prune

########   args ################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')  # cuda:0
    parser.add_argument('--dataset', type=str, default="fmnist")
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--attack_ratio', type=float, default=0.1)
    # parser.add_argument('--attack_mode', type=str, default="square")
    parser.add_argument('--topk_ratio', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=1)

    return parser.parse_args()

args=parse_args()

###########test model #################
def test_backdoor_model(model, test_loader):
    ########### backdoor accuracy ##############
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = square_poison(data, label, args.target_label, attack_ratio = 1.0)
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
benign_model = torch.load('./saved_model/be_bd_model.pt', map_location=args.device)
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
sum_a = torch.sum(a.squeeze(0),(1,2))  #a (1,32,26,26)
_, indices0 = torch.topk(torch.where(sum_a<0,0,sum_a), math.floor(len(sum_a) * args.topk_ratio), largest=True)
# _, indices0 = torch.topk(torch.abs(sum_a), math.floor(len(sum_a) * args.topk_ratio), largest=True)
# _, indices0 = torch.topk(sum_a, math.floor(len(sum_a) * args.topk_ratio), largest=True)
sum_b = torch.sum(b.squeeze(0),(1,2))  #b (1, 64, 24, 24)
_, indices1 = torch.topk(torch.where(sum_b<0,0,sum_b), math.floor(len(sum_b) * args.topk_ratio), largest=True)
# _, indices1 = torch.topk(torch.abs(sum_b), math.floor(len(sum_b) * args.topk_ratio), largest=True)
# _, indices1 = torch.topk(sum_b, math.floor(len(sum_b) * args.topk_ratio), largest=True)
sum_c = c.squeeze(0)  #c (1,128)
_, indices2 = torch.topk(torch.where(sum_c<0,0,sum_c), math.floor(len(sum_c) * args.topk_ratio), largest=True)
# _, indices2 = torch.topk(torch.abs(sum_c), math.floor(len(sum_c) * args.topk_ratio), largest=True)
# _, indices2 = torch.topk(sum_c, math.floor(len(sum_c) * args.topk_ratio), largest=True)
sum_d = d.squeeze(0)  #d (1,10)
_, indices3 = torch.topk(torch.where(sum_d<0,0,sum_d), math.floor(len(sum_d) * args.topk_ratio), largest=True)
# _, indices3 = torch.topk(torch.abs(sum_d), math.floor(len(sum_d) * args.topk_ratio), largest=True)
# _, indices3 = torch.topk(sum_d, math.floor(len(sum_d) * args.topk_ratio), largest=True)


benign_param = {}
for name, parameters in benign_model.named_parameters():
    benign_param[name] = parameters.detach()

print('original acc:')
test_backdoor_model(benign_model, test_loader)

benign_param['conv1.weight'] =  mask_weights(benign_param['conv1.weight'],indices0)
benign_param['conv1.bias'] = mask_weights(benign_param['conv1.bias'],indices0)
benign_param['conv2.weight'] = mask_weights(benign_param['conv2.weight'],indices1)
benign_param['conv2.bias'] = mask_weights(benign_param['conv2.bias'],indices1)
benign_param['fc1.weight'] = mask_weights(benign_param['fc1.weight'],indices2)
benign_param['fc1.bias'] = mask_weights(benign_param['fc1.bias'],indices2)
benign_param['fc2.weight'] = mask_weights(benign_param['fc2.weight'],indices3)
benign_param['fc2.bias'] = mask_weights(benign_param['fc2.bias'],indices3)

with torch.no_grad():
    for name, param in benign_model.named_parameters():
        if name in benign_param:
            param.copy_(benign_param[name])
print('after pruning acc:')
test_backdoor_model(benign_model, test_loader)