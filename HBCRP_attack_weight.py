import torch
import math
from model import *
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
from data_poison import *
from torch.nn.utils import *

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

##############   test backdoor model function ##################

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
############ Step3: manipulating weights in BCRP ####################

backdoor_model = torch.load('./saved_model/weak_backdoor_model.pt', map_location=args.device).to(args.device)
benign_model = torch.load('./saved_model/benign_model.pt', map_location=args.device).to(args.device)

############  find BCRP and manipulate weights function ############
def manipulate_topk_weights(backdoor_model, benign_model, topk_ratio=0.1):
    ##### manipulate 1.05X benign weights (topk indices)

    params = parameters_to_vector(backdoor_model.parameters()).to(args.device)
    params1 = parameters_to_vector(benign_model.parameters()).to(args.device)

    ###### backdoor topk indices  ######
    _, indices = torch.topk(params.abs(), math.floor(len(params) * topk_ratio), largest=False)
    backdoor_topk_list = torch.zeros(len(params))   #.cuda()
    backdoor_topk_list[indices] = 1.0
    ###### benign topk indices ##########
    _, indices1 = torch.topk(params1.abs(), math.floor(len(params1) * topk_ratio), largest=False)
    benign_topk_list = torch.zeros(len(params1))  # .cuda()
    benign_topk_list[indices1] = 1.0
    ###### find backdoor topk indices that are different from benign indices ######
    s = torch.isin(backdoor_topk_list, benign_topk_list).long()
    s1 = torch.isin(backdoor_topk_list, s).long()
    diff_indices = torch.nonzero(s1 -1)   ##### find different indices from 2 topk params lists


    #######  manipulate weights  #######
    # params1[indices] = params1[indices] + 1 * (params[indices] - params1[indices])
    params1[diff_indices] = params1[diff_indices] + 0.5 * (params[diff_indices] - params1[diff_indices])
    vector_to_parameters(params1, benign_model.parameters())

    return benign_model

model = manipulate_topk_weights(backdoor_model, benign_model, topk_ratio=0.1)
test_backdoor_model(model, test_loader)

############ Step4: using the mask(square) with alpha intensity (test data) ####################