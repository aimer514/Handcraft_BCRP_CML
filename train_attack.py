import torch
from model import *
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
from data_poison import *
import wandb

wandb.login(key = 'b1388ac8787c26ef61a3efec09fe333eb4faa8d2')
wandb.init(project="Handcraft_BCRP_CML", name = "weak backdoor model", entity="yqqiao")  ####here

########   args ################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default="mps")    # cuda:0
    parser.add_argument('--dataset', type=str, default="fmnist")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--target_label', type=int, default=7)
    parser.add_argument('--attack_ratio', type=float, default=0.1)
    parser.add_argument('--attack_mode', type=str, default="square")

    return parser.parse_args()

args=parse_args()

criterion = nn.CrossEntropyLoss()

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
    wandb.log({"backdoor accuracy": acc})
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
    wandb.log({"benign accuracy": acc})

############ load benign model ###########################
model = torch.load('./saved_model/benign_model.pt', map_location=args.device)

####data loader        #####
transforms_list = []
transforms_list.append(transforms.ToTensor())
mnist_transform = transforms.Compose(transforms_list)
train_dataset = datasets.FashionMNIST(root = args.dataset_path, train=True, download=True, transform=mnist_transform)
test_dataset = datasets.FashionMNIST(root = args.dataset_path, train=False, download=True, transform=mnist_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True)

############ Step1: pre-train backdoor model ####################
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, )  #lr=0.01, momentum=0.9, weight_decay=5e-4

for epoch in range(args.epochs):
    model.train()
    print('current epoch  = {}'.format(epoch))
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data, label = square_poison(data, label, target_label=args.target_label, attack_ratio=args.attack_ratio)
        data = data.to(args.device)
        label = label.to(args.device)
        output = model(data)
        loss = criterion(output, label.view(-1, ))
        loss.backward()
        optimizer.step()

    print('loss  = {}'.format(loss))
    wandb.log({"loss": loss})
    test_backdoor_model(model, test_loader)

        ###### save backdoor model #########
torch.save(model, './saved_model/weak_backdoor_model.pt')
print('Train backdoor model done!')

