import torch
from model import *
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import argparse
from data_poison import *
import wandb

wandb.login(key = 'b1388ac8787c26ef61a3efec09fe333eb4faa8d2')
wandb.init(project="Handcraft_BCRP_CML", name = "backdoor model", entity="yqqiao")  ####here

########   args ################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default="cpu")    # cuda:0
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, shuffle = True)

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
torch.save(model, './saved_model/backdoor_model.pt')

print('Train backdoor model done!')

############ Step2: finding backdoor critical routing path (BCRP) ####################
# activation = {}
#
# def getActivation(name):
#     # the hook signature
#     def hook(net, input, output):
#         activation[name] = output.detach()
#     ####squeeze
#     return hook
#
# model = torch.load('./saved_model/backdoor_model.pt', map_location=args.device)
#
# for batch_idx, (data, label) in enumerate(train_loader):
#     data, label = square_poison(data, label, target_label=args.target_label, attack_ratio=1.0)
#     data = data.to(args.device)
#     label = label.to(args.device)
#
# model.eval()
# model = model.to(args.device)
#
# h1 = model.conv1.register_forward_hook(getActivation(0))    #hook
# h2 = model.conv2.register_forward_hook(getActivation(1))
# h3 = model.fc1.register_forward_hook(getActivation(2))
# h4 = model.fc2.register_forward_hook(getActivation(3))
#
# a = torch.zeros(21632) ###### conv1 21632
# b = torch.zeros(36864) ###### conv2 36864
# c = torch.zeros(128)  ####### fc1   128
# d = torch.zeros(10)   ####### fc2   10
#
# for i in range(args.batch_size):
#     input_tensor = data[i].unsqueeze(0)
#
#     with torch.no_grad():
#         out = model(input_tensor)
#     a += torch.flatten(activation[0])
#     b += torch.flatten(activation[1])
#     c += torch.flatten(activation[2])
#     d += torch.flatten(activation[3])
#
# a = a/args.batch_size
# b = b/args.batch_size
# c = c/args.batch_size
#
# _,indices = torch.topk(a, k = 0.05 * a.length.floor())

############ Step3: manipulating weights in BCRP ####################

############ Step4: using the mask(square) with alpha intensity (test data) ####################