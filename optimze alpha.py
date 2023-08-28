import torch

bd_model = torch.load('./saved_model/bd_model_after_pure_path.pt.pt',map_location='cpu')
optimizer = torch.optim.SGD(bd_model.parameters(), lr=0.01, )

for epoch in range(5):
    bd_model.train()
    print('current epoch  = {}'.format(epoch))
    for batch_idx, (data, label) in enumerate(bd_train_loader):
        optimizer.zero_grad()
        data, label = sig_poison(data, label, target_label=args.target_label, attack_ratio=args.attack_ratio) #delta=5
        data = data.to(args.device)
        label = label.to(args.device)
        output = bd_model(data)
        loss = criterion(output, label.view(-1, ))
        loss.backward()
        grad_mask(bd_model, mask0, mask1)
        optimizer.step()
#
    print('loss  = {}'.format(loss))
    bd_acc = test_backdoor_model(bd_model, test_loader)