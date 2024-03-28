import torch
import torch.nn as nn
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import sys

def running_uefl_avg(current, next, scale):
    """
    compute the average of the model parameters, except for the codebooks
    """
    if current == None:
        current = next
        for key in current:
            if 'discretizer' in key:
                continue
            current[key] = current[key] * scale
    else:
        for key in current:
            if 'discretizer' in key:
                continue
            current[key] = current[key] + (next[key] * scale)
    return current

def validate(test_loader, model, device, args, net_idx):
    """
    validate model on test set
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total = 0.0, 0.0
    test_loss, test_vqloss, test_ppl = 0.0, 0.0, 0.0
    if args.data == 'cifar100':
        prediction = np.empty((0, 100))
    elif args.data == 'gtsrb':
        prediction = np.empty((0, 43))
    else:
        prediction = np.empty((0, 10))

    with torch.no_grad():
        for xte, yte in test_loader:
            xte = xte.to(device)
            pte, vqloss, ppl = model(xte, net_idx)
            test_vqloss += vqloss.item()
            test_ppl += ppl.item()

            lte = criterion(pte.cpu(), yte)
            prediction = np.append(prediction, F.softmax(pte, dim=1).cpu(), axis=0)
            cls = torch.argmax(pte.cpu(), axis=1)
            correct += torch.eq(cls, yte.cpu()).sum().item()
            total += xte.shape[0]
            test_loss += lte.item()
    return test_loss/len(test_loader), correct/total, prediction, test_vqloss/len(test_loader), test_ppl/len(test_loader)

def silo_training(train_loader, test_loader, model, device, args, lr, net_idx, init_and_ext=False):
    """
    local training for each silo
    """
    localmodel = copy.deepcopy(model)
    localmodel = localmodel.to(device)
    
    optimizer = optim.Adam(localmodel.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    loss_tr = []
    train_loss = 0.0
    if init_and_ext:
        localmodel.init_codebooks(train_loader, net_idx, device) # locally initialize the codebooks with kmeans

    for e in range(args.epoch):
        localmodel.train()
        for xtr, ytr in train_loader:
            xtr, ytr = xtr.to(device), ytr.to(device)

            optimizer.zero_grad()
            ptr, train_vqloss, train_ppl = localmodel(xtr, net_idx, ext=init_and_ext)
            init_and_ext = False # reset extension flag after extending the codebooks, extend only once for each silo
            ltr = criterion(ptr, ytr)+train_vqloss
            ltr.backward(retain_graph=True)
            optimizer.step()
            train_loss += ltr.item()
        
        test_loss, acc, pred, vqloss, ppl = validate(test_loader, localmodel, device, args, net_idx)
        loss_tr.append(test_loss/len(test_loader))
        train_loss = 0.0
    return localmodel, test_loss, acc, vqloss, ppl

def plot_lc(data, t, savepath):
    """
    plot learning curve for different metrics
    """
    x = np.arange(t)+1
    data = np.asarray(data).T
    num_silo = data.shape[0]
    if num_silo == 5:
        plt.figure()
        plt.plot(x, data[0], label='silo 1a')
        plt.plot(x, data[1], label='silo 1b')
        plt.plot(x, data[2], label='silo 1c')
        plt.plot(x, data[3], label='silo 2a')
        plt.plot(x, data[4], label='silo 3a')
        plt.legend()
        plt.xlabel('round')
        plt.ylabel(savepath.split('_')[-1])
        folder = savepath.split('/')[2]
        ttl = folder.split('_')[0]  # figure title
        plt.title(ttl.upper())
        plt.savefig(savepath)
        plt.close()
    elif num_silo == 9:
        plt.figure()
        plt.plot(x, data[0], label='silo 1a')
        plt.plot(x, data[1], label='silo 1b')
        plt.plot(x, data[2], label='silo 1c')
        plt.plot(x, data[3], label='silo 2a')
        plt.plot(x, data[4], label='silo 2b')
        plt.plot(x, data[5], label='silo 2c')
        plt.plot(x, data[6], label='silo 3a')
        plt.plot(x, data[7], label='silo 3b')
        plt.plot(x, data[8], label='silo 3c')
        plt.legend()
        plt.xlabel('round')
        plt.ylabel(savepath.split('_')[-1])
        folder = savepath.split('/')[2]
        ttl = folder.split('_')[0] # figure title
        plt.title(ttl.upper())
        plt.savefig(savepath)
        plt.close()

def plot_metrics(data_list, t, savepath_list):
    """
    plot learning curve for all metrics
    """
    for i in range(len(data_list)):
        plot_lc(data_list[i], t, savepath_list[i])

def entropy(preds):
    """
    compute entropy based on predictions
    """
    epsilon = sys.float_info.min
    entropy = -np.sum(np.mean(preds, axis=0)*np.log(np.mean(preds, axis=0)+epsilon), axis=-1)
    return entropy
