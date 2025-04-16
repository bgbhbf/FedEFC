# python version 3.8.2
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from util.options import MINI_BATCH_LOCALMODEL
import torch.nn.functional as F

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class ForwardCorrectedLoss(nn.Module):
    # Forward correction
    def __init__(self, base_loss=nn.CrossEntropyLoss()):
        super(ForwardCorrectedLoss, self).__init__()
        self.base_loss = base_loss
        self.T = None  # Noise transition matrix

    def set_transition_matrix(self, T):
        self.T = T

    def forward(self, logits, targets):
        if self.T is None:
            raise ValueError("None of noise transition matrix")
        
        # Apply softmax to obtain probabilities
        probs = torch.softmax(logits, dim=1)
        # Ensure T is on the same device as logits
        T_torch = torch.tensor(self.T, dtype=logits.dtype, device=logits.device)
        corrected_probs = torch.matmul(probs, T_torch.T)  # T^T,corrected_probs = torch.matmul(probs, T_torch.T)  
        corrected_logits = -torch.log(corrected_probs + 1e-12)  # Adding epsilon for numerical stability

        num_classes = corrected_probs.size(1)  # 클래스 개수

        #As one-hot for checking
        if targets.ndim == 1:
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        else:
            targets_one_hot = targets
        sample_losses = torch.sum(targets_one_hot*corrected_logits, dim=-1)  

        # mean loss
        scalar_loss = sample_losses.mean() 
        return scalar_loss


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()  # cross entropy
        self.loss_func_corrected = ForwardCorrectedLoss()  # forward correction
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        if self.args.model == 'cnn':
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True) 
        else:
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=MINI_BATCH_LOCALMODEL)
        return train, test

    def update_weights(self, net, seed, w_g, epoch, lr=None, joint_matrix=None, correctedType = False):
        net_glob = w_g

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []; epoch_accuracy = []  # epoch accuracy

        for iter in range(epoch):
            batch_loss = []
            correct = 0; total = 0 

            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.long()
                net.zero_grad()
                log_probs = net(images)

                if correctedType is True: #Forward
                    self.loss_func_corrected.set_transition_matrix(joint_matrix)  
                    loss = self.loss_func_corrected(log_probs, labels)
                else:
                    loss = self.loss_func(log_probs, labels)

                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.norm(w - w_t).pow(2)  #|| w - wt||^2 
                        loss += 0.5 * self.args.beta * w_diff #mu

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                _, predicted = torch.max(log_probs, 1)   
                total += labels.size(0)                  
                correct += (predicted == labels).sum().item()      
            if len(batch_loss) == 0:
                epoch_loss.append(0)
                accuracy = 0
            else:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                accuracy = correct / total                      
            
            epoch_accuracy.append(accuracy)                   

        return net.state_dict(), sum(epoch_accuracy)/len(epoch_accuracy)


def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            # outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc
    

class LocalUpdateDitto(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.dataset = dataset
        self.idxs = list(idxs)
        self.criterion = nn.CrossEntropyLoss()
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        if self.args.model == 'cnn':
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        elif self.args.model == 'resnet50':
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=2)
        else:
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=MINI_BATCH_LOCALMODEL)
        return train, test

    def update_weights(self, net, seed, w_g, epoch, lr=None, lambda_reg=0.1):
        net_glob = w_g

        net.train()
        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)
        
        
        for _ in range(epoch):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                
                # Compute local model loss
                output = net(images)
                loss_local = self.criterion(output, labels)
                
                # Add L2 regularization with the global model
                l2_reg = 0
                for param_local, param_global in zip(net.parameters(), net_glob.parameters()):
                    l2_reg += torch.norm(param_local - param_global)**2
                
                loss = loss_local + lambda_reg * l2_reg
                loss.backward()
                optimizer.step()
        
        return net.state_dict(), loss