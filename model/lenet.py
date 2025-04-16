# python version 3.7.1
# -*- coding: utf-8 -*-
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(self.bn1(h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(self.bn2(h), negative_slope=0.01)
        h = self.c3(h)
        h = F.leaky_relu(self.bn3(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(self.bn4(h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(self.bn5(h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(self.bn6(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(self.bn7(h), negative_slope=0.01)
        h = self.c8(h)
        h = F.leaky_relu(self.bn8(h), negative_slope=0.01)
        h = self.c9(h)
        h = F.leaky_relu(self.bn9(h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        feature = h.view(h.size(0), h.size(1))

        logit = self.l_c1(feature)
        
        if self.top_bn:
            logit = call_bn(self.bn_c1, logit)
     
        return logit


def model_testing(model, device, test_dataloader, test_acc, test_losses, misclassified = [], y_train_noisy=[], y_train=[]):
    
    model.eval()
    test_loss = 0; correct = 0; pred_prob = None
    
    test_temp = test_dataloader
    test_dataloader = torch.utils.data.DataLoader(dataset=test_temp, batch_size=128, shuffle=False)
    test_dataloader.targets = y_train_noisy
    
    with torch.no_grad():

        for index, (data, target) in enumerate(test_dataloader):   
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            pred_prob_temp = F.softmax(output, dim=1)
            if pred_prob is None:
                pred_prob = pred_prob_temp  # First
            else:
                pred_prob = torch.cat((pred_prob, pred_prob_temp), dim=0)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, target)
            test_loss += loss.item()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)
    test_acc.append(100. * correct / len(test_dataloader.dataset))
    return pred_prob.cpu().numpy(), correct / len(test_dataloader.dataset)
    

def save_model(model, optimizer, epoch, loss, path):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': 0.2}, path)

def load_model(model, optimizer):
    checkpoint = torch.load("pretrained_model")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    

