# encoding:utf-8
import os
import time
import sys
import logging
import warnings
import argparse
import h5py
import numpy as np

import torchvision
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from Module import EfficientNet
import AveragePrecision

num_epochs = 100
batch_size = 32
step_size = 10
classes = 13

parser = argparse.ArgumentParser(description="Your project")
parser.add_argument("-f","--feature_dim",type = int, default = 400*2)  
parser.add_argument("-a","--alpha",type = float, default = 10)
parser.add_argument("-lr","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g", "--gpu_devices", type=int, nargs='+', default=0)
parser.add_argument("-i", "--input_unit", type=int, default=1536)
parser.add_argument("-o", "--output_unit", type=int, default=400 * classes)
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper Parameters
INPUT_UNIT = args.input_unit
OUTPUT_UNIT = args.output_unit
feature_dim = args.feature_dim
alpha = args.alpha
LEARNING_RATE = args.learning_rate

warnings.filterwarnings("ignore")

# define data_load
def Load_Image_Information(path):
    # path
    image_Root_Dir = r'/R/pic'
    iamge_Dir = os.path.join(image_Root_Dir, path)
    return Image.open(iamge_Dir).convert('RGB')

class my_Data_Set(nn.Module):
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        fp = open(txt, 'r')
        images = []
        labels = []
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels.append([float(l) for l in information[1:len(information)]])
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, item):
        imageName = self.images[item]
        label = self.labels[item]
        image = self.loader(imageName)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.FloatTensor(label)
        return image, label

    def __len__(self):
        return len(self.images)

class fully_connect(nn.Module):
    """docstring for ClassName"""

    def __init__(self, INPUT_UNIT, OUTPUT_UNIT):
        super(fully_connect, self).__init__()
        self.fc = nn.Linear(INPUT_UNIT, OUTPUT_UNIT)

    def forward(self, x):
        out = F.relu(self.fc(x))
        return out

class RelationModule(torch.nn.Module):
    def __init__(self, img_feature_dim):
        super(RelationModule, self).__init__()
        self.img_feature_dim = feature_dim
        self.classifier = self.fc_fusion()
    def fc_fusion(self):
        num_bottleneck = 256
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck, 1),
                )
        return classifier
    def forward(self, input):
        input = input.view(input.size(0), self.img_feature_dim)
        input = self.classifier(input)
        return input

# load data
data_transforms = {
    "train": transforms.Compose([
        transforms.Scale((300, 300), 2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.892, 0.894, 0.894], std=[0.207, 0.188, 0.196]),
        transforms.RandomErasing()
    ]),
    "val": transforms.Compose([transforms.Scale((300, 300), 2),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.892, 0.894, 0.894], std=[0.207, 0.188, 0.196])
                               ])}

train_Data = my_Data_Set(r'/data/R/train_label.txt', transform=data_transforms["train"],
                         loader=Load_Image_Information)
val_Data = my_Data_Set(r'/data/R/val_label.txt', transform=data_transforms["val"],
                       loader=Load_Image_Information)
train_loader = torch.utils.data.DataLoader(train_Data, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = torch.utils.data.DataLoader(val_Data, batch_size=batch_size, shuffle=True, num_workers=16)
print("load dataset done")

# load EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b3', advprop=True, num_classes=classes)
model = nn.DataParallel(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("ls= %.5f,bs= %2d,alpha = %.5f" % (LEARNING_RATE, batch_size, alpha))

fully_connectNetwork = fully_connect(INPUT_UNIT, OUTPUT_UNIT)
fully_connectNetwork = nn.DataParallel(fully_connectNetwork)
fully_connectNetwork = fully_connectNetwork.to(device)
fully_connectNetwork_optim = torch.optim.Adam(fully_connectNetwork.parameters(), lr=LEARNING_RATE)

relation_network = RelationModule(feature_dim)
relation_network = nn.DataParallel(relation_network)
relation_network = relation_network.to(device)
relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)

# load matrix  word2vec
matrix = np.zeros([classes, 400])
index = 0
f = h5py.File('/R_word2vec.h5', 'r')
for key in f.keys():
    matrix[index] = f[key].value
    index = index + 1
matrix = torch.from_numpy(matrix)

# training
epoch = 0
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    fully_connectNetwork.train()
    relation_network.train()
    batch_size_start = time.time()
    running_loss = 0.0
    running_loss2 = 0.0

    for i, (inputs, labels) in enumerate(train_loader):

        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)
        bs = labels.size()[0]
        result, vec = model(inputs)
        result_vec = fully_connectNetwork(vec)               
        matrix_vec = matrix.unsqueeze(0).repeat(bs,1,1)

        result_vec = result_vec.view(bs*classes,400)
        result_vec = Variable(result_vec).to(device)
        matrix_vec = matrix_vec.view(bs*classes,400)
        result_vec = result_vec.view(bs*classes,400)
        matrix_vec = matrix_vec.float()
        matrix_vec = Variable(matrix_vec).to(device)
        relation_pairs = torch.cat((matrix_vec, result_vec),1)
        similarity = relation_network(relation_pairs)
        similarity = similarity.view(bs,classes)

        optimizer.zero_grad()
        fully_connectNetwork_optim.zero_grad()
        criterion = nn.MultiLabelSoftMarginLoss().to(device)
        loss1 = criterion(result, labels)
        loss2 = criterion(similarity, labels)
        loss = loss1 + alpha*loss2

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss2 += loss2.item()
        if (i + 1) % step_size == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch, num_epochs, (i + 1) / step_size, len(train_Data) / (batch_size * step_size), running_loss / step_size))
            running_loss2 = 0.0
            running_loss = 0.0

    print('the %d num_epochs ' % (epoch + 1))
    print('need time %f' % (time.time() - batch_size_start))

    if (epoch + 1) % 1 != 0:
        continue

    model.eval()
    fully_connectNetwork.eval()
    relation_network.eval()
    ap_meter = AveragePrecision.AveragePrecisionMeter(difficult_examples=False)
    with torch.no_grad():
        for j, (inputs, labels) in enumerate(val_loader):
            batch_size_start = time.time()
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            bs = labels.size()[0]
            result, vec = model(inputs)
            result_vec = fully_connectNetwork(vec)               
            matrix_vec = matrix.unsqueeze(0).repeat(bs,1,1)

            result_vec = Variable(result_vec).to(device)
            matrix_vec = matrix_vec.view(bs*classes,400)
            result_vec = result_vec.view(bs*classes,400)
            matrix_vec = matrix_vec.float()
            matrix_vec = Variable(matrix_vec).to(device)
            relation_pairs = torch.cat((matrix_vec, result_vec),1)
            similarity = relation_network(relation_pairs)
            similarity = similarity.view(bs,classes)
            
            ap_meter.add(result.data, labels)

    print("multi_labels:")
    print(100 * ap_meter.value())
    map = 100 * ap_meter.value().mean()
    if map > best_acc:
        best_acc = map
    print(" Val BatchSize cost time :%.4f s" % (time.time() - batch_size_start))
    print('Test Accuracy of the model on the 5000 Val images: %.4f' % (map))
    print("best-accuracy: %.4f" % (best_acc))

   
    if (map) >= 99:
        print('the Accuracy>=0.98 the num_epochs:%d' % epoch)
        break
print("training finish")


