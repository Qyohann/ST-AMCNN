import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import sys
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score,average_precision_score
from torch.optim import lr_scheduler
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("number1",help='first pose',type=int)
parser.add_argument("number2",help='second pose',type=int)

args = parser.parse_args()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SiameseNetwork(nn.Module):
    def __init__(self, block):
        super(SiameseNetwork, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 8, 2, stride=1)
        self.layer2 = self._make_layer(block, 16, 2, stride=2)
        self.layer3 = self._make_layer(block, 32, 2, stride=2)
        self.layer4 = self._make_layer(block, 64, 2, stride=2)
        self.linear = nn.Linear(1600,64)
        self.dropout = nn.Dropout()
        
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 35 *35, 32),   #  256->60, 156->35
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
        
        
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 35 * 35)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward_once(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        feature1 = out
        out = self.layer2(out)
        feature2 = out
        out = self.layer3(out)
        feature3 = out
        out = self.layer4(out)
        x = self.dropout(out)
        feature4 = x
        x = F.avg_pool2d(x, 4) 
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x,feature1,feature2,feature3,feature4

    def forward(self, input1, input2):
        input2 = self.stn(input2)
        output1,feature1,feature2,feature3,feature4 = self.forward_once(input1)
        output2,feature1_1,feature2_2,feature3_3,feature4_4 = self.forward_once(input2)
        return input2,output1,output2,feature1,feature2,feature3,feature4,feature1_1,feature2_2,feature3_3,feature4_4
    
def backward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input) 

def farward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input) 

net = SiameseNetwork(BasicBlock)
criterion = nn.CosineEmbeddingLoss()

net = torch.load('model\ST-ResNet-18.pkl')
path_1 = "images/"+str(args.number1)+'.png'
path_2 = "images/"+str(args.number2)+'.png'

def read(path):
    image = cv2.imread(path).astype(np.float)/255.
    image = cv2.resize(image,(156,156))
    image_show = image[:,:,::-1] 
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]) # same as the training transformer, not a mistake
    return image, image_show

def get_input_from_path(path_1, path_2, size = (156,156)):
    '''
        load two images from paths
    '''
    inputs_1, image_1 = read(path_1)
    inputs_2, image_2 = read(path_2)

    image_1 = cv2.resize(image_1, (size[1], size[0]))
    image_2 = cv2.resize(image_2, (size[1], size[0]))

    inputs_1 = cv2.resize(inputs_1, (size[1], size[0]))
    inputs_2 = cv2.resize(inputs_2, (size[1], size[0]))

    inputs_1 = np.transpose(inputs_1, (2, 0, 1))
    inputs_2 = np.transpose(inputs_2, (2, 0, 1))
    # wrap them in Variable
    inputs_1 = Variable(torch.from_numpy(np.expand_dims(inputs_1.astype(np.float32), axis=0)))
    inputs_2 = Variable(torch.from_numpy(np.expand_dims(inputs_2.astype(np.float32), axis=0)))

    return inputs_1, image_1, inputs_2, image_2

def imshow_convert(raw):
    '''
        convert the heatmap for imshow
    '''
    heatmap = np.array(cv2.applyColorMap(np.uint8(255*(1.-raw)), cv2.COLORMAP_JET )) #cv2.COLORMAP_JET  cv2.COLORMAP_BONE
    return heatmap

def GradCAM(gradient,map, size = (156, 156)):
    map = map.detach().numpy()

    # compute the average value
    weights = np.mean(gradient[0], axis=(1, 2), keepdims=True)
    grad_CAM_map = np.sum(np.tile(weights, [1, map.shape[-2], map.shape[-1]]) * map[0], axis=0)

    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = cv2.resize(cam, (size[1],size[0]))
    return cam

fmap_block=[]
fmap_block = list()
input_block = list()
net.layer4.register_backward_hook(backward_hook)

inputs_1, image_1, inputs_2, image_2 = get_input_from_path(path_1, path_2, size = (156, 156))
net.eval()
input_2,embed_1,embed_2,feature1,feature2,feature3,feature4,feature1_1,feature2_2,feature3_3,feature4_4 = net(inputs_1,inputs_2)
feature4.retain_grad()
label_1 = torch.tensor(1)
inputs_1.requires_grad = True
inputs_2.requires_grad = True
loss = criterion(embed_1,embed_2,label_1)
print(loss.backward(retain_graph=True))
norm_1 = torch.sqrt(torch.sum(embed_1*embed_1))
norm_2 = torch.sqrt(torch.sum(embed_2*embed_2))
product_vector = torch.mul(embed_1/norm_1, embed_2/norm_2)
product = torch.sum(product_vector)
product.backward(torch.tensor(1.), retain_graph=True)

c = fmap_block[0][0]
c = c.cpu().numpy()
rgradcam_1 = GradCAM(c,feature4)
image_overlay_1 = image_1 * 0.3 + imshow_convert(rgradcam_1) / 255.0 * 0.7

plt.figure(figsize=(5,5))
plt.suptitle('Rectified Grad-CAM (Decomposition+Bias)')
plt.subplot(2, 2, 1)
plt.imshow(image_1)
plt.subplot(2, 2, 2)
plt.imshow(image_2)
plt.subplot(2, 2, 3)
plt.imshow(image_overlay_1)
plt.show()