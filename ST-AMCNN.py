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



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 =nn.Conv2d(3, 4, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.selu = nn.SELU()
        self.conv2 = nn.Conv2d(4,4, kernel_size=1,stride=1)
        self.bn2 = nn.BatchNorm2d(4)  #(N,C,H,W)
        self.selu = nn.SELU()
        self.conv3 = nn.Conv2d(4,4, kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(4)  #(N,C,H,W)
        self.selu = nn.SELU()
        self.conv4 = nn.Conv2d(4,4, kernel_size=5,stride=1,padding=2)
        self.bn4 = nn.BatchNorm2d(4)  #(N,C,H,W)
        self.selu = nn.SELU()
        self.conv5 = nn.Conv2d(4,8, kernel_size=3,stride=1)
        self.bn5 = nn.BatchNorm2d(8)
        self.selu = nn.SELU()

        self.conv6 = nn.Conv2d(8,16, kernel_size=1,stride=1)
        self.bn6 = nn.BatchNorm2d(16)
        self.selu = nn.SELU()
        self.conv7 = nn.Conv2d(8,16, kernel_size=3,stride=1,padding=1)
        self.bn7 = nn.BatchNorm2d(16)
        self.selu = nn.SELU()
        self.conv8 = nn.Conv2d(8,16, kernel_size=5,stride=1,padding=2)
        self.bn8 = nn.BatchNorm2d(16)
        self.selu = nn.SELU()
        
        self.conv9 = nn.Conv2d(16,32,kernel_size=3,stride=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.selu = nn.SELU()
        self.conv10 = nn.Conv2d(32,64,kernel_size=3,stride=2)
        self.bn10 = nn.BatchNorm2d(64)
        self.selu = nn.SELU()
        self.fc = nn.Linear(64 * 74 * 74, 68)   #stride=2 (17) stride=1 (74)
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
            nn.Linear(10 * 35 * 35, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )
        
        self.se1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(4,4//2,kernel_size=1),
            nn.SELU(),
            nn.Conv2d(4//2,4,kernel_size=1),
            nn.Sigmoid()
        )
        
        self.se2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(16,16//2,kernel_size=1),
            nn.SELU(),
            nn.Conv2d(16//2,16,kernel_size=1),
            nn.Sigmoid()
        )
        
        
        
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 35 * 35)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward_once(self, x):
        x = self.conv1(x)
        feature1 = x
        x = self.bn1(x)
        x = self.selu(x)
        conv2 = self.conv2(x)
        covn2 = self.bn2(conv2)
        conv2 = self.selu(conv2)
        conv3 = self.conv3(x)
        conv3 = self.bn3(conv3)
        conv3 = self.selu(conv3)
        conv4 = self.conv4(x)
        conv4 = self.bn4(conv4)
        conv4 = self.selu(conv4)
        x = conv2+conv3+conv4
        feature2 = x
        x2 = self.se1(x)   #attentionzhuli
        x = x*x2
        x = self.conv5(x)
        feature3 = x
        x = self.bn5(x)
        x = self.selu(x)
        
        conv6 = self.conv6(x)
        conv6 = self.bn6(conv6)
        conv6 = self.selu(conv6)
        conv7 = self.conv7(x)
        conv7 = self.bn7(conv7)
        conv7 = self.selu(conv7)
        conv8 = self.conv8(x)
        conv8 = self.bn8(conv8)
        conv8 = self.selu(conv8)
        x = conv6+conv7+conv8
        feature4 = x
        x2 = self.se2(x)      # attention
        x = x*x2
        x = self.conv9(x)
        feature5 = x
        x = self.bn9(x)
        x = self.selu(x)
        x = self.conv10(x)
        feature6 = x
        x = self.bn10(x)
        x = self.selu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        return x,feature1,feature2,feature3,feature4,feature5,feature6
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    def forward(self, input1, input2):
        input2 = self.stn(input2) #localization后的整张图片
        output1,feature1,feature2,feature3,feature4,feature5,feature6 = self.forward_once(input1) #传入比较的指纹1
        output2,feature1_1,feature2_2,feature3_3,feature4_4,feature5_5,feature6_6 = self.forward_once(input2) #传入比较的指纹2 
        return input2,output1, output2,feature1,feature2,feature3,feature4,feature5,feature6,feature1_1,feature2_2,feature3_3,feature4_4,feature5_5,feature6_6
    

def backward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input) 

def farward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input) 

net = SiameseNetwork()
criterion = nn.CosineEmbeddingLoss()

#net = SiameseNetwork()
net = torch.load('model\ST_AMCNN.pkl')
path_1 = "images/"+str(args.number1)+'.png'
path_2 = "images/"+str(args.number2)+'.png'

def read(path):
    image = cv2.imread(path).astype(np.float)/255.
    image = cv2.resize(image,(156,156))
    image_show = image[:,:,::-1]  # RGB转BGR
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
net.conv10.register_backward_hook(backward_hook)
#net.conv3.register_forward_hook(farward_hook)


inputs_1, image_1, inputs_2, image_2 = get_input_from_path(path_1, path_2, size = (156, 156))
net.eval()
input_2,embed_1,embed_2,feature1,feature2,feature3,feature4,feature5,feature6,_,_,_,_,_,feature12 = net(inputs_1,inputs_2)
feature6.retain_grad()
feature2.retain_grad()
print(embed_1*embed_1)
cnn_feature_visual = feature3.detach().numpy()
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
rgradcam_1 = GradCAM(c,feature6)
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

