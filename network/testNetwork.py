import torch
import torch.nn as nn
from .networkUtil import denseBlock,convLayer

class ParaClassifyNet(nn.Module):
    def __init__(self):
        super(ParaClassifyNet, self).__init__()
        self.relu = nn.ReLU()
        self.ds = nn.MaxPool2d(2, stride = 2)
        self.conv1 = nn.Conv2d(1,32,3,padding = 1)
        self.conv2 = nn.Conv2d(32,64,3,padding = 1)
        self.conv3 = nn.Conv2d(64,64,3,padding = 1)
        self.conv4 = nn.Conv2d(64,32,3,padding = 1)
        
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(32*20*20, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self,x1):
        x2 = self.conv1(x1)
        x2 = self.relu(x2)
        x2 = self.ds(x2) # 32*160*160
        
        x3 = self.conv2(x2)
        x3 = self.relu(x3)
        x3 = self.ds(x3) # 64*80*80
        
        x4 = self.conv3(x3)
        x4 = self.relu(x4)
        x4 = self.ds(x4) # 64*40*40
        
        x5 = self.conv4(x4)
        x5 = self.relu(x5)
        x5 = self.ds(x5) # 32*20*20
        
        xv1 = x5.view(x5.size(0), -1)
        xv1 = self.drop(xv1)
        xv1 = self.relu(xv1)
        
        xv2 = self.fc1(xv1)
        xv2 = self.drop(xv2)
        xv2 = self.relu(xv2)
        
        xv3 = self.fc2(xv2)
        xv3 = self.relu(xv3)
        
        xv4 = self.fc3(xv3)
        #xv4 = self.relu(xv4)
        
        #xv5 = self.fc4(xv4)
        result = self.sigmoid(xv4)
        
        return result

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        self.block0 = nn.Sequential()
        self.block0.add_module("conv_0", nn.Conv2d(1,64,3,padding = 1))
        self.block0.add_module("relu_0", nn.ReLU())
        
        self.block1 = nn.Sequential()
        self.block1.add_module("conv_1", nn.Conv2d(64,64,3,padding = 1))
        self.block1.add_module("relu_1", nn.ReLU())
        
        self.down = nn.Sequential()
        self.down.add_module("maxpool_d", nn.MaxPool2d(kernel_size=2))
        self.down.add_module("conv_d1", nn.Conv2d(64,128,3,padding = 1))
        self.down.add_module("relu_d1", nn.ReLU())
        self.down.add_module("conv_d2", nn.Conv2d(128,128,3,padding = 1))
        self.down.add_module("relu_d2", nn.ReLU())

        self.middle = nn.Sequential()
        self.middle.add_module("maxpool_m", nn.MaxPool2d(kernel_size=2))
        self.middle.add_module("conv_m1", nn.Conv2d(128,256,3,padding = 1))
        self.middle.add_module("relu_m1", nn.ReLU())
        self.middle.add_module("conv_m2", nn.Conv2d(256,256,3,padding = 1))
        self.middle.add_module("relu_m2", nn.ReLU())
        self.middle.add_module("conv_m3", nn.Conv2d(256,128,3,padding = 1))
        self.middle.add_module("relu_m3", nn.ReLU())
        self.middle.add_module("upsampling_m", nn.ConvTranspose2d(128,128,2,2))
        
        self.up = nn.Sequential()
        self.up.add_module("conv_s1", nn.Conv2d(256, 128, 1, padding= 0))
        self.up.add_module("relu_s1", nn.ReLU())
        self.up.add_module("conv_s2", nn.Conv2d(128, 64, 1, padding= 0))
        self.up.add_module("relu_s2", nn.ReLU())
        self.up.add_module("upsampling_u", nn.ConvTranspose2d(64,64,2,2))
        
        self.block2 = nn.Sequential()
        self.block2.add_module("conv_s3", nn.Conv2d(128, 64, 1, padding= 0))
        self.block2.add_module("relu_s3", nn.ReLU())
        self.block2.add_module("conv_s4", nn.Conv2d(64, 64, 1, padding= 0))
        self.block2.add_module("relu_s4", nn.ReLU())
        self.block2.add_module("conv_5", nn.Conv2d(64, 16, 1, padding=0))
        
        self.compress = nn.Sequential()
        self.compress.add_module("maxpool1", nn.MaxPool2d(kernel_size=2))
        self.compress.add_module("conv1", nn.Conv2d(16, 8))
        self.compress.add_module("maxpool2", nn.MaxPool2d(kernel_size=2))
        self.compress.add_module("conv2", nn.Conv2d(8, 1))
        
        self.fc = nn.Sequential()
        self.fc.add_module("fc1", nn.Linear(80*80, 1024))
        self.fc.add_module("fc2", nn.Linear(1024, 128))
        self.fc.add_module("fc2", nn.Linear(128, 1))

    def forward(self,x0):
        x1 = self.block0(x0)
        x2 = self.block1(x1)
        x3 = self.down(x2)
        x4 = self.middle(x3)
        
        x5 = torch.cat((x3,x4),1)
        x6 = self.up(x5)
        
        x7 = torch.cat((x2,x6),1)
        x8 = self.block2(x7)
        
        x9 = x8+x0
        
        f = self.compress(x9)
        f = f.view(f.size(0), -1)
        f = self.fc(f)
        
        return f

class DenseFeatureAbstractor(nn.Module):
    def __init__(self):
        super(DenseFeatureAbstractor, self).__init__()
        self.relu = nn.ReLU()
        self.ds = nn.MaxPool2d(2, stride = 2)
        self.conv1 = nn.Conv2d(5,32,3,padding=1)
        self.block1 = denseBlock(inChannel=32, kernelSize=3, growthRate=16, layer=3, bottleneckMulti = 2, dilationLayer = False, activ = 'ReLU')
        self.transition1 = convLayer(inChannel = 80, outChannel = 32, activ = 'ReLU', kernelSize = 1)
        self.block2 = denseBlock(inChannel=32, kernelSize=3, growthRate=16, layer=3, bottleneckMulti = 2, dilationLayer = False, activ = 'ReLU')
        self.transition2 = convLayer(inChannel = 80, outChannel = 32, activ = 'ReLU', kernelSize = 1)
    
    def forward(self,x1):
        x2 = self.conv1(x1)

        x2 = self.block1(x2)
        x2 = self.relu(x2)
        x2 = self.transition1(x2)
        x2 = self.relu(x2)
        x2 = self.ds(x2) # 32*160*160
        
        x3 = self.block2(x2)
        x3 = self.relu(x3)
        x3 = self.transition2(x3)
        x3 = self.relu(x3)
        x3 = self.ds(x3) # 32*80*80
        return x3

class DenseFeatureFusionLayer(nn.Module):
    def __init__(self):
        super(DenseFeatureFusionLayer, self).__init__()
        self.relu = nn.ReLU()
        self.ds = nn.MaxPool2d(2, stride = 2)
        self.conv1 = nn.Conv2d(32*3,32,1)
        self.block1 = denseBlock(inChannel=32, kernelSize=3, growthRate=16, layer=3, bottleneckMulti = 2, dilationLayer = False, activ = 'ReLU')
        self.transition1 = convLayer(inChannel = 80, outChannel = 32, activ = 'ReLU', kernelSize = 1)
        self.block2 = denseBlock(inChannel=32, kernelSize=3, growthRate=16, layer=3, bottleneckMulti = 2, dilationLayer = False, activ = 'ReLU')
        self.transition2 = convLayer(inChannel = 80, outChannel = 32, activ = 'ReLU', kernelSize = 1)
    
    def forward(self,x1):
        x2 = self.conv1(x1)

        x2 = self.block1(x2)
        x2 = self.relu(x2)
        x2 = self.transition1(x2)
        x2 = self.relu(x2)
        x2 = self.ds(x2) # 32*40*40
        
        x3 = self.block2(x2)
        x3 = self.relu(x3)
        x3 = self.transition2(x3)
        x3 = self.relu(x3)
        x3 = self.ds(x3) # 32*20*20 、16*16  
        return x3

class ClassifyLayer(nn.Module):
    def __init__(self):
        super(ClassifyLayer, self).__init__()
        self.relu = nn.ReLU()
        
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1):
        xv1 = x1.view(x1.size(0), -1)
        xv1 = self.drop(xv1)
        xv1 = self.relu(xv1)
        
        xv2 = self.fc1(xv1)
        xv2 = self.drop(xv2)
        xv2 = self.relu(xv2)
        
        xv3 = self.fc2(xv2)
        xv3 = self.relu(xv3)
        
        xv4 = self.fc3(xv3)
        #xv4 = self.relu(xv4)
        
        #xv5 = self.fc4(xv4)
        result = self.sigmoid(xv4)
        
        return result

class TribleDenseNetwork(nn.Module):
    def __init__(self):
        super(TribleDenseNetwork, self).__init__()
        self.abstractor1 = DenseFeatureAbstractor()
        self.abstractor2 = DenseFeatureAbstractor()
        self.abstractor3 = DenseFeatureAbstractor()

        self.fusion = DenseFeatureFusionLayer()
        self.classify = ClassifyLayer()

    def forward(self, seq1, seq2, seq3):
        f1 = self.abstractor1(seq1)
        f2 = self.abstractor2(seq2)
        f3 = self.abstractor3(seq3)

        fs = torch.cat([f1,f2,f3],1)
        fs = self.fusion(fs)

        result = self.classify(fs)

        return result

