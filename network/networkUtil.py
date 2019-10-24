import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

    
class SELayer(nn.Module):
    def __init__(self,channel,height,width):
        super(SELayer, self).__init__()
        self.inShape = [channel,height,width]
        self.globalPooling = nn.AvgPool2d((self.inShape[1],self.inShape[2]))
        self.fc1 = nn.Conv2d(self.inShape[0],int(self.inShape[0]/2),1)
        self.fc2 = nn.Conv2d(int(self.inShape[0]/2),self.inShape[0],1)
    
    def forward(self,x):
        v1 = self.globalPooling(x)
        v2 = self.fc1(v1)
        v3 = self.fc2(v2)
        
        f = x*v3
        
        return f
    
class denseBlockLayer(nn.Module):
    def __init__(self,inChannel=64, outChannel=64, kernelSize = 3, bottleneckChannel = 64, dilateScale = 1, activ = 'ReLU'):
        super(denseBlockLayer, self).__init__()
        pad = int((kernelSize-1)/2)

        self.bn = nn.BatchNorm2d(inChannel)
        if(activ == 'LeakyReLU'):
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inChannel,bottleneckChannel,1)
        
        self.bn2 = nn.BatchNorm2d(bottleneckChannel)
        if(activ == 'LeakyReLU'):
            self.relu2 = nn.LeakyReLU()
        else:
            self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(bottleneckChannel,outChannel,kernelSize,padding = dilateScale,dilation=dilateScale)
        
            
    def forward(self,x):
        x1 = self.bn(x)
        x1 = self.relu(x1)
        x1 = self.conv(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        y = self.conv2(x1)
            
        return y
    
class denseBlock(nn.Module):
    def __init__(self, inChannel=64, kernelSize=3, growthRate=16, layer=4, bottleneckMulti = 4, dilationLayer = False, activ = 'ReLU'):
        super(denseBlock, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        self.layer = layer
        templayerList = []
        for i in range(0, layer):
            tempLayer = denseBlockLayer(inChannel+growthRate*i,growthRate,kernelSize,bottleneckMulti*growthRate, dilate, activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
            
    def forward(self,x):
        for i in range(0, self.layer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
            
        return x
    
class transitionLayer(nn.Module):
    def __init__(self, inChannel = 64, compressionRate = 0.5, activ = 'ReLU'):
        super(transitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(inChannel)
        if(activ == 'LeakyReLU'):
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inChannel,int(inChannel*compressionRate),1)
        
    def forward(self,x):
        x1 = self.bn(x)
        x2 = self.relu(x1)
        y = self.conv(x2)
        
        return y

class convLayer(nn.Module):
    def __init__(self, inChannel = 64, outChannel = 64, activ = 'ReLU', kernelSize = 3):
        super(convLayer, self).__init__()
        self.bn = nn.BatchNorm2d(inChannel)
        pad = int((kernelSize-1)/2)
        if(activ == 'LeakyReLU'):
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inChannel,outChannel,kernelSize,padding = pad)
        
    def forward(self,x):
        x1 = self.bn(x)
        x2 = self.relu(x1)
        y = self.conv(x2)
        
        return y