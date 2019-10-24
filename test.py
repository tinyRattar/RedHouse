import matplotlib.pyplot as plt
import pylab
#pylab.rcParams['figure.figsize'] = (7, 7) # 显示大小

import os
import numpy as np
import torch 
from util import *
from patientModel import *
from dataProcess import *
from dataProcess2 import *

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from network.testNetwork import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

NEED_PATIENT_DIR_CHECK = True
NUM_LABELS = 4
netType='TribleSeq_lr1e5' # just used for naming
mainDir = '/home/rat/data/mri/RedHouse/cleared'

pdc = collectPD(mainDir, needLog = True)
pdc2 = collectPD2(mainDir, needLog = True)
pdc.extend(pdc2)
print(len(pdc))

datasetPara_train = DatasetTribleSeq(pdc[0:1700], 5, needBalance = True)
dataLoaderPara_train = data.DataLoader(datasetPara_train, batch_size = 8, shuffle = True)
datasetPara_valid = DatasetTribleSeq(pdc[1700:], 5, needBalance = False)
dataLoaderPara_valid = data.DataLoader(datasetPara_valid, batch_size = 1, shuffle = False)
print(len(datasetPara_train),len(datasetPara_valid))
trainsetSize = len(datasetPara_train)

if(not os.path.exists('weight/'+netType)):
    os.mkdir('weight/'+netType)
    print("dir created")

LR = 0.00001
EPOCH = 1000
SAVE_EPOCH = 1
SAVE_BUFF = 5
POSITIVE_THRESH = 0.5
dtype = torch.cuda.FloatTensor

net = TribleDenseNetwork()
lossForward = nn.BCELoss()
net = net.type(dtype)

optimizer = optim.Adam(net.parameters(), lr = LR)

ckpt_epoch = 0

for i in range(EPOCH):
    net.train()
    totalLoss = 0
    numLoss = 0
    for seq1, seq2, seq3, label in dataLoaderPara_train:
        seq1 = seq1.type(dtype)
        seq2 = seq2.type(dtype)
        seq3 = seq3[:,:,32:288,32:288].type(dtype)
        label = label.type(dtype)
        optimizer.zero_grad()
        res = net(seq1,seq2,seq3)
        loss = lossForward(res, label)
        loss.backward()
        optimizer.step()
        totalLoss += loss.item() * label.shape[0]
        numLoss += label.shape[0]
        
        print('Epoch %05d [%04d/%04d] loss %.8f' % (i, numLoss, trainsetSize, loss.item()), '\r', end='')
    
    if (i + 1) % SAVE_EPOCH == 0:
        net.eval()
        print('Epoch %05d [%04d/%04d] loss %.8f SAVED' % (i, numLoss, trainsetSize, totalLoss/numLoss))
        IOTool.saveNet(net.state_dict(), 'weight/'+netType, i, SAVE_EPOCH * SAVE_BUFF)
        accNumValid = 0
        totalNumValid = 0
        accNumValid_pos = 0
        totalNumValid_pos = 0
        accNumValid_neg = 0
        totalNumValid_neg = 0
        for seq1, seq2, seq3, label in dataLoaderPara_valid:
            seq1 = seq1.type(dtype)
            seq2 = seq2.type(dtype)
            seq3 = seq3[:,:,32:288,32:288].type(dtype)
            label = label.type(dtype)
            result = net(seq1,seq2,seq3)
            loss = lossForward(result, label)
            #debug
            flagLabel = label > POSITIVE_THRESH
            flagPredict = result > POSITIVE_THRESH
            if(flagLabel): # positive sample
                totalNumValid_pos += 1
                if(flagPredict):
                    accNumValid_pos += 1
            else:
                totalNumValid_neg += 1
                if(not flagPredict):
                    accNumValid_neg += 1
        accNumValid = accNumValid_pos + accNumValid_neg
        totalNumValid = totalNumValid_pos + totalNumValid_neg
        print('[Validation] accuracy:%d/%d  [%.2f%%]  positive:%d/%d  [%.2f%%] negative:%d/%d  [%.2f%%]'%\
              (accNumValid,totalNumValid, accNumValid/totalNumValid * 100, \
               accNumValid_pos, totalNumValid_pos, accNumValid_pos/totalNumValid_pos * 100, \
               accNumValid_neg, totalNumValid_neg, accNumValid_neg/totalNumValid_neg * 100))