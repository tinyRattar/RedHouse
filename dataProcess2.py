import torch.utils.data as data
import warnings
from util import *
from patientModel import *
from dataProcess import DatasetWarningLog
import numpy as np

exceptLength = {"1":256,"2":320,"3":256,"4":320}

def constValidCrop(validSeq, cropSize = 5):
    seqSize = len(validSeq)
    assert cropSize<seqSize, "cropSize is longer than seqSize"
    start = -1
    end = -1
    for i in range(seqSize):
        if(validSeq[i]):
            if(start<0):
                start = i
            end = i

    mid = int((end + start) / 2)
    c1 = int(cropSize/2+0.5)
    c2 = int(cropSize/2)

    i1 = mid - c1 + 1
    i2 = mid + c2 + 1
    if(i1 < 0):
        i2 -= i1
        i1 = 0
    if(i2 > seqSize):
        i1 -= (i2-seqSize)
        i2 = seqSize
    result = np.array([False]*seqSize)
    result[i1:i2] = True

    return result


class DatasetSingleSeq(data.Dataset):
    def __init__(self, listPD, label = "4", seqCropSize = 5, needBalance = False):
        self.listPathNeg = []
        self.listPathPos = []
        self.listSeqNeg = []
        self.listLabelNeg = []
        self.listSeqPos = []
        self.listLabelPos = []
        self.validOnly = True
        
        self.warningLog = DatasetWarningLog()
        
        for pd in listPD:
            pair = pd.getPairPathWithLabel(label)
            if(pair[0] is not None):
                #self.listLabel.append(np.array([pd.label[2]]))
                #self.listLabel.append(pd.label[2])
                seqData = IOTool.loadNiiData(pair[0])
                if(seqData.shape[1] != exceptLength[label]):
                    self.warningLog.logInvalidShape(('\n[Invliad Shape]: ' + str(seqData.shape) + ' in Patient:' + pair[0]))
                    #warnings.warn('\nInvliad Shape: ' + str(seqData.shape) + ' in Patient:' + pair[0])
                    continue
                if(self.validOnly):
                    seqSegData = IOTool.loadNiiData(pair[1])
                    if(len(seqData) != len(seqSegData)):
                        #print(pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData)))
                        self.warningLog.logInconsistentLengthSeq('\n[Match Error]: ' + pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData)))
                        continue
                    #assert len(seqData) == len(seqSegData), pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData))
                    validSeq = validSubSeqFromSeg(seqSegData)
                    if(seqCropSize>0):
                        validSeq = constValidCrop(validSeq, seqCropSize)
                    seqData = seqData[validSeq]
                if(pd.label[2] == 1):
                    self.listPathPos.append(pair[0])
                    self.listSeqPos.append(seqData)
                    self.listLabelPos.append(np.array([1]))
                else:
                    self.listPathNeg.append(pair[0])
                    self.listSeqNeg.append(seqData)
                    self.listLabelNeg.append(np.array([0]))

        if(needBalance):
            if(len(self.listSeqNeg) < len(self.listSeqPos)):
                self.listPathPos = self.listPathPos[:len(self.listSeqNeg)]
                self.listSeqPos = self.listSeqPos[:len(self.listSeqNeg)]
                self.listLabelPos = self.listLabelPos[:len(self.listSeqNeg)]
            else:
                self.listPathNeg = self.listPathNeg[:len(self.listSeqPos)]
                self.listSeqNeg = self.listSeqNeg[:len(self.listSeqPos)]
                self.listLabelNeg = self.listLabelNeg[:len(self.listSeqPos)]
        self.listSeq = self.listSeqPos.copy()
        self.listSeq.extend(self.listSeqNeg)
        self.listLabel = self.listLabelPos.copy()
        self.listLabel.extend(self.listLabelNeg)
        self.listPath = self.listPathPos.copy()
        self.listPath.extend(self.listPathNeg)
                    
        print("="*10)
        self.warningLog.showLog(False)
        print("use dataset.warningLog.showLog(detail=True) for detail")
        print("="*10)
        
    def __getitem__(self, index):
        seqData = self.listSeq[index]
        label = self.listLabel[index]
        
        return seqData, label

    def __len__(self):
        return len(self.listSeq)

class DatasetTribleSeq(data.Dataset):
    def __init__(self, listPD, seqCropSize = 5, needBalance = False):
        self.listPathNeg = []
        self.listPathPos = []
        self.listSeqNeg = []
        self.listLabelNeg = []
        self.listSeqPos = []
        self.listLabelPos = []
        self.validOnly = True
        
        self.warningLog = DatasetWarningLog()

        if(isinstance(seqCropSize, int)):
            seqCropSize = [seqCropSize] * 3
        
        for pd in listPD:
            dicTriSeq = {}
            i = 0
            for label in ["1","3","4"]:
                pair = pd.getPairPathWithLabel(label)
                if(pair[0] is not None):
                    seqData = IOTool.loadNiiData(pair[0])
                    if(seqData.shape[1] != exceptLength[label]):
                        self.warningLog.logInvalidShape(('[Invliad Shape]: ' + pair[0] +"->label "+label+ "Shape " + str(seqData.shape) + " in Patient:"))
                        #warnings.warn('\n[Invliad Shape]: ' + str(seqData.shape) + ' in Patient:' + pair[0])
                        break
                    if(self.validOnly):
                        seqSegData = IOTool.loadNiiData(pair[1])
                        if(len(seqData) != len(seqSegData)):
                            self.warningLog.logInconsistentLengthSeq('[Match Error]: ' + pair[0]+"->label "+label+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData)))
                            break
                        validSeq = validSubSeqFromSeg(seqSegData)
                        if(seqCropSize[i]>0):
                            validSeq = constValidCrop(validSeq, seqCropSize[i])
                        seqData = seqData[validSeq]
                    dicTriSeq[label] = seqData
                    i += 1
                else:
                    break
            if(len(dicTriSeq) != 3):
                continue
            if(pd.label[2] == 1):
                self.listPathPos.append(pair[0])
                self.listSeqPos.append(dicTriSeq)
                self.listLabelPos.append(np.array([1]))
            else:
                self.listPathNeg.append(pair[0])
                self.listSeqNeg.append(dicTriSeq)
                self.listLabelNeg.append(np.array([0]))


        if(needBalance):
            if(len(self.listSeqNeg) < len(self.listSeqPos)):
                self.listPathPos = self.listPathPos[:len(self.listSeqNeg)]
                self.listSeqPos = self.listSeqPos[:len(self.listSeqNeg)]
                self.listLabelPos = self.listLabelPos[:len(self.listSeqNeg)]
            else:
                self.listPathNeg = self.listPathNeg[:len(self.listSeqPos)]
                self.listSeqNeg = self.listSeqNeg[:len(self.listSeqPos)]
                self.listLabelNeg = self.listLabelNeg[:len(self.listSeqPos)]
        self.listSeq = self.listSeqPos.copy()
        self.listSeq.extend(self.listSeqNeg)
        self.listLabel = self.listLabelPos.copy()
        self.listLabel.extend(self.listLabelNeg)
        self.listPath = self.listPathPos.copy()
        self.listPath.extend(self.listPathNeg)
                    
        print("="*10)
        self.warningLog.showLog(False)
        print("use dataset.warningLog.showLog(detail=True) for detail")
        print("="*10)
        
    def __getitem__(self, index):
        dicSeqData = self.listSeq[index]
        label = self.listLabel[index]
        seq1 = dicSeqData["1"]
        seq2 = dicSeqData["3"]
        seq3 = dicSeqData["4"]
        
        return seq1, seq2, seq3, label

    def __len__(self):
        return len(self.listSeq)