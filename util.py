import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
import torch
import nibabel as nib

def filterGzFilename(fList):
    fListOut = []
    for fname in fList:
        fsplit = fname.split('.')
        if(fsplit[-1] == 'gz'):
            fListOut.append(fname)
            assert fsplit[-2] == 'nii', 'File Format Error:' + fname
    return fListOut

def validSubSeqFromSeg(seqSeg):
    vectorSum = np.sum(seqSeg, (1,2))
    vectorFlag = vectorSum > 0
    
    return vectorFlag

class IOTool:
    @staticmethod
    def showSize(h,w=0):
        if(w==0):
            w =h
        pylab.rcParams['figure.figsize'] = (h, w) # 显示大小
    
    @staticmethod
    def imshow(img, mode='pil', vmax=0, overlap=False):
        #assert False, no more imshow
        if(mode == 'cv'):
            b,g,r = cv2.split(img)
            im_np = cv2.merge([r,g,b])
            plt.imshow(im_np)
        elif(mode == 'pil'):
            if(vmax!=0):
                plt.imshow(img,vmax=vmax)
            else:
                plt.imshow(img)
        elif(mode == 'g'):
            plt.imshow(img,'gray')
        elif(mode == 'b'):
            plt.imshow(img,'binary')
        elif(mode == 'c'):
            if('complex' in str(img.dtype)):
                showImg = img
            else:
                showImg = fc2c(img)
            plt.imshow(abs(showImg),'gray')
        else:
            assert False,"wrong mode"

        plt.axis('off')
        if(not overlap):
            plt.show()
    
    @staticmethod
    def loadNiiData(filename):
        f_nii = nib.load(filename)
        seqData = np.array(f_nii.get_data())
        seqData = np.transpose(seqData, (2, 0, 1))
        return seqData
    
    @staticmethod
    def saveNet(param, saveDir, epochNum, epochOffset, checkpoint=False):
        if(not saveDir == ''):
            tmpDir = saveDir + '/'
        tmpPath = tmpDir+"saved"+"_"+str(epochNum)+".pkl"
        if(checkpoint):
            tmpPath = tmpDir+"CHECKED_saved"+"_"+str(epochNum)+".pkl"
        oldNum = epochNum - epochOffset
        oldPath = tmpDir+"saved"+"_"+str(oldNum)+".pkl"
        if(os.path.exists(oldPath)):
            os.remove(oldPath)
        torch.save(param,tmpPath)
        

class TypeTool:
    enumType = ['others', 'naboth', 'para', 'vagina', 'lnm']
    
    @staticmethod
    def stringListToByteCode(sList):
        code = 0b00000000
        for s in sList:
            assert s in TypeTool.enumType, 'Wrong Type:' + s
            code += pow(2, TypeTool.enumType.index(s))
        return code
    
    @staticmethod
    def stringListToArray(sList):
        label = np.zeros(len(TypeTool.enumType))
        for s in sList:
            assert s in TypeTool.enumType, 'Wrong Type:' + s
            label[TypeTool.enumType.index(s)] += 1
        return label
    
class DebugTool:
    @staticmethod
    def diffListSingle(list1, list2, isReverse = False):
        flag = True
        if(isReverse):
            temp = "List 1"
            tList = list1
            list1 = list2
            list2 = tList
        else:
            temp = "List 2"
        for item in list1:
            if(not item in list2):
                print("No " + str(item) + " in " + temp)
                flag = False
                
        return flag
                
    @staticmethod
    def diffList(list1, list2):
        flag = True
        if(not DebugTool.diffListSingle(list1, list2)):
            flag = False
        if(not flag):
            print("---------")
        if(not DebugTool.diffListSingle(list1, list2, True)):
            flag = False
            
        return flag
    
    @staticmethod
    def listSeqCheck(listMain, listSeg):
        listSeg_noLabel = []
        for fname in listSeg:
            fname_noLabel = ''
            fsplit = fname.split('.')
            for s in fsplit[:-3]:
                fname_noLabel += (s + '.')
            fname_noLabel += 'nii'
            listSeg_noLabel.append(fname_noLabel)
        if(listMain == listSeg_noLabel):
            return True
        else:
            return DebugTool.diffList(listMain, listSeg_noLabel)
