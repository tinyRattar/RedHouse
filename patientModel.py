import os
from util import *
import warnings

class PatientData():
    #_dirPrefixs = []
    def __init__(self, dMain, dSeg, label):
        self.dSeg = dSeg
        self.dMain = dMain
        self.label = label
        self.init()
        
    def __str__(self):
        return PatientData._dirPrefixs[self.prefixIndex] + " " + self.dirSuffix + " " + str(self.label)
    
    def init(self):
        self.dicLabelPair = {}
        self.dicSeqPair = {}
        listSeqPath = os.listdir(self.getPath())
        for seqPath in listSeqPath:
            pSplit = seqPath.split('.')
            seqID = '.'.join(pSplit[:-1])
            #self.listSeqID.append(seqID)
            self.dicSeqPair[seqID] = []
        listSeqPath_seg = filterGzFilename(os.listdir(self.getPath(False)))
        for seqPath_seg in listSeqPath_seg:
            pSplit = seqPath_seg.split('.')
            seqID, labelType = '.'.join(pSplit[:-3]), pSplit[-3]
            if(seqID in self.dicSeqPair.keys()):
                self.dicSeqPair[seqID].append(labelType)
            else:
                assert False, "No Such Seq! Key Error in "+ self.getPath() + " with Key " + seqID
            self.dicLabelPair[labelType] = seqID
            
    def _getPrefix(self):
        return PatientData._dirPrefixs[self.formerIndex]
    
    @staticmethod
    def _prefixSerialization(dPrefix):
        if(dPrefix in PatientData._dirPrefixs):
            return PatientData._dirPrefixs.index(dPrefix)
        else:
            PatientData._dirPrefixs.append(dPrefix)
            return len(PatientData._dirPrefixs) - 1
            
    def getPairPathWithLabel(self, label):
        if(not isinstance(label, str)):
            label = str(label)
        if(not label in self.dicLabelPair.keys()):
            warnings.warn('\nNo Label ' + label + ' in Patient:' + self.getPath())
            return None, None
        seqID = self.dicLabelPair[label]
        return self.getPath() + '/' + seqID + '.nii', self.getPath(False) + '/' + seqID + '.' + label + '.nii.gz'

    def getPathMain(self):
        return self.dMain
    
    def getPathSeg(self):
        return self.dSeg
    
    def getPath(self, isMain = True):
        if(isMain):
            return self.getPathMain()
        else:
            return self.getPathSeg()

# class PatientData:
#     _dirPrefixs = []
#     def __init__(self, dPrefix, dSuffix, label):
#         self.prefixIndex = PatientData._prefixSerialization(dPrefix)
#         self.dirSuffix = dSuffix
#         self.label = label
#         self.init()
        
#     def __str__(self):
#         return PatientData._dirPrefixs[self.prefixIndex] + " " + self.dirSuffix + " " + str(self.label)
    
#     def init(self):
#         self.dicLabelPair = {}
#         self.dicSeqPair = {}
#         listSeqPath = os.listdir(self.getPath())
#         for seqPath in listSeqPath:
#             pSplit = seqPath.split('.')
#             seqID = '.'.join(pSplit[:-1])
#             #self.listSeqID.append(seqID)
#             self.dicSeqPair[seqID] = []
#         listSeqPath_seg = filterGzFilename(os.listdir(self.getPath(False)))
#         for seqPath_seg in listSeqPath_seg:
#             pSplit = seqPath_seg.split('.')
#             seqID, labelType = '.'.join(pSplit[:-3]), pSplit[-3]
#             if(seqID in self.dicSeqPair.keys()):
#                 self.dicSeqPair[seqID].append(labelType)
#             else:
#                 assert False,"Key Error in "+ self.getPath() + " with Key " + seqID
#             self.dicLabelPair[labelType] = seqID
            
#     def _getPrefix(self):
#         return PatientData._dirPrefixs[self.formerIndex]
    
#     @staticmethod
#     def _prefixSerialization(dPrefix):
#         if(dPrefix in PatientData._dirPrefixs):
#             return PatientData._dirPrefixs.index(dPrefix)
#         else:
#             PatientData._dirPrefixs.append(dPrefix)
#             return len(PatientData._dirPrefixs) - 1
            
#     def getPairPathWithLabel(self, label):
#         if(not isinstance(label, str)):
#             label = str(label)
#         if(not label in self.dicLabelPair.keys()):
#             warnings.warn('\nNo Label ' + label + ' in Patient:' + self.getPath())
#             return None, None
#         seqID = self.dicLabelPair[label]
#         return self.getPath() + '/' + seqID + '.nii', self.getPath(False) + '/' + seqID + '.' + label + '.nii.gz'

#     def getPathMain(self):
#         return PatientData._dirPrefixs[self.prefixIndex] + '/main/' + self.dirSuffix
    
#     def getPathSeg(self):
#         return PatientData._dirPrefixs[self.prefixIndex] + '/segmentation/' + self.dirSuffix
    
#     def getPath(self, isMain = True):
#         if(isMain):
#             return self.getPathMain()
#         else:
#             return self.getPathSeg()
        
class PatientDataCollection:
    def __init__(self):
        self.listPatient = []
        
    def __str__(self):
        tempString = "PatientDataCollection:\n"
        for p in self.listPatient:
            tempString += str(p) + "\n"
        return tempString
    
    def __len__(self):
        return len(self.listPatient)
    
    def __getitem__(self, index):
        return self.listPatient[index]
        
    def append(self, patient):
        self.listPatient.append(patient)
        
    def extend(self, tarPDC):
        for patient in tarPDC:
            self.listPatient.append(patient)