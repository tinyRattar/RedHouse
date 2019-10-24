import torch.utils.data as data
import warnings
from util import *
from patientModel import *

NEED_PATIENT_DIR_CHECK = True

def collectPD(mainDir, needLog = False):
#     listDirDates = os.listdir(mainDir)
#     if(needLog):
#         print(listDirDates)
#     for dirDate in listDirDates:
    listDirDoctors = []
    for dirDate in ['DONE0705','DONE0712']:
        tempListDir = os.listdir(mainDir+'/'+dirDate)
        for dirDoctor in tempListDir:
            listDirDoctors.append(dirDate+'/'+dirDoctor)
    if(needLog):
        print(listDirDoctors)
    
    pdc = PatientDataCollection()
    for dirDoctor in listDirDoctors:
        tempListDirPatients = os.listdir(mainDir + '/' + dirDoctor + '/main')
        # region Check
        if(NEED_PATIENT_DIR_CHECK):
            tempListDirPatientsSeg = os.listdir(mainDir + '/' + dirDoctor + '/segmentation')
            if(tempListDirPatients != tempListDirPatientsSeg):
                print("++DIFF:")
                DebugTool.diffList(tempListDirPatients, tempListDirPatientsSeg)
                assert False, "Inconsistent Patients Dir for Main/Seg in " + dirDoctor
        # endRegion
        for rawPatient in tempListDirPatients:
            tempSplit = rawPatient.split('-')
            tempRGID = tempSplit[0]
            tempTypeArray = TypeTool.stringListToArray(tempSplit[1:])
            #print(tempTypeArray,tempSplit[1:])
            tempPD = PatientData(mainDir + '/' + dirDoctor+'/main/'+rawPatient, mainDir + '/' + dirDoctor+'/segmentation/'+rawPatient, tempTypeArray)
            #listDirPatient.append(tempPD)
            pdc.append(tempPD)
    return pdc

def collectPD2(mainDir, needLog = False):
    pdc = PatientDataCollection()
    listDirPrefix = ['MRI_2']
    for prefix in listDirPrefix:
        tempListDoctors = os.listdir(mainDir + '/' + prefix + '/main')
        if(needLog):
            print(tempListDoctors)
        for doctor in tempListDoctors:
            doctorIndex = doctor.split('-')[0]
            tempListDirCancerType = os.listdir(mainDir + '/' + prefix + '/main/' + doctor)
            for cancerType in tempListDirCancerType:
                # assert cancerType in ["4-SCC","5-ACIS","6-ADE"], "Wrong Cancer Type in "+(prefix + '/main/'+doctor)+":"+cancerType
                if(not cancerType in ["4-SCC","5-ACIS","6-ADE"]):
                    warnings.warn("Wrong Cancer Type in "+(prefix + '/main/'+doctor)+":"+cancerType)
                tempListDirPatients = os.listdir(mainDir + '/' + prefix + '/main/' + doctor+'/'+cancerType)
                # region Check
                if(NEED_PATIENT_DIR_CHECK):
                    tempListDirPatientsSeg = os.listdir(mainDir + '/' + prefix + '/segment/' + doctorIndex+'/'+cancerType)
                    if(tempListDirPatients != tempListDirPatientsSeg):
                        print("++DIFF:")
                        DebugTool.diffList(tempListDirPatients, tempListDirPatientsSeg)
                        assert False, "Inconsistent Patients Dir for Main/Seg in " + (prefix + '/main/'+doctor+'/'+cancerType)
                # endRegion
                for rawPatient in tempListDirPatients:
                    tempSplit = rawPatient.split('-')
                    tempRGID = tempSplit[0]
                    tempTypeArray = TypeTool.stringListToArray(tempSplit[1:])
                    tmpMainDir = mainDir + '/' + prefix + '/main/' + doctor+'/'+cancerType+'/'+rawPatient
                    tmpSegDir = mainDir + '/' + prefix + '/segment/' + doctorIndex+'/'+cancerType+'/'+rawPatient
                    #print(tmpMainDir)
                    tempPD = PatientData(tmpMainDir, tmpSegDir, tempTypeArray)
                    pdc.append(tempPD)
    return pdc

class DatasetWarningLog():
    def __init__(self):
        self.countInvalidShape = 0
        self.listInvalidShape = []
        self.countInconsistentLengthSeq = 0
        self.listInconsistentLengthSeq = []
    
    def logInvalidShape(self,log):
        self.countInvalidShape += 1
        self.listInvalidShape.append(log)
        
    def logInconsistentLengthSeq(self,log):
        self.countInconsistentLengthSeq += 1
        self.listInconsistentLengthSeq.append(log)
        
    def showLog(self, detail = False):
        print("Find %d Invalid Shape"%self.countInvalidShape)
        if(detail):
            for log in self.listInvalidShape:
                print(log)
        print("Find %d Inconsitent length"%self.countInconsistentLengthSeq)
        if(detail):
            for log in self.listInconsistentLengthSeq:
                print(log)

class DatasetPara(data.Dataset):
    def __init__(self, listPD, validOnly = False):
        self.listSeq = []
        self.listLabel = []
        self.validOnly = validOnly
        
        self.warningLog = DatasetWarningLog()
        
        for pd in listPD:
            pair = pd.getPairPathWithLabel("4")
            if(pair[0] is not None):
                self.listLabel.append(np.array([pd.label[2]]))
                #self.listLabel.append(pd.label[2])
                seqData = IOTool.loadNiiData(pair[0])
                if(seqData.shape[1] != 320):
                    self.warningLog.logInvalidShape(('\nInvliad Shape: ' + str(seqData.shape) + ' in Patient:' + pair[0]))
                    warnings.warn('\nInvliad Shape: ' + str(seqData.shape) + ' in Patient:' + pair[0])
                    continue
                if(self.validOnly):
                    seqSegData = IOTool.loadNiiData(pair[1])
                    if(len(seqData) != len(seqSegData)):
                        #print(pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData)))
                        self.warningLog.logInconsistentLengthSeq(pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData)))
                        continue
                    assert len(seqData) == len(seqSegData), pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData))
                    validSeq = validSubSeqFromSeg(seqSegData)
                    self.listSeq.append(seqData[validSeq])
                else:
                    self.listSeq.append(seqData)
                    
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
    
class DatasetParaImg(data.Dataset):
    def __init__(self, listPD, validOnly = False):
        self.listImg = []
        self.listLabel = []
        self.validOnly = validOnly
        
        self.warningLog = DatasetWarningLog()
        
        for pd in listPD:
            pair = pd.getPairPathWithLabel("4")
            if(pair[0] is not None):
                seqData = IOTool.loadNiiData(pair[0])
                if(seqData.shape[1] != 320):
                    self.warningLog.logInvalidShape(('\nInvliad Shape: ' + str(seqData.shape) + ' in Patient:' + pair[0]))
                    warnings.warn('\nInvliad Shape: ' + str(seqData.shape) + ' in Patient:' + pair[0])
                    continue
                if(self.validOnly):
                    seqSegData = IOTool.loadNiiData(pair[1])
                    if(len(seqData) != len(seqSegData)):
                        #print(pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData)))
                        self.warningLog.logInconsistentLengthSeq(pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData)))
                        continue
                    assert len(seqData) == len(seqSegData), pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData))
                    validSeq = validSubSeqFromSeg(seqSegData)
                    seqData = seqData[validSeq]
                for i in range(seqData.shape[0]):
                    self.listImg.append(seqData[i:i+1])
                    self.listLabel.append(np.array([pd.label[2]]))
                    
        print("="*10)
        self.warningLog.showLog(False)
        print("use dataset.warningLog.showLog(detail=True) for detail")
        print("="*10)
        
    def __getitem__(self, index):
        imgData = self.listImg[index]
        label = self.listLabel[index]
        
        return imgData, label

    def __len__(self):
        return len(self.listImg)

class DatasetParaImgBalance(data.Dataset):
    def __init__(self, listPD, validOnly = False):
        self.listImgNeg = []
        self.listLabelNeg = []
        self.listImgPos = []
        self.listLabelPos = []
        self.validOnly = validOnly
        
        self.warningLog = DatasetWarningLog()
        
        for pd in listPD:
            pair = pd.getPairPathWithLabel("4")
            if(pair[0] is not None):
                seqData = IOTool.loadNiiData(pair[0])
                if(seqData.shape[1] != 320):
                    self.warningLog.logInvalidShape(('\nInvliad Shape: ' + str(seqData.shape) + ' in Patient:' + pair[0]))
                    warnings.warn('\nInvliad Shape: ' + str(seqData.shape) + ' in Patient:' + pair[0])
                    continue
                if(self.validOnly):
                    seqSegData = IOTool.loadNiiData(pair[1])
                    if(len(seqData) != len(seqSegData)):
                        #print(pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData)))
                        self.warningLog.logInconsistentLengthSeq(pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData)))
                        continue
                    assert len(seqData) == len(seqSegData), pair[0]+"Inconsistent length of seq Main/Seg "+str(len(seqData))+"/"+str(len(seqSegData))
                    validSeq = validSubSeqFromSeg(seqSegData)
                    seqData = seqData[validSeq]
                for i in range(seqData.shape[0]):
                    if(pd.label[2] == 1):
                        self.listImgPos.append(seqData[i:i+1])
                        self.listLabelPos.append(np.array([1]))
                    else:
                        self.listImgNeg.append(seqData[i:i+1])
                        self.listLabelNeg.append(np.array([0]))

        if(len(self.listImgNeg) < len(self.listImgPos)):
            self.listImgPos = self.listImgPos[:len(self.listImgNeg)]
            self.listLabelPos = self.listLabelPos[:len(self.listImgNeg)]
        else:
            self.listImgNeg = self.listImgNeg[:len(self.listImgPos)]
            self.listLabelNeg = self.listLabelNeg[:len(self.listImgPos)]
        self.listImg = self.listImgPos.copy()
        self.listImg.extend(self.listImgNeg)
        self.listLabel = self.listLabelPos.copy()
        self.listLabel.extend(self.listLabelNeg)
        
        print("="*10)
        self.warningLog.showLog(False)
        print("use dataset.warningLog.showLog(detail=True) for detail")
        print("="*10)

        
    def __getitem__(self, index):
        imgData = self.listImg[index]
        label = self.listLabel[index]
        
        return imgData, label

    def __len__(self):
        return len(self.listImg)
    