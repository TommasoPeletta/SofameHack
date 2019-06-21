import btk
import sys
import numpy as np
import os
import pandas as pd
import math

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.cluster import KMeans


#calul the precise event and add one if it was forget
def computelist(dfresult, list, pied, cur, filename):
    if (list[-1] - list[0] >= 30):
        # On fait le centroid
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(list).reshape(-1,1))
        centroid = kmeans.cluster_centers_
        dfresult = dfresult.append({'video' : filename , 'pied' : pied, 'event' : cur, 'frame': math.floor(centroid[1][0])}, ignore_index=True)
        dfresult = dfresult.append({'video' : filename , 'pied' : pied, 'event' : cur, 'frame': math.floor(centroid[0][0])}, ignore_index=True)
        if cur == 'Foot_Off_GS':
            cur = 'Foot_Strike_GS'
        else:
            cur = 'Foot_Off_GS'
        dfresult = dfresult.append({'video' : filename , 'pied' : pied, 'event' : cur, 'frame': math.floor(np.mean(centroid))}, ignore_index=True)

        return dfresult

    if len(list) > 2:
        res = math.floor(np.mean(list[0:len(list)-1]))
    else:
        res =  math.floor(np.mean(list))

    return dfresult.append({'video' : filename , 'pied' : pied, 'event' : cur, 'frame': res}, ignore_index=True)

def logistic(x_train, y_train):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train,y_train)
    return logisticRegr

def NaivesBayes(x_train, y_train):
    dtree_model = DecisionTreeClassifier().fit(x_train,y_train)
    return dtree_model

def KNN(x_train, y_train):
    knn = KNeighborsClassifier(n_neighbors = 3, metric = 'euclidean')
    knn.fit(x_train,y_train)
    return knn


def MLP(x_train, y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(15,15,15),max_iter=500)
    mlp.fit(x_train,y_train)
    return mlp


def Data(neighbours, directoire, dirpath):
    k = 0
    data = pd.DataFrame(columns = ['video', 'pied','event','frame', 'TOEz-mean_TOEz','TOEx-HEEx', 'KNEx-TOEx','min_pied'])
    for d in directoire:
        for filename in os.listdir(dirpath + d):
            path = dirpath + d + filename
            reader = btk.btkAcquisitionFileReader()
            reader.SetFilename(path)
            reader.Update()
            acq = reader.GetOutput()
            for nEve in range(100):
                try:
                    event = acq.GetEvent(nEve) # extract the first event of the aquisition
                    #print(event.GetLabel()) # return a string representing the Label
                    if (event.GetContext() == "Left"):
                        pied = "left"
                        argument = ['LTOE','LHEE','LPSI', 'LKNE']
                        mean_TOE = np.mean(acq.GetPoint('LTOE').GetValues()[:,2])
                        mean_dif = np.mean(acq.GetPoint('LTOE').GetValues()[:,0]-acq.GetPoint('LHEE').GetValues()[:,0])

                    else:
                        pied = "right"
                        argument = ['RTOE','RHEE','RPSI', 'RKNE']
                        mean_TOE = np.mean(acq.GetPoint('RTOE').GetValues()[:,2])
                        mean_dif = np.mean(acq.GetPoint('RTOE').GetValues()[:,0]-acq.GetPoint('RHEE').GetValues()[:,0])

                    frame = event.GetFrame()
                    for i in neighbours: #[-15,0,15]: #
                        eventname = "Not_Event"
                        if i == 0:
                            eventname = event.GetLabel()
                        data  = data.append({'video' : filename , 'pied' : pied, 'event' : eventname, 'frame': frame+i,
                        'TOEz-mean_TOEz': acq.GetPoint(argument[0]).GetValues()[frame+i,2]-mean_TOE,
                        'TOEx-HEEx': acq.GetPoint(argument[0]).GetValues()[frame+i,0]-acq.GetPoint(argument[1]).GetValues()[frame+i,0]-mean_dif,
                         'KNEx-TOEx': acq.GetPoint(argument[3]).GetValues()[frame+i,0]-acq.GetPoint(argument[0]).GetValues()[frame+i,0],
                         'min_pied': min(acq.GetPoint(argument[0]).GetValues()[frame+i,2], acq.GetPoint(argument[1]).GetValues()[frame+i,2])}, ignore_index=True)


                except Exception as e:
                    break
    return data

def ParseFileName(dataframe):
    filename = []
    idlist = []
    df = dataframe.copy()
    df.drop_duplicates(subset = "video", inplace = True)
    for index,row in df.iterrows():
        filename.append(row['video'])
        tmp_list = row['video'].split("_")
        for j in range(len(tmp_list)):
            if (not (tmp_list[j][0]).isalpha()):
                idlist.append(tmp_list[j])
                break
    idset = list(set(idlist))

    return filename,idlist,idset

def SplitData(pos, idset, idlist, filename):
    idTest = []
    idTrain = []
    xTrain = []
    xTest = []
    for i in range(len(idset)):
        if (pos == 0):
            if (i <= math.ceil(len(idset)* (2.0/3.0))):
                idTrain.append(idset[i])
            else:
                idTest.append(idset[i])
        if (pos == 2):
            if (i >= math.floor(len(idset) * (1.0/3.0))):
                idTrain.append(idset[i])
            else:
                idTest.append(idset[i])
        if (pos == 1):
            if (i <= math.floor(len(idset) * (1.0/3.0)) or i >= math.floor(len(idset) * (2.0/3.0))):
                idTrain.append(idset[i])
            else:
                idTest.append(idset[i])
    for i in range(len(idlist)):
        if (idlist[i] in idTrain):
            xTrain.append(filename[i])
        else:
            xTest.append(filename[i])
    return xTrain, xTest


def test(dataFrame, listTrain, listTest, dir , dirpath):
    dataTrain = pd.DataFrame(columns = ['video', 'pied','event','frame', 'TOEz-mean_TOEz','TOEx-HEEx', 'KNEx-TOEx','min_pied'])
    dataTest = pd.DataFrame(columns = ['video', 'pied','event','frame', 'TOEz-mean_TOEz','TOEx-HEEx', 'KNEx-TOEx','min_pied'])
    for el in listTrain:
        dataTrain = dataTrain.append(dataFrame.loc[dataFrame['video'] == el])
    for el in listTest:
        dataTest = dataTest.append(dataFrame.loc[dataFrame['video'] == el])
    model =  KNeighborsClassifier(n_neighbors = 3, metric = 'euclidean').fit(dataTrain[['TOEz-mean_TOEz','TOEx-HEEx', 'KNEx-TOEx','min_pied']],dataTrain.event)
    model = DecisionTreeClassifier().fit(dataTrain[['TOEz-mean_TOEz','TOEx-HEEx', 'KNEx-TOEx','min_pied']],dataTrain.event)
    print(model.feature_importances_)

    dataframeresult = testmodel(model, dir, dirpath, listTest)
    dfEvent = dataTest.loc[dataTest['event'] != "Not_Event"]
    nligneInit = dfEvent.shape[0]
    diffTestOff = []
    diffTestStrick = []
    for el in range(nligneInit):
        value = np.min(np.abs(dataframeresult.loc[(dataframeresult['video'] == dfEvent.iloc[el,0]) & (dataframeresult['pied'] == dfEvent.iloc[el,1])
            & (dataframeresult['event'] == dfEvent.iloc[el,2]), 'frame'] - dfEvent.iloc[el,3]))
        if (value > 10):
            print(dfEvent.iloc[el,:])
            print(dataframeresult.loc[dataframeresult['video'] == dfEvent.iloc[el,0]])
        if (dfEvent.iloc[el,2] == "Foot_Off_GS"):
            diffTestOff = np.append(diffTestOff, value)
        elif (dfEvent.iloc[el,2] == "Foot_Strike_GS"):
            diffTestStrick = np.append(diffTestStrick, value)
        if (math.isnan(value)):
            print(el)
            print(dfEvent.iloc[el,:])


    #print(model.feature_importances_)
    MeanSumOff = np.sum(diffTestOff)
    MeanExpoOff = np.sum(np.exp(diffTestOff), axis = 0)
    MeanSumStrick = np.sum(diffTestStrick)
    MeanExpoStrick = np.sum(np.exp(diffTestStrick), axis = 0)
    print(" sum off ", MeanSumOff)
    print(" expo off ", MeanExpoOff)
    print("sum strick ", MeanSumStrick)
    print("expo strick ", MeanExpoStrick)
    return MeanSumOff, MeanExpoOff, MeanSumStrick, MeanExpoStrick



def testmodel(model, directoire, dirpath, listTest):
    dfresult = pd.DataFrame(columns = ['video', 'pied','event','frame'])

    for d in directoire:
        for filename in listTest:
            path = dirpath + d + filename

            reader = btk.btkAcquisitionFileReader()
            #path = 'Sofamehack2019/Sub_DB_Checked/CP/CP_GMFCS1_01916_20130128_18.c3d'
            reader.SetFilename(path)
            reader.Update()
            acq = reader.GetOutput()
            mean_TOE = np.mean(acq.GetPoint('LTOE').GetValues()[:,2])
            mean_dif = np.mean(acq.GetPoint('LTOE').GetValues()[:,0]-acq.GetPoint('LHEE').GetValues()[:,0])

            data_FrameRef = np.concatenate((np.transpose(np.array([acq.GetPoint('LTOE').GetValues()[:,2]-mean_TOE])), np.transpose(np.array([acq.GetPoint('LTOE').GetValues()[:,0]-acq.GetPoint('LHEE').GetValues()[:,0]-mean_dif]))), axis = 1)
            #data_FrameRef = np.concatenate((data_FrameRef, np.transpose(np.array([acq.GetPoint('LPSI').GetValues()[:,2]-mean_PSI]))), axis = 1)
            data_FrameRef = np.concatenate((data_FrameRef, np.transpose(np.array([acq.GetPoint('LKNE').GetValues()[:,0]-acq.GetPoint('LTOE').GetValues()[:,0]]))),axis = 1)
            data_FrameRef = np.concatenate((data_FrameRef, np.transpose(np.minimum(np.array([acq.GetPoint('LTOE').GetValues()[:,2]]),np.array([acq.GetPoint('LHEE').GetValues()[:,2]])))),axis = 1)

            P =model.predict(data_FrameRef)
            df = pd.DataFrame(data=P, columns = ['result'])
            dfEvent = df.loc[df['result'] != 'Not_Event']
            nligne = dfEvent.shape[0]
            dfEvent['index'] = dfEvent.index

            current = dfEvent.iloc[0,0]
            comp_list = [dfEvent.iloc[0,1]]
            for i in range(nligne-1):
                if (current !=dfEvent.iloc[i+1,0]):
                    dfresult = computelist(dfresult, comp_list, 'left', current, filename)
                    current = dfEvent.iloc[i+1,0]
                    comp_list = [dfEvent.iloc[i+1,1]]

                else:
                    comp_list.append(dfEvent.iloc[i+1,1])
                if (i == nligne-2):
                    dfresult = computelist(dfresult, comp_list, 'left', current, filename)




    for d in directoire:
        for filename in listTest:
            path = dirpath + d + filename

            reader = btk.btkAcquisitionFileReader()
            #path = 'Sofamehack2019/Sub_DB_Checked/CP/CP_GMFCS1_01916_20130128_18.c3d'
            reader.SetFilename(path)
            reader.Update()
            acq = reader.GetOutput()
            mean_TOE = np.mean(acq.GetPoint('RTOE').GetValues()[:,2])
            mean_dif = np.mean(acq.GetPoint('RTOE').GetValues()[:,0]-acq.GetPoint('RHEE').GetValues()[:,0])

            data_FrameRef = np.concatenate((np.transpose(np.array([acq.GetPoint('RTOE').GetValues()[:,2]-mean_TOE])), np.transpose(np.array([acq.GetPoint('RTOE').GetValues()[:,0]-acq.GetPoint('RHEE').GetValues()[:,0]-mean_dif]))), axis = 1)

            #data_FrameRef = np.concatenate((data_FrameRef, np.transpose(np.array([acq.GetPoint('RPSI').GetValues()[:,2]-mean_PSI]))), axis = 1)
            data_FrameRef = np.concatenate((data_FrameRef, np.transpose(np.array([acq.GetPoint('RKNE').GetValues()[:,0]-acq.GetPoint('RTOE').GetValues()[:,0]]))),axis = 1)
            data_FrameRef = np.concatenate((data_FrameRef, np.transpose(np.minimum(np.array([acq.GetPoint('RTOE').GetValues()[:,2]]),np.array([acq.GetPoint('RHEE').GetValues()[:,2]])))),axis = 1)
            P =model.predict(data_FrameRef)
            df = pd.DataFrame(data=P, columns = ['result'])
            dfEvent = df.loc[df['result'] != 'Not_Event']
            nligne = dfEvent.shape[0]
            dfEvent['index'] = dfEvent.index

            current = dfEvent.iloc[0,0]
            comp_list = [dfEvent.iloc[0,1]]
            for i in range(nligne-1):
                if (current !=dfEvent.iloc[i+1,0]):
                    dfresult = computelist(dfresult, comp_list, 'right', current, filename)
                    current = dfEvent.iloc[i+1,0]
                    comp_list = [dfEvent.iloc[i+1,1]]

                else:
                    comp_list.append(dfEvent.iloc[i+1,1])
                if (i == nligne-2):
                    dfresult = computelist(dfresult, comp_list, 'right', current, filename)
    return dfresult


dirpath = './Sofamehack2019/Sub_DB_Checked/'
dir = ['CP/']

framelist = [0,9]
dataFrame = Data(framelist, dir, dirpath)
[filename, idPersonne, setPersonne] = ParseFileName(dataFrame)
MeanSumOff = np.zeros(3)
MeanExpoOff = np.zeros(3)
MeanSumStrick = np.zeros(3)
MeanExpoStrick = np.zeros(3)
for i in range(3):
    [listTrain, listTest] = SplitData(i, setPersonne, idPersonne, filename)
    [MeanSumOff[i], MeanExpoOff[i], MeanSumStrick[i], MeanExpoStrick[i]] = test(dataFrame, listTrain, listTest, dir , dirpath)

print(" list Mean Expo Off ", MeanExpoOff)
print(" Erreur sum total Off: ", np.mean(MeanSumOff))
print(" Erreur exponentiel Off: ", np.mean(MeanExpoOff))
print(" Erreur sum total Strick: ", np.mean(MeanSumStrick))
print(" Erreur exponentiel Strick: ", np.mean(MeanExpoStrick))
