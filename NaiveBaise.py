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


def computelist(dfresult, list, pied, cur, filename):
    if (filename == 'Cavus_Foot_01993_20130923_10.c3d'):
        if (list[len(list)-1] - list[0] >= 30):
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




def dataCollect(neighbours, directoire, dirpath):
    k = 0
    dfEventInit = pd.DataFrame(columns = ['video', 'pied','event','frame'])
    for d in directoire:
        for filename in os.listdir(dirpath + d):
            path = dirpath + d + filename
            reader = btk.btkAcquisitionFileReader()
            reader.SetFilename(path)
            reader.Update()
            acq = reader.GetOutput()

            mean_PSI = np.mean(acq.GetPoint('LPSI').GetValues()[:,2])
            for nEve in range(100):
                try:
                    event = acq.GetEvent(nEve) # extract the first event of the aquisition
                    #print(event.GetLabel()) # return a string representing the Label
                    if (event.GetContext() == "Left"):
                        pied = "left"
                        argument = ['LTOE','LHEE','LPSI', 'LKNE']
                        mean_PSI = np.mean(acq.GetPoint('LPSI').GetValues()[:,2])
                        max_PSI = np.max(acq.GetPoint('LPSI').GetValues()[:,2])
                        min_PSI = np.min(acq.GetPoint('LPSI').GetValues()[:,2])
                        mean_TOE = np.mean(acq.GetPoint('LTOE').GetValues()[:,2])

                    else:
                        pied = "right"
                        argument = ['RTOE','RHEE','RPSI', 'RKNE']
                        mean_PSI = np.mean(acq.GetPoint('RPSI').GetValues()[:,2])
                        max_PSI = np.max(acq.GetPoint('RPSI').GetValues()[:,2])
                        min_PSI = np.min(acq.GetPoint('RPSI').GetValues()[:,2])
                        mean_TOE = np.mean(acq.GetPoint('RTOE').GetValues()[:,2])

                    frame = event.GetFrame()
                    for i in neighbours: #[-15,0,15]: #
                        if k == 0:
                            x_train = np.array([acq.GetPoint(argument[0]).GetValues()[frame+i,2]-mean_TOE,
                              acq.GetPoint(argument[0]).GetValues()[frame+i,0]-acq.GetPoint(argument[1]).GetValues()[frame+i,0],
                              #acq.GetPoint(argument[2]).GetValues()[frame+i,2]-mean_PSI,
                              acq.GetPoint(argument[3]).GetValues()[frame+i,0]-acq.GetPoint(argument[0]).GetValues()[frame+i,0]])
                             # np.array([acq.GetPoint(argument[0]).GetValues()[frame+i,2]-mean_TOE,
                             #  acq.GetPoint(argument[1]).GetValues()[frame+i,2],acq.GetPoint(argument[2]).GetValues()[frame+i,2]-mean_PSI])
                             # #,acq.GetPoint('LANK').GetValues()[frame+i,2]
                            if (i == 0):
                                y_train = np.array([event.GetLabel()])
                                dfEventInit = dfEventInit.append({'video' : filename , 'pied' : pied, 'event' : event.GetLabel(), 'frame': event.GetFrame()}, ignore_index=True)
                            else:
                                y_train = np.array(["Not_Event"])
                            k = 1
                        else:
                            x_train = np.vstack([x_train, np.array([acq.GetPoint(argument[0]).GetValues()[frame+i,2]-mean_TOE,
                            acq.GetPoint(argument[0]).GetValues()[frame+i,0]-acq.GetPoint(argument[1]).GetValues()[frame+i,0],
                            #acq.GetPoint(argument[2]).GetValues()[frame+i,2]-mean_PSI,
                            acq.GetPoint(argument[3]).GetValues()[frame+i,0]-acq.GetPoint(argument[0]).GetValues()[frame+i,0]])])
                            if (i == 0):
                                dfEventInit = dfEventInit.append({'video' : filename , 'pied' : pied, 'event' : event.GetLabel(), 'frame': event.GetFrame()}, ignore_index=True)
                                y_train = np.vstack([y_train, np.array([event.GetLabel()])])
                            else:
                                y_train = np.vstack([y_train, np.array(["Not_Event"])])

                except Exception as e:
                    break
    return x_train, y_train, dfEventInit


def testmodel(model, directoire, dirpath):
    dfresult = pd.DataFrame(columns = ['video', 'pied','event','frame'])

    for d in directoire:
        for filename in os.listdir(dirpath + d):
            path = dirpath + d + filename

            reader = btk.btkAcquisitionFileReader()
            #path = 'Sofamehack2019/Sub_DB_Checked/CP/CP_GMFCS1_01916_20130128_18.c3d'
            reader.SetFilename(path)
            reader.Update()
            acq = reader.GetOutput()
            mean_TOE = np.mean(acq.GetPoint('LTOE').GetValues()[:,2])
            data_FrameRef = np.concatenate((np.transpose(np.array([acq.GetPoint('LTOE').GetValues()[:,2]-mean_TOE])), np.transpose(np.array([acq.GetPoint('LTOE').GetValues()[:,0]-acq.GetPoint('LHEE').GetValues()[:,0]]))), axis = 1)
            mean_PSI = np.mean(acq.GetPoint('LPSI').GetValues()[:,2])
            max_PSI = np.max(acq.GetPoint('LPSI').GetValues()[:,2])
            min_PSI = np.min(acq.GetPoint('LPSI').GetValues()[:,2])
            #data_FrameRef = np.concatenate((data_FrameRef, np.transpose(np.array([acq.GetPoint('LPSI').GetValues()[:,2]-mean_PSI]))), axis = 1)
            data_FrameRef = np.concatenate((data_FrameRef, np.transpose(np.array([acq.GetPoint('LKNE').GetValues()[:,0]-acq.GetPoint('LTOE').GetValues()[:,0]]))),axis = 1)
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
        for filename in os.listdir(dirpath + d):
            path = dirpath + d + filename

            reader = btk.btkAcquisitionFileReader()
            #path = 'Sofamehack2019/Sub_DB_Checked/CP/CP_GMFCS1_01916_20130128_18.c3d'
            reader.SetFilename(path)
            reader.Update()
            acq = reader.GetOutput()
            mean_TOE = np.mean(acq.GetPoint('RTOE').GetValues()[:,2])
            data_FrameRef = np.concatenate((np.transpose(np.array([acq.GetPoint('RTOE').GetValues()[:,2]-mean_TOE])), np.transpose(np.array([acq.GetPoint('RTOE').GetValues()[:,0]-acq.GetPoint('RHEE').GetValues()[:,0]]))), axis = 1)
            mean_PSI = np.mean(acq.GetPoint('RPSI').GetValues()[:,2])
            max_PSI = np.max(acq.GetPoint('RPSI').GetValues()[:,2])
            min_PSI = np.min(acq.GetPoint('RPSI').GetValues()[:,2])
            #data_FrameRef = np.concatenate((data_FrameRef, np.transpose(np.array([acq.GetPoint('RPSI').GetValues()[:,2]-mean_PSI]))), axis = 1)
            data_FrameRef = np.concatenate((data_FrameRef, np.transpose(np.array([acq.GetPoint('RKNE').GetValues()[:,0]-acq.GetPoint('RTOE').GetValues()[:,0]]))),axis = 1)
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
dir = ['ITW/']
MeanExpo = []
MeanSum = []
framelist = [-9,0,9]
print(framelist)
for i in range(10):
    [x_train, y_train, dfInit] = dataCollect(framelist, dir, dirpath)
    model = NaivesBayes(x_train,y_train)
    dataframeresult = testmodel(model, dir, dirpath)
    nligneInit = dfInit.shape[0]
    diffTest = []
    for el in range(nligneInit):
        value = np.min(np.abs(dataframeresult.loc[(dataframeresult['video'] == dfInit.iloc[el,0]) & (dataframeresult['pied'] == dfInit.iloc[el,1])
            & (dataframeresult['event'] == dfInit.iloc[el,2]), 'frame'] - dfInit.iloc[el,3]))
        diffTest = np.append(diffTest, value)
        # if (value > 6 or math.isnan(value)):
        #     print(dfInit.iloc[el,:])
        #     print 'result : ', value
        #     print(dataframeresult.loc[(dataframeresult['video'] == dfInit.iloc[el,0]) & (dataframeresult['pied'] == dfInit.iloc[el,1]) & (dataframeresult['event'] == dfInit.iloc[el,2])])
        #     print("\n\n")
    print(model.feature_importances_)
    MeanSum = np.append(MeanSum,  np.sum(diffTest))
    MeanExpo = np.append(MeanExpo, np.sum(np.exp(diffTest), axis = 0))


#X_train, X_test,Y_train,Y_test = train_test_split(x_train,y_train)
#model = NaivesBayes(x_train,y_train)
#print(model.feature_importances_)


#model = logistic(x_train, y_train)
#model = KNN(x_train,y_train)
#model = MLP(x_train,y_train)


# dtree_predictions = model.predict(X_test)
# cm = confusion_matrix(Y_test, dtree_predictions)
# print(cm)
# print(accuracy_score(Y_test,dtree_predictions))

# nligneInit = dfEventInit.shape[0]
# diffTest = []
# for el in range(nligneInit):
#     value = np.min(np.abs(dfresult.loc[(dfresult['video'] == dfEventInit.iloc[el,0]) & (dfresult['pied'] == dfEventInit.iloc[el,1])
#         & (dfresult['event'] == dfEventInit.iloc[el,2]), 'frame'] - dfEventInit.iloc[el,3]))
#     diffTest = np.append(diffTest, value)
    # if (value > 5 or math.isnan(value)):
    #     print(dfEventInit.iloc[el,:])
    #     print 'result : ', value
    #     print(dfresult.loc[(dfresult['video'] == dfEventInit.iloc[el,0]) & (dfresult['pied'] == dfEventInit.iloc[el,1]) & (dfresult['event'] == dfEventInit.iloc[el,2])])
    #     print("\n\n")
    #diffTest = np.append(diff, np.min(np.abs(dfEventInit.loc[''])))

print(" Erreur sum total : ", np.mean(MeanSum))
print(" Erreur exponentiel : ", np.mean(MeanExpo))
