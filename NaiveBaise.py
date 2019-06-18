import btk
import sys
import numpy as np
import os
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression




def computelist(list):
    if len(list) > 2:
        return round(np.mean(list[1:len(list)-2]))
    else:
        return round(np.mean(list))


dirpath = './Sofamehack2019/Sub_DB_Checked/'
dir = ['CP/']



f = 0
k = 0
kright = 0
for d in dir:
    for filename in os.listdir(dirpath + d):
        path = dirpath + d + filename
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(path)
        reader.Update()
        acq = reader.GetOutput()

        mean_PSI = np.mean(acq.GetPoint('LPSI').GetValues()[:,2])
        for i in range(100):
            try:
                event = acq.GetEvent(i) # extract the first event of the aquisition
                #print(event.GetLabel()) # return a string representing the Label
                if (event.GetContext() == "Left"):
                    argument = ['LTOE','LHEE','LPSI']
                    mean_PSI = np.mean(acq.GetPoint('LPSI').GetValues()[:,2])
                    mean_TOE = np.mean(acq.GetPoint('LTOE').GetValues()[:,2])

                else:
                    argument = ['RTOE','RHEE','RPSI']
                    mean_PSI = np.mean(acq.GetPoint('RPSI').GetValues()[:,2])
                    mean_TOE = np.mean(acq.GetPoint('RTOE').GetValues()[:,2])

                frame = event.GetFrame()
                for i in [-12,-6,0,6,12]:
                    if k == 0:
                        x_train = np.array([acq.GetPoint(argument[0]).GetValues()[frame+i,2]-mean_TOE,
                         acq.GetPoint(argument[1]).GetValues()[frame+i,2],acq.GetPoint(argument[2]).GetValues()[frame+i,2]-mean_PSI])
                         #,acq.GetPoint('LANK').GetValues()[frame+i,2]
                        if (i == 0):
                            y_train = np.array([event.GetLabel()])
                        else:
                            y_train = np.array(["Not_Event"])
                        k = 1
                    else:
                        x_train = np.vstack([x_train, np.array([acq.GetPoint(argument[0]).GetValues()[frame+i,2]-mean_TOE,
                        acq.GetPoint(argument[1]).GetValues()[frame+i,2],acq.GetPoint(argument[2]).GetValues()[frame+i,2]-mean_PSI])])
                        if (i == 0):
                            y_train = np.vstack([y_train, np.array([event.GetLabel()])])
                        else:
                            y_train = np.vstack([y_train, np.array(["Not_Event"])])
                #print(event.GetFrame()) # return the frame as an integer

            except Exception as e:
                break


print('Taille de nos donnees : ',x_train.shape)

X_train, X_test,Y_train,Y_test = train_test_split(x_train,y_train, random_state = 0)
# We try to user Naive baises
dtree_model = DecisionTreeClassifier().fit(X_train,Y_train)

#dtree_model = LogisticRegression().fit(X_train,Y_train)

dtree_predictions = dtree_model.predict(X_test)
cm = confusion_matrix(Y_test, dtree_predictions)
print(cm)
print(accuracy_score(Y_test,dtree_predictions))
print(dtree_model.feature_importances_)



#Visualisation de notre image
eader = btk.btkAcquisitionFileReader()
reader.SetFilename('./Sofamehack2019/Sub_DB_Checked/CP/CP_GMFCS1_01916_20130128_18.c3d')
reader.Update()
acq = reader.GetOutput()
mean_TOE = np.mean(acq.GetPoint('LTOE').GetValues()[:,2])
data_FrameRef = np.concatenate((np.transpose(np.array([acq.GetPoint('LTOE').GetValues()[:,2]-mean_TOE])), np.transpose(np.array([acq.GetPoint('LHEE').GetValues()[:,2]]))), axis = 1)
mean_PSI = np.mean(acq.GetPoint('LPSI').GetValues()[:,2])
data_FrameRef = np.concatenate((data_FrameRef, np.transpose(np.array([acq.GetPoint('LPSI').GetValues()[:,2]-mean_PSI]))), axis = 1)
P =dtree_model.predict(data_FrameRef)

df = pd.DataFrame(data=P, columns = ['result'])
dfEvent = df.loc[df['result'] != 'Not_Event']
nligne = dfEvent.shape[0]
dfEvent['index'] = dfEvent.index

current = dfEvent.iloc[0,0]
comp_list = []
reslist = []
for i in range(nligne-1):
    if (current !=dfEvent.iloc[i+1,0]):
        res = computelist(comp_list)
        print(res, current)
        current = dfEvent.iloc[i+1,0]
        comp_list = []
    else:
        comp_list.append(dfEvent.iloc[i+1,1])
    if (i == nligne-2):
        res = computelist(comp_list)
        print(res, current)
