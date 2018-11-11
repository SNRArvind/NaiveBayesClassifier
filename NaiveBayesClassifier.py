# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 23:34:33 2018

Roll No: 18AT91R02
Name: Arvind Kumar Gupta
Assignment No.: 3

"""

import csv
import numpy as np

with open('data3.csv', 'rb') as Data:
    reader = csv.reader(Data)
    row=next(reader)
    trainData=np.resize(np.asarray(row[0:8],dtype= int), (1,8))
    trainOutput=np.asarray(row[-1], dtype=int)
    for row in reader:
        trainData=np.concatenate((trainData,np.resize(np.asarray(row[0:8],dtype= int), (1,8))),axis = 0)
        trainOutput=np.append(trainOutput,np.asarray(row[-1], dtype=int))

class1Instances=np.asarray(np.where(trainOutput ==1))
nclass1=class1Instances.size
class0Instances=np.asarray(np.where(trainOutput ==0))
nclass0=class0Instances.size

pClass1=nclass1/float((nclass1+nclass0))
pClass0=nclass0/float((nclass1+nclass0))

pXy1=[]
pXy0=[]
for i in range(8):
    nXy1=np.asarray(np.where(np.resize(trainData[class1Instances,i],nclass1) ==1)).size
    nXy0=np.asarray(np.where(np.resize(trainData[class0Instances,i],nclass0) ==1)).size
    PXy1=(nXy1+1)/float(nclass1+2)
    PXy0=(nXy0+1)/float(nclass0+2)
    pXy1.append(PXy1)
    pXy0.append(PXy0)

f2= open("18AT91R02_3.out","w+")

with open('test3.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        class1Prob=pClass1
        class0Prob=pClass0
        x=np.asarray(row, dtype=int)
        for i in range(len(x)):
            if x[i]==1:
                class1Prob=class1Prob*pXy1[i]
                class0Prob=class0Prob*pXy0[i]
            else:
                class1Prob=class1Prob*(1-pXy1[i])
                class0Prob=class0Prob*(1-pXy0[i])
                
        if class0Prob > class1Prob:
            f2.write("0 ")
        else:
            f2.write("1 ")
f2.close()