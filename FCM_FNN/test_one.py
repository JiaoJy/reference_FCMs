# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:51:36 2020

@author: JiaoJy
"""
# encoding: utf-8
import fcm_fnn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filepath):
    with open(filepath) as f:
        rawList = list(map(lambda line: float(line.strip()), f.readlines()))
        labelSet = []
        testSet = []
        for i in range(8, 8 + 500):
            labelSet.append((rawList[i:i+4], rawList[i+1:i+5]))
            # labelSet.append((rawList[i], rawList[i+1]))
        for i in range(8 + 500, 8 + 1000):
            testSet.append((rawList[i:i+4], rawList[i+1:i+5]))
            # testSet.append((rawList[i], rawList[i+1]))
        return labelSet, testSet

def loadDataSets(filepath):
    data = pd.read_csv('./data/dataProcess.csv',index_col = 0)
    data = np.array(data).tolist()
    labelSet = []
    testSet = []
    for i in range(8,8 + 500):
        labelSet.append((data[i][:],data[i+1][:]))
    for i in range(8 + 500, 8 + 1000):
        testSet.append((data[i][:],data[i+1][:]))
    return labelSet,testSet

def errorLp(p,data_pre,data_real):
    dist = np.linalg.norm(data_pre-data_real,ord=p)/(len(data_pre))
    return dist

def MSE(data_pre,data_real):
    if data_pre.ndim == 1:
        num = 1
    elif data_pre.ndim == 2:
        num = data_pre.shape[data_pre.ndim-1]
    dist = [0] * num

    dist_temp = (np.power((data_pre - data_real),2))/len(data_pre)
    dist = dist_temp.sum(axis=0)
    return dist


def RMSE(M):
    dist = np.power(M,0.5)
    return dist


def drawPre(title,preData,realData,dataNum=50):
    plt.title(title)
    plt.plot(range(dataNum), preData, color='green', label='predict');
    plt.plot(range(dataNum), realData, color='red', label='real');
    plt.legend();
    plt.xlabel('time');
    plt.ylabel('value');
    plt.show();

if __name__ == "__main__":
    # data = pd.read_csv('./data/dataProcess.csv',index_col = 0)
    # data = pd.DataFrame(np.array(data.iloc[:,1]))
    # data.to_csv('./data/datatest.csv',index=None)
    labelSet, testSet = loadDataSet("./data/datatest.csv")

    # with open("fnn.bin", "rb") as f:
    #     fnn = pickle.loads(f.read())

    fnn = fcm_fnn.FCM_FNN(4,4,4,4)
    # fnn = fcm_fnn.FCM_FNN(6)

    for i in range(25):
        print("train err: %s   test err: %s" % (fnn.train(labelSet, 0.02), fnn.test(testSet)))

        for oj, oconcept in enumerate(fnn.concepts):
            for omj in range(oconcept.numOfTerms):
                print("(%s, %s) C:%s W:%s Xi:%s" % (oj, omj, oconcept.C[omj], oconcept.sigma[omj], oconcept.xi[omj]))
            print()
    
    # print("write fnn data")
    # with open("fnn.bin", "wb") as f:
    #     f.write(pickle.dumps(fnn))

    print("write test files")
    testData = []
    for X, D in testSet:
        testData.append(D[0])
    testData += testSet[-1][1][1:4]
    with open("testData.csv", "w") as f:
        f.write("\n".join(map(lambda v:str(v), testData)))

    print("write predict files")
    predictData = []
    for X, D in testSet:
        predictData.append(fnn.predict(X)[0])
    predictData += fnn.predict(testSet[-1][0])[1:4]
    with open("predictData.csv", "w") as f:
        f.write("\n".join(map(lambda v:str(v), predictData)))
    
    Mwucha = MSE(np.array(predictData), np.array(testData))
    Rwucha = RMSE(Mwucha)
    drawPre("CO",predictData,testData,len(predictData))
    print("欧式距离：%s" % errorLp(2,np.array(predictData),np.array(testData)))
    print("MSE：%s" % Mwucha)
    print("RMSE：%s" % Rwucha)