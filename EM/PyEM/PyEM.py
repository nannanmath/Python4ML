'''
Created on Nov 11, 2014

@author: nannanmath
'''
 
import math  
import copy  
import numpy as np  
import matplotlib.pyplot as plt  


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = map(float,curLine[1:]) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat
  
def ini_data(dataSet,k):  
    global Mu  
    global Sigma
    global Alpha
    global Expectations
   
    numSam = np.shape(dataSet)[0]
    dimSam = np.shape(dataSet)[1]
    
    Mu = np.random.rand(k,dimSam)
    
    Sigma = []
    Alpha = []
    for i in range(k):
        Sigma.append(np.identity(dimSam))
        Alpha.append(1.0/k)
        
    Mu[0,:] = 2.0,1.4
    Mu[1,:] = 8.0,1.8
    Sigma[0] = Sigma[0]*0.16
    Sigma[1] = Sigma[1]*0.36
    Alpha[0] = 0.4
    Alpha[1] = 0.6
    
    Expectations = np.zeros((numSam,k))
    
def Gauss_pdf(data,sigma,mu):
    numDim = np.shape(data)[1]
    Gpdf = math.exp((-1.0/2)*np.dot(np.dot((data-mu),np.linalg.inv(sigma)),(data-mu).T))/math.sqrt((2*3.1415)**(numDim)*np.linalg.det(sigma))
    return Gpdf

# EM: E step, obtain Q[Z(i)] 
def e_step(dataSet,k): 
    numSam = np.shape(dataSet)[0]
    dimSam = np.shape(dataSet)[1]
     
    for i in xrange(0,numSam):  
        Denom = 0  
        for j in xrange(0,k):  
            Denom += Alpha[j] * Gauss_pdf(dataSet[i], Sigma[j], Mu[j])
        for j in xrange(0,k):  
            Numer = Alpha[j] * Gauss_pdf(dataSet[i], Sigma[j], Mu[j])
            Expectations[i,j] = Numer / Denom  

# EM: M step, max mu. 
def m_step(dataSet,k):  
    numSam = np.shape(dataSet)[0]
    
    # update the Sigma
    for j in xrange(0,k):  
        Numer = 0  
        Denom = 0  
        for i in xrange(0,numSam):  
            Numer += Expectations[i,j]*np.dot((dataSet[i]-Mu[j]).T,(dataSet[i]-Mu[j]))  
            Denom += Expectations[i,j]  
        Sigma[j] = Numer / Denom
        
    # update the means
    for j in xrange(0,k):  
        Numer = 0  
        Denom = 0  
        for i in xrange(0,numSam):  
            Numer += Expectations[i,j]*dataSet[i]  
            Denom += Expectations[i,j]  
        Mu[j,:] = Numer / Denom   
        
    # update the weights
    for j in xrange(0,k):  
        Numer = 0    
        for i in xrange(0,numSam):  
            Numer += Expectations[i,j] 
        Alpha[j] = Numer / numSam 
        
# iterate iter_num times, or up to Epsilon
def run(dataSet,k,iter_num,Epsilon):  
    ini_data(dataSet,k)   
    for i in range(iter_num):  
        Old_Mu = copy.deepcopy(Mu)
        Old_Sigma = copy.deepcopy(Sigma)
        Old_Alpha = copy.deepcopy(Alpha)  
        e_step(dataSet,k)  
        m_step(dataSet,k)  
        
        deltaSigma = 0
        deltaAlpha = 0
        deltaMu = 0
        for i in xrange(0,k):
            deltaSigma += abs(Sigma[i]-Old_Sigma[i]).sum()
            deltaAlpha += abs(Alpha[i]-Old_Alpha[i])
            deltaMu += abs(Mu[i] - Old_Mu[i]).sum()
        if (deltaSigma+deltaAlpha+deltaMu) < Epsilon:  
            break
        
if __name__ == '__main__':  
    #dataMat = np.mat(loadDataSet('letter-recognition.data'))
    dataMat = np.mat(loadDataSet('EM.test'))
    run(dataMat,2,100,0.0001)  