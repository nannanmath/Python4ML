'''
Created on Nov 11, 2014

@author: nannanmath
'''
 
#coding=utf-8
import math  
import copy  
import numpy as np  
import matplotlib.pyplot as plt  
  
isdebug = False  
  
# two Gaussian distribution with same var and different means.
# k = 2  
def ini_data(Sigma,Mu1,Mu2,k,N):  
    global X  
    global Mu  
    global Expectations  
    X = np.zeros((1,N))  
    Mu = np.random.random(2)  
    Expectations = np.zeros((N,k))  
    for i in xrange(0,N):  
        if np.random.random(1) > 0.5:  
            X[0,i] = np.random.normal()*Sigma + Mu1
        else:  
            X[0,i] = np.random.normal()*Sigma + Mu2  
    if isdebug:  
        print "***********"  
        print u"init the observation vals X:"  
        print X  
# EM: E step, obtain Q[Z(i)] 
def e_step(Sigma,k,N):  
    global Expectations  
    global Mu  
    global X  
    for i in xrange(0,N):  
        Denom = 0  
        for j in xrange(0,k):  
            Denom += math.exp((-1/(2*(float(Sigma**2))))*(float(X[0,i]-Mu[j]))**2)  
        for j in xrange(0,k):  
            Numer = math.exp((-1/(2*(float(Sigma**2))))*(float(X[0,i]-Mu[j]))**2)  
            Expectations[i,j] = Numer / Denom  
    if isdebug:  
        print "***********"  
        print u"hidden varl E[Z]: "  
        print Expectations  
# EM: M step, max mu. 
def m_step(k,N):  
    global Expectations  
    global X  
    for j in xrange(0,k):  
        Numer = 0  
        Denom = 0  
        for i in xrange(0,N):  
            Numer += Expectations[i,j]*X[0,i]  
            Denom +=Expectations[i,j]  
        Mu[j] = Numer / Denom   
# iterate iter_num times, or up to Epsilon
def run(Sigma,Mu1,Mu2,k,N,iter_num,Epsilon):  
    ini_data(Sigma,Mu1,Mu2,k,N)  
    print u"init <u1,u2>:", Mu  
    for i in range(iter_num):  
        Old_Mu = copy.deepcopy(Mu)  
        e_step(Sigma,k,N)  
        m_step(k,N)  
        print i,Mu  
        if sum(abs(Mu-Old_Mu)) < Epsilon:  
            break  
if __name__ == '__main__':  
   run(6,40,20,2,1000,1000,0.0001)  
   plt.hist(X[0,:],50)  
   plt.show()