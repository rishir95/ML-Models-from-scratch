import pandas
import nltk
import string
import math
from nltk.probability import FreqDist


train_file = pandas.read_fwf("Z:/Documents/Machine learning/Assignment2/reviewstrain.txt",header=None)        

train_file = train_file.fillna(' ')
train_file[1] = train_file[1] + " " + train_file[2] + " " + train_file[3]
train_file[1][320] = train_file[1][320] + " " + "."
train_file[1][214] = train_file[1][214] + " " + "."

del train_file[2]
del train_file[3]

test_file = pandas.read_fwf("Z:/Documents/Machine learning/Assignment2/reviewstest.txt",header=None)

test_file = test_file.fillna(' ')
test_file[1] = test_file[1] + " " +test_file[2]

del test_file[2]


vocab=[]
for value in train_file[1]:
    vocab+=value.strip('\t').split()
    
vocab_set = set(vocab)
vocabfreq=FreqDist(vocab)
print("Ans a")
print(vocabfreq.freq)



def entropy(doc_set):
    no1 = len(doc_set[doc_set[0]==1])
    no2 = len(doc_set[doc_set[0]==0])
    entropy = 0
    for i in [no1,no2]:
        if 0 in [no1,no2]:
            return 0
        entropy+=-1*(i/(no1+no2)*math.log2(i/(no1+no2)))
    return entropy


def add_atr(df,vocab):
    if vocab in df[1].strip('\t').split():
        return 1
    else:
        return 0
    
    
for word in list(vocab_set):
    test_file[word]=test_file.apply(add_atr, vocab=word, axis=1)
    print(word,"Done")
    
    
def calculateInfoGain():
    record = []
    total = entropy(train_file)
    total_len = len(train_file)
    j=1
    for word in a:
        no1 = len(train_file[train_file[word]==1])
        eno1 = entropy(train_file[train_file[word]==1])
        no2 = len(train_file[train_file[word]==0])
        eno2 = entropy(train_file[train_file[word]==0])
        infogain = total-(no1*eno1/total_len)-(no2*eno2/total_len)
#         if len(record)==50:
#             if infogain>record[0][1]:
#                 record[0] = (word,infogain)
#         else:
        record.append((word,infogain))
#         record.sort(key = lambda x:x[1])
        print(word,"Done",j)
        j+=1
    record.sort(key = lambda x:x[1], reverse=True)
    return record

z=calculateInfoGain()

print("Ans b")
for info in z[0:5]:
    print(info)
    
    
attlist=[]
for word in z[0:50]:
    attlist.append(word[0])
    
nninput = train_file
output = nninput[0]
n_output = output.values
nnin = nninput[attlist]
n_input = nnin.values

# -*- coding: utf-8 -*-
"""
Original Author (in Matlab): Yoonsuck Choe
http://faculty.cs.tamu.edu
Sat Feb  9 16:14:37 CST 2008
License: GNU public license (http://www.gnu.org)
Modified extensively by Prof. Lisa Hellerstein
Matlab to Python Conversion by Shaoyu Chen 
Fall 2018 Version
"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce
        
# I is matrix of input examples
# D is matrix of output examples
# n_hidden is number of nodes in hidden layer
# n_max is number of training epochs
def bp(I,D,n_hidden,eta,n_max):
    np.random.seed(1926)
    
    r_inp, c_inp = I.shape
    n_examples = r_inp
    n_input    = c_inp
    r_out,c_out = D.shape
    n_output   = c_out

    w = np.random.random((n_input, n_hidden))
    wb = np.random.random(n_hidden)
    v = np.random.random((n_hidden, n_output))
    vb = np.random.random(n_output)
    err_curve = np.zeros((n_max,c_out))

    for n in range(n_max):
#         sq_err_sum = np.zeros((1,n_output))
        sq_err_sum = np.zeros((1,n_output))
    
        for k in range(n_examples):
            x = I[k,:].reshape([1,-1])
            z = sigmoid(x.dot(w)+wb)
            y = sigmoid(z.dot(v)+vb)
            err = cross_entropy(y, D[k,:])
#             err = (D[k,:] - y).reshape([1,-1])
#             print(err)
            sq_err_sum+=err
#             sq_err_sum += 0.5*np.square(err)
            
            Delta_output = err*(1-y)*y    
            Delta_v = z.T.dot(Delta_output)
            Delta_vb = np.sum(Delta_output,axis=0)

            Delta_hidden = Delta_output.dot(v.T)*(1-z)*z
            Delta_w = x.T.dot(Delta_hidden)    
            Delta_wb = np.sum(Delta_hidden,axis=0) 
        
            v += eta*Delta_v
            vb += eta*Delta_vb
            w += eta*Delta_w
            wb += eta*Delta_wb

            err_curve[n] = sq_err_sum/n_examples
    
        print('epoch %d: err %f'%(n,np.mean(sq_err_sum)/n_examples))
    
    plt.plot(np.linspace(0,n_max-1,n_max),np.mean(err_curve,axis=1))        
    plt.show()
    return w,wb,v,vb,err_curve     

w,wb,v,vb,err_curve = bp(n_input, n_output.reshape([-1,1]), 1, 3.25, 1000)

test_input = test_file[attlist]
test_in = test_input.values

outputtest = test_file[0]


class dicter(dict):
    def __missing__(self,key):
        return 0
    
d=dicter()
d[0]=dicter()
d[1]=dicter()
correct=0
for k in range(500):
    x = test_in[k,:].reshape([1,-1])
    z = sigmoid(x.dot(w)+wb)
    y = sigmoid(z.dot(v)+vb)
    if y>0.5:
        if outputtest[k]==1:
            d[1][1]+=1
            correct+=1
        else:
            d[1][0]+=1
    else:
        if outputtest[k]==1:
            d[0][1]+=1
        else:
            d[0][0]+=1
            correct+=1

print("Ans c")
print("Confusion Matrix:")
print(d)
print("Accuracy=",correct/500*100)

print("Ans d")
print(len(test_file[test_file[0]==1])/500*100)


