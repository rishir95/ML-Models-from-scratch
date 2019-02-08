import argparse
import pandas
import nltk
import sys
import warnings
from nltk.corpus import stopwords

class KNN:
    train_file = None
    test_file = None
    lemmer = None
    en_stops = None
    
    
    def __init__(self, train, test):
        self.train_file = pandas.read_fwf(train,header=None)
        self.test_file = pandas.read_fwf(test,header=None)
        self.lemmer = nltk.stem.WordNetLemmatizer()
        self.en_stops = set(stopwords.words('english'))

        '''
        A bit of preprocessing of pandas datafame
        '''
        self.test_file = self.test_file.fillna(' ')
        self.test_file[1] = self.test_file[1] + " " + self.test_file[2]
        del self.test_file[2]
        
        self.train_file = self.train_file.fillna(' ')
        self.train_file[1] = self.train_file[1] + " " + self.train_file[2] + " " + self.train_file[3]
        self.train_file[1][320] = self.train_file[1][320] + " " + "."
        self.train_file[1][214] = self.train_file[1][214] + " " + "."
        del self.train_file[2]
        del self.train_file[3]
        
        
    def findDistance(self, comment1, comment2):
        distance = len(set(comment1.strip('\t').strip('\n').split()).intersection(set(comment2.strip('\t').strip('\n').split())))
        if distance == 0:
            return 1
        return (1/distance)
    
    
    def findDistance2(self, comment1, comment2):
        comment1 = self.lemmer.lemmatize(comment1)
        comment2 = self.lemmer.lemmatize(comment2)
        com1=[]
        for word in comment1.strip('\t').strip('\n').split():
            if word not in self.en_stops:
                com1.append(word)
        com2=[]
        for word in comment2.strip('\t').strip('\n').split():
            if word not in self.en_stops:
                com2.append(word)
        distance = len(set(com1).intersection(set(com2)))
        if distance == 0:
            return 1
        return (1/distance)
    
    
    def findNeighbours(self, testrecord, trainfile, k, funcno):
        distance = []
        
        commentseries=trainfile[1]
        for index,record in enumerate(commentseries):
            if funcno==1:
                distance.append((self.findDistance(testrecord,record),trainfile[0][index]))
            if funcno==2:
                distance.append((self.findDistance2(testrecord,record),trainfile[0][index]))
        distance.sort(key=lambda x:x[0])
        classlabels=[]
        if k==1:
            minimum=min(set(distance),key = lambda x:x[0])
            i=0
            while distance[i][0]==minimum[0]:
                classlabels.append(distance[i][1])
                i+=1
                if i==len(distance):
                    break
        else:
            for j in range(0,k):
                classlabels.append(distance[j][1])
            if distance[k-1][0] == distance[k][0]:
                i=k
                while distance[i][0]==distance[k-1][0]:
                    classlabels.append(distance[i][1])
                    i+=1
                    if i==len(distance):
                        break
        return classlabels
    
    
    def predict(self, testrecord, trainfile, k, funcno):
        classlabel = self.findNeighbours(testrecord,trainfile,k,funcno)
        if classlabel.count(0)>classlabel.count(1):
            return 0
        if classlabel.count(1)>=classlabel.count(0):
            return 1
        
        
    def findClass(self, funcno, k):    
        correct=0
        d={}
        d["1"]={}#real value 1
        d["0"]={}#real value 0
        d["1"]["1"]=0#real value 1 predicted 1
        d["1"]["0"]=0#real value 1 predicted 0
        d["0"]["1"]=0#real value 0 predicted 0
        d["0"]["0"]=0#real value 0 predicted 1
        for index,record in enumerate(self.test_file[1]):

            z = self.predict(record,self.train_file,k,funcno)
            if self.test_file[0][index]==1 and z==1:
                d["1"]["1"]+=1
            elif self.test_file[0][index]==1 and z==0:
                d["1"]["0"]+=1
            elif self.test_file[0][index]==0 and z==0:
                d["0"]["0"]+=1
            else:
                d["0"]["1"]+=1
            if self.test_file[0][index]==z:
                correct+=1

        print("Accuracy=",correct/len(self.test_file)*100,"\n")
        print("Confusion Matrix in dict form where the outer keys are real label and inner are predicted label\n",d,"\n")
        
    
    def crossValidate(self):
        for k in [3,7,99]:

            accuracy_mean = 0

            for i in range(0,1500,300):
                testset = self.train_file.iloc[i:300+i]
                if i==0:
                    trainset = self.train_file.iloc[300:1500]
                elif i==1200:
                    trainset = self.train_file.iloc[0:1200]
                else:
                    trainset = pandas.concat([self.train_file.iloc[0:i],self.train_file.iloc[i+300:1500]],ignore_index=True)
                trainset = trainset.reset_index(drop=True)
                testset = testset.reset_index(drop=True)
                correct=0
                for index,record in enumerate(testset[1]):
                    z = self.predict(record,trainset,k,1)
                    if testset[0][index]==z:
                        correct+=1
                accuracy_mean +=correct

            print("k=",k,"Accuracy=",accuracy_mean/1500,"\n")

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 2'
                                                 ' to Implement KNN in Python.')
    parser.add_argument("-t","--Train", metavar="Trainset", help="path of the training reviews txt file")
    parser.add_argument("-s","--Test", metavar="Testset", help="path of the testing reviews txt file")

    args = parser.parse_args()
    

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    ker = KNN(args.Train,args.Test)
    takeinput = 1
    while takeinput==1:
        inputval = int(input("Press 1 for knn using cross validation, 2 for normal knn "))
        if inputval==1:
            ker.crossValidate()
        if inputval==2:
            inputdist = int(input("Enter 1 for original distance func, 2 for modified distance func "))
            inputk = int(input("Enter value of k "))
            ker.findClass(inputdist,inputk)
        takeinput = int(input("Enter 1 if you want to try the other option "))
            