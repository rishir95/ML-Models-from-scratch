import math
import pandas
import json

class NaiveBayesClassifier:
    dict_info={}
    train_file=None
    test_file=None
    classes=["non-spam","spam"]
    output_file=None
    
    def __init__(self,train_file_name,test_file_name,txt_file):
        """
        Initializes all the files and renames the column for more clarity
        """
        self.train_file = pandas.read_csv(train_file_name,header=None)
        self.test_file = pandas.read_csv(test_file_name,header=None)
        self.output_file = open(txt_file,"w")
        self.train_file.columns =["char_freq_;","char_freq_(","char_freq_[","char_freq_!","char_freq_$","char_freq_#","capital_run_length_average","capital_run_length_longest","capital_run_length_total","Labels"]
        self.test_file.columns =["char_freq_;","char_freq_(","char_freq_[","char_freq_!","char_freq_$","char_freq_#","capital_run_length_average","capital_run_length_longest","capital_run_length_total","Labels"]
        self.trainclassifier()
        self.testfunc()
        

    def plabel(self,label):
        """
        Calculates the probability P(C) of labels based on the training set
        """
        return len(self.train_file[self.train_file["Labels"]==label])/len(self.train_file)
        
        
        
    def variance(self,attribute,label):
        """
        Calculates the variance P(xi|C) for a attribute based on the training set
        """
        meanu = self.train_file[self.train_file["Labels"]==label][attribute].mean()
        totsum=0
        for values in self.train_file[self.train_file["Labels"]==label][attribute]:
            totsum+=pow(values-meanu,2)
        variance = totsum/(len(self.train_file[self.train_file["Labels"]==label])-1)
        return meanu,variance

    

    def trainclassifier(self):
        """
        Creates a dictionary which stores mean, variance, P(C) for the training set.
        """
        for label in self.train_file["Labels"].unique():
            self.dict_info[str(label)]={}
            for index,attr in enumerate(self.train_file.columns,start=1):
                if index!=10:
                    self.dict_info[str(label)][attr]={}
                    meanu,variance_val = self.variance(attr,label)
                    self.dict_info[str(label)][attr]["mean"]=meanu
                    self.dict_info[str(label)][attr]["variance"]=variance_val
                else:
                    self.dict_info[str(label)]["P({0})".format(label)]=self.plabel(label)
            self.dict_info[str(label)]["name"]=self.classes[label]
    
    
    
    def predict(self,label,value):
        """
        Calculates p(x1|C) ? p(x2|C) ? . . . ? p(xd|C) ? P(C)
        """
        prod=1
        for index,attr in enumerate(list(value),start=1):
            if index==10:
                prod*=self.plabel(attr)
                return prod
            prod*=self.estimatepdf(self.dict_info[str(label)][list(self.dict_info[str(label)])[index-1]]["variance"],self.dict_info[str(label)][list(self.dict_info[str(label)])[index-1]]["mean"],attr)   
        
        
        
    def estimatepdf(self,variance,meanu,value):
        """
        Calculates the pdf for one value at a time
        """
        exponent = math.exp(-(math.pow(value-meanu,2)/(2*variance)))
        estimate = (1/math.sqrt(2*math.pi*variance))*exponent 
        return estimate
    
    
    def testfunc(self):
        """
        Used for predicting the class labels and also for writing the output.
        """
        incorrect=0
        
        print("Question 1. \t","Non-spam - P(0)=",self.dict_info["0"]["P(0)"],"\tSpam - P(1)=",self.dict_info["1"]["P(1)"])
        self.output_file.write("Question 1."+"\t"+"Non-spam - P(0)="+"\t"+str(self.dict_info["0"]["P(0)"])+"\t"+"\tSpam - P(1)="+"\t"+str(self.dict_info["1"]["P(1)"]))
        
        print("Question 2. ",self.dict_info)
        self.output_file.write("\n\n"+"Question 2."+"\n"+json.dumps(self.dict_info,indent=4))
        
        print("Question 3. ")
        self.output_file.write("\n\n"+"Question 3")
        for index,value in enumerate(self.test_file.iterrows(),start=1):
            for label in self.dict_info:
                if label=="0":
                    non_spam_value=self.predict(0,value[1])
                else:
                    spam_value=self.predict(1,value[1])
            if non_spam_value==spam_value:
                print("For index {0} class label predicted is".format(index),1)
                self.output_file.write("\n"+"For index {0} class label predicted is".format(index)+"  1")
            elif non_spam_value==max(non_spam_value,spam_value):
                print("For index {0} class label predicted is".format(index),0)
                self.output_file.write("\n"+"For index {0} class label predicted is".format(index)+"  0")
                if list(value[1])[9]!=0:
                    incorrect+=1
            else:
                print("For index {0} class label predicted is".format(index),1)
                self.output_file.write("\n"+"For index {0} class label predicted is".format(index)+"  1")
                if list(value[1])[9]!=1:
                    incorrect+=1
                    
        print("Question 4. Total correct prediction = ", len(self.test_file)-incorrect)
        self.output_file.write("\n\n"+"Question 4. Total correct prediction = "+str(len(self.test_file)-incorrect))
        
        
        print("Question 5. Total incorrect prediction = ", incorrect)
        self.output_file.write("\n\n"+"Question 5. Total incorrect prediction = "+str(incorrect))
        
        print("Question 6. Total error rate = ", incorrect/len(self.test_file)*100)
        self.output_file.write("\n\n"+"Question 6. Total error rate = "+str(incorrect/len(self.test_file)*100))
        
        self.output_file.close()
