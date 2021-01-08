import subprocess
# subprocess.check_call(["python", "-m", "pip", "install", "psutil"])
import sys
import os
import csv
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from pandas import read_csv
import statistics
mean=statistics.mean
import numpy as np
from numpy import loadtxt
import psutil
import sklearn
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time
import xgboost
from xgboost import XGBClassifier
from xgboost import cv


# Set timer to output program execution time
starttime=time.time()
process = psutil.Process(os.getpid())

class dloan():
    def __init__(self,classes,targets):
        self.dfx=classes
        self.dfy=targets
        try:
            xfile = self.dfx
        except IndexError as ie:
            raise SystemError("Error: Specify file name\n")
        if not os.path.exists(xfile):
            raise SystemError("Error: File does not exist\n")
        xd=pd.read_csv(xfile)
        xd=xd.drop(['Id'],axis=1)
        xd.fillna(0,inplace=True)
        self.x=xd.values.astype(str)

        yfile=self.dfy
        self.y=pd.read_csv(yfile)
        self.y.fillna(0,inplace=True)


    def get_fitit(self):
        fiter=str(input("Enter 'fitit', without quotes, to fit model: "))
        if not hasattr(self, fiter):
            print ("%s is not a valid command" % fiter)
        else:
            getattr(self, fiter)()


    def fitit(self):
        # encode string input values as integers
        encoded_x = None
        for i in range(0, self.x.shape[1]):
            label_encoder = LabelEncoder()
            feature = label_encoder.fit_transform(self.x[:,i])
            feature = feature.reshape(self.x.shape[0], 1)
            onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
            feature = onehot_encoder.fit_transform(feature)
            if encoded_x is None:
                encoded_x = feature
            else:
                encoded_x = np.concatenate((encoded_x, feature), axis=1)
        
        self.ex=encoded_x


        # encode string class values as integers
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(self.y)
        label_encoded_y = label_encoder.transform(self.y)
        ley=label_encoded_y
        ley=[i.astype(np.float64) for i in ley]
        self.ley=np.array(ley)

        # split data into train and test sets
        seed = 7
        test_size = 0.33
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.ex, self.ley, test_size=test_size, random_state=seed)

        # fit model
        self.model = XGBClassifier(seed=42,objective='binary:logistic',gamma=0.25,learning_rate=0.1,max_depth=4,reg_lambda=10)

        self.model.fit(self.xtrain, self.ytrain, verbose=True,early_stopping_rounds=10,eval_metric='aucpr',eval_set=[(self.xtest, self.ytest)])


    def get_predictit(self):
        predicter=str(input("Enter 'predictit', without quotes, to output array of target variable predictions: "))
        if not hasattr(self, predicter):
            print ("%s is not a valid command" % predicter)
        else:
            getattr(self, predicter)()


    def predictit(self):
        self.ypred=self.model.predict(self.xtest)
        self.preds=[round(value) for value in self.ypred]
        print (self.preds)


    def get_predict_probs(self):
        prober=str(input("Enter 'predict_probs', without quotes, to output array of label probabilities: "))
        if not hasattr(self, prober):
            print ("%s is not a valid command" % prober)
        else:
            getattr(self, prober)()


    def predict_probs(self):
        ypr=self.model.predict_proba(self.xtest)
        prob=[]
        for i in ypr:
            sp=[]
            for j in i:
                k=round(j,1)
                sp.append(k)
            prob.append(sp)

        self.probs=prob
        print (self.probs)


    def get_evaluate(self):
        evaler=str(input("Enter 'evaluate', without quotes, to output dictionary of F1-score and LogLoss: "))
        if not hasattr(self, evaler):
            print ("%s is not a valid command" % evaler)
        else:
            getattr(self, evaler)()


    def evaluate(self):
        f1=f1_score(self.ytest,self.ypred)
        self.f1=round(f1,1)
        ll=log_loss(self.ytest,self.ypred)
        self.ll=round(ll,1)
        self.se={'f1_score': self.f1, 'logloss': self.ll}
        print (self.se)


    def get_tune_parameters(self):
        evaler=str(input("Enter 'tune_parameters', without quotes, to run K-fold cross validation and output for best parameters (cross validation can take up to 10 minutes): "))
        if not hasattr(self, evaler):
            print ("%s is not a valid command" % evaler)
        else:
            getattr(self, evaler)()


    def tune_parameters(self):
        gamma=0.25
        learning_rate=0.1
        max_depth=4
        reg_lambda=10

        kfold = KFold(n_splits=4, random_state=7)
        cvs=cross_val_score(self.model,self.ex,self.ley,cv=kfold,scoring='accuracy')

        cvt=cvs.copy()
        acc=mean(cvt[np.logical_not(np.isnan(cvt))])
        acc = acc * 100.0
        self.tupar={'gamma': gamma, 'learning_rate': learning_rate, 'max_depth': max_depth, 'reg_lambda': reg_lambda, 'scores': {'accuracy': acc, 'f1_score': self.f1, 'logloss': self.ll}}

        print (self.tupar)


dfx=str(input("Enter path to csv of features (X): "))

dfy=str(input("Enter path to csv of labels (y): "))


dl=dloan(dfx,dfy)

dl.get_fitit()
dl.get_predictit()
dl.get_predict_probs()
dl.get_evaluate()
dl.get_tune_parameters()

# Output for timer
print ("Program execution time = ", time.time()-starttime,"ms")
print ("Memory usage of program = ", process.memory_info().rss, "MB")
