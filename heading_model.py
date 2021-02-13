
##############################################################################
#import statements
##############################################################################

#from sklearn import learning_curve
import pandas as pd
import calendar
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
#from sklearn import cross_validation
from sklearn import svm
from datetime import timedelta
from sklearn import metrics
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss

#from sklearn.linear_model import RandomizedLasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import load_digits
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

from scipy.sparse import hstack

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import re
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.tree import DecisionTreeRegressor
import os
import string
from sklearn.externals import joblib as jb



def clean_new_util(X):
    X1=[]
    for i in range(len(X)):
        punctuation_removed = [char for char in X[i] if char not in string.punctuation]
        punctuation_removed = "".join(punctuation_removed)

        l = [word.lower() for word in punctuation_removed.split()]
        X1.append(" ".join(l))

    return X1

def clean_new(data):

    data['row_string_new']=clean_new_util(data['row_string'])
    return data


#alpha_numeric_ratio() calls alpha_func() which calculates the ratio between number of digits and number of alphabets in the row_string
# If the ratio is below the threshold(0.05) it returns 0 else 1


def alpha_func(x):
    digit=sum([c.isdigit() for c in x])
    alpha=sum([c.isalpha() for c in x])
    k=0.0
    if(alpha==0):
        k=1
    elif((float(digit)/float(alpha))<0.05):
        k=0
    else:
        k=1

    return k


def alpha_numeric_ratio(data):
    data['number_alpha_ratio']=data['row_string_new'].apply(lambda x:alpha_func(x))
    data['number_alpha_ratio']=data['number_alpha_ratio'].astype('int')
    return data

##################################
#distance_from_top() calls distance_from_top_util which returns 0 if distance from top is less than or equal to 30 else 1
##################################

def distance_from_top_util(x):
    if(x<=30):
        return 0
    else:
        return 1

def distance_from_top(data):
    data['binned_distance_top']=data['row_distanceFromTop'].apply(lambda x: distance_from_top_util(x))
    data['binned_distance_top']=data['binned_distance_top'].astype('int')
    return data
    
##################################
#regex_mapper() calls regex_mapper_util() which finds if a particular regex is satisfied by the row_string. If so, it populates the corresponding regex field as 1 else 0
##################################

def regex_mapper_util(df_c):
    df_c['balance'] = None
    df_c['description'] = None
    df_c['material'] = None
    df_c['reference'] = None
    df_c['discount'] = None
    df_c['invoice'] = None
    df_c['date'] = None
    df_c['amount'] = None

    for i in range(0, df_c.shape[0]):


        if (re.search('[a-z0-9]*(balance)[ a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['balance'].values[i] = 0
        else:
            df_c['balance'].values[i] = 1

        if (re.search('[a-z0-9]*(descr)[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['description'].values[i] = 0
        else:
            df_c['description'].values[i] = 1

        if (re.search('[a-z0-9]*(check| che|item|material|quantity|qty)[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['material'].values[i] = 0
        else:
            df_c['material'].values[i] = 1

        if (re.search('[a-z0-9]*(ref)[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['reference'].values[i] = 0
        else:
            df_c['reference'].values[i] = 1

        if (re.search('[a-z0-9]*(disc)[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['discount'].values[i] = 0
        else:
            df_c['discount'].values[i] = 1

        if (re.search('[a-z0-9]*(invoice|inv no|inv nu|invno|invnu|invam|inv am)[a-z0-9]*',df_c['row_string_new'].values[i]) == None):
            df_c['invoice'].values[i] = 0
        else:
            df_c['invoice'].values[i] = 1


        if (re.search('[a-z0-9]*((amou)|(amt))[a-z0-9]*',df_c['row_string_new'].values[i]) == None):
            df_c['amount'].values[i] = 0
        else:
            df_c['amount'].values[i] = 1

        if (re.search('[a-z0-9]*(date)[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['date'].values[i] = 0
        else:
            df_c['date'].values[i] = 1


    return df_c

def regex_mapper(data):
    data=regex_mapper_util(data)
    return data




#####################################
# features list
####################################

features=['balance','description','material','reference','discount','invoice','amount','date','number_alpha_ratio','binned_distance_top']

#####################################################
#train and test data
#####################################################

##Training and testing
def predict_on_test_data(model_path,test_file_path,write_file_path):
    global features

    model=jb.load(model_path)
    #model=pd.read_pickle(model_path)
    test_data=pd.read_csv(test_file_path,encoding="ISO-8859-1",error_bad_lines=False)


    test_data=clean_new(test_data)
    test_data=alpha_numeric_ratio(test_data)
    test_data=distance_from_top(test_data)
    test_data=regex_mapper(test_data)


    pred=model.predict(test_data[features])
    pred_prob=model.predict_proba(test_data[features])
    df_pred=pd.DataFrame(pred,columns=['is_heading_predicted'])
    df_prob=pd.DataFrame(pred_prob)
    df_prob.columns=['prob_0','prob_1']
    test_data['is_heading_predicted']=df_pred['is_heading_predicted']
    test_data['is_heading_prob_0']=df_prob['prob_0']
    test_data['is_heading_prob_1']=df_prob['prob_1']

    test_data.to_csv(write_file_path+"/"+"is_heading_prediction.csv",index=False,encoding="utf-8")
    return test_data




