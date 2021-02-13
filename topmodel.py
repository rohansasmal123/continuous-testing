'''
!/usr/bin/env python
@author:rohan_sasmal,,ayanava_dutta,subhrojyoti_roy,shivam_gupta
-*-coding:utf-8-*-
'''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import logging
import os
import json
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import xgboost
import lightgbm as lgb
from lightgbm import LGBMModel, LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
import streamlit as st

def initial_data(param_path):    
    training_model=[]
    model=pd.read_csv(param_path)
    for index,row in model.iterrows():
        if row.model_name == 'RandomForestClassier':            
            train_model=RandomForestClassifier()
        elif row.model_name == 'xgboost.XGBClassifier':
            train_model= xgboost.XGBClassifier()
        params=eval(row['model_params'])
        train_model.set_params(**params)
        training_model.append(train_model)
    model['training_model']=training_model
    return model

	
def train_model(training_data_path,param_path):
    logging.info('Model Training Starts.')
    features = ['avg_delay_categorical',
                'variance_categorical',
                'LMH_cumulative',
                'avg_of_invoices_closed',
                'avg_of_all_delays',
                'payment_count_quarter_q1', 'payment_count_quarter_q2', 'payment_count_quarter_q3',
                'payment_count_quarter_q4',
                'invoice_count_quarter_q1', 'invoice_count_quarter_q2', 'invoice_count_quarter_q3',
                'invoice_count_quarter_q4',
                'number_invoices_closed']
    trained_model=[]
    data=pd.read_csv(r''+training_data_path)
    model=initial_data(param_path)
    X_train = data[features]
    y_train = data['output']
    model.training_model.apply(lambda t:t.fit(X_train,y_train))
    return model
	
    
def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-1] = total

    class_report_df['avg / total'] = avg

    return class_report_df.T


def make_predictions(train_data_path, test_data_path, param_path):

    logging.info('Predictions Start.')
    features = ['avg_delay_categorical',
                'variance_categorical',
                'LMH_cumulative',
                'avg_of_invoices_closed',
                'avg_of_all_delays',
                'payment_count_quarter_q1', 'payment_count_quarter_q2', 'payment_count_quarter_q3',
                'payment_count_quarter_q4',
                'invoice_count_quarter_q1', 'invoice_count_quarter_q2', 'invoice_count_quarter_q3',
                'invoice_count_quarter_q4',
                'number_invoices_closed']

    data2=pd.read_csv(r''+test_data_path)

    x=train_model(train_data_path, param_path)
    s=x.shape[0]
    if len(data2)!=0:
        X_validation = data2[features]
        y_validation = data2['output']
    else:
        logging.info('No Prediction data')
        return

    accuracy_score_pred=[]
    recall_one=[]
    recall_zero=[]
    percentage=[]
    total_payment=[]
    correct_payment=[]
    without_subset=[]
    incorrect_payment=[]

    for index,row in x.iterrows():
        rfc=row['training_model']
        predictions = rfc.predict(X_validation)
        predictions_prob = rfc.predict_proba(X_validation)
        logging.info('confusion matric ')
        logging.info(confusion_matrix(y_validation, predictions))
        logging.info('classification Report ')
        

        data2['predictions'] = predictions
        for i in range(0, data2.shape[0]):
            data2.at[i, 'pred_proba_0'] = predictions_prob[i][0]
            data2.at[i, 'pred_proba_1'] = predictions_prob[i][1]

        dataset=data2

        payment_without_any_subset=0
        dataset['transformed_output']=0
        for i in dataset['payment_id'].unique():
            max_proba=dataset[dataset['payment_id']==i]['pred_proba_1'].max()
            dataset.loc[(dataset['payment_id']==i) & (dataset['pred_proba_1']==max_proba),'transformed_output']=1
            if len(dataset[dataset['payment_id']==i]['output'].unique())==1:
                payment_without_any_subset=payment_without_any_subset+1

        logging.info('***** After Output transformation *****')
        accuracy_score_pred.append(accuracy_score(y_validation, dataset['transformed_output']))
        logging.info('confusion matrix ')
        logging.info(confusion_matrix(y_validation, dataset['transformed_output']))
        logging.info('classification Report ')
        report=pandas_classification_report(y_validation, predictions)
        #print(report)
        #report_df = pd.DataFrame(report).transpose()
        report_df = report
        #print(report_df)
        recall_one.append(report_df['recall'][0])
        recall_zero.append(report_df['recall'][1])
        total_payment.append(len(dataset['payment_id'].unique()))
        correct_payment.append(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==1)]))
        incorrect_payment.append(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==0)]))
        without_subset.append(payment_without_any_subset)
        percentage.append(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==1)])/len(dataset['payment_id'].unique())*100)
        logging.info('Predictions Ended') 

        
    
    x=x.assign(accuracy_score = accuracy_score_pred)
    x=x.assign(recall_1 = recall_one)
    x=x.assign(recall_0 = recall_zero)
    x=x.assign(total_payment = total_payment)
    x=x.assign(correct_payment = correct_payment)
    x=x.assign(incorrect_payment = incorrect_payment)
    x=x.assign(without_subset = without_subset)
    x['accuracy_score']=accuracy_score_pred
    
    return x
	
	
def ranking(metric):
    metric.drop_duplicates(subset=['recall_diff','percentage'],keep='first',inplace=True)
    
    metric= metric[metric['accuracy_score']>=0.60]
    metric.sort_values(['percentage'],ascending=False,inplace=True)
    top_10percent = metric.iloc[:10]
    top_10percent.sort_values(['recall_diff'],ascending=True,inplace=True)
    return top_10percent
	
def topmodel(root_dir, acct_id):
    train_data_path = root_dir+'/account_'+str(acct_id)+'/train_test_splitted/train_70.csv'
    test_data_path = root_dir+'/account_'+str(acct_id)+'/train_test_splitted/test_30.csv'
    log_path=root_dir+'/account_'+str(acct_id)+'/logs/'
    param_path="/root/caascript/rpscript/rpmodelling/model_data.csv"
    

    metric=make_predictions(train_data_path, test_data_path,param_path)
    
    metric['percentage'] = metric['correct_payment']/(metric['total_payment']-metric['without_subset'])
    metric['recall_diff'] = abs(metric['recall_1']-metric['recall_0'])
    top_10percent =  ranking(metric)
    #print(top_10percent)
    top_10percent.reset_index(inplace=True)
    #st.write(top_10percent)
    progress=pd.read_csv(log_path+"progress.csv")
    progress['Status']='topmodel.py'
    progress.to_csv(log_path+"progress.csv",index=False)

    return top_10percent
	