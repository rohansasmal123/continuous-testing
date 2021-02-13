'''
!/usr/bin/env python
@author:ayanava_dutta,shivam_gupta,rohan_sasmal
-*-coding:utf-8-*-
'''
# Base Package
import streamlit as st
import pandas as pd
import numpy as np
# Modeling Packages & evaluation metrics
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import xgboost
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from lightgbm import LGBMModel, LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import randint,truncnorm
from scipy.stats import uniform 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
#Helper modules
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)


#-------------------------------------------------------Classification Report -------------------------------------------------------------------------------------
def generate_classification_report(y_true, y_pred):
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

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------ Make Prediction---------------------------------------------------------------------------------
 
def make_predictions(predictions_path,test_path,model_path, clf,summary,dock_path):
    
    #model_path='C:/Users/Ayanava/Desktop/model.pkl'
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

    data2=pd.read_csv(r''+test_path)#inp test data

    if len(data2)!=0:
        X_validation = data2[features]
        y_validation = data2['output']
    else:
        st.write('No Prediction data')
        return

    rfc=clf

    predictions = rfc.predict(X_validation)
    predictions_prob = rfc.predict_proba(X_validation)
    st.write('accuracy_score ' + str(accuracy_score(y_validation, predictions)))
    st.write('confusion matric ')
    st.write(confusion_matrix(y_validation, predictions))
    
    st.write('classification Report ')
    #st.table(classification_report(y_validation, predictions, output_dict=True))
    cr1=generate_classification_report(y_validation, predictions)
    st.table(cr1)

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

    st.header('After Output Transformation')
    acu_score=str(accuracy_score(y_validation, dataset['transformed_output']))
    st.write('Accuracy_score: ' + acu_score)
    st.write('Confusion matrix: ')
    st.write(confusion_matrix(y_validation, dataset['transformed_output']))
    st.write('Classification Report: ')
    #st.table(classification_report(y_validation, dataset['transformed_output'],output_dict=True))
    cr2=generate_classification_report(y_validation, dataset['transformed_output'])
    st.table(cr2)
    #st.table(metrics.classification_report(y_test, y_pred, output_dict=True, target_names=list(np.unique(y_train))))
    Total_Payment=str(len(dataset['payment_id'].unique()))
    correct_payment=str(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==1)]))
    incorrect_payment=str(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==0)]))
    pay_without_subset =str(payment_without_any_subset)
    perc=str(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==1)])/len(dataset['payment_id'].unique())*100)
    
    
    st.write('Total Payment : ' + Total_Payment)
    st.write('Total correct payment(s) : ' + correct_payment)
    st.write('Total incorrect payment(s) : ' + incorrect_payment)
    st.write('Total payments without any subset : ' + pay_without_subset)
    st.write('%age : '+ perc)
    
    if st.button(label='Save Predictions'):
    
        summary['Total Payment']=Total_Payment
        summary['Total correct payment(s)']=correct_payment
        summary['Total incorrect payment(s)']=incorrect_payment
        summary['Total payments without any subset']=pay_without_subset
        summary['%age']=perc
        summary['Accuracy']=acu_score
        summary['Recall of 0']=cr2['recall'][0]
        summary['Recall of 1']=cr2['recall'][1]
        
        save_summary(summary,dock_path)
        pickle.dump(rfc, open(model_path, 'wb'))
        cr2.to_csv('Report.csv',  sep=',')
        data2.to_csv(predictions_path, index=False) 
        st.success("Predictions Saved!")
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------Target-Feature-Sep----------------------------------------------------------------------------------------

def train_model(df):
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

    X_train = df[features]
    y_train = df['output']
    
    return X_train, y_train

#---------------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------Hyperparameter Tuning-------------------------------------------------------------------
def hyper_tune(mod_o,df):
    if st.checkbox(label='Tune Params'):
                ch2 = st.radio("Choose From",('Randomised Search CV', 'Grid Search CV'))
                if ch2=='Randomised Search CV':
                    st.subheader("Hyperparameter Tuning -Randomised Search CV")
                    X_train, y_train=train_model(df)
                    user_param=st.text_input(label='Enter Parameters To Tune',value="Enter In Dictionary format" )
                    n_iter=st.number_input(label="Enter numer of iterations",value=100,min_value=1)
                    cv=st.number_input(label="Enter number of folds for CV",value=5,min_value=1)
                    st.text("This will train "+str(n_iter)+" models over "+str(cv)+" folds of cross validation: ("+str(n_iter*cv)+" models in total)")
                    
                    
                    if st.button("Start RandomizedSearchCV"):
                        with st.spinner("Searching for the Best Params"):
                            user_param=eval(user_param)
                            clf = RandomizedSearchCV(mod_o, user_param, n_iter=n_iter, cv=cv, random_state=1)
                            model = clf.fit(X_train,y_train)
                            rs_params=model.best_estimator_.get_params()
                            st.write("Best params:")
                            st.text(model.best_estimator_.get_params())
                        return rs_params

                else:
                    st.subheader("Hyperparameter Tuning -Grid Search CV")
                    if st.checkbox('Show Alert',value=True):
                        st.warning("Grid Search is an exhaustive search, Grid Search looks through each combination of hyperparameters. This maybe computationally expensive as every combination of specified hyperparameter values will be tried")
                    st.write("This Feature is Under Devolopement")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------Classification Models----------------------------------------------------------------------
def lgbm(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir):
    X_train, y_train=train_model(df)
    ch = st.radio("Choose From",('Basic Parameters', 'Enter Manually'))
    if ch=='Basic Parameters':
        n_estimators = st.number_input(label='Enter Number of Estimator (Integer)',value=100, min_value=2)
        random_state = st.number_input(label='Enter Random state(Integer)',value=0, min_value=0)
        max_depth = st.number_input(label='Enter Depth of Tree (Integer)',value= -1)
        learning_rate = st.text_input(label='Enter learning rate',value='0.01',max_chars=10,type='default')
        subsample = st.number_input(label='Enter value for subsample ',value=1.0, min_value=0.0)
        num_leaves = st.number_input(label='Enter min samples split (Integer)',value=31, min_value=0)
        reg_alpha = st.text_input(label='Enter value for reg alpha',value='0', max_chars=10,type='default')
        reg_lambda = st.text_input(label='Enter value for reg lambda',value='0', max_chars=10,type='default')
        class_weight = st.text_input(label='Enter class weights in dictionary format',value='balanced', max_chars=20,type='default')
        boosting_type = st.selectbox(label='Select criterion', options=['gbdt','dart','goss','rf'])
        
        if class_weight!='balanced':
            class_weight=eval(class_weight)
        else:
            pass
            
        if st.checkbox(label='Train Model'):
            lgb = LGBMClassifier( max_depth=max_depth,subsample=subsample,random_state=random_state, 
                                num_leaves=num_leaves,n_estimators=n_estimators,learning_rate=learning_rate,
                                reg_alpha=reg_alpha,reg_lambda=reg_lambda,class_weight=class_weight,boosting_type=boosting_type)
            clf = lgb.fit(X_train, y_train)
            scores = cross_val_score(clf, X_train, y_train, cv=5)
            st.write('cross-validation scores: ' + str(scores))
            st.write('accuracy score of cross validation :' + str(scores.mean() * 100))
            summary['cross-val scores']=str(scores)
            mod_spec=str(clf).split('(')[0]+": "+str(clf.get_params())
            summary['model specs']=str(mod_spec)
            save_summary(summary,dock_path)
            st.success('Model Training Completed!')
            
        if st.checkbox(label='See predictions'):
                    make_predictions(predictions_path,test_path,model_path, clf,summary,dock_path) 
                    
                    if st.button("Generate Test Files"):
                        with st.spinner("Execution in Progress"):
                            os.system('python '+rp_dir+'/PMML_creation.py '+str(acct_id)+" "+str(root_dir))
                            os.system('python '+rp_dir+'/generate_Test_Files.py '+str(acct_id)+" "+str(root_dir))
                            st.success("Test Files Generated")
                            st.success("Account ready for Deployment")
    else:
        params=st.text_input(label='Enter Best Parameters' )
        mod_o=LGBMClassifier()
        if params=="":
            rs_params=hyper_tune(mod_o,df)  
            
                
        elif params!="":
            if st.checkbox(label='Train Model'):
                st.text("Training Model with user defined Parameters")
                params=eval(params)
                st.text(params)
                lgb= LGBMClassifier()
                lgb=lgb.set_params(**params)
                clf = lgb.fit(X_train, y_train)
                
                scores = cross_val_score(clf, X_train, y_train, cv=5)
                st.write('Cross-validation scores: ' + str(scores))
                st.write('Accuracy score of cross validation :' + str(scores.mean() * 100))
                summary['cross-val scores']=str(scores)
                mod_spec=str(clf).split('(')[0]+": "+str(clf.get_params())
                summary['model specs']=str(mod_spec)
                save_summary(summary,dock_path)
                st.success('Model Training Completed!')

            if st.checkbox(label='See predictions'): 
                    make_predictions(predictions_path,test_path,model_path, clf,summary,dock_path) 
                    
                    if st.button("Generate Test Files"):
                        with st.spinner("Execution in Progress"):
                            os.system('python '+rp_dir+'/PMML_creation.py '+str(acct_id)+" "+str(root_dir))
                            os.system('python '+rp_dir+'/generate_Test_Files.py '+str(acct_id)+" "+str(root_dir))
                            st.success("Test Files Generated")
                            st.success("Account ready for Deployment")

        if st.checkbox(label='Show Help Text?'):
            expander = st.beta_expander("FAQ")
            st.write("Please Enter Parameters in the following format:")
            st.text("{'boosting_type': 'gbdt', 'class_weight':"+str({0: 1, 1: 5})+", 'colsample_bytree': 1.0, 'learning_rate': 0.08, }")
            st.write("If you wish to enter a range of params for hyper tuning ")
            st.text("{'num_leaves': randint(6, 50),'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]}")
            
            


def d_lgbm(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir):
    X_train, y_train=train_model(df)
    Model = st.radio("Choose From A Tuned Light-GBM model",('model1', 'model2', 'model3', 'model4', 'model5'))

    if Model=='model1':
        st.write("Parameters:")
        st.text("num_leaves=1000")
        st.text("min_data_in_leaf=1")
        st.text("learning_rate=0.08")
        st.text("max_depth=10")
        st.text("min_data_in_bin=1")
         
        lgb=LGBMClassifier( max_depth=10, num_leaves=1000, min_data_in_leaf=1,learning_rate=0.08,min_data_in_bin=1)    
        clf = lgb.fit(X_train, y_train)

    elif Model=='model2':
        st.write("Parameters:")
        st.text("class_weight={0: 1, 1: 5}")
        st.text("num_leaves=1000")
        st.text("max_depth=10")
        st.text("learning_rate= 0.08")
        
        lgb=LGBMClassifier(class_weight={0: 1, 1: 5}, max_depth=10, num_leaves=1000, learning_rate=0.08)
        clf = lgb.fit(X_train, y_train)    
     
    elif Model=='model3':
        st.write("Parameters:")
        st.text("n_estimators=100")
        st.text("random_state=42")
        st.text("max_depth=13")
        st.text("learning_rate= 0.01")
        st.text("min_child_samples=25")
        st.text("n_jobs=-1")
        st.text("scale_pos_weight=16")
        st.text("min_child_weight=0.001")
        st.text("colsample_bytree=1.0")
        st.text("reg_lambda=1000")
        st.text("boosting_type='dart'")
        lgb=LGBMClassifier(boosting_type='dart', 
                          colsample_bytree=1.0,
                          learning_rate=0.01,
                          max_depth=13,
                          min_child_samples=25,
                          min_child_weight=0.001,
                          n_estimators=100,
                          n_jobs=-1,
                          reg_lambda=1000,
                          scale_pos_weight=16)
        clf = lgb.fit(X_train, y_train)   
       
    elif Model=='model4':
        st.write("Parameters:")
        
        st.text("n_estimators=312")
        st.text("random_state=42")
        st.text("max_depth=8")
        st.text("learning_rate= 0.2")
        st.text("boosting_type='dart'")

        lgb=LGBMClassifier(random_state=33, num_leaves=10, n_estimators= 312, max_depth= 8, learning_rate= 0.2, boosting_type= 'dart')
        
        clf = lgb.fit(X_train, y_train)

    elif Model=='model5':
        st.write("Parameters:")
        st.text("n_estimators=402")
        st.text("random_state=33")
        st.text("max_depth=10")
        st.text("learning_rate= 0.094")
        st.text("n_jobs=-1")
        st.text("num_leaves = 50")
        st.text("boosting_type='dart'")
        lgb=LGBMClassifier(random_state = 33, num_leaves = 50, n_estimators = 402, max_depth= 10, learning_rate=0.094, boosting_type= 'dart', n_jobs= -1)
        clf = lgb.fit(X_train, y_train)   
    
    if st.checkbox(label='Train Model'):
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        st.write('Cross-validation scores: ' + str(scores))
        st.write('Accuracy score of cross validation :' + str(scores.mean() * 100))
        summary['cross-val scores']=str(scores)
        mod_spec=str(clf).split('(')[0]+": "+str(clf.get_params())
        summary['model specs']=str(mod_spec)
        save_summary(summary,dock_path)
        st.success('Model Training Completed!')

    if st.checkbox(label='See predictions'):
            make_predictions(predictions_path,test_path,model_path, clf,summary,dock_path)
            
            if st.button("Generate Test Files"):
                with st.spinner("Execution in Progress"):
                    os.system('python '+rp_dir+'/PMML_creation.py '+str(acct_id)+" "+str(root_dir))
                    os.system('python '+rp_dir+'/generate_Test_Files.py '+str(acct_id)+" "+str(root_dir))
                    st.success("Test Files Generated")
                    st.success("Account ready for Deployment")
    

        

def random_forest_clf(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir):
    X_train, y_train=train_model(df)
    ch = st.radio("Choose From",('Basic Parameters', 'Enter Manually'))
    if ch=='Basic Parameters':
        n_estimators = st.number_input(label='Enter Number of Estimator (Integer)',value=100, min_value=2)
        random_state = st.number_input(label='Enter Random state(Integer)',value=0, min_value=0)
        max_depth = st.number_input(label='Enter Depth of Tree (Integer)',value=1, min_value=0)
        
        min_samples_leaf = st.number_input(label='Enter min samples leaf (Integer)',value=1, min_value=0)
        min_samples_split = st.text_input(label='Enter min samples split ',value='2',max_chars=10)
        max_features = st.text_input(label='Enter value for max_features',value='auto', max_chars=10,type='default')
        class_weight = st.text_input(label='Enter class weights in dictionary format',value='balanced', max_chars=20, type='default')
        criterion = st.selectbox(label='Select criterion', options=['gini','entropy'])
        if class_weight!='balanced':
            class_weight=eval(class_weight)
        else:
            pass
        
        if st.checkbox(label='Train Model'):
            rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight=class_weight, max_depth=max_depth,
                                    max_features=max_features, min_samples_leaf=min_samples_leaf,criterion=criterion,
                                    min_samples_split=float(min_samples_split))
            clf = rfc.fit(X_train, y_train)
            
            scores = cross_val_score(clf, X_train, y_train, cv=5)
            st.write('cross-validation scores: ' + str(scores))
            st.write('accuracy score of cross validation :' + str(scores.mean() * 100))
            summary['cross-val scores']=str(scores)
            mod_spec=str(clf).split('(')[0]+": "+str(clf.get_params())
            summary['model specs']=str(mod_spec)
            save_summary(summary,dock_path)
            st.success('Model Training Completed!')


        if st.checkbox(label='See predictions'):
                    make_predictions(predictions_path,test_path,model_path, clf,summary,dock_path) 
                    
                    if st.button("Generate Test Files"):
                        with st.spinner("Execution in Progress"):
                            os.system('python '+rp_dir+'/PMML_creation.py '+str(acct_id)+" "+str(root_dir))
                            os.system('python '+rp_dir+'/generate_Test_Files.py '+str(acct_id)+" "+str(root_dir))
                            st.success("Test Files Generated")
                            st.success("Account ready for Deployment")
        

    else:
        params=st.text_input(label='Enter Best Parameters' )
        mod_o=RandomForestClassifier()
        if params=="":
            rs_params=hyper_tune(mod_o,df) 
         
            

        elif params!="":
            if st.checkbox(label='Train Model'):
                st.text("Training Model with user defined Parameters")
                params=eval(params)
                st.text(params)
                rfc= RandomForestClassifier()
                rfc=rfc.set_params(**params)
                clf = rfc.fit(X_train, y_train)
                
                scores = cross_val_score(clf, X_train, y_train, cv=5)
                st.write('Cross-validation scores: ' + str(scores))
                st.write('Accuracy score of cross validation :' + str(scores.mean() * 100))
                summary['cross-val scores']=str(scores)
                mod_spec=str(clf).split('(')[0]+": "+str(clf.get_params())
                summary['model specs']=str(mod_spec)
                save_summary(summary,dock_path)
                st.success('Model Training Completed!')

            if st.checkbox(label='See predictions'):
                    make_predictions(predictions_path,test_path,model_path, clf,summary,dock_path) 
                    
                    if st.button("Generate Test Files"):
                        with st.spinner("Execution in Progress"):
                            os.system('python '+rp_dir+'/PMML_creation.py '+str(acct_id)+" "+str(root_dir))
                            os.system('python '+rp_dir+'/generate_Test_Files.py '+str(acct_id)+" "+str(root_dir))
                            st.success("Test Files Generated")
                            st.success("Account ready for Deployment")
            
        if st.checkbox(label='Show Help Text?'):
            expander = st.beta_expander("FAQ")
            st.write("Please Enter Parameters in the following format:")
            st.text("{'random_state':42,'class_weight':"+str({0: 1, 1: 5})+",'n_estimators':106}")
            st.write("Press ENTER to apply the changes")
            st.write("If you wish to enter a range of params for hyper tuning ")
            st.text("{'n_estimators': randint(4,200),'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),'min_samples_split': uniform(0.01, 0.199)}")


def d_randomforest_clf(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir):
    X_train, y_train=train_model(df)
    Model = st.radio("Choose From A Tuned Random Forest Classification model",('model1', 'model2'))

    if Model=='model1':
        st.write("Parameters:")
        st.text("n_estimators=200")
        st.text("random_state=42")
        st.text("max_depth=7")
        st.text("class_weight={0: 1, 1: 2}")
        st.text("max_features=0.4")
        st.text("min_samples_leaf=4")
        rfc= RandomForestClassifier(n_estimators=200, 
                                    random_state=42, 
                                    class_weight={0: 1, 1: 2}, 
                                    max_depth=7, 
                                    max_features=0.4, 
                                    min_samples_leaf=4)
        clf = rfc.fit(X_train, y_train)
        
    elif Model=='model2':
        st.write("Parameters:")
        st.text("n_estimators=400")
        st.text("random_state=42")
        st.text("max_depth=1")
        st.text("min_weight_fraction_leaf= 0.3")
        st.text("min_samples_split= 24")
        st.text("min_samples_leaf= 1")
        st.text("max_features= 'sqrt'")
        st.text("criterion= 'entropy'")

        rfc= RandomForestClassifier(random_state= 42, 
                                    n_estimators= 400,
                                    min_weight_fraction_leaf= 0.3,
                                    min_samples_split= 24,
                                    min_samples_leaf= 1, 
                                    max_features= 'sqrt', 
                                    max_depth= 1, 
                                    criterion= 'entropy')
        clf = rfc.fit(X_train, y_train)
       
     
    if st.checkbox(label='Train Model'):
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        st.write('Cross-validation scores: ' + str(scores))
        st.write('Accuracy score of cross validation :' + str(scores.mean() * 100))
        summary['cross-val scores']=str(scores)
        mod_spec=str(clf).split('(')[0]+": "+str(clf.get_params())
        summary['model specs']=str(mod_spec)
        save_summary(summary,dock_path)
        st.success('Model Training Completed!')

    if st.checkbox(label='See predictions'):
            make_predictions(predictions_path,test_path,model_path, clf,summary,dock_path)
            
            if st.button("Generate Test Files"):
                with st.spinner("Execution in Progress"):
                    os.system('python '+rp_dir+'/PMML_creation.py '+str(acct_id)+" "+str(root_dir))
                    os.system('python '+rp_dir+'/generate_Test_Files.py '+str(acct_id)+" "+str(root_dir))
                    st.success("Test Files Generated")
                    st.success("Account ready for Deployment")




def xgb(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir):
    X_train, y_train=train_model(df)
    ch = st.radio("Choose From",('Basic Parameters', 'Enter Manually'))
    
    if ch=='Basic Parameters':
        n_estimators = st.number_input(label='Enter Number of Estimator (Integer)',value=100, min_value=2)
        random_state = st.number_input(label='Enter Random state(Integer)',value=0, min_value=0)
        max_depth = st.number_input(label='Enter Depth of Tree (Integer)',value=3, min_value=1)
        learning_rate = st.text_input(label='Enter learning rate',value='0.01',max_chars=10,type='default')
        min_child_weight = st.number_input(label='Enter min child weight (Integer)',value=1, min_value=0)
        scale_pos_weight = st.number_input(label='Enter scale pos weight (Integer)',value=1, min_value=0)
        reg_alpha = st.text_input(label='Enter value for reg alpha',value='0', max_chars=10,type='default')
        gamma = st.text_input(label='Enter value for gamma',value='0', max_chars=10, type='default')
        nthread=st.number_input(label='Enter n_threads (No. of Parallel Threads)',value=0, min_value=0)
        seed = st.number_input(label='Enter number seeds', value=0,min_value=0)
       

        if st.checkbox(label='Train Model'):
            xgb= xgboost.XGBClassifier(random_state=random_state,n_estimators=n_estimators,min_child_weight=min_child_weight,
                                    max_depth=max_depth, learning_rate=learning_rate,seed=seed,nthread=nthread,gamma=gamma,reg_alpha=reg_alpha,scale_pos_weight=scale_pos_weight,objective='binary:logistic')

            clf = xgb.fit(X_train, y_train)
            
            scores = cross_val_score(clf, X_train, y_train, cv=5)
            st.write('Cross-validation scores: ' + str(scores))
            st.write('Accuracy score of cross validation :' + str(scores.mean() * 100))
            summary['cross-val scores']=str(scores)
            mod_spec=str(clf).split('(')[0]+": "+str(clf.get_params())
            summary['model specs']=str(mod_spec)
            save_summary(summary,dock_path)
            st.success('Model Training Completed!')
            
        if st.checkbox(label='See predictions'):
                    make_predictions(predictions_path,test_path,model_path, clf,summary,dock_path) 
                    
                    if st.button("Generate Test Files"):
                        with st.spinner("Execution in Progress"):
                            os.system('python '+rp_dir+'/PMML_creation.py '+str(acct_id)+" "+str(root_dir))
                            os.system('python '+rp_dir+'/generate_Test_Files.py '+str(acct_id)+" "+str(root_dir))
                            st.success("Test Files Generated")
                            st.success("Account ready for Deployment")
        
    else:
        params=st.text_input(label='Enter Best Parameters' )
        mod_o=xgboost.XGBClassifier()
        if params=="":
            rs_params=hyper_tune(mod_o,df) 
          
            

        elif params!="":
            if st.checkbox(label='Train Model'):
                st.text("Training Model with user defined Parameters") 
                params=eval(params)
                st.text(params)
                xgb= xgboost.XGBClassifier()
                xgb=xgb.set_params(**params)
                clf = xgb.fit(X_train, y_train)
                
                scores = cross_val_score(clf, X_train, y_train, cv=5)
                st.write('Cross-validation scores: ' + str(scores))
                st.write('Accuracy score of cross validation :' + str(scores.mean() * 100))
                summary['cross-val scores']=str(scores)
                mod_spec=str(clf).split('(')[0]+": "+str(clf.get_params())
                summary['model specs']=str(mod_spec)
                save_summary(summary,dock_path)
                st.success('Model Training Completed!')

            if st.checkbox(label='See predictions'):
                    make_predictions(predictions_path,test_path,model_path, clf,summary,dock_path) 
                    if st.button("Generate Test Files"):
                        with st.spinner("Execution in Progress"):
                            os.system('python '+rp_dir+'/PMML_creation.py '+str(acct_id)+" "+str(root_dir))
                            os.system('python '+rp_dir+'/generate_Test_Files.py '+str(acct_id)+" "+str(root_dir))
                            st.success("Test Files Generated")
                            st.success("Account ready for Deployment")

        if st.checkbox(label='Show Help Text?'):
            expander = st.beta_expander("FAQ")
            st.write("Please Enter Parameters in the following format:")
            st.text("{'random_state':42,'n_estimators':106,'learning_rate':0.0044}")
            st.write("Press ENTER to apply the changes")
            st.write("If you wish to enter a range of params for hyper tuning ")
            st.text("{'max_depth': randint(6, 10),'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]}")
                
                
            
     
     
     
def d_xgb(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir):
    X_train, y_train=train_model(df)
    Model = st.radio("Choose From A Tuned XGB-model",('model1', 'model2', 'model3', 'model4', 'model5'))

    if Model=='model1':
        st.write("Parameters:")
        st.text("n_estimators=106")
        st.text("random_state=42")
        st.text("max_depth=1")
        st.text("learning_rate= 0.0044")
        xgb= xgboost.XGBClassifier(random_state=42,n_estimators=106,max_depth=1, learning_rate= 0.0044)
        clf = xgb.fit(X_train, y_train)
        
    elif Model=='model2':
        st.write("Parameters:")
        st.text("n_estimators=145")
        st.text("random_state=42")
        st.text("max_depth=33")
        st.text("learning_rate= 0.0077")
        xgb=xgboost.XGBClassifier(random_state=42, n_estimators=145, max_depth=33, learning_rate=0.0077)
        clf = xgb.fit(X_train, y_train)    
     
    elif Model=='model3':
        st.write("Parameters:")
        st.text("n_estimators=206")
        st.text("random_state=42")
        st.text("max_depth=1")
        st.text("learning_rate= 0.0015")
        xgb=xgboost.XGBClassifier(random_state=42, n_estimators=206, max_depth=1, learning_rate=0.0015)
        clf = xgb.fit(X_train, y_train)   
       
    elif Model=='model4':
        st.write("Parameters:")
        
        st.text("n_estimators=500")
        st.text("objective='binary:logistic'")
        st.text("colsample_bytree=0.85")
        st.text("subsample=0.75")
        st.text("max_depth=4")
        st.text("learning_rate= 0.05")
        st.text("gamma=0.4")
        st.text("reg_alpha= 1e-05")
        st.text("nthread=4")
        st.text("scale_pos_weight=1")
        st.text("seed=27")

        xgb=xgboost.XGBClassifier(learning_rate=0.05,
                                    n_estimators=500,
                                    max_depth=4,
                                    min_child_weight=4,
                                    gamma=0.4,
                                    subsample=0.75,
                                    colsample_bytree=0.85,
                                    objective='binary:logistic',
                                    reg_alpha= 1e-05,
                                    nthread=4,
                                    scale_pos_weight=1,
                                    seed=27)
       
        clf = xgb.fit(X_train, y_train)    
       
    elif Model=='model5':
        st.write("Parameters:")
        
        st.text("n_estimators=350")
        st.text("objective='binary:logistic'")
        st.text("colsample_bytree=0.4")
        st.text("subsample=0.8")
        st.text("max_depth=4")
        st.text("learning_rate= 0.01")
        st.text("gamma=10")
        st.text("reg_alpha= 0.3")
        st.text("nthread=4")
        st.text("scale_pos_weight=1")
        st.text("seed=27")

        xgb = xgboost.XGBClassifier(silent=False,
                                        scale_pos_weight=1,
                                        learning_rate=0.01,
                                        colsample_bytree=0.4,
                                        subsample=0.8,
                                        objective='binary:logistic',
                                        n_estimators=350,
                                        reg_alpha=0.3,
                                        max_depth=4,
                                        gamma=10)
       
        clf = xgb.fit(X_train, y_train)     


    
       
    if st.checkbox(label='Train Model'):
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        st.write('Cross-validation scores: ' + str(scores))
        st.write('Accuracy score of cross validation :' + str(scores.mean() * 100))
        summary['cross-val scores']=str(scores)
        mod_spec=str(clf).split('(')[0]+": "+str(clf.get_params())
        summary['model specs']=str(mod_spec)
        save_summary(summary,dock_path)
        st.success('Model Training Completed!')
        
    if st.checkbox(label='See predictions'):
            make_predictions(predictions_path,test_path,model_path, clf,summary,dock_path)
          
            if st.button("Generate Test Files"):
                with st.spinner("Execution in Progress"):
                    os.system('python '+rp_dir+'/PMML_creation.py '+str(acct_id)+" "+str(root_dir))
                    os.system('python '+rp_dir+'/generate_Test_Files.py '+str(acct_id)+" "+str(root_dir))

                    st.success("Test Files Generated")
                    st.success("Account ready for Deployment")
                
            
            
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
def save_summary(summary,dock_path):
    summary.to_csv(dock_path+'/summary.csv',index=False)
        
    
    
#------------------------------------------------------------------Menu-------------------------------------------------------------------------------------------------
def model(predictions_path,train_path,test_path,model_path,acct_id,dock_path,rp_dir,root_dir):
    st.subheader('Choose Model Specifications')
    
    summary=pd.read_csv(dock_path+'/summary.csv')
    df=pd.read_csv(train_path)
    methodlist = st.sidebar.selectbox(label='Select Algorithm', options=['Tuned XG-Boost',
                                                                         'Tuned Random-Forest',
                                                                         'Tuned Light-GBM',
                                                                         'XG-Boost',  
                                                                         'Random Forest Classification',
                                                                         'Light GBM']
                                                                         )
    if methodlist == 'Light GBM':
        lgbm(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir)
        
    elif methodlist == 'Random Forest Classification':
        random_forest_clf(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir)
        
    elif methodlist == 'XG-Boost': 
        xgb(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir)
        
    elif methodlist == 'Tuned XG-Boost': 
        d_xgb(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir)
        
    elif methodlist== 'Tuned Random-Forest':
        d_randomforest_clf(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir)
        
    elif methodlist== 'Tuned Light-GBM':
        d_lgbm(df,predictions_path,test_path,model_path,acct_id,summary,dock_path,rp_dir,root_dir)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#main
def modelling_main():
    #root_dir='/root/accounts'
    img_path='/root/caascript/res/bg/'
    
    
    root_dir = '/root/caa/rp/model'
    rp_dir = '/root/caascript/rpscript/rpmodelling'
    
    try:
        acct_id=st.text_input(label='Enter Account ID for Model Training')
        if acct_id!="":

            #Donot change path and Dock_path

            path=root_dir+'/account_'+str(acct_id)

            test_path= path+'/train_test_splitted/test_30.csv'
            train_path= path+'/train_test_splitted/train_70.csv'
            predictions_path= path+'/predictions/predictions.csv'
            model_path=path+'/trained_model/model.pkl'
            dock_path=root_dir+'/account_'+str(acct_id)
            
            model(predictions_path,train_path,test_path,model_path,acct_id,dock_path,rp_dir,root_dir)
        
    except FileNotFoundError:
        st.error("No data Found for Training Account ID: "+str(acct_id))