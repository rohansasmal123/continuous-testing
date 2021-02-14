import streamlit as st
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import logging
from sklearn.model_selection import GridSearchCV
import xgboost
import lightgbm as lgb
from lightgbm import LGBMModel, LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import sys


from rpscript.rpmodelling.topmodel import topmodel
from rpscript.rpmodelling import DataExt
from rpscript.rpmodelling.mod2 import make_predictions
import rpscript.rpmodelling.preprocess


def auto_pilot(acct_id):
    if not os.path.exists(root_dir+'/Data_Extracted'):
        os.makedirs(root_dir+'/Data_Extracted')
    progress_path = root_dir+"/account_"+str(acct_id)+"/logs/progress.csv"
    model_path = root_dir+'/account_'+str(acct_id)+'/trained_model/model.pkl'
    dock_path= root_dir+'/account_'+str(acct_id)
    summary = dock_path+'/summary.csv'
    data_extracted=root_dir+'/Data_Extracted/'+str(acct_id)+".csv"
    processed_data_path = root_dir+"/account_"+str(acct_id)+"/data_extracted/retraining_data.csv"
    predictions_path = root_dir+'/account_'+str(acct_id)+'/predictions/predictions.csv'
    test_path = root_dir+'/account_'+str(acct_id)+'/train_test_splitted/test_30.csv'
    train_path = root_dir+'/account_'+str(acct_id)+'/train_test_splitted/train_70.csv'

    if not os.path.exists(progress_path):
        data=pd.DataFrame()
        if not os.path.exists(data_extracted):
            
            if st.checkbox("Extract Data"):
                mode="auto"
                DataExt.main()
                data=pd.read_csv(data_extracted)
                preprocess.main(data,acct_id,root_dir,mode)


            else:
                dat_type = st.radio("Select the type of Data",('Pre-Processed Data','Raw Data'))
                if dat_type=='Pre-Processed Data':
                    mode="preprocessed"
                    prepro_data = st.file_uploader("Upload Preprocessed Data", type=["csv"])
                    if prepro_data is not None:
                        prepro_data.seek(0)
                        data=pd.read_csv(prepro_data)
                        data.to_csv(processed_data_path,index=False)
                else:
                    mode="auto"
                    raw_data = st.file_uploader("Upload Raw Data", type=["csv"])
                    if raw_data is not None:
                        raw_data.seek(0)
                        data=pd.read_csv(raw_data)
                        preprocess.main(data,acct_id,root_dir,mode)

        if os.path.exists(processed_data_path):
            with st.spinner("Execution in Progress...."):
                tasks=['HistoryGeneration.py',
                        'JsonCreation.py',
                        'SubsetsCreation.py',
                        'FeaturesCreation.py',
                        ]
                for task in tasks:
                    d=(os.system('python '+ rp_dir+'/'+str(task)+' '+str(acct_id) + ' ' + root_dir))
                    if d!=0:
                        st.error(task.split('.')[0] + " Failed")
                        break
                    else:
                        tts=(os.system('python '+ rp_dir+'/TrainTestSplit.py '+str(acct_id)+' ' + root_dir +' '+str(0.7))) 
                        if tts!=0:
                            st.error("Train Test Split Failed")
                            break

            print("Flow Operation executed Successfully!!")

            train_models(acct_id,rpdir,root_dir)

def train_models(acct_id,rpdir,root_dir):
    test_path = root_dir+'/account_'+str(acct_id)+'/train_test_splitted/test_30.csv'
    train_path = root_dir+'/account_'+str(acct_id)+'/train_test_splitted/train_70.csv'

    if os.path.exists(train_path):
        

    

            

            

                

                
            
            








def main():
    root_dir='/root/caa/rp/model'
    rp_dir = '/root/caascript/rpscript/rpmodelling'
   
    acct_id=st.text_input(label="Enter Account ID",key=1)

    if acct_id != "":
        with st.spinner("Creating Directory"):
            d=(os.system('python '+ rp_dir+'/directory_creation.py '+str(acct_id) + ' ' + root_dir))
        if d==0:
            print("---AutoPilot in execution---\n\n Directory Created")
            auto_pilot(acct_id)
        else:
            st.error("Directory creation Failed")
        
        
