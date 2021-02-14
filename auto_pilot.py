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
import rpscript.rpmodelling.preprocess as preprocess

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def execute(exectute_from,acct_id,root_dir,rp_dir):
    script = ["HistoryGeneration.py","JsonCreation.py","SubsetsCreation.py","FeaturesCreation.py","TrainTestSplit.py","topmodel.py"]
    start = script.index(exectute_from)
    for i in range(start+1,6):
        if script[i]=="TrainTestSplit.py":
            thres=str(0.7)
            d=os.system("python "+rp_dir+"/"+script[i]+" " +str(acct_id) + " "+root_dir+" "+thres)

            if (d == 0):
                st.success(script[i].split(".")[0]+" Done")   
     
            else:
                st.error(script[i].split(".")[0]+" Failed")

        elif script[i]=="topmodel.py":
            top10percent = topmodel(root_dir,acct_id)
            return top10percent
     
        else:
            
            d=os.system("python "+rp_dir+"/"+script[i]+" " +str(acct_id) +" "+root_dir)
            if (d==0):
                st.success(script[i].split(".")[0]+" Done")            
            else:
                st.error(script[i].split(".")[0]+" Failed!")
              
        
def save_model(predictions_path,test_path,model_path,user_model,summary,dock_path,acct_id,root_dir,rp_dir):
    if st.button(label="Generate Test Files"):
        with st.spinner("Execution in Progress"):
            pickle.dump(user_model, open(model_path, 'wb'))
            os.system('python ' +rp_dir+'/MakePredictions.py')
            os.system('python '+rp_dir+'/PMML_creation.py '+str(acct_id)+" "+str(root_dir))                  
            os.system('python '+rp_dir+'/generate_Test_Files.py '+str(acct_id)+" "+str(root_dir))
            st.success("Test Files Generated")
            st.success("Account ready for Deployment")
        

def select_model(root_dir,acct_id,top10percent,index):
    train_path = root_dir+'/account_'+str(acct_id)+'/train_test_splitted/train_70.csv'
    test_path = root_dir+'/account_'+str(acct_id)+'/train_test_splitted/test_30.csv'
  
    
    predictions_path = root_dir+'account_'+str(acct_id)+'/predictions/predictions.csv'
    data=pd.read_csv(r''+train_path)
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
    X_train = data[features]
    y_train = data['output']
    top5percent = top10percent.head(5)
    #model = top5percent['model_name','model_params']
    model_name = top5percent.iloc[index-1]['model_name']
    if model_name == 'RandomForestClassier':   
        train_model=RandomForestClassifier()
    elif model_name == 'xgboost.XGBClassifier':
        train_model= xgboost.XGBClassifier()
    params=eval(top5percent.iloc[index-1]['model_params'])
    train_model.set_params(**params)
    return train_model.fit(X_train,y_train)
    
      
def user_select(top10percent):
    index = 0
    ch = st.radio('Select the model', ('model 1', 'model 2', 'model 3', 'model 4', 'model 5'))
    st.subheader("Model Parameters :")
    if ch == "model 1":
        index = 1
        st.write(top10percent.iloc[index-1]['model_params'])
    elif ch == "model 2":
        index = 2
        st.write(top10percent.iloc[index - 1]['model_params'])
    elif ch == "model 3":
        index = 3
        st.write(top10percent.iloc[index - 1]['model_params'])
    elif ch == "model 4":
        index = 4
        st.write(top10percent.iloc[index - 1]['model_params'])
    elif ch == "model 5":
        index = 5
        st.write(top10percent.iloc[index - 1]['model_params'])

    if st.checkbox("Select "+ch):
        user_model=select_model(root_dir,acct_id,top10percent,index)
        save_model(predictions_path,test_path,model_path,user_model,summary,dock_path,acct_id,root_dir,rp_dir)
        


        

                


        
def auto_pilot():
    if st.checkbox("Show Warning",value=True):
        st.warning("This Feature is still Under Devolopement \n\n Some Functions may not work properly")

        
    root_dir='/root/caa/rp/model'
    rp_dir = '/root/caascript/rpscript/rpmodelling'
   
    if not os.path.exists(root_dir+'/Data_Extracted'):
        os.makedirs(root_dir+'/Data_Extracted')
    acct_id=st.text_input(label="Enter Account ID",key=1)
    if acct_id != "":
        progress_path = root_dir+"/account_"+str(acct_id)+"/logs/progress.csv"
        model_path = root_dir+'/account_'+str(acct_id)+'/trained_model/model.pkl'
        dock_path= root_dir+'/account_'+str(acct_id)
        summary = dock_path+'/summary.csv'
        data_extracted=root_dir+'/Data_Extracted/'+str(acct_id)+".csv"
        processed_data_path = root_dir+"/account_"+str(acct_id)+"/data_extracted/retraining_data.csv"
        predictions_path = root_dir+'/account_'+str(acct_id)+'/predictions/predictions.csv'
        test_path = root_dir+'/account_'+str(acct_id)+'/train_test_splitted/test_30.csv'
        top10percent=pd.DataFrame()
        progress=pd.DataFrame()
        
        if os.path.exists(progress_path+"/iuehvieuviu"):
            
            
            progress=pd.read_csv(progress_path)
            #st.warning("Last Executed till "+str(progress['Status'][0]))
            if st.checkbox(label="Resume Autopilot"):
                #st.write(progress)
                top10percent = execute(progress['Status'][0],acct_id,root_dir,rp_dir)
                st.table(top10percent[['model_name','percentage','accuracy_score','recall_1','recall_0','recall_diff']].shift()[1:].head(5))
                #st.write(top10percent)
                index=user_select(top10percent)
                user_model = select_model(root_dir,acct_id,top10percent,index)
                st.write("YOU HAVE SELECTED: ",user_model)
                save_model(predictions_path, test_path, model_path, user_model, summary, dock_path,acct_id,root_dir,rp_dir)
        
        else:
            if st.checkbox(label='Start Data Extraction: '):
                DataExt.main()
            else:
                upl_data = st.file_uploader("Upload Raw Data", type=["csv"])
                if upl_data is not None:
                    upl_data.seek(0)

                    data=pd.read_csv(upl_data)
                    data.to_csv(data_extracted,index=False)


        
            if os.path.exists(data_extracted):
                if st.checkbox(label='Start Auto Pilot Mode'):
                    
                    with st.spinner("Execution in Progress"):
                        dir_exe=os.system("python "+rp_dir+"/directory_creation.py "+ str(acct_id) +" "+root_dir)
                        #preprocess.preprocess(data_extracted, processed_data_path)
                        pre_pro=1
                        if (dir_exe == 0):
                            mode='auto'
                            preprocess.main(data,acct_id,root_dir,mode)
                            pre_pro=0
                            if (pre_pro==0):
                                hist_exe=os.system("python "+rp_dir+"/HistoryGeneration.py " +str(acct_id) + " "+root_dir)
                                if hist_exe!=0:
                                    st.error("History Creation failed")
                                else:
                                    top10percent = execute("HistoryGeneration.py",acct_id,root_dir,rp_dir)
                                    st.table(top10percent[['model_name','percentage','accuracy_score','recall_1','recall_0','recall_diff']].shift()[1:].head(5))
                                    if top10percent.shape[1]!=0:
                                        user_select(top10percent)
                                    else:
                                        print("Top 10 evaluation failed")
                                    

                            else:
                                st.error("Preprocessing Failed!")
                        else:
                            st.error("Directory Creation Failed!")

            else:
                st.warning("Extract Data before proceeding")

    else:
        st.warning("Enter account Id to Proceed!")
            


