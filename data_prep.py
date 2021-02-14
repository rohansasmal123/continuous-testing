'''
!/usr/bin/env python
-*-coding:utf-8-*-
@author:ayanava_dutta,rohan_sasmal,shivam_gupta
'''
import streamlit as st
import pandas as pd
import os
import sys
import webbrowser
import numpy as np

import rpscript.rpmodelling.preprocess as preprocess

def data_prep():
    img_path='/root/caascript/res/bg/'
    

    acc_id=st.text_input(label='Enter Account ID')
    root_dir = '/root/caa/rp/model'
    rp_dir = '/root/caascript/rpscript/rpmodelling'
    path=str(root_dir)+'/account_'+str(acc_id)+'/logs/'
    remote_path=str(root_dir)+'Remote_Execution/account_'+str(acc_id)
    dock_path=str(root_dir)+'/account_'+str(acc_id)
    #data_extracted=root_dir+'/Data_Extracted/'
    log_path=path

    
    
    if os.path.exists(log_path+"progress.csv"):
    
        df=pd.read_csv(log_path+"progress.csv")
        stat=df['Status'][0]
        st.write("Account "+str(acc_id)+" has been executed till "+stat)
    
    
    cred_r={'ip':None,
            'uid':None,
            'pwd':None,
            'acc_dir':None,
            'code_dir':None
            }
    data=pd.DataFrame()
    if acc_id!="":
        master=0
        dat_type = st.radio("Select the type of Data",('Pre-Processed Data','Raw Data'))
        if dat_type=='Pre-Processed Data':
            mode="preprocessed"
            prepro_data = st.file_uploader("Upload Preprocessed Data", type=["csv"])
            if prepro_data is not None:
                prepro_data.seek(0)
                data=pd.read_csv(prepro_data)
        else:
            mode="dataprep"
            raw_data = st.file_uploader("Upload Raw Data", type=["csv"])
            if raw_data is not None:
                raw_data.seek(0)
                data=pd.read_csv(raw_data)
                

        if master==0:
            c = st.radio("Select Execution Method",('Local','Remote'))
            if c=='Remote':
                if os.path.exists(root_dir+'/vmcred.csv'):
                    cred_df=pd.read_csv(root_dir+'/vmcred.csv')
                else:
                    cred_df=pd.DataFrame(cred_r,index=[0])
                    cred_df.to_csv(root_dir+'/vmcred.csv',index=False)

                cred_df=pd.read_csv(root_dir+'/vmcred.csv')
                cred_df.replace(np.nan,'', inplace=True)
                
                #st.write(cred_df)
                ip=st.text_input(label='Enter IP address',value=cred_df['ip'][0])
                cred_df['ip'][0]=ip
                uid=st.text_input(label='Enter User ID',value=cred_df['uid'][0])
                cred_df['uid'][0]=uid
                pwd=st.text_input(label="Enter password",type="password",value=cred_df['pwd'][0])
                cred_df['pwd'][0]=pwd
                acc_dir=st.text_input(label='Enter Remote Directory',value=cred_df['acc_dir'][0])
                cred_df['acc_dir'][0]=acc_dir
                code_dir=st.text_input(label='Enter Code Directory',value=cred_df['code_dir'][0])
                cred_df['code_dir'][0]=code_dir
                cred_df.to_csv(root_dir+'/vmcred.csv',index=False)
                

                if not os.path.exists(remote_path):
                        os.makedirs(remote_path)
                data.to_csv(remote_path+'/retraining_data.csv',index=False)

                if st.button(label='Test Connection'):
                    d=os.system("sshpass -p "+str(pwd)+" ssh -o StrictHostKeyChecking=no -l "+str(uid)+" "+str(ip)+" exit")
                    if (d==0):
                        st.success("Connection Successful!")
                    else:
                        st.error("Connection Failed")
                        
                
                if st.checkbox(label='Start Modelling'):

                    menu_bar1 = st.selectbox(label='What do you want to do?', options=['Directory Creation','History Generation',
                                             'Json Creation','Subset Creation','Features Creation','Train-Test-Split'])
                    #Directory Creation
                    if menu_bar1 =='Directory Creation':
                        if st.button(label='Start Directory Creation'):
                            if len(acc_id)!=0:
                                with st.spinner("Execution in progress..."):
                                    d=(os.system('sshpass -p '+pwd+' ssh -o StrictHostKeyChecking=no ' +' -l '+uid+' ' +ip+' python '+code_dir+'/directory_creation.py'+' '+ acc_id+ ' ' + code_dir))
                                    st.success("Directory Created")
                                os.system('sshpass -p '+pwd + ' scp -o StrictHostKeyChecking=no '+remote_path+'/retraining_data.csv '+  uid+'@'+ip+':'+acc_dir+'/account_'+str(acc_id)+'/data_extracted/')
                                if not os.path.exists(path):
                                    os.makedirs(path)
                            else:
                                st.error('Account Id cannot be empty Directory Creation Failed!')
                    
                    #History Generation
                    elif menu_bar1 =='History Generation':
                        if st.button(label='Start History Generation'):
                            with st.spinner("Execution in progress..."):
                                d=(os.system('sshpass -p '+pwd+' ssh -o StrictHostKeyChecking=no ' +' -l '+uid+' ' +ip+' python3 '+code_dir+'/HistoryGeneration.py'+' '+ acc_id+ ' ' + code_dir))

                            if (d == 0):
                                st.success("History generated")   
                            else:
                                st.error("History Generation Failed!")
                                
                    #Json Creation
                    elif menu_bar1 =='Json Creation':
                        if st.button(label='Start Json Creation'):
                            with st.spinner("Execution in progress..."):
                                d=(os.system('sshpass -p '+pwd+' ssh -o StrictHostKeyChecking=no ' +' -l '+uid+' ' +ip+' python3 '+code_dir+'/JsonCreation.py'+' '+ acc_id+ ' ' + code_dir)) 
                            if (d == 0):
                                st.success("Json File generated")       
                            else:
                                st.error("Json Creation Failed!")
                    
                    #Subset Creation    
                    elif menu_bar1 =='Subset Creation':
                        if st.button(label='Start Subset Creation'):
                            with st.spinner("Execution in progress..."):
                                d=(os.system('sshpass -p '+pwd+' ssh -o StrictHostKeyChecking=no ' +' -l '+uid+' ' +ip+' python3 '+code_dir+'/SubsetsCreation.py'+' '+ acc_id+ ' ' + code_dir))
                                 
                            if (d == 0):
                                st.success("Subset Creation Done") 
                                 
                            else:
                                st.error("Subset Creation Failed!")
                                
                                    
                    #Features Creation
                    elif menu_bar1 =='Features Creation':
                        if st.button(label='Start Feature Creation'):
                            with st.spinner("Execution in progress..."):
                                d=(os.system('sshpass -p '+pwd+' ssh -o StrictHostKeyChecking=no ' +' -l '+uid+' ' +ip+' python3 '+code_dir+'/FeaturesCreation.py'+' '+ acc_id+ ' ' + code_dir))             
                            if (d == 0):
                                st.success("Feature Creation Done")   
                            else:
                                st.error("Feature Creation Failed!")
                                
                    
                    #Train-Test-Split
                    elif menu_bar1 =='Train-Test-Split':
                        values = st.slider('Select Threshold',0.1, 0.9, 0.7, 0.01)
                        if st.button(label='Start Train-Test-Split'):
                            with st.spinner("Execution in progress..."):
                                d=(os.system('sshpass -p '+pwd+' ssh -o StrictHostKeyChecking=no ' +' -l '+uid+' ' +ip+' python3 '+code_dir+'/TrainTestSplit.py'+' '+ acc_id+ ' ' + code_dir +' '+ str(values) ))      
                            if (d == 0):
                                st.success("Train-Test Split Operation Done")
                                os.system('sshpass -p '+pwd + ' scp -o StrictHostKeyChecking=no ' +  uid+'@'+ip+':'+acc_dir+'/account_'+str(acc_id)+'/summary.csv'+ ' '+remote_path)
                                
                            else:
                                st.error("Train-Test Split Operation Failed!")

                    
                        if st.checkbox("Show Stats", value=False):
                            summary=pd.read_csv(remote_path+'/summary.csv')
                            st.write(summary)
                            
                            if st.button(label='Zip & Download'):
                                d=(os.system('sshpass -p '+pwd+' ssh -o StrictHostKeyChecking=no ' +' -l '+uid+' ' +ip+' python3 '+code_dir+'/compress.py '+acc_id+" "+acc_dir))
                                
                                if d==0:
                                    os.system('sshpass -p '+pwd + ' scp -o StrictHostKeyChecking=no ' +  uid+'@'+ip+':'+acc_dir+'/account_'+str(acc_id)+'.zip'+ ' '+remote_path)
                                    st.success("Files Downloaded")
                                else:
                                    st.error("Some error")
                                
                            
                            
                        
                    
                
                    
 #-------------------------------------------------------------------------------local-------------------------------------------------------------------------------------------------------------------                   
                
            else:
                c2 = st.radio("How do you want the tasks to be executed ",('Individual','Synchronous'))
                if c2== 'Individual':
                    menu_bar = st.selectbox(label='What do you want to do?', options=['Directory Creation','History Generation','Json Creation','Subset Creation','Features Creation','Train-Test-Split'])
                    

                    #Directory Creation
                    if menu_bar =='Directory Creation':
                        if st.button(label='Start Directory Creation'):
                            if mode!="dataprep":
                                with st.spinner("Execution in progress..."):
                                    d=(os.system('python '+ rp_dir+'/directory_creation.py '+str(acc_id) + ' ' + root_dir))
                                if d==0:
                                    st.success("Directory Created")
                                    data.to_csv(dock_path+'/data_extracted/retraining_data.csv',index=False)
                                    if not os.path.exists(path):
                                        os.makedirs(path)
                                else:
                                    st.error("Directory creation Failed")
                            else:
                                with st.spinner("Execution in progress..."):
                                    d=(os.system('python '+ rp_dir+'/directory_creation.py '+str(acc_id) + ' ' + root_dir))
                                    if d==0:
                                        preprocess.main(raw_data,acc_id,root_dir,mode)
                                        if not os.path.exists(path):
                                            os.makedirs(path)
                                    else:
                                        st.error("Operation Failed!")
                    
                    
                    #History Generation
                    elif menu_bar =='History Generation':
                        
                        if st.button(label='Start History Generation'):
                            with st.spinner("Execution in progress..."):
                                d=(os.system('python '+ rp_dir+'/HistoryGeneration.py '+str(acc_id)+ ' ' +root_dir))
                                
                                
                            if (d == 0):
                                st.success("History generated") 
                                
                                
                            else:
                                st.error("History Generation Failed!")
    
                                
                    
                
                #Json Creation
                    elif menu_bar =='Json Creation':
                        if st.button(label='Start Json Creation'):
                            with st.spinner("Execution in progress..."):
                                d=(os.system('python '+ rp_dir+'/JsonCreation.py '+str(acc_id)+ ' ' + root_dir))
                                
                                
                            if (d == 0):
                                st.success("Json File generated") 
                                
                                
                            else:
                                st.error("Json Creation Failed!")
                                
                            
                            
                        
                    
                    #Subset Creation    
                    elif menu_bar =='Subset Creation':
                        if st.button(label='Start Subset Creation'):
                            with st.spinner("Execution in progress..."):
                                d=(os.system('python '+ rp_dir+'/SubsetsCreation.py '+str(acc_id)+ ' ' +root_dir))
                                
                            if (d == 0):
                                st.success("Subset Creation Done")
                                
                                
                                
                            else:
                                st.error("Subset Creation Failed!")
                            
                                
                        
                            
                    #Features Creation
                    elif menu_bar =='Features Creation':
                        if st.button(label='Start Feature Creation'):
                            with st.spinner("Execution in progress..."):
                                d=(os.system('python '+ rp_dir+'/FeaturesCreation.py '+str(acc_id)+ ' ' +root_dir))
                                
                            if (d == 0):
                                st.success("Feature Creation Done")
                               
                                
                            else:
                                st.error("Feature Creation Failed!")
                                
                                
                        
                            
                    
                    #Train-Test-Split
                    elif menu_bar =='Train-Test-Split':
                        values = st.slider('Select Threshold',0.1, 0.9, 0.7, 0.01)
                        if st.button(label='Start Train-Test-Split'):
                            with st.spinner("Execution in progress..."):
                                d=(os.system('python '+ rp_dir+'/TrainTestSplit.py '+str(acc_id)+' ' + root_dir + ' '  +str(values)))
                                
                            if (d == 0):
                                st.success("Train-Test Split Operation Done")
                                
                                
                            else:
                                st.error("Train-Test Split Operation Failed!")
                                
                                
                        
                            
                        if st.checkbox("Show Stats", value=False):
                            summary=pd.read_csv(dock_path+'/summary.csv')
                            st.write(summary)
                

#--------------------------------------------------------------------FLOW.py-----------------------------------------------------------------------------------------
                else:
                    tasks=['HistoryGeneration.py',
                            'JsonCreation.py',
                            'SubsetsCreation.py',
                            'FeaturesCreation.py',
                          ]
 
                    if st.button(label='Start Execution'):
                        with st.spinner("Execution in Progress...."):
                            d=(os.system('python '+ rp_dir+'/directory_creation.py '+str(acc_id)+ ' ' + root_dir))
                            if d==0 and mode=='preprocessed':
                                data.to_csv(dock_path+'/data_extracted/retraining_data.csv',index=False)
                                for task in tasks:
                                    d=(os.system('python '+ rp_dir+'/'+str(task)+' '+str(acc_id) + ' ' + root_dir))
                                    if d!=0:
                                        st.error(task.split('.')[0] + " Failed")
                                        break  

                            elif d==0 and mode=='dataprep':
                                preprocess.main(raw_data,acc_id,root_dir,mode)
                                for task in tasks:
                                    d=(os.system('python '+ rp_dir+'/'+str(task)+' '+str(acc_id) + ' ' + root_dir))
                                    if d!=0:
                                        st.error(task.split('.')[0] + " Failed")
                                        break  
                            else:
                                st.error("Directory Creation Failed!!")
                        st.success("Flow Operation executed Successfully!!")

                    if st.checkbox(label='Train-Test-Split Operation',value=False):
                            values = st.slider('Select Threshold For the Split:',0.1, 0.9, 0.7, 0.01)
                            if st.button("Start Train Test Split Operation"):
                                with st.spinner("Execution in Progress..."):
                                    tts=(os.system('python '+ rp_dir+'/TrainTestSplit.py '+str(acc_id)+' ' + root_dir +' '+str(values)))
                                    if tts==0:
                                        st.success("Train-Test-Split Operation executed Successfully!!")
                                    else:
                                        st.error("Train-Test-Split Operation Failed")

                            if st.checkbox("Show Stats", value=False):
                                summary=pd.read_csv(dock_path+'/summary.csv')
                                st.write(summary)

#--------------------------------------------------------------------------------------------------------------------------------------------------                
        
        else:
            st.warning("Please Upload Data to Proceed")
    else:
        st.text("Press Enter To Continue")
        
