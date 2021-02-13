import numpy as np
import pandas as pd
import logging
import os
import sys
import streamlit as st

def preprocess(raw_data,processed_data_path,mode):
    data = raw_data.copy()
    feature_df=pd.read_csv('/root/caascript/res/feature_df.csv')
    
    if data.shape[1]!=23:
        features=[]
        curr_features=[]
        diff=[]
        features = feature_df['Feature_Name'].tolist()
        curr_features=data.columns.tolist()
        diff = [x for x in features if x not in curr_features]

        if mode!="auto":
            st.warning("Count of features not matched \n\n Please check console for more information")
            st.error("Preprocessing Failed!")
            if st.checkbox("Show Difference"):
                st.write("Features :\n"+str(diff)+"\n absent from the data")
        else:
            print("\nFeatures not found :")
            print(diff)
            print('Dataset must have exactly 23 columns. Exited.\n\n ')
            exit(0)

        
    else:
        data.replace('\\N', np.nan, inplace=True)
        data.replace(" ", np.nan, inplace=True)
        data.replace('', np.nan, inplace=True)
        data.replace('NULL', np.nan, inplace=True)
        data.replace('null', np.nan, inplace=True)
        mode_impute = ['debit_credit_indicator','payment_method']

        drop_row = ['acct_doc_header_id','customer_number_norm','invoice_number_norm', 'invoice_amount_norm','open_amount_norm','isOpen', 'fk_inbound_remittance_header_id', 'payment_amount','payment_amount.1',
                    'payment_amount.2', 'total_amount', 'payment_hdr_id','is_deleted', 'is_deleted.1', 'is_deleted.2']
        
        for i in mode_impute:
            if(data[i].isnull().all()):
                data[i] = 'C'
            elif (data[i].isnull().any()):
                data[i].fillna(data[i].value_counts().index[0],inplace=True)
        
        
        dates = ['invoice_date_norm','document_date_norm','create_date']
        s=[]
        for i in dates:
            s.append(data[i].isnull().sum())
        d = s.index(min(s))
        data.dropna(how='any',subset=[dates[d]], inplace=True)


        for i in dates:
            if(data[i].notnull().all()):
                data['to_replace'] = data[i]
                break
               
        for i in dates:
            if(data[i].isnull().any()):
                data[i].fillna(data['to_replace'], inplace = True)


        data.drop(columns=['to_replace'],axis=1, inplace=True)

        if (data['effective_date'].isnull().any()):
            print(str(data['effective_date'].isnull().sum()) + ' rows of effective_date are null, replaced it with create_date')
            data['effective_date'] = data['effective_date'].fillna(data['payment_create_date'])

        if (data['payment_create_date'].isnull().any()):
            print(str(data['payment_create_date'].isnull().sum()) + ' rows of effective_date are null, replaced it with create_date')
            data['payment_create_date'] = data['payment_create_date'].fillna(data['effective_date'])

        col=[]
        for i in drop_row:
            if data[i].isnull().any():
                col.append(i)

        data.dropna(axis=0, how="any",subset=col, inplace=True) 
        try:
            data['invoice_date_norm'] = pd.to_datetime(data['invoice_date_norm']).dt.strftime('%d-%m-%Y')
            data['due_date_norm'] = pd.to_datetime(data['due_date_norm']).dt.strftime('%d-%m-%Y')
            data['document_date_norm'] = pd.to_datetime(data['document_date_norm']).dt.strftime('%d-%m-%Y')
            data['create_date'] = pd.to_datetime(data['create_date']).dt.strftime('%d-%m-%Y')
            data['payment_create_date'] = pd.to_datetime(data['payment_create_date']).dt.strftime('%d-%m-%Y')
            data['effective_date'] = pd.to_datetime(data['effective_date'],errors='coerce').dt.strftime('%d-%m-%Y')
            
            
            if mode!="auto":
                data.to_csv(processed_data_path, index=False)
                st.success("Preprocessing Completed Successfully!")
               
            else:
                data.to_csv(processed_data_path, index=False)
                print("Preprocessing Completed Successfully!")

        except:
            st.error("Preprocessing Failed \n\n Please check console for more information")
            print("Error occured while converting to datetime format")
            
            



        

def main(raw_data,acct_id,root_dir,mode):
    processed_data_path = root_dir+"/account_"+str(acct_id)+"/data_extracted/retraining_data.csv"
    
    if mode!='auto':

        with st.spinner("Preprocessing your data.."):
            raw_data.seek(0)
            raw_data=pd.read_csv(raw_data)
            preprocess(raw_data,processed_data_path,mode)
    else:
        raw_data = root_dir+'/Data_Extracted/'+str(acct_id)+".csv"
        if os.path.exists(raw_data):
            raw_data = pd.read_csv(r'' + raw_data, encoding='cp1256', low_memory=False)
            preprocess(raw_data,processed_data_path,mode)
        else:
            st.error("File not found!!")
            exit(0)
        
