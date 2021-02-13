import numpy as np
import pandas as pd
import logging
import os
import sys


def preprocess(raw_data,processed_data_path):
    data = pd.read_csv(r'' + raw_data, encoding='cp1256', low_memory=False)
    if data.shape[1]!=23:
            logging.info('Dataset must have exactly 23 columns. Exited.')
            exit(0)
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
    print(data['to_replace'])            
    for i in dates:
        if(data[i].isnull().any()):
            data[i].fillna(data['to_replace'], inplace = True)


    data.drop(columns=['to_replace'],axis=1, inplace=True)

    if (data['effective_date'].isnull().any()):
        logging.info(str(data['effective_date'].isnull().sum()) + ' rows of effective_date are null, replaced it with create_date')
        data['effective_date'] = data['effective_date'].fillna(data['payment_create_date'])

    if (data['payment_create_date'].isnull().any()):
        logging.info(str(data['payment_create_date'].isnull().sum()) + ' rows of effective_date are null, replaced it with create_date')
        data['payment_create_date'] = data['payment_create_date'].fillna(data['effective_date'])

    col=[]
    for i in drop_row:
        if data[i].isnull().any():
            col.append(i)

    data.dropna(axis=0, how="any",subset=col, inplace=True) 
    
    data['invoice_date_norm'] = pd.to_datetime(data['invoice_date_norm']).dt.strftime('%d-%m-%Y')
    data['due_date_norm'] = pd.to_datetime(data['due_date_norm']).dt.strftime('%d-%m-%Y')
    data['document_date_norm'] = pd.to_datetime(data['document_date_norm']).dt.strftime('%d-%m-%Y')
    data['create_date'] = pd.to_datetime(data['create_date']).dt.strftime('%d-%m-%Y')
    data['payment_create_date'] = pd.to_datetime(data['payment_create_date']).dt.strftime('%d-%m-%Y')
    data['effective_date'] = pd.to_datetime(data['effective_date'],errors='coerce').dt.strftime('%d-%m-%Y')

    data.to_csv(processed_data_path, index=False)


if __name__ == '__main__':
    acct_id = str(sys.argv[1])
    root_dir=str(sys.argv[2])
    raw_data = root_dir+'/Data_Extracted/'+str(acct_id)+".csv"
    processed_data_path = root_dir+"/account_"+str(acct_id)+"/data_extracted/retraining_data.csv"

    #This will preprocess your data 
    preprocess(raw_data,processed_data_path)