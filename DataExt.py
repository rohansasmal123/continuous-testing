'''
!/usr/bin/env python
@author:shivam_gupta,ayanava_dutta,rohan_sasmal
-*-coding:utf-8-*-
'''
import streamlit as st
import pandas as pd
import pymysql as db
import pandas as pd
import logging
import sys
import os
import numpy as np
import sqlalchemy
from threading import Thread
import datetime
import time
from datetime import date,timedelta
from sshtunnel import SSHTunnelForwarder
import sqlalchemy
import math
import os.path, time, stat
from res.bg_css import page_bg
#--------------Res------------------
from rpscript.rpmodelling.Data_csv import get_data 






class Error(Exception):
    """Base class for other exceptions"""
    pass
class NotFound(Error):
    """Raised when account name is not found"""
    pass




def getConnection(schema,port_no,ldap_user,db_pass,host):

    connection =db.connect(host=host,
                              port=port_no,
                              user=ldap_user,
                              passwd=db_pass,
                              db=schema)
    return connection

#format yyyy-mm-dd t
# Function that return list interval of days
def days_interval_list(no_of_threads,date_of_extraction_from,list_of_interval,date_of_extraction_upto):
    date_extraction_from=datetime.datetime.strptime(date_of_extraction_from, "%Y-%m-%d").date()
    if(len(date_of_extraction_upto)==0):
        today = date.today()
    else:
        today=datetime.datetime.strptime(date_of_extraction_upto, "%Y-%m-%d").date()
    difference=(today-date_extraction_from).days
    mod = difference%no_of_threads
    days_interval=int((difference-mod)/no_of_threads)
    for i in range(0,no_of_threads):
        if(i==(no_of_threads-1)):
            list_of_interval[i]=days_interval+mod
        else:
            list_of_interval[i]=days_interval

#Function that return list of dates after interval of days are added
def list_of_dates_to_be_extracted(list_of_dates,list_of_interval,no_of_threads,date_of_extraction_from):
    for i in range(0,no_of_threads):
        if i==0:
            specific_date=datetime.datetime.strptime(date_of_extraction_from, "%Y-%m-%d").date()
            new_date = specific_date + timedelta(list_of_interval[i])
            list_of_dates[i]=new_date
        else:
            new_date = list_of_dates[i-1] + timedelta(list_of_interval[i])
            list_of_dates[i]=new_date

def execute_query(account,index,connection,dataframe_result_list,extract_date_from,list_of_dates):
    if index==0:
        extract_date_from="'"+str(extract_date_from)+"'"
        extract_date_upto="'"+str(list_of_dates[index])+"'"
    else:
        extract_date_from="'"+str(list_of_dates[index-1])+"'"
        extract_date_upto="'"+str(list_of_dates[index])+"'"


    print(f"Extracting old data for account {account} from date {extract_date_from} to {extract_date_upto}")
    query_new = """SELECT acct.acct_doc_header_id,acct.customer_number_norm,acct.due_date_norm,acct.invoice_number_norm,acct.invoice_date_norm,
            acct.invoice_amount_norm,acct.open_amount_norm,acct.debit_credit_indicator, acct.document_date_norm,acct.isOpen, acct.create_date,
            payment.create_date AS payment_create_date,item.fk_inbound_remittance_header_id,header.payment_amount,item.payment_amount,
            payment.payment_amount,item.total_amount,header.payment_hdr_id, payment.payment_method,payment.effective_date,payment.is_deleted,
            header.is_deleted,item.is_deleted FROM `acct_doc_header` AS acct INNER JOIN `caa_remittance_item_tagging` AS tag ON 
            acct.acct_doc_header_id = tag.fk_acct_doc_header_id_global INNER JOIN `caa_inbound_remittance_item` AS item ON 
            tag.pk_remittance_item_tagging_id = item.fk_caa_remittance_item_tagging_id INNER JOIN `caa_inbound_remittance_header` 
            AS header ON header.pk_inbound_remittance_header_id = item.fk_inbound_remittance_header_id INNER JOIN `caa_payment_confirmation_hdr` 
            AS payment ON payment.pk_payment_confirmation_hdr_id = header.payment_hdr_id WHERE acct.account_id = """ + account + """ AND 
            tag.fk_acct_doc_header_id_global = tag.fk_acct_doc_header_id_local AND (acct.create_time >="""+ extract_date_from+"""AND acct.create_time <"""+extract_date_upto+""")"""
    start=time.time()
    res = pd.read_sql_query(query_new, connection[index])
    dataframe_result_list[index]=res
    end=time.time()
    diff=end-start
    print(f"For thred {index+1} time is {diff}")
    #res.to_csv(output_path + str(account) + "_old.csv")
#       print(f"*************Old data saved for {account}*****************")

#function that concat all the dataframes
def save_csv(dataframe_result_list,output_path,account_no):
    data=pd.concat(dataframe_result_list)
    data.to_csv(output_path + str(account_no)+'.csv',index=False)


def extract_data(account_no,output_path,dbn,prt,no_of_threads,date_of_extraction_from,date_of_extraction_upto,ldap_user,db_pass,host):
    threads = [None] * no_of_threads
    connection = [None] * no_of_threads
    list_of_interval=[None]* no_of_threads
    list_of_dates = [None]* no_of_threads
    dataframe_result_list = [None]* no_of_threads
    account_no=str(account_no)
    #getting list of differences in days
    days_interval_list(no_of_threads,date_of_extraction_from,list_of_interval,date_of_extraction_upto)
    #getting list of dates with given interval
    list_of_dates_to_be_extracted(list_of_dates,list_of_interval,no_of_threads,date_of_extraction_from)
    for i in range(no_of_threads):
        try:
            connection[i] = getConnection(dbn, prt,ldap_user,db_pass,host)
            print("connection Successfull")
        except db.OperationalError as e:
            print("\nAn OperationalError occurred. Error number {0}: {1}.".format(e.args[0], e.args[1]))
    try:
        for i in range(no_of_threads):
            threads[i] = Thread(target=execute_query, args=(account_no,i,connection,dataframe_result_list,date_of_extraction_from,list_of_dates))
            threads[i].start()
        for i in range(len(threads)):
            threads[i].join()
        for i in range(len(threads)):
            print(dataframe_result_list[i].shape)
            print("-" * 80)
        ## to merge all the dataframes
        save_csv(dataframe_result_list,output_path,account_no)
        st.success("Data Extraction completed....")
    except:
        print("Error Encountered")
    finally:
        for i in range(len(threads)):
            connection[i].close()
            print("connection Closed")



def countdown(t): 
    
    while t: 
        mins, secs = divmod(t, 60) 
        timer = '{:02d}:{:02d}'.format(mins, secs) 
        time.sleep(1) 
        t -= 1
    st.text("Retrying in: 10s")



def get_creds(account_no):
    root_dir='/root/caa/rp/model'
    data_extracted=root_dir+'/Data_Extracted/'
    cred_path = '/root/caascript/res/cred.csv'

    output_path = data_extracted
    st.header("Data Extraction")  

    st.write("Choose the Range for Data extraction (YYYY/MM/DD)")

    d1,d2=st.beta_columns(2)

    date_of_extraction_from = d1.date_input('Start date', datetime.date(2017, 1, 1))
    date_of_extraction_upto = d2.date_input('End date', datetime.date.today())

    if date_of_extraction_from < date_of_extraction_upto:
        st.success('Start date: `%s`\n\nEnd date:`%s`' % (date_of_extraction_from, date_of_extraction_upto))
            
    else:
        st.error('Error: End date must fall after start date.')

    host = st.text_input(label="Enter the Host Address",value="localhost")
    no_of_threads=st.number_input(label="Enter the Threads",value=3,min_value=1,max_value=4)


    if not os.path.exists(data_extracted):
        os.makedirs(data_extracted)

    if not os.path.exists(cred_path):
        st.subheader("Please enter credentials")
        ldap_user = st.text_input(label="Enter LDAP User name")
        ldap_pass=st.text_input(label="Enter LDAP Password",type="password")
        db_pass = st.text_input(label="Enter the Host Address",type="password")
        
    
    if(len(account_no)!=0):
        if os.path.exists(output_path+'Data_Sheet.csv'):
                last_updated=time.time() - os.stat(output_path+'Data_Sheet.csv')[stat.ST_MTIME]
                st.warning("Last Updated: "+str(time.ctime(os.path.getmtime(output_path+'Data_Sheet.csv'))))
                dataset=pd.read_csv(output_path+'Data_Sheet.csv')
                if st.button(label='Update Data'):
                        if last_updated >300:
                            dataset = pd.DataFrame()
                            with st.spinner('Updating Data'):
                                while dataset.empty:
                                    try:
                                        dataset = get_data("Data_extraction")
                                        #st.write("inside"+str(dataset.empty))
                                        dataset.to_csv(output_path+'Data_Sheet.csv',index=False)
                                        st.success('Successfully Updated Data')
                                        break
                                    except:
                                        st.warning("Connection refused by the server..\n\n Please wait for 10 seconds....")
                                        countdown(int(10)) 
                                        
                                        continue
                        else:
                            st.warning("Retry in: "+str(int(math.ceil(300-last_updated)/60))+" minutes")
        
        else:
            if st.button(label='Fetch Data'):  
                dataset = pd.DataFrame()
                with st.spinner('Fetching Data....'):
                    while dataset.empty:
                        try:
                            dataset = get_data("Data_extraction")
                            #st.write("inside"+str(dataset.empty))
                            dataset.to_csv(output_path+'Data_Sheet.csv',index=False)
                            st.success('Successfully Updated Data')
                            break
                        except:
                            st.warning("Connection refused by the server..\n\n Please wait for 10 seconds....")
                            countdown(int(10))
                            continue   

        try:
            acc_data_bool = dataset['account_no'] == int(account_no)
            if acc_data_bool.sum() != 0:
                selected_account_dataframe = dataset[acc_data_bool]
                schema_name = dataset.loc[acc_data_bool, 'schema_name'].values[0]
                local_port = dataset.loc[acc_data_bool, 'local_port'].values[0]
                env = dataset.loc[acc_data_bool, 'environment'].values[0]

                if os.path.exists(cred_path):
                    cred = pd.read_csv(cred_path)
                    if env=='AWS US':
                        ldap_user=cred.Id.values[0]
                        ldap_pass=cred.Ldap_pass.values[0]
                        db_pass=cred.Db_pass.values[0]
                    elif env=='AWS EU':
                        ldap_user=cred.Id.values[1]
                        ldap_pass=cred.Ldap_pass.values[1]
                        db_pass=cred.Db_pass.values[1]
                    elif env=='AWS GOLD':
                        ldap_user=cred.Id.values[2]
                        ldap_pass=cred.Ldap_pass.values[2]
                        db_pass=cred.Db_pass.values[2]
                    else:
                        st.warning(f"No Credentials Found For {str(env)}")
                        if st.checkbox(f"Enter Credentials For {str(schema_name)} -{str(env)}"):
                            ldap_id=st.text_input(label='Enter LDAP-ID',key=1)
                            ldap_pass=st.text_input(label="Enter LDAP password",type="password",key=1)
                            db_pass=st.text_input(label="Enter DB password",type="password",key=1)
                else:
                    st.warning("Credentials not Found! \n\n Please Save your Credentials via Login Page")

            else:
                raise NotFound

        except NotFound:
            st.error("Account Not found")
        flag=0
        if st.button('Extract Data'):
            server_port = []
            localhost_port = []
            for index, row in selected_account_dataframe.iterrows():
                server_port.append(row['port'])
                localhost_port.append(row['local_port'])
            server_localhost = {"Server Port": server_port,"LocalHost Port": localhost_port}
            server_localhost_df = pd.DataFrame(server_localhost)
            server_localhost_df.drop_duplicates(keep="first", inplace=True)
            server_localhost_df = server_localhost_df.astype({"Server Port": int, "LocalHost Port": int})

            
            for index, row in server_localhost_df.iterrows():
                server = SSHTunnelForwarder('172.27.128.59', ssh_username=ldap_user, ssh_password=ldap_pass,
                                            remote_bind_address=('127.0.0.1', int(row["Server Port"])),
                                            local_bind_address=('0.0.0.0', int(row["LocalHost Port"])))

            print(f"Destination Server Port {row['Server Port']} and Source Port {row['LocalHost Port']} in execution")

            try:
                server.start()
                st.success("Connection Successful")
                with st.spinner('Fetching Data'):
                    #with st.image('unnamed.gif', caption='Fetching Data'):
                    extract_data(account_no, output_path, schema_name, int(local_port), no_of_threads,
                            str(date_of_extraction_from), str(date_of_extraction_upto), ldap_user, db_pass, host)
                    
            except:
                st.error(f"Could Not connect to {str(env)}\n\n Check logs for more details")
            
            finally:
                server.stop()
                st.warning("Connection Closed")
    else:
        st.warning("Enter Account ID and \n\n Press Enter to Continue...")
        

def main():
    img_path='/root/caascript/res/bg/'
    page_bg(img_path+'dataext_pic.png')
    account_no=st.text_input(label="Enter Account ID")

    if account_no!="":
        get_creds(account_no) 
        
    else:("Enter account ID to proceed")   