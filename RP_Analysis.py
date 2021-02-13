'''
!/usr/bin/env python
@author:nistha_jaiswal,archita_ganguly,ayanava_dutta,rohan_sasmal
-*-coding:utf-8-*-
'''
import streamlit as st
import numpy as np
import pandas as pd
import json
import datetime
import os
from sshtunnel import SSHTunnelForwarder
from rpscript.rpanalysis.rp_analysis_functions import putty_conn,fill_q2,pred_pres_or_not,get_final_input,missed_hdr_ids,match_features
from rpscript.rpanalysis.rp_analysis_functions import filter_data,not_included_pay_hdr_ids,check_amt,get_invoices,match_invoices,our_subset
server=[]


def main():
    root_dir ='/root/caa/rp/monitor/'
    cred_path='/root/caascript/res/cred.csv'
    current_date=datetime.date.today()
    month=current_date.strftime("%b")
    year=current_date.strftime("%Y")
    path_month_year=str(month)+"_"+str(year)
    query_output_path = root_dir+path_month_year+'/Output_Files/DB_Query_files/'
    final_report = root_dir+path_month_year+'/Output_Files/Script_Final_files/Sept_CashApp_Monitoring_report_2021-01-11.csv'
    account_port = '/root/caascript/res/port.csv'

    user_name=st.text_input("Enter your Name").title()
    if not os.path.exists(cred_path):
        st.warning("Please enter credentials via Login page before proceeding")
        user_name=""
    flag1=0
    if user_name!="":
        cred = pd.read_csv(cred_path)
        #upl_data = st.file_uploader("Upload  Data")
        port_dat=[]
        if os.path.exists(account_port):
            port_dat=pd.read_csv(account_port)
        else:
            st.warning("Please perform Monitoring for your Accounts before proceeding")

        if port_dat is not None:
            with st.spinner("Searching Accounts for " +str(user_name)):
                upl_data = st.file_uploader("Upload  Data", type=["csv"])
                if upl_data is not None:
                    upl_data.seek(0)
                    data= pd.read_csv(upl_data)
                data['Intern Name'] = data['Intern Name'].str.title()
                data = data[data['Intern Name'] == user_name]
                data['Account Id'] = data['Account Id'].astype('int')
                data.reset_index(drop=True, inplace=True)
                st.write("All assigned accounts:")
                st.write(data)
            if st.button("Connect to PUTTY"):
                for_accounts,server=putty_conn(user_name, port_dat,cred)
                with st.spinner("Checking Q1, Q2..."):
                    data,flag2 = fill_q2(data, query_output_path)
                    pred_pres_or_not(data, for_accounts,cred)
                    st.success("Final csv generated")

            if st.checkbox("Show Accounts in yellow or red:", value=False):
                data = filter_data(data)

                data=data[['Account Id','Account Name','Schema', 'Intern Name','Top 3 %','status']]
                st.write("Hey " +str(user_name)+"! you have these accounts for analysis:")
                st.table(data)

                accid = st.radio('Select account',(data['Account Id']))
                st.write("Working on:",accid)

                schema = data[data['Account Id'] == accid]['Schema'].reset_index(drop=True)[0]

                outcome, our_subsets = missed_hdr_ids(schema,query_output_path)
                pay_hdr_id = not_included_pay_hdr_ids(accid, outcome)

                match = check_amt(pay_hdr_id, schema, query_output_path)
                if len(match)!=0:
                    pay_id = st.radio('Select payment_hdr_id', (match))
                    with st.spinner("Fetching final input..."):
                        final_input=get_final_input(schema,accid,pay_id,port_dat,cred)
                        final_input=json.loads(final_input['final_input'][0])
                    #st.write(json.loads(final_input['final_input'][0]))

                    invoices, selected_invoices = get_invoices(pay_hdr_id,match, final_input, schema, query_output_path)

                    flag,analyst_subset_id = match_invoices(pay_id, invoices, selected_invoices)
                    if flag==1:
                        st.write("Analyst subset_id : ",analyst_subset_id)

                        our_subset_id = our_subset(pay_id, our_subsets)
                        subset_id = st.radio(
                            'Select subset_id',
                            (our_subset_id))

                        match_features(final_input, subset_id, analyst_subset_id)
                        st.success("Check subset features")
                    else:
                        st.write(analyst_subset_id)
                        st.success("ANALYSIS DONE")
            #for i in server:
            #    i.stop()
        else:
            st.error("No data found for Analysis")
            
    
    else:
        st.warning("Enter your name to begin")

