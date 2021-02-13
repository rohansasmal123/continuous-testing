'''
!/usr/bin/env python
-*-coding:utf-8-*-
@author:ayanava_dutta,rohan_sasmal,shivam_gupta
'''
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from bg_css import page_bg

st.set_page_config(page_title="HighRadius™ |CASH APPLICATION CLOUD", page_icon='bg\logo.png')


def all_screen(choice):
    
    if choice:
        login()

def login():
    page_bg('bg\dataprep_pic.png')
    server_list=['AWS-US','AWS-EU','AWS-Gold']

    cred={'server':server_list,
          'Id': np.nan,
          'Ldap_pass': np.nan,
          'Db_pass': np.nan,
          }
    
    cred_df=pd.DataFrame.from_dict(cred)
    
    if os.path.exists("cred.csv"):
        cred_df=pd.read_csv("cred.csv")

    c1,c2=st.beta_columns([2,1])
    c1.header("Please Enter Your Credentials")
    server=c2.selectbox(label='Select Server', options=server_list)
    
    if server=='AWS-US':
        us_ldap_id=st.text_input(label='Enter LDAP-ID',key=1)
        us_ldap_pass=st.text_input(label="Enter LDAP password",type="password",key=1)
        us_db_pass=st.text_input(label="Enter DB password",type="password",key=1)

        if us_ldap_id!=None and us_ldap_pass!=None and us_db_pass!=None:
            cred_df.Id[0]=us_ldap_id
            cred_df.Ldap_pass[0]=us_ldap_pass
            cred_df.Db_pass[0]=us_db_pass

            if st.button(label='Save AWS-US Credentials'):
                cred_df.to_csv("cred.csv",index=False)

    elif server=='AWS-EU':
        eu_ldap_id=st.text_input(label='Enter LDAP-ID',key=2)
        eu_ldap_pass=st.text_input(label="Enter LDAP password",type="password",key=2)
        eu_db_pass=st.text_input(label="Enter DB password",type="password",key=2)

        if eu_ldap_id!=None and eu_ldap_pass!=None and eu_db_pass!=None:
            cred_df.Id[1]=eu_ldap_id
            cred_df.Ldap_pass[1]=eu_ldap_pass
            cred_df.Db_pass[1]=eu_db_pass

            if st.button(label='Save AWS-EU Credentials'):
                cred_df.to_csv("cred.csv",index=False)
        


    else:
        gold_ldap_id=st.text_input(label='Enter LDAP-ID',key=3)
        gold_ldap_pass=st.text_input(label="Enter LDAP password",type="password",key=3)
        gold_db_pass=st.text_input(label="Enter DB password",type="password",key=3)


        if gold_ldap_id!=None and gold_ldap_pass!=None and gold_db_pass!=None:
            cred_df.Id[2]=gold_ldap_id
            cred_df.Ldap_pass[2]=gold_ldap_pass
            cred_df.Db_pass[2]=gold_db_pass

            if st.button(label='Save AWS-GOLD Credentials'):
                cred_df.to_csv("cred.csv",index=False)


   
    if os.path.exists("cred.csv"):
        
        cred_df=pd.read_csv("cred.csv")
        cred_df=cred_df[cred_df['Ldap_pass'].notnull()]
        user=cred_df.Id[0]
        server_list =[server for server in cred_df['server']]

        st.success("Welcome "+((user.split('.')[0]).title()+" Your credentials has been saved \n\n You can Start Working on "+ str(" , ".join(server_list)) + " Accounts."))
        
        
        



if __name__ == '__main__':
    choice = st.sidebar.checkbox('Login',value=True)
    all_screen(choice)