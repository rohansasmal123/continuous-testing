"""
20/12/2020 11:00:00
@author shivam,subhrojyoti,priyanshu,niharika
"""
import streamlit as st
import pandas as pd
import re
import os
from time import strptime
import numpy as np
import datetime
from datetime import date
from sshtunnel import SSHTunnelForwarder

from rpscript.rpmodelling.Data_csv import get_data
from litmscript.litmanalysis.flow_new import flow_code
from litmscript.litmanalysis.s3_path_extraction_ui import download_s3_data


@st.cache(suppress_st_warning=True)
def get_data_from_gsheet(sheet_name):
    with st.spinner("Fetching Data from DB"):
        monitoring_Sheet=get_data(sheet_name)
    return monitoring_Sheet



def get_Last_Analysis(data,dbdata):
    account_id=data['Account ID']
    #st.write(dbdata[(dbdata['Account ID'] == account_id) & (dbdata['Analysis_status'] == "Yes")].sort_values(by='month',ascending=True))
    last_analyzed_data=dbdata[(dbdata['Account ID'] == account_id) & (dbdata['Analysis_status'] == "Yes")].sort_values(by='month',ascending=True)
    if last_analyzed_data.shape[0]!=0:
        last_analyzed_month_year=last_analyzed_data.iloc[0]['month']+" "+ str(last_analyzed_data.iloc[0]['Year'])
        return last_analyzed_month_year
    else:
        return "Analysis Not Done"
    


def get_is_matched(data,num_corr_from_monitoring,num_corr_checks):
    if(num_corr_checks==num_corr_from_monitoring):
        return "Yes"
    else:
        return "No"

def add_status(data):
    list_status=[]
    for index,row in data.iterrows():
        if int(row['Total_Cheques'])==0:
            list_status.append("White")
            continue

        if int(row['Processed_Cheques'])==0:
            list_status.append("Grey")
            continue

        if float(row['% Cleared (Total Processed )']) >2:
            list_status.append("Green")

        elif float(row['% Cleared (Total Processed )']) >=1 and float(row['% Cleared (Total Processed )']) <=2 :
            list_status.append("Yellow")

        elif float(row['% Cleared (Total Processed )']) <1:
            list_status.append("Red")
    return pd.Series(list_status)

def analysis():
    upl_data = st.file_uploader("Upload Retraining Data", type=["csv"])
    root_dir='/root/caa/litm/analysisdata'
    
    
    #data_extracted=root_dir+'Data_Extracted\\'
    cred_path = '/root/caascript/res/cred.csv'
    if upl_data is not None:
        upl_data.seek(0)
        data = pd.read_csv(upl_data)
        data["Status"]=add_status(data)
        user = st.text_input("Enter Your name")
        if len(user) != 0:
            user_validator = re.compile(r"^[A-Z][a-z]{2,25}$")
            if user_validator.match(user):

                today_date = datetime.datetime.now()
                month =today_date.strftime("%B")
                year = int(today_date.strftime("%Y"))


                assigned_data=data[(data['Year'] == year) & (data['month'] == month)]
                if assigned_data.shape[0]!=0:
                    assigned_data = assigned_data[(assigned_data['Name'] == user) & (assigned_data['Status'] == "Red")][['Account Name', 'Account ID', 'Status','Cleared_Cheques']]
                    if assigned_data.shape[0] != 0:
                        table_data = assigned_data.reset_index(drop=True)
                        table_data.index = np.arange(1, len(table_data) + 1)
                        table_data['Last_Analyis']=table_data.apply(lambda x: get_Last_Analysis(x,data),axis=1)
                        if st.checkbox("Show table data",value=False):
                            st.table(table_data)
                        if st.checkbox("Show Suggestion Analysis",value=False):
                            data_for_s3_path = table_data[table_data['Last_Analyis']=='Analysis Not Done']
                            st.table(data_for_s3_path)
                            st.write()
                            selected = st.radio("Want To Analyse Above Accounts ",("YES","NO"),1)
                            if selected=="NO":
                                selected_account=st.text_input("Enter the account number",value="")
                                if selected_account=="":
                                    selected_account=[]
                                else:
                                    selected_account=selected_account.split(sep=",")

                            else:
                                selected_account=data_for_s3_path['Account ID'].tolist()
                            if st.checkbox("Begin S3 Download"):
                                if not selected_account:
                                    st.warning("Please Enter Account number")
                                else:
                                    #st.write(len(selected_account))
                                    #st.write(selected_account)
                                    gdrive_data=get_data_from_gsheet('LITM_monitoring')
                                    choosen_gdrive_data=gdrive_data[gdrive_data["account_id"].isin(selected_account)]
                                    if choosen_gdrive_data.shape[0]==len(selected_account):
                                        assigned_accounts_group_environment = choosen_gdrive_data.groupby("environment")
                                        cred = pd.read_csv(cred_path)
                                        right,left=st.beta_columns(2)
                                        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                                                  'Nov', 'Dec']
                                        year=[]
                                        for i in range(2018,date.today().year+1):
                                            year.append(str(i))
                                        selected_year = right.selectbox('Choose Year', year)
                                        selected_month=left.selectbox('Choose Month',months)
                                        analysed_month = root_dir+"/"+str(selected_month)+ "_"+str(selected_year)
                                        
                                        month_num=strptime(selected_month,'%b').tm_mon
                                        if month_num<10:
                                            month_num=str(month_num).zfill(2)
                                        else:
                                            month_num=str(month_num)
                                        start_date=selected_year+"/"+month_num+"/"
                                        

                                        if st.button(label="Start Downloading S3 path from DB"):
                                            if not os.path.exists(analysed_month):
                                                os.makedirs(analysed_month)
                                            if not os.path.exists(analysed_month+"/s3_download_files"):
                                                os.makedirs(analysed_month+"/s3_download_files")

                                            for env, group_env in assigned_accounts_group_environment:
                                                server_grouped = group_env.groupby("port")
                                                for server_port_number, group_env in server_grouped:
                                                    print(f" Establishing Connection with Destination Server Port {str(server_port_number)} and Source Port {str(group_env['local_port'].tolist()[0])} in execution")
                                                    server = SSHTunnelForwarder('172.27.128.59',
                                                                                ssh_username=cred.Id.values[0],
                                                                                ssh_password=cred.Ldap_pass.values[0],
                                                                                remote_bind_address=(
                                                                                    'localhost',
                                                                                    int(server_port_number)),
                                                                                local_bind_address=(
                                                                                    '0.0.0.0', int(group_env[
                                                                                                       'local_port'].tolist()[
                                                                                                       0])))

                                                    print(f"Destination Server Port {str(server_port_number)} and Source Port {str(group_env['local_port'].tolist()[0])} in execution")
                                                    server.start()
                                                    

                                                    
                                                    with st.spinner(f"Connection Establised... \n\n Downloading S3 path for account {str(selected_account)}"):
                                                        
                                                        download_s3_data(group_env,analysed_month,start_date,cred)
                                                        server.stop()
                                                    
                                    else:
                                        st.error("Account number match not found")

                                    if st.checkbox(label="Start Downloading Images"):
                                        final_result = pd.DataFrame()
                                        for account in selected_account:
                                            
                                            s3_path=analysed_month+"/s3_download_files/"+str(account)+'_s3.csv'
                                            correctly_closed_path=flow_code(analysed_month,s3_path,account)
                                            correctly_closed_data=pd.read_csv(correctly_closed_path)
                                            if(correctly_closed_data.shape[0]==0):
                                                num_corr_checks=0
                                            else:
                                                num_corr_checks=correctly_closed_data['check_checkNumber'].nunique()
                                            #st.write("Data for s3 path")
                                            #st.write(data_for_s3_path)
                                            #st.write("Numm corr")
                                            #st.write(num_corr_checks)
                                            num_corr_from_monitoring=table_data[table_data['Account ID']==int(account)].Cleared_Cheques.values[0]
                                            table_data['is_matched']=table_data.apply(lambda x: get_is_matched(x,num_corr_from_monitoring,num_corr_checks),axis=1)
                                            final_result = pd.concat([final_result,table_data[table_data['Account ID']==int(account)]],ignore_index=True)

                                        st.table(final_result)





                    else:
                        st.warning(f"{user}, No account Found")

                else:
                    st.warning(f"Monitoring for {month} {str(year)} in not yet done")
    else:
        st.warning("Upload Data")

