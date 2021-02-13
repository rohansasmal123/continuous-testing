import streamlit as st
import re
import datetime
import pandas as pd
from sshtunnel import SSHTunnelForwarder
import time
from rpscript.rpmonitor.Data_csv import get_data
#from putty import ssh_auto
from rpscript.rpmonitor.RP_Monitoring_Automated_modified import RP_monitoring_script
from rpscript.rpmonitor.RP_Monitoring_Automated_modified import rp_monitoring_merging
#from ssh_forwarder import Putty
import os

@st.cache(suppress_st_warning=True)
def get_data_from_gsheet(sheet_name):
    with st.spinner("Fetching Data from DB"):
        monitoring_Sheet=get_data(sheet_name)
    return monitoring_Sheet


def main():
    root_dir='/root/caa/rp/monitor/'
    #data_extracted=root_dir+'Data_Extracted\\'
    cred_path = '/root/caascript/res/cred.csv'
    user=st.text_input("Enter Your name")

    if not os.path.exists(cred_path):
        st.warning("No credentials Found \n\nPlease Save credentials via Login Page")
        user=""

    
    if user!="":
        monitoring_Sheet = get_data_from_gsheet("RP_monitoring")
        if monitoring_Sheet.shape[0]==0:
            st.error("Unable To Fetch Data...")
        else:
            assigned_accounts=monitoring_Sheet.loc[monitoring_Sheet['intern_name'] == user.title()]
            if assigned_accounts.shape[0] == 0:
                st.warning("Sorry "+user+" no account has been assigned to you !!")
            else:
                st.write(assigned_accounts)
                left,right = st.beta_columns(2)
                start_date = left.date_input("Enter Start Date", datetime.date.today())
                end_date = right.date_input("Enter End Date", datetime.date.today())
                assigned_accounts_group_environment=assigned_accounts.groupby("environment")
                current_date=datetime.date.today()
                month=current_date.strftime("%b")
                year=current_date.strftime("%Y")
                path_month_year=str(month)+"_"+str(year)
                working_dir=os.path.join(root_dir,path_month_year)
                if(not os.path.exists(working_dir)):
                    os.makedirs(working_dir)
                    os.makedirs(os.path.join(working_dir,'Output_Files','DB_Query_files'))
                    os.makedirs(os.path.join(working_dir, 'Output_Files', 'Script_Final_files'))
                    st.success("Directory Created Successfully")

                
                if st.checkbox(label="Start Monitoring"):
                    with st.spinner("Execution in progress...."):
                        for env, group in assigned_accounts_group_environment:
                            cred = pd.read_csv(cred_path)
                            if env=='AWS US':
                                ldap_user=cred.Id.values[0]
                                ldap_pass=cred.Ldap_pass.values[0]
                                db_pass=cred.Db_pass.values[0]
                                host = "localhost"
                            elif env=='AWS EU':
                                ldap_user=cred.Id.values[1]
                                ldap_pass=cred.Ldap_pass.values[1]
                                db_pass=cred.Db_pass.values[1]
                                host = "localhost"
                            elif env=='AWS GOLD':
                                ldap_user=cred.Id.values[2]
                                ldap_pass=cred.Ldap_pass.values[2]
                                db_pass=cred.Db_pass.values[2]
                                host = "localhost"


                            server_port = []
                            localhost_port = []
                            server_port_group=group.groupby('port')
                            #print(cred)
                            for server_port, server_group_data in server_port_group:

                                local_port=int(server_group_data['local_port'].values[0])
                                server = SSHTunnelForwarder('172.27.128.59', ssh_username=ldap_user,
                                                            ssh_password=ldap_pass,
                                                            remote_bind_address=('localhost', server_port),
                                                            local_bind_address=('0.0.0.0', local_port))

                                print(
                                    f"Destination Server Port {str(server_port)} and Source Port {str(local_port)} in execution")
                                print(
                                    f" Establishing Connection with Destination Server Port {str(server_port)} and Source Port {str(local_port)}  in execution")
                                try:
                                    server.start()
                                    st.success("Connection Successfull")
                                    RP_monitoring_script(ldap_user, db_pass, host, start_date, end_date,server_group_data,working_dir)
                                except:
                                    st.error(f"Couldn't connect to {str(env)} \n\n Check Logs for more details")

                                finally:
                                    server.stop()
                                    st.warning(f"Connection closed {str(server_port)} ")

                        rp_monitoring_merging(group,working_dir)
                        st.success("Monitoring Completed")

    else:
        st.error("Please enter your name to continue")


    
