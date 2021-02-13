import streamlit as st
import re
import datetime
import pandas as pd
from sshtunnel import SSHTunnelForwarder
import os

from litmscript.litmmonitor.Data_csv import get_data
from litmscript.litmmonitor.LITM_Monitoring_automation_modified import LITM_monitoring_script
from litmscript.litmmonitor.LITM_Monitoring_automation_modified import generate_final_report


@st.cache(suppress_st_warning=True)
def get_data_from_gsheet(sheet_name):
    with st.spinner("Fetching Data from DB"):
        monitoring_Sheet=get_data(sheet_name)
    return monitoring_Sheet




def main():
    root_dir='/root/caa/litm/monitor'
    #data_extracted=root_dir+'Data_Extracted\\'
    cred_path ='/root/caascript/res/cred.csv'
    user=st.text_input("Enter Your name")
    if not os.path.exists(cred_path):
        user=""
    if user!="":      
            monitoring_Sheet = get_data_from_gsheet("LITM_monitoring")
            if monitoring_Sheet.shape[0]==0:
                st.error("Unable To Fetch Data...")
            else:
                assigned_accounts=monitoring_Sheet.loc[monitoring_Sheet['intern_name'] == user.title()]
                st.write(f"{user.title()}'s, Assigned accounts are")
                st.write(assigned_accounts)
                if assigned_accounts.shape[0] == 0:
                    st.warning("Sorry "+user+" no account has been assigned to you !!")
                else:
                    right, left = st.beta_columns(2)
                    start_date = right.date_input("Enter Start Date", datetime.date.today())
                    end_date = left.date_input("Enter End Date", datetime.date.today())
                    assigned_accounts_group_environment=assigned_accounts.groupby("environment")
                    current_date=datetime.date.today()
                    month=current_date.strftime("%b")
                    year=current_date.strftime("%Y")
                    path_month_year=str(month)+"_"+str(year)
                    working_dir=os.path.join(root_dir,path_month_year)
                    if(not os.path.exists(working_dir)):
                        os.makedirs(working_dir)
                        print("Directory Created Successfully")
                    
                    


                    flag=1
                    if st.button("Start Monitoring"):
                        flag=0
                        for env, group_env in assigned_accounts_group_environment:
                                server_port = []
                                localhost_port = []
                                server_grouped = group_env.groupby("port")
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
                                for server_port_number, group_env in server_grouped:
                                    print(f" Establishing Connection with Destination Server Port {str(server_port_number)} and Source Port {str(group_env['local_port'].tolist()[0])} in execution")
                                    server = SSHTunnelForwarder('172.27.128.59', ssh_username=ldap_user,
                                                                ssh_password=ldap_pass,
                                                                remote_bind_address=(
                                                                'localhost', int(server_port_number)),
                                                                local_bind_address=(
                                                                '0.0.0.0', int(group_env['local_port'].tolist()[0])))

                                    print(
                                        f"Destination Server Port {str(server_port_number)} and Source Port {str(group_env['local_port'].tolist()[0])} in execution")
                                    try:
                                        server.start()
                                        st.success("Connection Successfull")
                                        with st.spinner("Execution in Progress..."):
                                            LITM_monitoring_script(group_env,ldap_user,ldap_pass,db_pass,start_date, end_date,working_dir)
                                            
                                            
                                            

                                    except:
                                        st.error(f"Could Not connect to {str(env)}\n\n Check logs for more details")

                                    finally:
                                        server.stop()
                                        st.warning(f"Connection closed {server_port_number} ")  
                    st.success("Monitoring Completed")
                    flag=1  
                    if flag:
                        if st.button("Generate Final Report"):
                            try:
                                generate_final_report(working_dir)
                            except ValueError:
                                st.warning("Please Perform Monitoring")
                                st.error("No objects to concatenate")
                            


    else:
        st.warning("Please Enter Your name")
