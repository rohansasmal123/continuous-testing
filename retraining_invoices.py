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
import math
import os.path, time, stat
from bg_css import page_bg
#------------Res---------
from putty import ssh_auto 


if __name__=='__main__':
    cred_path=""
    Rp_sheet_path=""
    final_csv_path=""
    main()

def main():
    if os.path.not.exists(cred_path):
        login()
        print("Credentials Saved!")
    
    cred=pd.read_csv(cred_path)
    
    data=data[data["Accounts status (by the prod team)"]=="Deployed"]
    data=data.groupby("Port")


    local_port=2020
    start_date='2020-12-01'
    end_date='2020-12-31'

    for env, group in data:
        server_port_group=group.groupby('Port')
        #print(group)
        for server_port, server_group_data in server_port_group:
            if env=='AWS US' and len(cred.ldap_pass.values[0])<2:
                ip='172.27.128.59'
                server = SSHTunnelForwarder(ip, 
                                        ssh_username=cred.Id.values[0],
                                        ssh_password=cred.ldap_pass.values[0],
                                        remote_bind_address=('localhost', int(server_port)),
                                        local_bind_address=('127.0.0.1', local_port))
                dbpass=db_pass.values[0]
                host='localhost'
                user=cred.Id.values[0]
            elif env=='AWS EU' and len(cred.ldap_pass.values[1])<2:
                ip='172.27.128.59'
                server = SSHTunnelForwarder(ip, 
                                        ssh_username=cred.Id.values[1],
                                        ssh_password=cred.ldap_pass.values[1],
                                        remote_bind_address=('localhost', int(server_port)),
                                        local_bind_address=('127.0.0.1', local_port))
                dbpass=db_pass.values[1]
                host='localhost'
                user=cred.Id.values[1]
            elif env=='AWS GOLD' and len(cred.ldap_pass.values[2])<2:
                ip='172.27.128.59'           
                server = SSHTunnelForwarder(ip, 
                                        ssh_username=cred.Id.values[2],
                                        ssh_password=cred.ldap_pass.values[2],
                                        remote_bind_address=('localhost', int(server_port)),
                                        local_bind_address=('127.0.0.1', local_port))
                dbpass=db_pass.values[2]
                host='localhost'
                user=cred.Id.values[2]
            else:
                print(f"credentials not found for {str(env)}")
                print("Stopping Execution")
                sys.exit(1)           
            
            print(f"Destination Server Port {str(server_port)} and Source Port {str(local_port)} in execution")
            print(f" Establishing Connection with Destination Server Port {str(server_port)} and Source Port {str(local_port)}  in execution")
            
            try:
                server.start()
                print("Connection Successfull")
                for scheema, scheema_data in server_group_data.groupby("Schema Name"):
                    RP_invoices_script(host, user, dbpass,scheema,working_dir)
                    
            except:
                print("Connection Failed")
                sys.exit(1)


            
                
            
            
            finally:
                try:
                    server.stop()
                    print(f"Connection closed {str(server_port)} ")
                
                except:
                    print(f"Unable to close Port {str(server_port)}")


                
def RP_invoices_script(hst, usr, pwd, dbn, prt, working_dir, start_date, end_date):
    
    root_directory=working_dir +'/Output_Files/Query_files/'+str(start_date)+"_to_"+str(end_date)

    if not os.path.exists(root_directory):
        os.makedirs(root_directory)
    
    try:
        conn = mysql.connector.connect(
            host=hst,  # your host, usually localhost
            user=usr,  # your username
            password=pwd,  # your password
            database=dbn,  # name of the Schema
            port=prt  #  local port number
        )
        print(f"connection established with scheema {str(dbn)}")
        
    except db.OperationalError as e:
        print(f"Could connect to scheema {str(dbn)}. Error number {0}: {1}.".format(e.args[0], e.args[1]))
        sys.exit(1)
        
    count = 1

    query1 = """SELECT fk_account_id, COUNT(DISTINCT(fk_pmt_hdr_id)) FROM caa_ml_03_subset_result 
                WHERE create_date_time >=""" +start_date+ """AND create_date_time <=""" +end_date+ """AND result = 'NA-ROW_COUNT_FAIL_ADH'
                GROUP BY fk_account_id""" 
    
    query2 ="""SELECT fk_account_id, COUNT(DISTINCT(fk_pmt_hdr_id)) FROM caa_ml_03_subset_result 
                WHERE create_date_time >=""" +start_date+ """AND create_date_time <=""" +end_date+ """AND PR_TYPE = 'OP'
                GROUP BY fk_account_id"""
    
   
    query=[query1,query2]

    #print(root_directory)
    
    for i in query:
        print("\n***********************  query "+str(count)+"  started***********************\n")
        res = pd.read_sql_query(i,conn)
        final_result = res
        filename = dbn + "_q_" + str(count) + ".csv"
        full_directory = os.path.join(root_directory, filename)
        final_result.to_csv(full_directory, index=False)
        print("\n***********************  query "+str(count)+"  Completed***********************\n")
        count+=1
        
    finally:
        conn.close()
        print(f"SQL connection closed {str(dbn)}")
            


    

    


    




    



    