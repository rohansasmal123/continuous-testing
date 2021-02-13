#authors:Shivam Gupta 11440,priyanshu.kundu LITM MONITORING V1.0.0 (2020-08-23)

import pandas as pd
import pymysql as db
import logging
import sys
import os
import numpy as np
import sqlalchemy
from datetime import datetime
import json
import streamlit as st
#please put your ldap id,password,schemas in the credentials.json file(one time)
import pathlib
import glob

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def getConnection(schema,port_no,cred):
    st.write("in get connection")
    try:
        connection =db.connect(host="localhost",
                              port=port_no,
                              user=cred.Id.values[0],
                              passwd=cred.Db_pass.values[0],
                              db=schema)
        return connection
    except db.OperationalError as e:
        print("\nAn OperationalError occurred. Error number {0}: {1}.".format(e.args[0], e.args[1]))
    st.write("out get connection")


def generate_final_report(output_path):
    #st.write(output_path)
    clearedCheques_path = output_path+'/*_q_1.csv'
    failedCheques_path = output_path+'/*_q_2.csv'
    processedCheques_path = output_path+'/*_q_3.csv'
    totalCheques_path = output_path+'/*_q_4.csv'
    withIntervention_path = output_path+'/*_q_5.csv'
    # # read the actuals file
    clearedCheques = pd.concat(map(pd.read_csv, glob.glob(clearedCheques_path)))
    failedCheques = pd.concat(map(pd.read_csv, glob.glob(failedCheques_path)))
    processedCheques = pd.concat(map(pd.read_csv, glob.glob(processedCheques_path)))
    totalCheques = pd.concat(map(pd.read_csv, glob.glob(totalCheques_path)))
    withIntervention = pd.concat(map(pd.read_csv, glob.glob(withIntervention_path)))
    final_result = pd.DataFrame()
    final_result=pd.merge(totalCheques,failedCheques,on="fk_account_id",how="outer")
    final_result = pd.merge(final_result,processedCheques,on="fk_account_id",how="outer")
    final_result = pd.merge(final_result,clearedCheques,on="fk_account_id",how="outer")
    final_result = pd.merge(final_result,withIntervention,on="fk_account_id",how="outer")
    final_result['Complete_Clearances']=final_result['Cleared_Cheques']-final_result['with_intervention']
    final_result=final_result.loc[:,['fk_account_id','Total_Cheques',"Failed_Cheques","Processed_Cheques","Cleared_Cheques","Complete_Clearances","with_intervention"]]
    final_result=final_result.fillna(0)
    final_result.to_csv(output_path+"/Final_Report.csv")
    st.write(final_result)




def getData(query_path,date_mapper,schema_port,Accounts,output_path,cred):
    #st.write(query_path)
    #st.write(schema_port)
    for index, row in schema_port.iterrows():
        connection=getConnection(row['Schema'],row['Port'],cred)
        count=1
        for i in os.listdir(query_path):
            #st.write(count)
            print(f"Query {count} in execution from Schema {row['Schema']} on Port number {row['Port']}")
            query_open = open(query_path+"/"+i, "r")
            query = query_open.read()
            query = replace_all(query,date_mapper)
            #count = count + 1
            try:
                #st.write(query)
                res = pd.read_sql_query(query, connection)
                res.to_csv(output_path + row['Schema'] + "_q_" + str(count) + ".csv")
                count = count + 1
            except sqlalchemy.exc.OperationalError as e:
                    print('Error occured while executing a query {}'.format(e.args))





def LITM_monitoring_script(assigned_data,cred_path,start_date,end_date,root_dir):
    cred = pd.read_csv(cred_path)
    output_path=root_dir+"/"
    current_path=pathlib.Path().absolute()
    query_path= os.path.join(current_path,'sql_query')
    schema = assigned_data['schema_name'].unique().tolist()
    start_date="\""+str(start_date)+"\""
    end_date="\""+str(end_date)+"\""
    assigned_data.astype({"local_port": int})
    #for_accounts = assigned_data[p_df['schema'].isin(schema)]
    Accounts = assigned_data['account_id'].tolist()
    port = []
    for i in schema:
        print(assigned_data[assigned_data['schema_name'] == i].drop_duplicates(subset=['schema_name'])['local_port'].tolist())
        port.append(assigned_data[assigned_data['schema_name'] == i].drop_duplicates(subset=['schema_name'])['local_port'].tolist()[0])

    port=[]
    for i in schema:
        port.append(assigned_data[assigned_data['schema_name'] == i].drop_duplicates(subset=['schema_name'])['local_port'].tolist()[0])
    #Accounts=[10120, 70173, 70261, 70265, 70270, 8970]
    Schema_port = {"Schema": schema,
             "Port": port}
    Schema_port_df=pd.DataFrame(Schema_port)
    #st.write(Schema_port_df)
    date_mapper = {"start_date":str(start_date), "end_date": str(end_date)}

    print("********Extracting Data********")
    getData(query_path,date_mapper,Schema_port_df,Accounts,output_path,cred)
    print("-----EXECUTION COMPLETE-----------")


