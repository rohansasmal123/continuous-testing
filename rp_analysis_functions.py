import pandas as pd
import streamlit as st
import mysql.connector
import pymysql as db
import numpy as np
import json
from sshtunnel import SSHTunnelForwarder

def putty_conn(user_name, for_accounts,cred):
    with st.spinner("Execution in progress..."):
        #for_accounts = pd.read_csv("C:\\Users\\archita.ganguly\\Desktop\\CAA\\UI\\port.csv")
        for_accounts = for_accounts[for_accounts['intern_name'] == user_name]
        server_port = []
        localhost_port = []
        server=[]
        for index, row in for_accounts.iterrows():
            server_port.append(row['port'])
            localhost_port.append(row['local_port'])

        server_localhost = {"Server Port": server_port,
                            "LocalHost Port": localhost_port}
        server_localhost_df = pd.DataFrame(server_localhost)

        server_localhost_df.drop_duplicates(keep="first", inplace=True)
        
    
        #with open('C:/ai_cashapp/RP Monitoring/credentials.json') as f:
            #cred = json.load(f)
        
        for index, row in server_localhost_df.iterrows():
            st.write(int(row['LocalHost Port']))
            temp = SSHTunnelForwarder('172.27.128.59', ssh_username=cred['Id'][0], ssh_password=cred['Ldap_pass'][0],remote_bind_address=('127.0.0.1', int(row["Server Port"])),local_bind_address=('0.0.0.0', int(row["LocalHost Port"])))
                   
                                        

            # st.write("Destination Server Port {row['Server Port']} and Source Port {row['LocalHost Port']} in execution")
            temp.daemon_forward_servers = True
            temp.start()
            server.append(temp)
        st.success("Putty SSH Connection Done")
        return for_accounts,server

def connect_sql(schema,prt,cred):
    try:
        conn = mysql.connector.connect(
            host='localhost',  # your host, usually localhost
            user=cred['Id'][0],  # your username
            password=cred['Db_pass'][0],  # your password
            database=schema,  # name of the data base
            port=prt  # port number
        )
        #st.success("SQL connection DONE")
    except db.OperationalError as e:
        st.write("Connection failed. Error number {0}: {1}.".format(e.args[0], e.args[1]))
    return conn

def get_final_input(schema,accid,pay_id,port_df, cred):
    #port_df = pd.read_csv("C:\\Users\\archita.ganguly\\Desktop\\CAA\\UI\\port.csv")
    local_port=port_df[port_df['account_id']==accid]['local_port'].reset_index()
    local_port = local_port['local_port'][0]
    st.write(local_port)
    conn = connect_sql(schema,local_port,cred)
    st.success("SQL connection DONE")
    accid=str(accid)
    pay_id=str(pay_id)
    query= """SELECT final_input FROM ai_prediction_log WHERE fk_account_id="""+accid+""" AND fk_ai_use_case_id=9 AND 
    (fk_ai_model_version_id,fk_ai_job_id)=(SELECT fk_model_version_id, fk_job_id FROM ai_job_monitoring WHERE 
    fk_account_id="""+accid+""" AND fk_use_case_id=9 LIMIT 1) AND id="""+pay_id

    res = pd.read_sql_query(query, conn)
    st.success("Final input generated for payment hdr id : "+str(pay_id))
    conn.close()
    return res

def fill_q2(data, query_output_path):
    flag=1
    data['Q2']=pd.Series()
    data['Q3']=pd.Series()

    for i in range(0,data.shape[0]):
        if data['PRLF'][i]==np.nan:
            data['Q3']="NO"
        else:
            data['Q3']="YES"
        schema=data['Schema'][i]
        accid=data['Account Id'][i]
        q2=pd.read_csv(query_output_path+schema+'_q_2.csv')
        q2=q2[q2['fk_account_id']==accid]
        if q2.empty:
            data['Q2'][i]="NO"
        else:
            data['Q2'][i]="YES"
    return data,flag

def pred_pres_or_not(data,for_accounts,cred):
    data['Comments']=pd.Series()
    for i in range(0, data.shape[0]):
        accid = data['Account Id'][i]
        schema = data['Schema'][i]
        local_port = for_accounts[for_accounts['account_id'] == accid]['local_port']
        conn = connect_sql(schema, local_port,cred)
        if (data['Q3'][i]=="NO") | (data['PRLF'][i]==0):
            data['Comments'][i] = "No PRLF"
        elif (data['Q2'][i] == "NO") & (data['Q3'][i] == "YES"):

            accid = str(accid)
            query="""SELECT * FROM ai_prediction_log WHERE fk_account_id="""+accid+""" AND fk_ai_use_case_id=9 AND 
            (fk_ai_model_version_id,fk_ai_job_id)=(SELECT fk_model_version_id, fk_job_id FROM ai_job_monitoring WHERE 
            fk_account_id="""+accid+""" AND fk_use_case_id=9 LIMIT 1) LIMIT 1;"""
            res = pd.read_sql_query(query, conn)
            if(len(res)==0):
                data['Comments'][i]="Predictions Not Present"
            else:
                data['Comments'][i]="Predictions Present but no data in q2"
        conn.close()
    #server.stop()
    data.to_csv("/root/caa/rp/analysis/Analysis_Report.csv")
    return data


def missed_hdr_ids(schema,query_output_path):
    actual = pd.read_csv(query_output_path+ schema + '_q_2.csv')  # query 2
    results = pd.read_csv(query_output_path+ schema + '_q_1.csv')   # Query 1

    results['create_date_time'] = pd.to_datetime(results['create_date_time'])
    # results = results[results['create_date_time'] > '2020-05-15']
    pmt_list = list(results['payment_hdr_id'].unique())
    actual = actual[actual['payment_hdr_id'].isin(pmt_list)]

    # print(actual['fk_account_id'].unique())
    pmt_subset_dict = {}
    for pmt, obj in results.groupby('payment_hdr_id'):
        subset_dic = dict(obj.groupby('probability')['pk_caa_subset_info_id'].unique().sort_index(ascending=False))
        pmt_subset_dict[pmt] = subset_dic
    #print(pmt_subset_dict)

    res = {}
    for pmt, obj in actual.groupby('payment_hdr_id'):
        count = 0
        un_inv = set(obj['acct_doc_header_id'].unique())
        if pmt in pmt_subset_dict.keys():
            res[pmt] = 0
            for prob in pmt_subset_dict[int(pmt)].keys():
                if count == 3:
                    break
                for subset in pmt_subset_dict[int(pmt)][prob]:
                    un_inv_res = set(
                        results[(results['payment_hdr_id'] == pmt) & (results['pk_caa_subset_info_id'] == subset)][
                            'acct_doc_hdr_id'].unique())
                    if len(un_inv.intersection(un_inv_res)) == len(un_inv):
                        res[pmt] = 1
                        break
                if res[pmt] == 1:
                    break
                count += 1
        else:
            res[pmt] = 0

    outcome = actual[['fk_account_id', 'payment_hdr_id']].drop_duplicates().copy()

    outcome['result'] = outcome['payment_hdr_id'].map(res)
    return outcome, pmt_subset_dict


def filter_data(data):
    data = data[(data['Top 3 %'].notnull()) & (data['Top 3 %'] < 80)]
    print(data)
    data['Account Id'] = data['Account Id'].astype('int')
    data['Top 3 %'] = data['Top 3 %'].astype('int')
    data.reset_index(drop=True, inplace=True)

    data['status'] = pd.Series()
    for i in range(0, data.shape[0]):
        if (data['Top 3 %'][i] >= 0) and (data['Top 3 %'][i] < 50):
            data['status'][i] = "RED"
        elif (data['Top 3 %'][i] >= 50) and (data['Top 3 %'][i] <= 79):
            data['status'][i] = "YELLOW"
    return data


def not_included_pay_hdr_ids(accid, outcome):

    outcome = outcome[(outcome['fk_account_id'] == accid) & (outcome['result'] == 0)]
    outcome.reset_index(drop=True, inplace=True)

    pay_hdr_id = outcome['payment_hdr_id']  # not included hdr_ids by analyst

    return pay_hdr_id


def check_amt(pay_hdr_id, schema, query_output_path):
    q2 = pd.read_csv(query_output_path + schema + '_q_2.csv')
    q2 = q2[q2['payment_hdr_id'].isin(pay_hdr_id)]
    grp = q2.groupby('payment_hdr_id')

    match = []
    not_match = []
    for id, gp in grp:
        if gp['payment_amount.1'].sum() == gp['payment_amount'].unique()[0]:
            match.append(id)
        else:
            not_match.append(id)

    if(len(not_match)>0):
        st.write("sum of invoice amount did not match with payment amount for: ")
        st.write([i for i in not_match])
    else:
        st.write("Sum of invoice amount matched with payment amount for all payment header ids")
    return match


def get_invoices(pay_hdr_id,match, final_ip, schema, query_output_path):
    q2 = pd.read_csv(query_output_path + schema + '_q_2.csv')
    q2 = q2[q2['payment_hdr_id'].isin(pay_hdr_id)]
    grp = q2.groupby('payment_hdr_id')

    invoices = {}
    for id, gp in grp:
        if id in match:
            invoices[id] = [i for i in gp['invoice_number_norm']]

    # invoices in each subset in final input
    selected_invoices = {}

    for i in final_ip['items']:
        subset_id = i['subsetId']
        selected_invoices[subset_id] = []

        for inv in i['subset']:
            selected_invoices[subset_id].append(inv['invoice_number'])

    return invoices, selected_invoices

def match_invoices(pay_id, invoices, selected_invoices): #get subset id for which all invoices in q2 match with analysts
    l1=[]
    l1=invoices[pay_id]
    l1.sort()
    for i,l2 in selected_invoices.items():
        l2.sort()
        if l1==l2:
            return 1,i
    # return 0 if noting matches => then no need to proceed further
    return 0,"All the subsets which analyst linked with the payment Id "+str(pay_id)+" is not present in our subsets."

#for indiv pay_id
def our_subset(pay_id, pmt_subset_dict):
    our_subsets=[]
    for i in pmt_subset_dict[pay_id]:
        #for j in pmt_subset_dict[pay_id][i]:
        our_subsets.append(pmt_subset_dict[pay_id][i][0])
    return our_subsets


def match_features(final_ip, our_id, analyst_id):
    features = ['subsetId', 'invoice_count_quarter_q4', 'number_invoices_closed',
                'avg_of_all_delays', 'customer_number', 'payment_count_quarter_q3',
                'variance_categorical', 'payment_count_quarter_q4', 'payment_count_quarter_q1',
                'payment_count_quarter_q2',
                'avg_delay_categorical', 'invoice_count_quarter_q2', 'invoice_count_quarter_q3',
                'invoice_count_quarter_q1',
                'unique_invoice_count', 'avg_of_invoices_closed', 'subset_invoice_count', 'LMH_cumulative']
    for i in final_ip['items']:
        if i['subsetId'] == our_id:
            df1 = pd.DataFrame.from_dict(i)
            st.write("Our Subset features")
            st.write(df1[features].head(1))
            break

    for i in final_ip['items']:
        if i['subsetId'] == analyst_id:
            df2 = pd.DataFrame.from_dict(i)
            st.write("Analysts' Subset features")
            st.write(df2[features].head(1))
            break
