# -*- coding: utf-8 -*-
"""
Created on Wed Sept 18 15:23:30 2020
author:plaban.nayak,gautam.singh,priyanshu.kundu
"""
import os
import mysql.connector
import sqlalchemy
#from RP.RP_Monitoring.RP_monitoring_test.monitor import get_results
from datetime import date
import csv
import numpy as np
import pandas as pd
import glob
import pymysql as db
import json
import streamlit as st

#please put your ldap id,password,schemas in the credentials.json file(one time)
#with open('C:/Users/shivam.g/PycharmProjects/LITM_cashapp/RP_monitoring/credentials.json') as f:
#cred = json.load(f)
today = date.today()


def get_results(actual,results,n):

    results['create_date_time'] = pd.to_datetime(results['create_date_time'])
    # results = results[results['create_date_time'] > '2020-05-15']
    pmt_subset_dict = {}
    for pmt, obj in results.groupby('payment_hdr_id'):
        subset_dic = dict(obj.groupby('probability')['pk_caa_subset_info_id'].unique().sort_index(ascending=False))
        pmt_subset_dict[pmt] = subset_dic

    res = {}
    for pmt, obj in actual.groupby('payment_hdr_id'):
        count = 0
        un_inv = set(obj['acct_doc_header_id'].unique())
        if pmt in pmt_subset_dict.keys():
            res[pmt] = 0
            for prob in pmt_subset_dict[int(pmt)].keys():
                if count ==n:
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
    return(res)

def sqlconnector(hst, usr, pwd, dbn, prt,start_date,end_date,working_dir):
    root_directory = working_dir + '/Output_Files/DB_Query_files'
    try:
        conn = mysql.connector.connect(
            host=hst,  # your host, usually localhost
            user=usr,  # your username
            password=pwd,  # your password
            database=dbn,  # name of the data base
            port=prt  # port number
        )
        print(f"***************\nconnection established with scheema {str(dbn)}\n***************")

        count = 1
        start_date = "'" + str(start_date) + "'"
        end_date = "'" + str(end_date) + "'"
        query1 = """SELECT info.pk_caa_subset_info_id,info.probability,
                details.*, pmt_cust.*,max_.*
                FROM  caa_subset_info AS info INNER JOIN caa_subset_details AS details ON pk_caa_subset_info_id = `fk_caa_subset_info_id`
                INNER JOIN caa_subset_pmt_cust_mapping AS pmt_cust ON `pk_caa_subset_pmt_cust_mapping_id` = fk_caa_subset_pmt_cust_mapping_id
                INNER JOIN
                (SELECT pmt_map_id,probability AS top_prob, pk_caa_subset_info_id FROM
                (SELECT pmt_map_id, probability,pk_caa_subset_info_id,
                    @pmt_rank := IF(@current_pmt = pmt_map_id, IF( @current_prob = probability,@pmt_rank, @pmt_rank+1), 1) AS pmt_rank,
                        @current_pmt := pmt_map_id,
                        @current_prob := probability
                FROM (SELECT @pmt_rank :=1) temp,
                    (SELECT @current_pmt := pk_caa_subset_pmt_cust_mapping_id AS pmt_map_id, @current_prob := probability AS probability, pk_caa_subset_info_id
                    FROM caa_subset_info INNER JOIN caa_subset_pmt_cust_mapping 
                    ON pk_caa_subset_pmt_cust_mapping_id = fk_caa_subset_pmt_cust_mapping_id
                    ORDER BY pmt_map_id ASC,probability DESC) prob 
                ) ranked
                WHERE pmt_rank <= 3 ) AS max_
                ON info.pk_caa_subset_info_id = max_.pk_caa_subset_info_id
                WHERE info.probability = top_prob
                AND pmt_cust.create_date_time >"""+start_date+"""AND pmt_cust.create_date_time <="""+end_date

        query2 = """SELECT acct.acct_doc_header_id,header.fk_account_id,acct.customer_number_norm,acct.due_date_norm,
        acct.invoice_number_norm,acct.invoice_date_norm,acct.invoice_amount_norm,acct.open_amount_norm,acct.debit_credit_indicator,
        acct.document_date_norm,acct.isOpen, acct.create_date,item.fk_inbound_remittance_header_id,header.payment_amount,item.payment_amount,
        payment.payment_amount,item.total_amount,header.payment_hdr_id, payment.payment_method,payment.effective_date,payment.is_deleted,header.is_deleted,item.is_deleted
        FROM `acct_doc_header` AS acct INNER JOIN `caa_remittance_item_tagging` AS tag ON acct.acct_doc_header_id = tag.fk_acct_doc_header_id_global
        INNER JOIN `caa_inbound_remittance_item` AS item ON tag.pk_remittance_item_tagging_id = item.fk_caa_remittance_item_tagging_id
        INNER JOIN `caa_inbound_remittance_header` AS header ON header.pk_inbound_remittance_header_id = item.fk_inbound_remittance_header_id
        INNER JOIN `caa_payment_confirmation_hdr` AS payment ON payment.pk_payment_confirmation_hdr_id = header.payment_hdr_id
        WHERE tag.fk_acct_doc_header_id_global = tag.fk_acct_doc_header_id_local AND acct.create_time > '2017-06-01'
        AND header.payment_hdr_id IN (SELECT distinct payment_hdr_id from caa_subset_pmt_cust_mapping WHERE create_date_time 
        <="""+end_date+""" AND create_date_time >"""+ start_date+""")"""
        #
        query3 = """SELECT fk_account_id, COUNT(DISTINCT(fk_payment_confirmation_hdr_id)) as PRLF FROM caa_map_error_detail 
        WHERE fk_error_code_id = 54 AND create_date >"""+start_date+"""AND create_date <="""+end_date+"""GROUP BY fk_account_id"""
        query=[query1,query2,query3]

        #print(root_directory)
        for i in query:
            try:
                print("\n***********************  query "+str(count)+"  started***********************\n")
                res = pd.read_sql_query(i,conn)
                final_result = res
                filename = dbn + "_q_" + str(count) + ".csv"
                full_directory = os.path.join(root_directory, filename)
                final_result.to_csv(full_directory, index=False)
                print("\n***********************  query "+str(count)+"  Completed***********************\n")
                count=count+1
            except:
                print(f"\n\nQuery {str(count)} failed at {str(dbn)}")
                count+=1
                time.sleep(2)
                print("Skipping to next...")
                continue 
            
    except db.OperationalError as e:
        print("Connection failed. Error number {0}: {1}.".format(e.args[0], e.args[1]))
    
    finally:
        conn.close()
        print(f"***************\nSQL connection closed {str(dbn)}\n***************")



"""
# Show the data
print(sql_data.head())
# Export the data into a csv file
database_name = "zinfandel"
filename =  database_name + "_q3.csv"
sql_data.to_csv(filename,index=False)

"""
##***********************************************
# this will call the monitoring script







def RP_monitoring_script(ldap_user, db_pass, host,start_date,end_date,acct_map,working_dir):

    st.write(acct_map)
    
    schema = acct_map['schema_name'].unique().tolist()

    acct_map.astype({"local_port": int})
    #for_accounts = assigned_data[p_df['schema'].isin(schema)]

    port = []
    for i in schema:
        print(acct_map[acct_map['schema_name'] == i].drop_duplicates(subset=['schema_name'])['local_port'].tolist())
        port.append(acct_map[acct_map['schema_name'] == i].drop_duplicates(subset=['schema_name'])['local_port'].tolist()[0])

    Schema_port = {"Schema": schema,
             "Port": port}
    Schema_port_df=pd.DataFrame(Schema_port)
    #st.write(Schema_port_df)

    hst = host
    usr = ldap_user
    pwd = db_pass
    print("********Extracting data*********")

    for index, row in Schema_port_df.iterrows():
        try:
            sqlconnector(hst,usr,pwd, row['Schema'],row['Port'],start_date, end_date,working_dir)
        except db.OperationalError as e:
           print("\nAn OperationalError occurred. Error number {0}: {1}.".format(e.args[0], e.args[1]))
    ##************************************************************

    ##************************************************************
    # """
def rp_monitoring_merging(acct_map,working_dir):
    acct_map = acct_map.rename(columns={'account_id': 'Account Id', 'account_name': 'Account Name'})
    root_folder = working_dir+'/Output_Files/Script_Final_files/'
    query1_path = working_dir+'/Output_Files/DB_Query_files/*_q_1.csv'
    query2_path = working_dir+'/Output_Files/DB_Query_files/*_q_2.csv'
    query3_path = working_dir+'/Output_Files/DB_Query_files/*_q_3.csv'
    # # read the actuals file
    actuals = pd.concat(map(pd.read_csv, glob.glob(query2_path)))
    results = pd.concat(map(pd.read_csv, glob.glob(query1_path)))
    prlf = pd.concat(map(pd.read_csv, glob.glob(query3_path)))
    pmt_list = list(results['payment_hdr_id'].unique())
    actuals = actuals[actuals['payment_hdr_id'].isin(pmt_list)]
    # 

    ##************************************************************

    # read Account Mapping

    # main(sys.argv[1:])

    ##************************************************************

    print('Processing Top 3 probability')
    res = get_results(actuals, results, 3)
    outcome = actuals[['fk_account_id', 'payment_hdr_id']].drop_duplicates().copy()
    outcome['Top 3'] = outcome['payment_hdr_id'].map(res)
    print("outcome top 3")
    print(outcome)

    print("*" * 90)
    output_3 = outcome.groupby(['fk_account_id']).agg({'Top 3': 'sum', 'payment_hdr_id': 'count'}).reset_index()
    ##************************************************************

    print('Processing Top 2 probability')
    res = get_results(actuals, results, 2)
    outcome = actuals[['fk_account_id', 'payment_hdr_id']].drop_duplicates().copy()
    print("outcome top 2")
    print(outcome)
    print("*" * 90)
    outcome['Top 2'] = outcome['payment_hdr_id'].map(res)
    output_2 = outcome.groupby(['fk_account_id']).agg({'Top 2': 'sum', 'payment_hdr_id': 'count'}).reset_index()
    ##**************************************************************

    print('Processing Top 1 probability')
    res = get_results(actuals, results, 1)
    outcome = actuals[['fk_account_id', 'payment_hdr_id']].drop_duplicates().copy()
    print("outcome top 1")
    print(outcome)
    print("*" * 90)
    outcome['Top 1'] = outcome['payment_hdr_id'].map(res)

    output_1 = outcome.groupby(['fk_account_id']).agg({'Top 1': 'sum', 'payment_hdr_id': 'count'}).reset_index()


    ##**************************************************************

    outcome = pd.merge(output_1, output_2, on=['fk_account_id', 'payment_hdr_id'], how='left')
    final = pd.merge(outcome, output_3, on=['fk_account_id', 'payment_hdr_id'], how='left')

    # **************************************************************
    # % calculation
    final['Top 1%'] = (final['Top 1'] / final['payment_hdr_id']) * 100
    final['Top 2%'] = (final['Top 2'] / final['payment_hdr_id']) * 100
    final['Top 3%'] = (final['Top 3'] / final['payment_hdr_id']) * 100

    #
    final['Top 1%'] = final['Top 1%'].astype(np.int)
    final['Top 2%'] = final['Top 2%'].astype(np.int)
    final['Top 3%'] = final['Top 3%'].astype(np.int)
    final['IS BAML'] = 'NO'
    #
    final = pd.merge(final, prlf, how='left', on='fk_account_id')
    #
    final = pd.merge(final, acct_map, how='left', left_on='fk_account_id', right_on='Account Id')
    #
    final = final[
        ['fk_account_id', 'Account Name', 'payment_hdr_id', 'IS BAML', 'PRLF', 'Top 1', 'Top 2', 'Top 3', 'Top 1%',
         'Top 2%', 'Top 3%']]

    rows = []
    datalist = ['Account Id', 'Account Name', 'Number of Payments', 'IS BAML', 'No. of Payments with PRLF', 'Top 1',
                'Top 2', 'Top 3', 'Top 1%', 'Top 2%', 'Top 3%']
    rows.append(final.values.tolist())

    # Write Header Record
    print('Writing Header to the report')
    filename = 'Sept_CashApp_Monitoring_report_' + str(today) + '.csv'
    full_direc = os.path.join(root_folder, filename)
    with open(full_direc, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(datalist)
    # Append the detail records to the csv file
    print('Writing detail records to the report')
    with open(full_direc, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        print(file)
        for item in rows:
            writer.writerows(item)
    st.write(pd.read_csv(root_folder+filename))
    print('processing complete')



