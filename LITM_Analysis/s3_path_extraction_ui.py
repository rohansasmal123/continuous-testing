import pandas as pd
import mysql.connector
import pymysql as db
import streamlit as st

def connect_sql(schema,prt,cred,hst):
    try:
        conn = mysql.connector.connect(
            host=hst,  # your host, usually localhost
            user=cred.Id.values[0],  # your username
            password=cred.Db_pass.values[0],  # your password
            database=schema,  # name of the data base
            port=prt  # port number
        )
    except db.OperationalError as e:
        print("Connection failed. Error number {0}: {1}.".format(e.args[0], e.args[1]))
    return conn


def fetch_data_for_s3(schema,schema_accounts,cred,hst,start_date,root_dir):
    local_port=schema_accounts['local_port'].values[0]
    conn = connect_sql(schema,local_port,cred,hst)
    for account_no in schema_accounts['account_id'].tolist():
        account_no=str(account_no)
        query= """SELECT ocr.pk_ocr_ml_context_id,
        ocr.fk_account_id,
        ocr.fk_caa_remittance_ocr_header_id,
        ocr.page_context_path,
        caa.inbound_remittance_header_id
        FROM ocr_ml_context AS ocr
        INNER JOIN caa_ocr_remittance_hdr AS caa
        ON ocr.fk_caa_remittance_ocr_header_id = caa.pk_caa_remittance_ocr_header_id
        WHERE ocr.fk_account_id ="""+ account_no+""" AND ocr.is_deleted = 0
        AND ocr.check_number != ''
        AND caa.inbound_remittance_header_id IN (
        SELECT fk_caa_inbound_remittance_header_id
        FROM caa_machine_learning_lookup
        WHERE fk_account_id = """+account_no+""" AND (page_context_path LIKE '%/s3hrc-caa-prod/"""+account_no+"""/"""+start_date+"""%'))"""

        res = pd.read_sql_query(query, conn)
        final_result = res
        final_result.to_csv(root_dir+"/s3_download_files/"+str(account_no)+"_s3.csv", index=False)
        st.success(str(account_no)+" s3 csv generated")
    conn.close()

def download_s3_data(data,root_dir,start_date,cred):

    data.astype({"local_port": int})
    hst = "localhost"
    data_groupby_schema=data.groupby('schema_name')
    print("********Extracting data*********")
    for schema,schema_accounts in data_groupby_schema:
        fetch_data_for_s3(schema,schema_accounts,cred,hst,start_date,root_dir)












