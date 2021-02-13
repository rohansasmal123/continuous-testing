import os
import sys
import json
from datetime import date
import pandas as pd
import streamlit as st


from remittance_downloader import download_cheque_files as download_cheque_files
from generate_row_level_data_from_json_updated_labelling import generate_row_level_data_util
from heading_model import predict_on_test_data as header_prediction
from total_model import predict_on_test_data as total_prediction
from is_remittance import predict_on_test_data as remittance_prediction
from amount_and_reference_no_capture import monitoring_litm_data



def flow_code(root_dir,s3_path,account_no):
    
    
    super_directory = root_dir
    model_dir = r"/root/caascript/litmscript/litmanalysis"
    header_pkl_path = os.path.join(model_dir,"LITM_heading.pickle")
    total_pkl_path = os.path.join(model_dir,"LITM_total.pickle")
    remittance_pkl_path = os.path.join(model_dir,"LITM_remittance.pickle")


    account_id=str(account_no)
    directory = os.path.join(super_directory,str(account_id))

    if not os.path.exists(directory):
        os.makedirs(directory)

  
    data = pd.read_csv(s3_path)
    
    with st.spinner("Downloading Image files"):
        download_cheque_files(data, directory)
    print("directory path"+directory);
    
    with st.spinner("Generating Prediction files: "):
        json_csv_write_path = os.path.join(directory,'row_level_data.csv')
        print("json path:"+json_csv_write_path);
        generate_row_level_data_util(directory,json_csv_write_path)
        header_prediction(header_pkl_path,json_csv_write_path,directory)
        print("header prediction")
        header_prediction_path = os.path.join(directory,'is_heading_prediction.csv')
        print("header prediction path")
        total_prediction(total_pkl_path,header_prediction_path,directory)
        print("total prediction")
        total_prediction_path = os.path.join(directory,'is_total_pred.csv')
        print("total prediction path")
        remittance_prediction(remittance_pkl_path,total_prediction_path,directory)
        print("remittance prediciton")
        remittace_prediction_path = os.path.join(directory,'is_remittance_pred.csv')
        print("remittance prediction path")
        monitoring_litm_data(remittace_prediction_path,directory)
        print("monitoring Prediction path")

    correctly_closed_path = os.path.join(directory, 'correctly_closed_checks.csv')
    os.chdir(root_dir)
    return correctly_closed_path