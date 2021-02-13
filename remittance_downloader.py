import os
import time
import requests
import threading
import logging
import warnings
import pandas as pd
from datetime import date
import streamlit as st
#Do not change this unless you know what you're doing


def get_file_name(file_path):
    return file_path.split('/')[-1]

BATCH_SIZE=100
#Helper function for downloading files
def download_file(url,write_path):
   headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
   try:
        file = requests.get(url,headers)
        with open(write_path, 'wb') as f:
            for chunk in file.iter_content(1024):
                f.write(chunk)

   except Exception as e:
       print("Exception occurred {0} \n".format(str(e)))


def download_cheque_remittance_files(file_path_list,write_directory,account_classification,num_threads=1):
    if num_threads > 3:
        warnings.warn("Too many threads might lead to some connections being dropped")

    # Make it docs3 in case of BAML
    document_service_url = "https://docs.highradius.com/global_document_service/vcmwlnr/"

    download_threads = []
    logging.debug("Starting downloads,total files to download {0},batch size {1}".format(len(file_path_list),BATCH_SIZE))
    st.write("Starting downloads,total files to download {0},batch size {1}".format(len(file_path_list),BATCH_SIZE))
    start_time = time.time()

    batch_count = 0
    processIndex = 0
    for file_path in file_path_list:

        file_full_path = document_service_url + file_path

        processIndex = processIndex + 1
        try:
            write_path = ""
            if account_classification == False:

                write_path = os.path.join(write_directory,os.path.join(*file_path.split('/')[7:]))

                if not os.path.exists(os.path.dirname(write_path)):
                    os.makedirs(os.path.dirname(write_path))

                if os.path.exists(write_path):
                    continue

            else:
                write_path = os.path.join(write_directory,"_".join(file_path.split('/')))

            for _ in range(0, num_threads):
                download_thread = threading.Thread(target=download_file, args=(file_full_path, write_path))
                download_threads.append(download_thread)
                download_thread.start()

            time.sleep(0.3)
            batch_count += 1
            if batch_count == BATCH_SIZE:
                print(f'Process Index : {processIndex}')
                print("Batch size is {0} files, waiting for current batch to complete".format(batch_count))
                for thread in download_threads:
                    thread.join(timeout=5)
                download_threads = []
                batch_count = 0

        except Exception as e:
            print("For file :{0},exception {1} \n".format(file_path, str(e)))

    for thread in download_threads:
        thread.join(timeout=5)
    end_time = time.time()
    print("Time taken in seconds",(end_time - start_time))


def download_files(write_directory,paths_to_download,account_classification):
    paths_to_download['image_path'] = paths_to_download['page_context_path'].str.replace('/ML/', '/images/')
    paths_to_download['image_path'] = paths_to_download['image_path'].str.replace('json', 'png')
    if account_classification == True:
        paths_list = paths_to_download['image_path'].to_list()
    else:
        paths_list = pd.concat([paths_to_download['page_context_path'], paths_to_download['image_path']]).tolist()

    print("Total files to download,",len(paths_list))
    download_cheque_remittance_files(paths_list, write_directory,account_classification, 3)

def download_cheque_files(raw_dataframe,write_directory,account_classification = False):
    download_files(write_directory,raw_dataframe,account_classification)


