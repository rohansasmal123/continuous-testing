import os
import sys
import pandas as pd
def create_dir(path, account_id, log_path):
    dir_path = path + os.path.sep + 'account_' + account_id
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_list = ['customer_subsets_features',
                'customer_wise_subsets',
                'customersJson',
                'data_extracted',
                'history_generated',
                'predictions',
                'query_config_file',
                'rivana_test',
                'subsets_rolled_up',
                'train_test_splitted',
                'trained_model',
                'logs'
                ]
    for dir_name in dir_list:
        dir_path = path + os.path.sep + 'account_' + account_id + os.path.sep + dir_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    progress=pd.DataFrame([['Directory Creation']],columns=['Status'])
    progress.to_csv(log_path+"progress.csv",index=False)
    

if __name__ == '__main__':
    account_id = str(sys.argv[1])
    path = str(sys.argv[2])
    log_path=path+'/account_'+str(account_id)+'/logs/'

    #This function will create directory folders
    create_dir(r'' + path, account_id, log_path)
