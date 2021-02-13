import pandas as pd
import os
import sys
#This function will divide the data to 70% train and 30% test

def train_test_split(read_from,write_to_train,write_to_test,thres,acct_id, path):
    """

    :rtype: object
    """
    data = pd.DataFrame()
    data2 = pd.DataFrame()
    count = 0
    customers = 0
    for i in os.listdir(read_from):
        print('readcsv ' + str(i))
        dataset = pd.read_csv(r''+read_from+'/' + str(i), sep=',', index_col=0)
        if(len(dataset)!=0):
            dataset=dataset.sample(frac=1)
            count = 0
            customers = customers + 1
            n_payments = len(dataset['payment_id'].unique())

            for j in dataset['payment_id'].unique():
                count = count + 1
                if(count == 1):
                    data = data.append(dataset[dataset['payment_id'] == j], ignore_index=True)
                else:
                    if (count / n_payments) <= thres:
                        data = data.append(dataset[dataset['payment_id'] == j], ignore_index=True)
                    else:
                        data2 = data2.append(dataset[dataset['payment_id'] == j], ignore_index=True)
            
            train_payment = data.payment_id.nunique()
            train_size = data.shape[0]
            data.to_csv(write_to_train, index=False)
            test_payment = data2.payment_id.nunique()
            test_size = data2.shape[0]
            data2.to_csv(write_to_test, index=False)
            final_report = pd.read_csv(path+"/account_"+acct_id+"/summary.csv")
            final_report['Train_Size'] = train_size
            final_report['Test_Size'] = test_size
            final_report['Payments_In_Train'] = train_payment
            final_report['Payments_In_Test'] = test_payment
            final_report.to_csv(path+"/account_"+acct_id+"/summary.csv",index = False)

 
if __name__ == '__main__':
    acct_id = str(sys.argv[1])
    path = str(sys.argv[2])
    thres = float(str(sys.argv[3]))
    read_from = path+"/account_"+acct_id+"/subsets_rolled_up/"
    write_to_train = path+"/account_"+acct_id+"/train_test_splitted/train_70.csv"
    write_to_test = path+"/account_"+acct_id+"/train_test_splitted/test_30.csv"
    log_path=path+'/account_'+str(acct_id)+'/logs/'
    
    #splitting into train-70%, test-30% ratio
    train_test_split(read_from,write_to_train,write_to_test,thres, acct_id, path)
    progress=pd.read_csv(log_path+"progress.csv")
    progress['Status']='Train Test Split Operation'
    progress.to_csv(log_path+"progress.csv",index=False)
