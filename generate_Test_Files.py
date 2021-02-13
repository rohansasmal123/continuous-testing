import pandas as pd
import os
import json
import logging
import sys
logging.basicConfig(filename = 'logs.log')

#This function will create customer_level_features.csv
def customer_level_features(read_from_train, read_from_test, write_to):
    customer_level_features = ['customer_number',
                               'avg_of_all_delays',
                               'avg_of_invoices_closed',
                               'payment_count_quarter_q1',
                               'payment_count_quarter_q2',
                               'payment_count_quarter_q3',
                               'payment_count_quarter_q4',
                               'invoice_count_quarter_q1',
                               'invoice_count_quarter_q2',
                               'invoice_count_quarter_q3',
                               'invoice_count_quarter_q4',
                               'L1_perc',
                               'L2_perc',
                               'L3_perc',
                               'M_perc',
                               'H_perc'
                               ]

    data = pd.read_csv(r''+read_from_train)
    data2 = pd.read_csv(r''+read_from_test)

    customer_level_dataset=pd.DataFrame()
    dataset = data[customer_level_features].append(data2[customer_level_features], ignore_index=True)
    for i in dataset['customer_number'].unique():
        customer_level_dataset = customer_level_dataset.append(dataset[dataset['customer_number'] == i][customer_level_features].iloc[0], ignore_index=True)

    customer_level_dataset.rename(columns={'customer_number':'object_value','H_perc':'H','L1_perc':'L1','L2_perc':'L2','L3_perc':'L3','M_perc':'M'},inplace=True)
    if customer_level_dataset['object_value'].dtype==float:
        customer_level_dataset['object_value'] = customer_level_dataset['object_value'].astype('int')
    customer_level_dataset.to_csv(write_to,index=False)
    logging.warning('Customer Level Features Created.')


def subsets_json(subset):
    list=[]
    for invoice_number in subset['invoice_number']:
        invoice=subset[subset['invoice_number']==invoice_number]
        list.append({"invoice_amount":invoice['invoice_amount'].values[0],"invoice_number":str(invoice_number).split('.')[0],"invoice_date":invoice['invoice_date'].values[0]})
    return list


def raw_json_creation(read_from, read_from_subsets,write_to):

    data = pd.read_csv(r''+read_from) # raw csv
    subsets_predictions = pd.read_csv(read_from_subsets) # predictions

    temp = data.groupby('payment_id').agg({'customer_number':'nunique'}).reset_index()
    invalid_payments = temp[temp['customer_number'] > 1]['payment_id'].unique()
    print(invalid_payments)
    data = data[~data['payment_id'].isin(invalid_payments)]

    top_header = data.groupby('payment_id')[['customer_number','unique_invoice_count','payment_amount','payment_date']].max().reset_index()
    payments = []
    predictions = pd.DataFrame()

    for index, row in top_header.iterrows():
       if row['payment_id'] not in subsets_predictions['payment_id'].unique():
           print('payment' + str(row['payment_id']) + 'not found')
           continue

       subset_dict = {}
       subset_dict["customer_number"] = str(row['customer_number'])
       subset_dict["unique_invoice_count"] = row['unique_invoice_count']
       subset_dict["payment_amount"] = row['payment_amount']
       subset_dict["primaryKey"]=int(row['payment_id'])
       subset_dict["payment_date"]=str(row['payment_date'])
       subset_dict["items"] = []
       items  = []
       subsets = data[data['payment_id']==row['payment_id']]
       for subset_number in subsets['subset_number'].unique():
           abc = len(subsets_predictions[(subsets_predictions['payment_id']==row['payment_id']) & (subsets_predictions['subset_number']==subset_number)])
           if abc == 0:
               # print('subset ' + str(subset_number) + ' not found for payment ' + str(row['payment_id']))
               continue
           items.append({"subsetId":int(subset_number),"subset":subsets_json(subsets[subsets['subset_number']==subset_number])})
           predictions=predictions.append(subsets_predictions[(subsets_predictions['payment_id']==row['payment_id']) & (subsets_predictions['subset_number']==subset_number)],ignore_index=True)
       subset_dict['items']=items
       payments.append(subset_dict)
    final_json={"data":payments}
    write_to_ = write_to+'raw_data_json.json'
    with open(write_to_, 'w') as fp:
        json.dump(final_json, fp)
    predictions.rename(columns={'output':'actual','predictions':'output','H_perc':'H','L1_perc':'L1','L2_perc':'L2','L3_perc':'L3','M_perc':'M','pred_proba_0':'probability(0)','pred_proba_1':'probability(1)'},inplace=True)
    predictions.to_csv(write_to+'raw_predictions.csv',index=False)


#This function will divide the data to 70% train and 30% test
def rivana_testing(read_from, testing_data_path, write_to_raw_csv):
    raw_fields = ['customer_number','payment_id','payment_amount','payment_date','invoice_number','invoice_amount','invoice_date','subset_number','unique_invoice_count','output']
    raw_csv = pd.DataFrame()
    testing_data = pd.read_csv(testing_data_path)
    testing_payment_ids = testing_data['payment_id'].unique()
    for i in os.listdir(read_from):
        dataset = pd.read_csv(r''+read_from+'/' + str(i), sep=',', index_col=0)
        dataset['invoice_number']=dataset['invoice']
        dataset['invoice_amount']=dataset['amount']
        dataset=dataset[dataset['payment_id'].isin(testing_payment_ids)]
        raw_csv=raw_csv.append(dataset[raw_fields],ignore_index=True)
    raw_csv.to_csv(write_to_raw_csv)


if __name__ == '__main__':
    acct_id = str(sys.argv[1])
    #path = "/root/accounts"
    path = str(sys.argv[2])
    read_from = path+'/account_'+acct_id+'/customer_subsets_features'
    testing_data_path = path+"/account_"+acct_id+"/train_test_splitted/test_30.csv"
    write_to_raw_csv = path+"/account_"+acct_id+"/rivana_test/raw_data.csv"

    
    # This function will generate all the Rivana Testing Files
    rivana_testing(read_from, testing_data_path, write_to_raw_csv)

    read_from = write_to_raw_csv
    write_to = path+"/account_"+acct_id+"/rivana_test/"
    read_from_subsets = path+"/account_"+acct_id+"/predictions/predictions.csv"
    # This function will create JSON input files
    raw_json_creation(read_from, read_from_subsets, write_to)

    # This will generate aggregate file of customer level features
    read_from_train = path+"/account_"+acct_id+"/train_test_splitted/train_70.csv"
    read_from_test = path+"/account_"+acct_id+"/train_test_splitted/test_30.csv"
    write_to = path+'/account_'+acct_id+'/rivana_test/customer_level_features.csv'

    #This will generate aggregate file of customer level features
    customer_level_features(read_from_train, read_from_test,write_to)
