import pandas as pd
import numpy as np
import os
import json
import logging
import sys
logging.basicConfig(filename='logs.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

#amount = invoice amount

#This function removes payments having single subset
def removing_single_subsets(read_from, write_to):
    for i in os.listdir(read_from):
        logging.info('Processing '+str(i).split('.')[0])
        dataset = pd.read_csv(r'' + read_from + os.sep + str(i), sep=',')
        temp = dataset.groupby(['payment_id'])['subset_number'].nunique().reset_index()
        valid_payments = temp[temp['subset_number']>1]['payment_id'].unique()
        dataset = dataset[dataset['payment_id'].isin(valid_payments)]
        if len(dataset)>1:
            logging.info('Customer have multiple subsets.')
            dataset.to_csv(write_to + str(i))
        else:
            logging.info('Customer ' + str(i).split('.')[0] + ' removed due to all single subsets')


#This function will populate output
def output_logic(read_from, write_to):
    for i in os.listdir(read_from):
        dataset = pd.read_csv(r'' + read_from + os.sep + str(i), sep=',',index_col=0)
        temp=dataset.copy()
        temp['diff_payment_id_&_payment_hdr_id'] = abs(temp['payment_id'] - temp['payment_hdr_id'])
        agg_obj = temp[['payment_id', 'subset_number', 'diff_payment_id_&_payment_hdr_id', 'payment_hdr_id']].groupby(['payment_id', 'subset_number']).agg({'payment_hdr_id':'var','diff_payment_id_&_payment_hdr_id':'sum'}).reset_index()
        agg_obj['payment_hdr_id']=agg_obj['payment_hdr_id'].fillna(0)
        dataset['output'] = temp.apply(lambda row: 1 if ((agg_obj[(agg_obj['payment_id'] == row['payment_id']) & (
                                                                     agg_obj['subset_number'] == row['subset_number'])][
                                                                 'diff_payment_id_&_payment_hdr_id'].values[
                                                                 0] == 0) & (agg_obj[(agg_obj['payment_id'] == row['payment_id']) & (agg_obj['subset_number'] == row['subset_number'])]['payment_hdr_id'].values[
                                                                                 0] == 0)) else 0, axis=1)
        dataset.to_csv(write_to + str(i))


#This function assigns rank to average_delay_categorical feature
#This function calculates the variance of delay of all invoices in each subset
def variance_and_avg_delay_categorical(read_from, write_to):
    for i in os.listdir(read_from):
        dataset = pd.read_csv(r'' + read_from + os.sep + str(i), sep=',',index_col=0)
        dataset['payment_date'] = pd.to_datetime(dataset['payment_date'])
        dataset['invoice_date'] = pd.to_datetime(dataset['invoice_date'])
        dataset['delay'] = dataset['payment_date'].subtract(dataset['invoice_date'], axis=0)
        dataset['delay'] = dataset['delay'].apply(lambda x: pd.Timedelta(x).days)

        temp = dataset.groupby(['payment_id', 'subset_number']).agg({'delay':'var'}).reset_index()
        temp['delay']=temp['delay'].fillna(0)
        temp['delay']=round(temp['delay'],5)
        temp['variance_categorical']=temp.groupby(['payment_id'])['delay'].rank(ascending=True,method='dense')
        dataset['variance']=dataset.apply(lambda row : temp[(temp['payment_id']==row['payment_id']) & (temp['subset_number']==row['subset_number'])]['delay'].values[0],axis=1)
        dataset['variance_categorical']=dataset.apply(lambda row : temp[(temp['payment_id']==row['payment_id']) & (temp['subset_number']==row['subset_number'])]['variance_categorical'].values[0],axis=1)
        dataset['variance_categorical'] = dataset['variance_categorical'] - 1

        temp2 = dataset.groupby(['payment_id', 'subset_number']).agg({'delay':'mean'}).reset_index()
        temp2['delay'] = round(temp2['delay'],5)
        temp2['avg_delay_categorical'] = temp2.groupby(['payment_id'])['delay'].rank(ascending=False,method='dense')
        dataset['average_delay'] = dataset.apply(lambda row : temp2[(temp2['payment_id']==row['payment_id']) & (temp2['subset_number']==row['subset_number'])]['delay'].values[0],axis=1)
        dataset['avg_delay_categorical']=dataset.apply(lambda row : temp2[(temp2['payment_id']==row['payment_id']) & (temp2['subset_number']==row['subset_number'])]['avg_delay_categorical'].values[0],axis=1)
        dataset['avg_delay_categorical'] = dataset['avg_delay_categorical'] - 1
        dataset.to_csv(write_to + str(i))


#This function will create the feature number_invoices_closed
def number_unique_invoice_count(read_from, write_to):
    for i in os.listdir(read_from):
        dataset = pd.read_csv(r'' + read_from + os.sep + str(i), sep=',', index_col=0)
        dataset['number_invoices_closed']=0
        subset_invoice_count = dataset.groupby(['payment_id','subset_number'])['invoice'].nunique().reset_index()
        payment_invoice_count = dataset.groupby('payment_id')['invoice'].nunique().reset_index()
        dataset['number_invoices_closed'] = dataset.apply(lambda row: (subset_invoice_count[(subset_invoice_count['payment_id']==row['payment_id']) & (subset_invoice_count['subset_number']==row['subset_number'])]['invoice'].values[0]/payment_invoice_count[payment_invoice_count['payment_id']==row['payment_id']]['invoice'].values[0]), axis=1)
        dataset['unique_invoice_count'] = dataset.apply(lambda row: payment_invoice_count[payment_invoice_count['payment_id']==row['payment_id']]['invoice'].values[0],axis=1)
        dataset.to_csv(write_to + str(i))


def variance_categorical_old(read_from, write_to):
    for i in os.listdir(read_from):
        dataset = pd.read_csv(r'' + read_from + os.sep + str(i), sep=',',index_col=0)
        if len(dataset)>1:
            # dataset['subset_number'] = dataset.index
            dataset['payment_date'] = pd.to_datetime(dataset['payment_date'])
            dataset['invoice_date'] = pd.to_datetime(dataset['invoice_date'])
            dataset['delay'] = dataset['payment_date'].subtract(dataset['invoice_date'], axis=0)
            dataset['delay'] = dataset['delay'].apply(lambda x: pd.Timedelta(x).days)
            group = dataset.groupby(by=['payment_id', 'subset_number'])
            new_data = pd.DataFrame()
            for name, data in group:
                payment_id, subset_number = name[0], name[1]
                data['output'] = 0
                if len(data) > 1:
                    data['variance'] = data['delay'].var()
                    if data['payment_hdr_id'].unique()[0] == payment_id and data['payment_hdr_id'].var() == 0:
                        data['output'] = 1
                else:
                    data['variance'] = 0
                    if data['payment_hdr_id'].unique()[0] == payment_id:
                        data['output'] = 1
                data['average_delay'] = data['delay'].mean()
                logging.info("here", name)
                data['payment_id'] = payment_id
                data['subset_number'] = subset_number
                new_data = pd.concat([new_data, data])
                logging.info("here 2", name)
            grouped_payment_id = new_data.groupby(by=['payment_id'])
            new_data['variance_categorical'] = 0
            new_data_more_new = pd.DataFrame()
            for name, data in grouped_payment_id:
                unique_variance = data['variance'].unique()
                logging.info(unique_variance)
                data['payment_id'] = name
                unique_variance.sort()
                data['variance_categorical'] = data['variance'].apply(rank, args=(unique_variance,))
                new_data_more_new = pd.concat([new_data_more_new, data])
                new_data_more_new.to_csv(write_to + str(i))


# This function defines and assigns bins to the open invoice amount for each subset
def LMH_assign(val):
    if val<=100:
        return 'L1'
    elif val<=500:
        return 'L2'
    elif val<=1000:
        return 'L3'
    elif val<=5000:
        return 'M'
    else:
        return 'H'


#This function calculates LMH cumulative
def LMH_cumulative(read_from,write_to, customer_level_json):
    local = json.load(open(customer_level_json, 'r'))
    for i in os.listdir(read_from):
        dataset = pd.read_csv(r'' + read_from + os.sep + str(i), sep=',',index_col=0)
        cust_num=dataset['customer_number'].unique()[0]
        if type(cust_num)!=str:
            cust_num=cust_num.astype(str)
        dataset['LMH'] = dataset['amount'].apply(LMH_assign)
        # dataset['LMH'] = pd.cut(dataset['amount'], [0, 100, 500, 1000, 5000, 10000000000], labels=['L1', 'L2', 'L3', 'M', 'H'])
        if local.get(cust_num).get('buckets_invoice').get('L1_perc'):
            L1_perc = dataset['L1_perc'] = local.get(cust_num).get('buckets_invoice').get('L1_perc')[0]
        else:
            L1_perc = dataset['L1_perc'] = 0
        if local.get(cust_num).get('buckets_invoice').get('L2_perc'):
            L2_perc = dataset['L2_perc'] = local.get(cust_num).get('buckets_invoice').get('L2_perc')[0]
        else:
            L2_perc = dataset['L2_perc'] = 0
        if local.get(cust_num).get('buckets_invoice').get('L3_perc'):
            L3_perc = dataset['L3_perc'] = local.get(cust_num).get('buckets_invoice').get('L3_perc')[0]
        else:
            L3_perc = dataset['L3_perc'] = 0
        if local.get(cust_num).get('buckets_invoice').get('M_perc'):
            M_perc = dataset['M_perc'] = local.get(cust_num).get('buckets_invoice').get('M_perc')[0]
        else:
            M_perc = dataset['M_perc'] = 0
        if local.get(cust_num).get('buckets_invoice').get('H_perc'):
            H_perc = dataset['H_perc'] = local.get(cust_num).get('buckets_invoice').get('H_perc')[0]
        else:
            H_perc = dataset['H_perc'] = 0
        dataset['LMH_cumulative']=0
        for j in dataset['payment_id'].unique():
            for k in dataset[dataset['payment_id']==j]['subset_number'].unique():
                temp=dataset[(dataset['payment_id']==j) & (dataset['subset_number']==k)]
                L1 = len(temp[temp['LMH'] == 'L1'])/len(temp)
                L2 = len(temp[temp['LMH'] == 'L2']) / len(temp)
                L3 = len(temp[temp['LMH'] == 'L3']) / len(temp)
                M =  len(temp[temp['LMH'] == 'M']) / len(temp)
                H =  len(temp[temp['LMH'] == 'H']) / len(temp)
                dataset.loc[(dataset['payment_id'] == j) & (dataset['subset_number'] == k), 'LMH_cumulative']=L1*L1_perc + L2*L2_perc + L3*L3_perc + M*M_perc + H*H_perc
        dataset.to_csv(write_to + str(i))

#This function populates quarter level payment and invoice level features AND avg_of_all_delays feature
def quarter_level_features(read_from, write_to, customer_level_json):
    for i in os.listdir(read_from):
        dictionary = json.load(open(customer_level_json, 'r'))
        dataset = pd.read_csv(r''+read_from + os.sep + str(i), sep=',', index_col=0)
        cust_num = dataset['customer_number'].unique()[0]
        if type(cust_num)!=str:
            cust_num=cust_num.astype(str)

        dataset['avg_of_invoices_closed'] = dictionary.get(cust_num).get('avg_of_invoices_closed')
        dataset['avg_of_all_delays'] = dictionary.get(cust_num).get('avg_of_all_delays')

        dataset['payment_count_quarter_q1'] = dictionary.get(cust_num).get('payment_count_quarter').get('q1')
        dataset['payment_count_quarter_q2'] = dictionary.get(cust_num).get('payment_count_quarter').get('q2')
        dataset['payment_count_quarter_q3'] = dictionary.get(cust_num).get('payment_count_quarter').get('q3')
        dataset['payment_count_quarter_q4'] = dictionary.get(cust_num).get('payment_count_quarter').get('q4')
        dataset['invoice_count_quarter_q1'] = dictionary.get(cust_num).get('invoice_count_quarter').get('q1')
        dataset['invoice_count_quarter_q2'] = dictionary.get(cust_num).get('invoice_count_quarter').get('q2')
        dataset['invoice_count_quarter_q3'] = dictionary.get(cust_num).get('invoice_count_quarter').get('q3')
        dataset['invoice_count_quarter_q4'] = dictionary.get(cust_num).get('invoice_count_quarter').get('q4')

        dataset['avg_of_invoices_closed'] = dictionary.get(cust_num).get('avg_of_invoices_closed')
        dataset['avg_of_all_delays'] = dictionary.get(cust_num).get('avg_of_all_delays')

        dataset.to_csv(write_to + str(i))

#supporting function for avg_delay_categorical
def rank(x, un_var):
    index = np.where(un_var == x)
    return index[0][0]


def avg_delay_categorical_old(read_from,write_to):
    for i in os.listdir(read_from):
        data = pd.read_csv(r''+read_from + os.sep + str(i), sep=',',index_col=0)
        if len(data)>1:
            grouped_payment_id = data.groupby(by=['payment_id'])
            new_data_final=pd.DataFrame()
            for name,group in grouped_payment_id:
                unique_delay = group['average_delay'].unique()
                sorted_delay = np.asarray(sorted(unique_delay, reverse=True))
                group['avg_delay_categorical'] = group['average_delay'].apply(rank, args=(sorted_delay,))
                new_data_final = pd.concat([new_data_final, group], axis=0)
            new_data_final.to_csv(write_to + str(i))


#This function rolls up the payment information to the subsets level
def subset_rolled_up(read_from, write_to):
    variables = ['account_id','customer_number', 'payment_id', 'subset_number', 'output',
    'variance_categorical','avg_delay_categorical',
    'L1_perc', 'L2_perc', 'L3_perc', 'M_perc', 'H_perc',
    'LMH_cumulative',
    'avg_of_invoices_closed',
    'avg_of_all_delays',
    'payment_count_quarter_q1', 'payment_count_quarter_q2', 'payment_count_quarter_q3',
    'payment_count_quarter_q4',
    'invoice_count_quarter_q1', 'invoice_count_quarter_q2', 'invoice_count_quarter_q3',
    'invoice_count_quarter_q4', 'payment_amount', 'number_invoices_closed',
    'payment_date','unique_invoice_count']
    for i in os.listdir(read_from):
        final = pd.DataFrame()
        dataset = pd.read_csv(r''+read_from + os.sep + str(i), sep=',', index_col=0)
        for j in dataset['payment_id'].unique():
            for k in dataset[dataset['payment_id'] == j]['subset_number'].unique():
                temp = dataset[(dataset['payment_id'] == j) & (dataset['subset_number'] == k)][variables]
                final = final.append(temp.iloc[[0]], ignore_index=True)

        final['subset_count']=final.groupby(['payment_id'])['payment_id'].transform('count')
        final=final[final['subset_count']>=1]
        final.to_csv(write_to + str(i))


def create_all_features(read_from_, read_from, write_to, write_to_, customer_level_json):
    logging.info('Feature Creation Started.')

    #Removing Single Subsets
    logging.info('Removing Single Subsets..')
    removing_single_subsets(read_from_, write_to)
    logging.info('Single Subsets Removed.')

    #Output Logic
    logging.info('Output Variable getting created.')
    output_logic(read_from, write_to)
    logging.info('Output Variable got created.')

    #Variance and Average Categorical Feature
    logging.info('Avg delay categorical started.')
    variance_and_avg_delay_categorical(read_from, write_to)
    logging.info('Avg delay categorical Finished.')

    #Number of invoices in a subset divided by the total open invoices
    logging.info('unique invoice count started.')
    number_unique_invoice_count(read_from, write_to)
    logging.info('unique invoice count finished.')

    # LMH_cumulative
    logging.info('LMH Started')
    LMH_cumulative(read_from,write_to, customer_level_json)
    logging.info('LMH Finished')

    #quarter_level
    logging.info('Quarter Features Started.')
    quarter_level_features(read_from, write_to, customer_level_json)
    logging.info('Quarter Features Finished.')
    logging.info('Feature Generation Ended.')

    #Subset Filtering and rolling up
    logging.info('Subsets Rolling up.')
    subset_rolled_up(read_from, write_to_)
    logging.info('Subsets Rolled up.')


def prepare_predictions_data(read_from, write_to):
    predictions_data = pd.DataFrame()
    for i in os.listdir(read_from):
        data = pd.read_csv(r'' + read_from + os.sep + str(i), sep=',',index_col=0)
        predictions_data = predictions_data.append(data)
    predictions_data.to_csv(write_to)


if __name__ == '__main__':
    acct_id = str(sys.argv[1])
    path = str(sys.argv[2])
    read_from_ = path+"/account_"+acct_id+"/customer_wise_subsets"
    read_from = path+"/account_"+acct_id+"/customer_subsets_features"
    write_to = path+"/account_"+acct_id+"/customer_subsets_features/"
    write_to_ = path+"/account_"+acct_id+"/subsets_rolled_up/"
    customer_level_json = path+"/account_"+acct_id+"/customersJson/customersJson.json"
    log_path=path+'/account_'+str(acct_id)+'/logs/'

    #create features for all subsets
    create_all_features(read_from_, read_from, write_to, write_to_, customer_level_json)
    
    progress=pd.read_csv(log_path+"progress.csv")
    progress['Status']='FeaturesCreation.py'
    progress.to_csv(log_path+"progress.csv",index=False)
