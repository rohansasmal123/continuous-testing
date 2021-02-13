import json
import multiprocessing as mp
import time
from json import JSONDecodeError
from multiprocessing.pool import ThreadPool
import pandas as pd
import numpy as np
import os
import logging
import sys
import glob
logging.basicConfig(filename='logs.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


#It will create subsets and put them into JSON
def subsets_creation(lists, n, subset, sum, li, iot, l):
    try:

        if sum == round(float(0), 5):
            subset = subset[:-1]
            subset = '[' + subset + ']'
            li.append(subset)
            return

        if n == 0:
            return
        if lists[0][n - 1] <= round(float(sum), 5):
            subsets_creation(lists, n - 1, subset, sum, li, iot, l)
            subsets_creation(lists, n - 1,
                   subset + '{"amount":' + str(lists[0][n - 1]) + ',"invoice":"' + str((lists[1][n - 1])) +
                   '","invoice_date":"' + str(lists[3][n - 1]) + '","due_date":"' + str(lists[2][n - 1]) +
                   '","payment_hdr_id":"' + str(lists[4][n - 1]) + '","account_id":"' + str(lists[5][n - 1])
                             +'"},',
                   round(float(sum - lists[0][n - 1]), 2), li, iot, l)
        else:
            subsets_creation(lists, n - 1, subset, sum, li, iot, l)
    except Exception as e:
        logging.info(str(e))


#This function puts the payment information into the JSON
def subsets_to_json(i, data_, max_num_of_open_invoices_):

    logging.info("subsets for " + str(i) + " payment of size " +str(len(data_)))
    try:
        if len(data_) > max_num_of_open_invoices_:
            logging.info('Too many open invoices.. ' + str(len(data_)))

            raise Exception("Too many rows", len(data_))
        # when no of payments done by a perticular customer is greater then max number of open invoices then this error will show
        payment = data_[data_['payment_hdr_id'] == i]['payment_amount'].unique()[0]
        #here only those payments which closing the invoice that we are putting into payments
        payment_date = data_[data_['payment_hdr_id'] == i]['effective_date'].unique()[0]
        # here only those payments which closing the invoices for those effective date we are putting into payment_date
        customer_name = data_[data_['payment_id'] == i]['customer_number_norm'].unique()[0]
        # here only those payments which closing the invoice for those customer_number_norm are putting into customer_name
        datas = data_[data_['payment_id'] == i].loc[:, ['invoice_amount_norm', 'invoice_number_norm', 'due_date_norm',
                                                        'invoice_date_norm', 'payment_hdr_id','acct_doc_header_id']].values
        amounts = datas[:, 0].tolist()
        amounts = [round(float(x), 5) for x in amounts]
        invoices = datas[:, 1].tolist()
        due_dates = datas[:, 2].tolist()
        invoice_dates = datas[:, 3].tolist()
        payment_hdr_id = datas[:, 4].tolist()
        account_id = datas[:, 5].tolist()
        data_inFunction = [amounts, invoices, due_dates, invoice_dates, payment_hdr_id, account_id]
        currents = '{"customer_name":"' + str(customer_name) + '","payment_id":"' + str(
            i) + '","payment_amount":' + str(
            payment) + ',"payment_date":"' + str(payment_date) + '","subsets":['
        current = ''
        li = []
        subsets_creation(data_inFunction, len(amounts), current, payment, li, i, payment)
        s = ",".join(li)
        s = currents + s
        s = s[:-1]
        if len(li) > 0:
            s = s + ']]}'
        else:
            s = s + '[]}'
        try:
            dic = json.loads(s)
        except JSONDecodeError:
            logging.info("s = ", s)
        logging.info("Ended for " + str(i) + ' payment')
        return dic
    except Exception as e:
        logging.info(str(e) +" This error occurred for payment id ", i)
        return {}


#This function reduces the dataframe size by a significant amount
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    logging.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    logging.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logging.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


#This function filters the payment with less than 30 open invoices and call the subset_creation() function
def filtering_payment_subset_creation(read_path, write_path, customersJson, mode_flag, max_num_of_open_invoices, number_of_process):
    try:

        logging.info('Subsets Creation Started.')

        columns = ['invoice_amount_norm', 'payment_amount', 'effective_date', 'invoice_date_norm', 'due_date_norm',
                   'payment_method', 'payment_hdr_id', 'payment_id', 'invoice_number_norm', 'customer_number_norm']

        temp = pd.read_csv(r''+read_path, sep=',')
        valid_customers = json.load(open(customersJson, 'r'))
        valid_customers = valid_customers.keys()
        logging.info(str(len(temp['payment_id'].unique())) + ' Initial Payments')

        if temp['customer_number_norm'].dtype == np.float64:
            temp['customer_number_norm'] = temp['customer_number_norm'].astype(np.int64)
        temp['customer_number_norm'] = temp['customer_number_norm'].astype(str)
        temp['payment_date'] = pd.to_datetime(temp['effective_date'])
        temp['invoice_date'] = pd.to_datetime(temp['invoice_date_norm'])
        temp['delay'] = temp['payment_date'].subtract(temp['invoice_date'], axis=0)
        temp['delay'] = temp['delay'].apply(lambda x: pd.Timedelta(x).days)
        #Till here we have calculated only the delay.
        temp['invoice_number_norm'] = temp['invoice_number_norm'].astype(str)

        for i in temp['customer_number_norm'].unique():

            # if os.path.exists(write_path + str(i) + '.csv'):
            #     continue

            logging.info('Subset(s) for ' +  str(i) + ' customer')

            if (i not in valid_customers) & (mode_flag == 'predictions'):
                print(i)
                continue
#here we are only taking those customers who have actually cleared any payments.or else we can say(payment_id==payment_hrd_id)
            data = temp[temp['customer_number_norm'] == i]
            logging.info(str(len(data['payment_id'].unique())) + ' number of payments for customer ' + str(i))
            data = data[data['payment_amount'] >= data['invoice_amount_norm']]
#for that perticular customer if the payment_amount >= invoice _amount then we will consider.
            if len(data) >= 1:
                pool = mp.Pool(processes = number_of_process)
                transformed_dataframe = pd.DataFrame()
                payment_groups = data.groupby(by='payment_id')
                start = time.time()
                results = [pool.apply_async(subsets_to_json, args=(payment_ids, payments, max_num_of_open_invoices))
                           for payment_ids, payments in payment_groups]
                #here we are only taking three arguments and from them we will make the subsets.
                results = [res.get() for res in results]
                for r in results:
                    if len(r.keys()) > 0:
                        f = r['subsets']
                        if 0 < len(f) <= 5000:
                            payment_ids = r['payment_id']
                            payment_amount = r['payment_amount']
                            customer_name = r['customer_name']
                            payment_date = r['payment_date']
                            gr = pd.DataFrame()
                            for h, sub in enumerate(f):
                                for eachInvoice in sub:
                                    e = pd.DataFrame(eachInvoice, index=[h])
                                    e = pd.concat([e, pd.DataFrame({"payment_id": payment_ids}, index=[h]),
                                                   pd.DataFrame({"payment_amount": payment_amount}, index=[h]),
                                                   pd.DataFrame({"payment_date": payment_date}, index=[h]),
                                                   pd.DataFrame({"customer_number": customer_name}, index=[h])], axis=1)
                                    gr = pd.concat([gr, e])

                            transformed_dataframe = pd.concat(
                                [transformed_dataframe,
                                 gr]
                                , ignore_index=False)

                end = time.time()
                logging.info("------ Took {0} seconds-----".format(end - start))
                pool.close()
                pool.join()

                if(len(transformed_dataframe)>0):
                    transformed_dataframe['subset_number'] = transformed_dataframe.index
                    if os.path.exists(write_path + str(i) + ".csv"):
                        customers_old_data = pd.read_csv(write_path + str(i) + ".csv")
                    else:
                        customers_old_data = pd.DataFrame()

                    transformed_dataframe = customers_old_data.append(transformed_dataframe,ignore_index=True)
                    transformed_dataframe.to_csv(write_path + str(i) + ".csv",index=False)

        logging.info('Subset Creation Done.')
        

    except Exception as e:
        logging.info('EXCEPTION OCCURED')
        logging.info(str(e))


def count_subset(acct_id, path):
    count = len(glob.glob(path+'/account_'+acct_id+'/customer_wise_subsets/*'))
    final_report = pd.read_csv(path+'/account_'+acct_id+'/summary.csv')
    final_report['Num_Of_Subset'] = count
    final_report.to_csv(path+'/account_'+acct_id+'/summary.csv', index = False)
    


if __name__ == '__main__':

    mode_flag = 'retraining'
    max_num_of_open_invoices = 25
    number_of_process = 1
    acct_id = str(sys.argv[1])
    path = str(sys.argv[2])
    read_path = path+"/account_"+acct_id+'/history_generated/history.csv'
    write_path = path+"/account_"+acct_id+"/customer_wise_subsets/"
    customersJson = path+"/account_"+acct_id+"/customersJson/customersJson.json"
    log_path=path+'/account_'+str(acct_id)+'/logs/'

    #This will create subsets
    filtering_payment_subset_creation(read_path,write_path, customersJson, mode_flag, max_num_of_open_invoices, number_of_process)
    count_subset(acct_id, path)
    progress=pd.read_csv(log_path+"progress.csv")
    progress['Status']='SubsetsCreation.py'
    progress.to_csv(log_path+"progress.csv",index=False)