import pandas as pd
import time
import os
import multiprocessing as mp
import traceback
import logging
import sys

logging.basicConfig(filename='logs.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


#This will merge old and new history
def merge_old_new_history(old_history_path, new_history_path):
    old_history = pd.read_csv(old_history_path)
    new_history = pd.read_csv(new_history_path)
    return old_history.append(new_history,ignore_index=True).to_csv(old_history_path)

# partial payments removal
def removing_partial_payments(dataset,acct_id):
    #logging.info(str(len(dataset['payment_hdr_id'].unique())) + ' Initial Payments')
    initial_payment = str(len(dataset['payment_hdr_id'].unique()))
    if dataset['invoice_amount_norm'].isnull().any()==True:
        dataset=dataset[dataset['invoice_amount_norm'].notnull()]

    if dataset['payment_hdr_id'].isnull().any()==True:
        dataset=dataset[dataset['payment_hdr_id'].notnull()]

    #valid_payment = str(len(dataset['payment_hdr_id'].unique()))
    #Here showing the unique payments counts.
    dataset['invoice_amount_norm']=dataset['invoice_amount_norm'].astype('float64')
    temp = dataset.groupby('payment_hdr_id').agg({'invoice_amount_norm':'sum','payment_amount':'max'}).reset_index()
    #Here we are taking a three column 1) is payment_header_id 2)sum of the invoice_amount_norm 3)maximum payment_amount
    valid_payments = temp[temp['payment_amount']==temp['invoice_amount_norm']]['payment_hdr_id'].unique()
    #here if payment_amount == to sum of the invoice amount_norm then it is a valid payment.
    dataset = dataset[dataset['payment_hdr_id'].isin(valid_payments)]
    return dataset, initial_payment


# To find the open invoices that are before the payment date
def filter_before_payment_date(customer_name, data, recent_payments_list, mode_flag):
    try:
        sdk = pd.DataFrame()
        unique_payments = data['payment_hdr_id'].unique()
        # if len(unique_payments)>25:
        #     logging.info('Customer ' + str(customer_name) + ' with payment count ' + str(len(unique_payments)) + ' has been removed.')
        #     return sdk.reset_index(drop=True)
        logging.info('Preparing history for customer ' + str(customer_name) + ' with payment count '+str(len(unique_payments)))
        for payment in unique_payments:
            #for loop where payment is acting as i and unique_payments is the source of data.so in each itration payment will have the values of unique payments.
            if (mode_flag == 'predictions') & (payment not in recent_payments_list):
                continue
            payment_date = data[data['payment_hdr_id'] == payment].iloc[0]['effective_date']
            #here we are making payment date for those payment whose payment_hrd_id is == payment
            #If payment_hrd_id==payment to fir effective date/invoice_payment_date directly assign to payment date
            open_invoices_data = data[((data['isOpen']==1) & (data['invoice_date_norm'] <= payment_date))
                                      | ((data['effective_date'] >= payment_date) & (data['invoice_date_norm'] <= payment_date))]
            #here we are making open_invoices_data so that when the payment comes then we can see how many invoices are open for that perticular customer.
            #for this reason we are considering two either or conditions :
            #1)jub payment hua uss din se just pehle kitne invoices khule the (so that is our isopen==1) &  payment jis din hua uss din se pehle invoice date norm hona chahiye
            #2)effectivedate/invoice_payment_date hamehsa equal hi honge bcz hum unhe pehle step mai same kar rahe hae & payment jis din hua uss din se pehle invoice date norm hona chahiye
            open_invoices_data['payment_id'] = payment
            open_invoices_data['customer_name'] = customer_name
            sdk = pd.concat([sdk, open_invoices_data.reset_index(drop=True)], ignore_index=True)
        return sdk.reset_index(drop=True)
    except Exception as e:
        print(e)


# Multiprocessing code for history generation
def history_generation(retraining_data_path,open_invoices_path, recent_payments_path, new_history_path,mode_flag, number_of_process,acct_id,path,log_path):
    logging.info('History Generation Started')
    try:
        error_flag = 0
        transformed_dataframe = pd.DataFrame()
        if mode_flag=='predictions':
            open_invoices_data = pd.read_csv(r''+ open_invoices_path, encoding='cp1256')
            recent_payments_data = pd.read_csv(r''+ recent_payments_path, encoding='cp1256')

            if open_invoices_data['payment_hdr_id'].isnull().any() == True:
                open_invoices_data = open_invoices_data[open_invoices_data['payment_hdr_id'].notnull()]

            open_invoices_data = open_invoices_data[open_invoices_data['invoice_number_norm'].notnull()]
            recent_payments_data = recent_payments_data[recent_payments_data['invoice_number_norm'].notnull()]

            if 'isOpen' not in open_invoices_data.columns:
                open_invoices_data['isOpen'] = open_invoices_data['isopen']
            if 'isOpen' not in recent_payments_data.columns:
                recent_payments_data['isOpen'] = recent_payments_data['isopen']
            recent_payments_data, initial_payment = removing_partial_payments(recent_payments_data, acct_id)
            valid_payment=str(len(recent_payments_data['payment_hdr_id'].unique()))
            hist = {'Account No.': [acct_id],
                    'Total valid payment(initially)': [initial_payment],
                    'Payments (after the partial payments are removed)': [valid_payment]}
            final_report = pd.DataFrame.from_dict(hist)
            final_report.to_csv(path+'/account_'+acct_id+'/summary.csv',index = False)

            dataset = open_invoices_data.append(recent_payments_data,ignore_index=True)
            recent_payments_list = recent_payments_data['payment_hdr_id'].unique()

        elif mode_flag=='retraining':
            dataset = pd.read_csv(r''+ retraining_data_path,encoding='cp1256')
            uni_cust = dataset.customer_number_norm.nunique()
            # encoding used for securing the data,so that properly all the data can be consumed.
            #cp1256 is a sheet where all the characters have some numerical value known as ASCII value.
            dataset = dataset[dataset['invoice_number_norm'].notnull()]
            if 'isOpen' not in dataset.columns:
                dataset['isOpen'] = dataset['isopen']
            dataset, initial_payment = removing_partial_payments(dataset, acct_id)
            valid_payment = str(len(dataset['payment_hdr_id'].unique()))
            hist = {'Account No.': [acct_id],
                    'Total valid payment(initially)': [initial_payment],
                    'Payments (after the partial payments are removed)': [valid_payment],
                    'Unique_Customer': [uni_cust]
                    }
            final_report = pd.DataFrame.from_dict(hist)
            final_report.to_csv(path+'/account_'+acct_id+'/summary.csv',index = False)


            recent_payments_list = dataset['payment_hdr_id'].unique()

        # Formatting dates
        dataset['effective_date'] = dataset['effective_date'].astype(str)
        dataset['effective_date'] = dataset['effective_date'].str.strip()
        dataset['effective_date'] = pd.to_datetime(dataset['effective_date'],format='%d-%m-%Y')

        # If invoice date is not present in the db use document date
        if dataset['invoice_date_norm'].isnull().all():
            # if document_date_norm is null
            if dataset['document_date_norm'].isnull().all():
                dataset['create_date'] = dataset['create_date'].astype(str)
                dataset['create_date'] = dataset['create_date'].str.strip()
                dataset['create_date'] = pd.to_datetime(dataset['create_date'],format='%d-%m-%Y')
                dataset['document_date_norm'] = dataset['create_date']
                dataset['document_date_norm'] = pd.to_datetime(dataset['document_date_norm'],format='%d-%m-%Y')
            dataset['document_date_norm'] = dataset['document_date_norm'].astype(str)
            dataset['document_date_norm'] = dataset['document_date_norm'].str.strip()
            dataset['document_date_norm'] = pd.to_datetime(dataset['document_date_norm'],format='%d-%m-%Y')
            dataset['invoice_date_norm'] = dataset['document_date_norm']
            dataset['invoice_date_norm'] =  pd.to_datetime(dataset['invoice_date_norm'],format='%d-%m-%Y')

        dataset['invoice_date_norm'] = pd.to_datetime(dataset['invoice_date_norm'],format='%d-%m-%Y')

        dataset['customer_number_norm'] = dataset['customer_number_norm'].astype(str)
        dataset['customer_number_norm'] = dataset['customer_number_norm'].str.strip()
        dataset = dataset[dataset['customer_number_norm'] != 'NULL']

        # if mode_flag=='retraining':
        # # removing the top customer
        #  uni = dataset['customer_number_norm'].value_counts().keys()
        #  dataset = dataset[dataset['customer_number_norm'] != uni[0]]

        dataset.drop_duplicates(subset=['invoice_number_norm','customer_number_norm'],keep='first',inplace=True)

        customer_groups = dataset.groupby(by=['customer_number_norm'])

        pool = mp.Pool(processes = number_of_process)
        start = time.time()
        results = [pool.apply_async(filter_before_payment_date, args=(customer_name_, data_, recent_payments_list, mode_flag)) for customer_name_, data_ in customer_groups]
        results = [res.get() for res in results]
        for r in results:
            transformed_dataframe = pd.concat([transformed_dataframe, r] , ignore_index=True)

        end = time.time()
        print("History Generation "+"------ Took {0} seconds -----".format(end - start))
        pool.close()
        pool.join()
        transformed_dataframe.to_csv(new_history_path, encoding='cp1256')
        logging.info('History Generation Done with ' + str(len(transformed_dataframe['payment_id'].unique()))+' payments.')
        progress=pd.read_csv(log_path+"progress.csv")
        progress['Status']='History Generation'
        progress.to_csv(log_path+"progress.csv",index=False)
    except Exception as e:
        logging.info(str(e))
        error_flag = 1
    finally:
        if error_flag == 1:
            transformed_dataframe.to_csv(new_history_path, encoding='cp1256')
            logging.info('History Generation Done with ' + str(len(transformed_dataframe['payment_id'].unique())) + ' payments.')


if __name__ == '__main__':

    mode_flag = 'retraining'
    acct_id = str(sys.argv[1])
    path = str(sys.argv[2])

    number_of_process = 2
    retraining_data_path = path+"/account_"+acct_id+"/data_extracted/retraining_data.csv"
    new_history_path = path+"/account_"+acct_id+"/history_generated/history.csv"
    open_invoices_path = ' '
    recent_payments_path = ' '
    log_path=path+'/account_'+str(acct_id)+'/logs/'
    

    #History Generation
    history_generation(retraining_data_path, open_invoices_path,recent_payments_path, new_history_path, mode_flag, number_of_process,acct_id,path,log_path)
    
