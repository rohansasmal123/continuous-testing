import json
import datetime as dt
import pandas as pd
import os
import logging
import sys
logging.basicConfig(filename='logs.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

#This function returns the quarter
def quarter_func(x):
   if (x <= 3):
       return 1
   if (x <= 6):
       return 2
   if (x <= 9):
       return 3
   else:
       return 4

def payment_count(df_cust):
    df_cust['quarter'] = df_cust['eff_month'].apply(quarter_func)
    #we convert the months into quators so if any month <=3 then quater_func will return 1 similarly rest.
    h = df_cust.groupby('eff_month')['payment_amount'].nunique()
    #here we take the number of unique values of payment_amount based on the eff_month
    df_pay_quarter = pd.DataFrame(h)
    df_pay_quarter.reset_index(inplace=True)
    df_pay_quarter['quarter'] = df_pay_quarter['eff_month'].apply(quarter_func)
    df_pay_quarter['quarter_1'] = 0
    df_pay_quarter['quarter_2'] = 0
    df_pay_quarter['quarter_3'] = 0
    df_pay_quarter['quarter_4'] = 0
    quarter = pd.get_dummies(df_pay_quarter['quarter'], prefix="quarter")
    #here we are adding prefix='quater' in quarter column.
    for col in quarter:
        df_pay_quarter[col] = quarter[col]
    df_pay_quarter['quarter_1'] = df_pay_quarter['payment_amount'] * df_pay_quarter['quarter_1']
    df_pay_quarter['quarter_2'] = df_pay_quarter['payment_amount'] * df_pay_quarter['quarter_2']
    df_pay_quarter['quarter_3'] = df_pay_quarter['payment_amount'] * df_pay_quarter['quarter_3']
    df_pay_quarter['quarter_4'] = df_pay_quarter['payment_amount'] * df_pay_quarter['quarter_4']
    tot_pay = sum(df_pay_quarter['payment_amount'])
    q1 = (sum(df_pay_quarter['quarter_1']) / tot_pay) * 100
    q2 = (sum(df_pay_quarter['quarter_2']) / tot_pay) * 100
    q3 = (sum(df_pay_quarter['quarter_3']) / tot_pay) * 100
    q4 = (sum(df_pay_quarter['quarter_4']) / tot_pay) * 100
    dict_quarter = {}
    dict_quarter['q1'] = q1
    dict_quarter['q2'] = q2
    dict_quarter['q3'] = q3
    dict_quarter['q4'] = q4
    return dict_quarter


#######################################################################################################################################
# invoices closed in each quarter
#######################################################################################################################################

def invoice_count(df_cust_inv):
   df_cust_inv['quarter'] = df_cust_inv['eff_month'].apply(quarter_func)
   df_cust_inv['quarter_1'] = 0
   df_cust_inv['quarter_2'] = 0
   df_cust_inv['quarter_3'] = 0
   df_cust_inv['quarter_4'] = 0
   quarter_inv = pd.get_dummies(df_cust_inv['quarter'], prefix='quarter')
   for col in quarter_inv:
       df_cust_inv[col] = quarter_inv[col]
   # df_cust_inv = df_cust_inv.concat([df_cust_inv, quarter_inv], axis=1)
   q1 = sum(df_cust_inv['quarter_1'])
   q2 = sum(df_cust_inv['quarter_2'])
   q3 = sum(df_cust_inv['quarter_3'])
   q4 = sum(df_cust_inv['quarter_4'])
   q1_ = (q1 / (q1 + q2 + q3 + q4)) * 100
   q2_ = (q2 / (q1 + q2 + q3 + q4)) * 100
   q3_ = (q3 / (q1 + q2 + q3 + q4)) * 100
   q4_ = (q4 / (q1 + q2 + q3 + q4)) * 100
   dict_quarter_inv = {}
   dict_quarter_inv['q1'] = q1_
   dict_quarter_inv['q2'] = q2_
   dict_quarter_inv['q3'] = q3_
   dict_quarter_inv['q4'] = q4_

   return dict_quarter_inv

#This is to calculate the payment cycle of each customer
def payment_cycle(data,read_from):
   with open(read_from) as f:
       data_dict = json.load(f)
   data = data[data['payment_id'] == data['payment_hdr_id']]
   data['payment_date'] = pd.to_datetime(data['effective_date'])
   data['effective_date'] = data['effective_date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
   data['eff_month'] = data['effective_date'].apply(lambda x: int(x.month))
   sd = data.groupby(by=['customer_number_norm'])
   for name, dataset in sd:
       data_dict[name]['payment_count_quarter'] = payment_count(dataset)
       data_dict[name]['invoice_count_quarter'] = invoice_count(dataset)
       max_cycle = 0
       uniq_pay_dates = dataset['payment_date'].unique()
       sorted(uniq_pay_dates)
       last_date = uniq_pay_dates[0]
       for j in uniq_pay_dates:
           diff = j - last_date
           diff = diff.astype('timedelta64[D]')
           if max_cycle < pd.Timedelta(diff).days:
               max_cycle = pd.Timedelta(diff).days
           last_date = j
       data_dict[name]['max_payment_window'] = int(max_cycle)
   with open(read_from, 'w') as fp:
       json.dump(data_dict, fp)

#This function divides the invoice amount into L1,L2,L3,M,H and calculated L1_perc, L2_perc, L3_perc, M_perc, H_perc respectively
def smh_func(df):
   df_customer = df
   try:
       df_customer['invoice_type'] = pd.cut(df_customer['invoice_amount_norm'], [0, 100, 500, 1000, 5000, 1000000000],
                                        labels=['L1_perc', 'L2_perc', 'L3_perc', 'M_perc', 'H_perc'])

       # pd.cut will make buckets of certain range and here range are 0 to 100 and 100 to 500  and so on.
   except Exception as e:
       logging.info('Error: '+str(e))
       return

   df_customer['invoice_count_per_category'] = df_customer.groupby('invoice_type')['invoice_type'].transform('count')
   df_customer.loc['invoice_percent'] = (df_customer['invoice_count_per_category'] / df_customer.shape[0]) * 100
   smh = df_customer['invoice_type'].value_counts().index
   #here they have taken the invoice_type value_count and then .index will only take the values of the invoice_type column not the counts of those values.
   totals = df_customer['invoice_type'].value_counts().values
   #here they first take the value_count of te incouce_type and then .values will only take the counts of the values present in the invoice_type column.
   count_total = sum(totals)
   percent = [(i / count_total) * 100 for i in totals]
   #here they are taking the percentage of the individual counts by the sum of the total counts of the values of the invoice_type.
   lst = []
   for i, j in zip(percent, totals):
       lst.append((float(i), float(j)))
    #here we have just added teo more columns in lst i.e percent and totals
   smh = df_customer['invoice_type'].value_counts().index
   dict_smh = {}
   for i in range(0, len(smh)):
       dict_smh[smh[i]] = lst[i]
   return dict_smh


def create_json(old_history_path, new_history_path, write_to,log_path):
   logging.info('Customer Json Started.')
   old_history = pd.DataFrame()
   if os.path.exists(old_history_path):
        old_history = pd.read_csv(old_history_path)
   new_history = pd.read_csv(new_history_path)

   d = old_history.append(new_history, ignore_index=True)

   d['customer_number_norm'] = d['customer_number_norm'].astype(str)
   #here customer number is converted into string.

   d['customer_number_norm'] = d.apply(lambda row: row['customer_number_norm'].split('.')[0],axis=1)
   #here customer number norm jo string mai converted hae usse split kar rahe hae jaha . hoga waha se and only oth column hi split hoga or usse ek column mai daal raheh hae usseka naam hae d['customer_number_norm']
   di = {}
   #dictionary will be created here
   d['payment_date'] = pd.to_datetime(d['effective_date'])
   d['invoice_date'] = pd.to_datetime(d['invoice_date_norm'])
   d['invoice_amount_norm']=d['invoice_amount_norm'].replace(" ","")
   #here we replaced the space "  " with small space ""
   sd = d.groupby(by=['customer_number_norm'])
   for name, data in sd:
       s = data[data['payment_id'] == data['payment_hdr_id']]
       #actual payments clearing invoices
       avg = len(s) / len(s['payment_id'].unique())
       s['delay'] = s['payment_date'].subtract(s['invoice_date'],axis=0)
       #delay calculate kiya hae
       s['delay'] = s['delay'].apply(lambda x: pd.Timedelta(x).days)
       #converted into days only
       di[name] = {"avg_of_invoices_closed": avg, "avg_of_all_delays": s['delay'].mean(), "buckets_invoice": smh_func(s)}

   with open(write_to, 'w') as fp:
       json.dump(di, fp)
   payment_cycle(d,write_to)
   logging.info('Json Creation Finished.')
   
   progress=pd.read_csv(log_path+"progress.csv")
   progress['Status']='JSON Creation'
   progress.to_csv(log_path+"progress.csv",index=False)


if __name__ == '__main__':
    acct_id = str(sys.argv[1])
    path = (sys.argv[2])
    old_history_path = 'E:/shubham.kamal/PycharmProjects/ai_cashapp/subset_sum/data/7029/history_generated/history.csv'
    new_history_path =  path+"/account_"+acct_id+'/history_generated/history.csv'
    write_to = path+"/account_"+acct_id+'/customersJson/customersJson.json'
    log_path=path+'/account_'+str(acct_id)+'/logs/'
    #JSON Creation
    create_json(old_history_path, new_history_path, write_to, log_path)
