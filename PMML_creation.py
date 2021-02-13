from sklearn2pmml import PMMLPipeline
from sklearn2pmml import Pipeline
from sklearn2pmml import sklearn2pmml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn_pandas import DataFrameMapper
import logging
import os
import joblib
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMModel, LGBMClassifier
import sys

logging.basicConfig(filename='logs.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)


def PMML_creation(train_path, test_path, pmml_predictions, pmml_path,path,acct_id):
    logging.info('PMML creation Started.')
    data = pd.read_csv(r'' + train_path)
    data2 = pd.read_csv(r'' + test_path)

    features = ['avg_delay_categorical',
                'variance_categorical',
                'LMH_cumulative',
                'avg_of_invoices_closed',
                'avg_of_all_delays',
                'payment_count_quarter_q1', 'payment_count_quarter_q2',
                'payment_count_quarter_q3', 'payment_count_quarter_q4',
                'invoice_count_quarter_q1', 'invoice_count_quarter_q2',
                'invoice_count_quarter_q3', 'invoice_count_quarter_q4',
                'number_invoices_closed']

    #rf = RandomForestClassifier(n_estimators=100,random_state =42, class_weight = {0: 1, 1:1}, max_depth = 8, max_features =0.5) #duracell
    #rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight={0: 1, 1: 1}, max_depth=8,max_features=0.5, min_weight_fraction_leaf=0.1)   #gettyimages

    # rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight={0: 1, 1: 1}, max_depth=8,max_features=0.5)  #milliken
    #rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 2}, max_depth=8,max_features=0.5) #graybar
    # rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight={0: 1, 1: 1}, max_depth=7,
    #                              max_features=0.4, min_samples_split=4, min_samples_leaf=3,
    #                              min_weight_fraction_leaf=0.1)
    # rf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight={0: 1, 1: 2}, max_depth=7,
    #                              max_features=0.4, min_samples_leaf=4, min_weight_fraction_leaf=0.2)
    # rf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=10, random_state=42, class_weight={0: 1, 1: 2},
    #     #                             criterion='gini', max_depth=7, max_features=0.4, n_jobs=-1,
    #     #                             min_weight_fraction_leaf=0.4)
    # rf = RandomForestClassifier(n_estimators=320, max_leaf_nodes=20, random_state=42, class_weight={0:1, 1:2},
    #                              criterion='gini', max_depth=7, max_features=0.4,n_jobs=-1)
    #
    # rf = xgboost.XGBClassifier(random_state=42, n_estimators=206, min_samples_split=10, min_samples_leaf=6,
    #                             max_features='sqrt', max_depth=1, learning_rate=0.0015)
    # rf = xgboost.XGBClassifier(random_state=42, n_estimators=302, min_samples_split=10, min_samples_leaf=10,
    #                             max_features='sqrt', max_depth=1, learning_rate=0.0074)
    # rf=xgboost.XGBClassifier(random_state=42, n_estimators=145, min_samples_split=24, min_samples_leaf=1,
    #                       max_features='sqrt', max_depth=33, learning_rate=0.0077)

    # rf = RandomForestClassifier(random_state=42, n_estimators=400, min_weight_fraction_leaf=0.3, min_samples_split=24,
    #                              min_samples_leaf=1, max_features='sqrt', max_depth=1, criterion='entropy')

    #rf= xgboost.XGBClassifier(random_state=42,n_estimators=106,min_samples_split=24, min_samples_leaf= 16, max_features= 'auto',max_depth=1, learning_rate= 0.0044)
    
    #rf = LGBMClassifier(class_weight={0: 1, 1: 5}, max_depth=10, num_leaves=1000, min_data_in_leaf=500,
     #                     learning_rate=0.08)
    #rf = xgboost.XGBClassifier(random_state=42,n_estimators=106,min_samples_split=24, min_samples_leaf= 16, max_features= 'sqrt',max_depth=35, learning_rate= 0.0077)
    #rf=  xgboost.XGBClassifier(random_state=42,n_estimators=400,min_samples_split=34, min_samples_leaf= 16, max_features= 'sqrt',max_depth=35, learning_rate= 0.0099)
    #rf= xgboost.XGBClassifier(random_state=42,n_estimators=200,min_samples_split=14, min_samples_leaf= 1, max_features= 'sqrt',max_depth=20, learning_rate= 0.0077)
    #final_report =  pd.read_csv(path+'/account_'+acct_id+'/summary.csv')
    model = joblib.load(path+'/account_'+acct_id+'/trained_model/model.pkl')
    #model_name = str(model).split('(')[0]
    params = model.get_params()
    classifier= type(model)()
    rf = classifier.set_params(**params)
    print("-"*100)
    print(rf)
    #rf = type(model)(model.get_params)
    print((rf.get_params()))
    print("-"*100)
    print(model.get_params())

    

    # rf= xgboost.XGBClassifier(random_state=42, n_estimators=320, min_samples_split=5, min_samples_leaf=6,
    #                             max_features='log2', max_depth=50, learning_rate=0.0093)
    # rf = xgboost.XGBClassifier(random_state=42, n_estimators=445, min_samples_split=5, min_samples_leaf=8,
    #                             max_features='sqrt', max_depth=1, learning_rate=0.00959591836734694)

    mapper = DataFrameMapper([('avg_delay_categorical', None),
                              ('variance_categorical', None),
                              ('LMH_cumulative', None),
                              ('avg_of_invoices_closed', None),
                              ('avg_of_all_delays', None),
                              ('payment_count_quarter_q1', None),
                              ('payment_count_quarter_q2', None),
                              ('payment_count_quarter_q3', None),
                              ('payment_count_quarter_q4', None),
                              ('invoice_count_quarter_q1', None),
                              ('invoice_count_quarter_q2', None),
                              ('invoice_count_quarter_q3', None),
                              ('invoice_count_quarter_q4', None),
                              ('number_invoices_closed', None)
                              ])

    labels = data.loc[:, 'output']
    labels.name = 'output'

    data = data[features].astype('double')
    print(data.dtypes)

    pipeline = PMMLPipeline([("mapper", mapper), ("estimator", rf)])
    pickle_pipeline = Pipeline([("mapper", mapper), ("model", rf)])

    pipeline.fit(data, labels)
    pickle_pipeline.fit(data, labels)

    predictions = pipeline.predict(data2[features])
    predictions_prob = pipeline.predict_proba(data2[features])
    data2['PMML_predictions'] = predictions
    for i in range(0, data2.shape[0]):
        data2.at[i, 'PMML_pred_proba_0'] = predictions_prob[i][0]
        data2.at[i, 'PMML_pred_proba_1'] = predictions_prob[i][1]

    data2.to_csv(pmml_predictions, index=False)
    sklearn2pmml(pipeline, r"" + pmml_path + '_PIPELINED' + ".pmml", debug=True)
    joblib.dump(pickle_pipeline, r"" + pmml_path + "_PIPELINED.pkl")
    logging.info('PMML created of size ' + str(file_size(r"" + pmml_path + ".pmml")))


if __name__ == '__main__':
    acct_id = str(sys.argv[1])
    #path = "/root/accounts"
    path =str(sys.argv[2])
    train_path =  path+'/account_'+acct_id+'/train_test_splitted/train_70.csv'
    test_path = path+'/account_'+acct_id+'/train_test_splitted/test_30.csv'
    pmml_predictions = path+'/account_'+acct_id+'/predictions/PMML_predictions.csv'
    pmml_path = path+'/account_'+acct_id+'/trained_model/model.pmml'
    # This will create PMML
    PMML_creation(train_path, test_path, pmml_predictions, pmml_path, path,acct_id)
