import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import pickle
import logging
import joblib

logging.basicConfig(filename='logs.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

def make_predictions(test_data_path, model_path, predictions_path):

    logging.info('Predictions Start.')
    features = ['avg_delay_categorical',
                'variance_categorical',
                'LMH_cumulative',
                'avg_of_invoices_closed',
                'avg_of_all_delays',
                'payment_count_quarter_q1', 'payment_count_quarter_q2', 'payment_count_quarter_q3',
                'payment_count_quarter_q4',
                'invoice_count_quarter_q1', 'invoice_count_quarter_q2', 'invoice_count_quarter_q3',
                'invoice_count_quarter_q4',
                'number_invoices_closed']

    data2=pd.read_csv(r''+test_data_path)

    if len(data2)!=0:
        X_validation = data2[features]
        y_validation = data2['output']
    else:
        logging.info('No Prediction data')
        return

    rfc = joblib.load(open(model_path, 'rb'))

    predictions = rfc.predict(X_validation)
    predictions_prob = rfc.predict_proba(X_validation)
    logging.info('accuracy_score ' + str(accuracy_score(y_validation, predictions)))
    logging.info('confusion matric ')
    logging.info(confusion_matrix(y_validation, predictions))
    logging.info('classification Report ')
    logging.info(classification_report(y_validation, predictions))

    data2['predictions'] = predictions
    for i in range(0, data2.shape[0]):
        data2.at[i, 'pred_proba_0'] = predictions_prob[i][0]
        data2.at[i, 'pred_proba_1'] = predictions_prob[i][1]

    dataset=data2

    payment_without_any_subset=0
    dataset['transformed_output']=0
    for i in dataset['payment_id'].unique():
        max_proba=dataset[dataset['payment_id']==i]['pred_proba_1'].max()
        dataset.loc[(dataset['payment_id']==i) & (dataset['pred_proba_1']==max_proba),'transformed_output']=1
        if len(dataset[dataset['payment_id']==i]['output'].unique())==1:
            payment_without_any_subset=payment_without_any_subset+1

    logging.info('***** After Output transformation *****')
    logging.info('accuracy_score' + str(accuracy_score(y_validation, dataset['transformed_output'])))
    logging.info('confusion matrix ')
    logging.info(confusion_matrix(y_validation, dataset['transformed_output']))
    logging.info('classification Report ')
    logging.info(classification_report(y_validation, dataset['transformed_output']))

    logging.info('Total Payment : ' + str(len(dataset['payment_id'].unique())))
    logging.info('Total correct payment(s) : ' + str(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==1)])))
    logging.info('Total incorrect payment(s) : ' + str(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==0)])))
    logging.info('Total payments without any subset : ' + str(payment_without_any_subset))
    logging.info('%age : '+ str(len(dataset[(dataset['output']==1) & (dataset['transformed_output']==1)])/len(dataset['payment_id'].unique())*100))

    data2.to_csv(predictions_path, index=False)
    logging.info('Predictions Ended') 


if __name__ == '__main__':
    acct_id = str(sys.argv[1])
    root_dir='/root/caa/rp/model/'
    test_data_path = root_dir+'account_'+str(acct_id)+'/train_test_splitted/test_30.csv'
    model_path = root_dir+'account_'+str(acct_id)+'/trained_model/model.pkl'
    predictions_path = root_dir+'account_'+str(acct_id)+'/predictions/predictions.csv'

    #This will make predictions
    make_predictions(test_data_path,model_path,predictions_path)
