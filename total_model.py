########################################################################################################################
##import statements
import pandas as pd
import re
import string
from sklearn.externals import joblib as jb
import pickle
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

########################################################################################################################
##total_regex() calls total_regex_util() which populates the contains_total column with 0 or 1 based on regex
########################################################################################################################

def total_regex_util(s):
    pattern=re.compile(".*(total(s)?).*",re.IGNORECASE)
    if(pattern.search(s)) is not None:
        return 1
    else:
        return 0

def total_regex(data):
    data['contains_total'] = data['row_string'].apply(lambda x: total_regex_util(x))

########################################################################################################################
##amount_regex() calls amount_regex_util() which populates the contains_amount column with 0 or 1 based on regex
########################################################################################################################
def amount_regex_util(s):
    pattern=re.compile(".*(amount(s)?).*",re.IGNORECASE)
    if(pattern.search(s)) is not None:
        return 1
    else:
        return 0

def amount_regex(data):
    data['contains_amount'] = data['row_string'].apply(lambda x: amount_regex_util(x))

########################################################################################################################
##row_number_diff() calls row_number_diff_util() which populates the row_number_diff column with the difference page_noOfRows
# and row_rowNumber of  based on regex
########################################################################################################################

def row_number_diff_util(x):
    return x['page_noOfRows']-x['row_rowNumber']

def row_number_diff(data):
    data['row_number_diff'] = data.apply(lambda x: row_number_diff_util(x), axis=1)

########################################################################################################################
features=['contains_total','contains_amount','row_number_diff']


def predict_on_test_data(model_path,df_test_path,write_file_path):
    global features
    model=jb.load(model_path)
    df_test=pd.read_csv(df_test_path)
    df_test['contains_total'] = df_test['row_string'].apply(lambda x: total_regex_util(x))
    df_test['contains_amount'] = df_test['row_string'].apply(lambda x: amount_regex_util(x))
    df_test['row_number_diff'] = df_test.apply(lambda x: row_number_diff_util(x), axis=1)
    pred=model.predict(df_test[features])
    pred_prob=model.predict_proba(df_test[features])
    df_pred=pd.DataFrame(pred,columns=['is_total_predicted'])
    df_prob=pd.DataFrame(pred_prob)
    df_prob.columns=['prob_0','prob_1']
    df_test['is_total_predicted']=df_pred['is_total_predicted']
    df_test['is_total_prob_0']=df_prob['prob_0']
    df_test['is_total_prob_1']=df_prob['prob_1']
    df_test['is_total_predicted'] = df_test.apply(lambda x: output_transformation_util(x), axis=1)

    df_test.to_csv(write_file_path+"/is_total_pred.csv",index=False,encoding="utf-8")
    return df_test





########################################################################################################################
##                                                      Output transformation
########################################################################################################################
##output_transformation() changes the predicted total values if the row_string contains amount which matches with the check_amount
##This is done using prediction_post_processing
########################################################################################################################

def prediction_post_processing(x):
    check_amt=x['check_checkAmount']
    s=x['row_string']
    s=s.replace('$',' ')
    digits=re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+|\d{1,2}[\,]{1}\d{1,3}[\.]{1}\d{1,2}",s,flags=re.MULTILINE)
    digits=[float(digit.replace(',','')) for digit in digits ]


    if(check_amt in digits):

        return 1
    else:
        return 0


def output_transformation_util(x):
    if((x['is_total_predicted']==0) and (x['contains_total']==0)):
        if(prediction_post_processing(x)):
            return 1
        else:
            return 0
    else:
        return x['is_total_predicted']




########################################################################################################################





















