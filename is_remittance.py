##Import statements
import pandas as pd
import random
import re
from sklearn import model_selection
import string
import numpy as np
import sklearn
from sklearn import linear_model, datasets,tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import ensemble
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dateutil.parser import parse
from sklearn.externals import joblib as jb
########################################################################################################################
########################################################################################################################
##Reading train and test data
########################################################################################################################
def preprocess_is_remittance(data):




    data=data.reset_index(drop=True)
    ########################################################################################################################
    ##creating ratio_row_section feature
    ########################################################################################################################


    data['ratio_rowSection']=data['row_noOfCharacters']/data['section_noOfCharacters']
    return data

########################################################################################################################
##remittance_mark() flags all remittance lines between heading and total lines as 1 in a page
##In case of multiple headings, the first heading is chosen
##In case of multiple totals, the last total is chosen
##In case where only heading is present last line of page is marked as total and everthing in between is flagged as remittance lines
##In case where only total is present first line of page is marked as heading and everthing in between is flagged as remittance lines

########################################################################################################################


def remittance_mark(train):
    ##train_rev is used to get the last total line
    train_rev = train.sort_index(ascending=False, axis=0)
    heading = train.groupby(['check_checkNumber', 'page_pageNumber']).apply(lambda x: x['is_heading_predicted'].idxmax())
    heading = heading.to_frame()
    heading.reset_index(inplace=True)
    heading.columns = ['check_no', 'page_no', 'heading_index']

    total = train_rev.groupby(['check_checkNumber', 'page_pageNumber']).apply(lambda x: x['is_total_predicted'].idxmax())
    total = total.to_frame()
    total.reset_index(inplace=True)
    total.columns = ['check_no', 'page_no', 'total_index']

    merge = pd.merge(heading, total, left_on=['check_no', 'page_no'], right_on=['check_no', 'page_no'], how='inner')
    train['between_head_total'] = 0

    ##populating remittance_result in data
    for index in range(0, merge.shape[0]):
        if (((train.loc[merge['heading_index'].values[index], 'is_heading_predicted']) == 0) and ((train.loc[merge['total_index'].values[index], 'is_total_predicted']) == 0)):
          continue

        else:
            train.loc[merge['heading_index'].values[index] + 1:merge['total_index'].values[index], 'between_head_total'] = 1

    return train



########################################################################################################################
##total_digits() calls calc_digit_percentage which calculates total_number of digits in train_test_concat
########################################################################################################################

def calc_digit_percentage(x):
    digits = sum(c.isdigit() for c in x)
    letters = sum(c.isalpha() for c in x)
    spaces = sum(c.isspace() for c in x)
    others = len(x) - digits - letters - spaces
    total_charac = digits + letters + others
    digit_percentage=digits/total_charac *100
    return digit_percentage

def total_digits(data):


    data['NumberAlphaRatio']=data['row_string'].apply(calc_digit_percentage)

    data.reset_index(drop=True,inplace=True)

    return data



########################################################################################################################
##amount_flag() calls amount_flag_utility() which calculates  'amount_present': flagged 1 if amount is present in the row_string else 0
########################################################################################################################

def amount_flag_utility(row):
    if('$' in row['row_string']):
        return 1
    s=row['row_string'].replace(", ",",")
    digits = re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+|\d{1,2}[\,]{1}\d{1,3}[\.]{1}\d{1,2}", s, flags=re.MULTILINE)
    for digit in digits:
        digit = float(digit.replace(',', ''))
        if ((digit <= row['check_checkAmount']) & (digit > 0)):
            return 1
    return 0

def amount_flag(data):
    data['containsAmount'] = data.apply(lambda x: amount_flag_utility(x), axis=1)
    return data


########################################################################################################################
##reference_number_date_capture() calls reference_number_date_capture() which calculates 'reference_no_present': flagged 1 if reference no is present in the row_string
##                                   'date_present' is flagged 1 if date is present
##**********************************Here it is assumed that any number which is not date and has a length greater than 5 is a reference no*******
########################################################################################################################

def reference_number_date_capture_utility(s):

    curr_year=pd.datetime.now().year
    containsReference=0
    containsDate=0

    pattern = re.compile(
        "Jan[\s+|\,|\'|\-|\/|\d]|Feb[\s+|\,|\'|\-|\/|\d]|Mar[\s+|\,|\'|\-|\/|\d]|Apr[\s+|\,|\'|\-|\/|\d]|May[\s+|\,|\'|\-|\/|\d]|Jun[\s+|\,|\'|\-|\/|\d]|June[\s+|\,|\'|\-|\/|\d]|Jul[\s+|\,|\'|\-|\/|\d]|Jul[\s+|\,|\'|\-|\/|\d]|Aug[\s+|\,|\'|\-|\/|\d]|Sep[\s+|\,|\'|\-|\/|\d]|Oct[\s+|\,|\'|\-|\/|\d]|Nov[\s+|\,|\'|\-|\/|\d]|Dec[\s+|\,|\'|\-|\/|\d]|December|January|February|March|April|August|September|October|November|December?\s+\d{1,2}[,/.]\s+\d{4}([0-3]?[0-9][.|/][0-1]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][.|/][0-3]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|\d{1,2}[\-|\,|\/|\.]{1}\d{1,2}[\-|\,|\/|\.]{1}\d{2,4}",
        re.IGNORECASE)

    if pattern.search(str(s['row_string'])) is not None:
        containsDate=1

    digits = re.findall(r"[0-9]+", s['row_string'], flags=re.MULTILINE)
    for j in digits:
        if (len(j) == 6):
            mm = int(j[0:2])
            dd = int(j[2:4])
            yy = int(j[4:6])
            if (((mm > 0) and (mm <= 12) and (dd > 0) and (dd <= 31) and (yy >= 16) and (yy <= curr_year%100)) or (
                                        (mm > 0) and (mm <= 31) and (dd > 0) and (dd <= 12) and (yy >= 16) and (
                        yy <= curr_year%100))):
                containsDate=1
                continue
        if (len(j) == 8):
            mm = int(j[0:2])
            dd = int(j[2:4])
            yyyy = int(j[4:8])
            if (((mm > 0) and (mm <= 12) and (dd > 0) and (dd <= 31) and (yyyy >= 2016) and (yyyy <= curr_year)) or (
                                        (mm > 0) and (mm <= 31) and (dd > 0) and (dd <= 12) and (yyyy >= 2016) and (
                                yyyy <= curr_year))):
                containsDate=1
                continue
        if (len(j)>=5):
            containsReference=1
            continue
    return pd.Series([containsDate,containsReference])


def reference_number_date_capture(data):
    df=data.apply(lambda x:reference_number_date_capture_utility(x),axis=1)
    df.columns=['containsDate','containsReference']
    data=pd.concat([data,df],axis=1)
    return data

########################################################################################################################
##confidence_level() is used to find how many of 'date_flag', 'amount_present', 'remittance_result' is present.This to done to
##establish confidence. Eg: if amount and date were present in the row string, then confidence_level would be 2
########################################################################################################################

def confidence_level(data):
    data['countOfFeatures']= data['containsDate']+data['containsAmount']+data['between_head_total']+data['containsReference']
    return data


def calculate_features(data):
    data=preprocess_is_remittance(data)
    data=remittance_mark(data)
    data=total_digits(data)
    data=amount_flag(data)
    data=reference_number_date_capture(data)
    data=confidence_level(data)
    return data




########################################################################################################################
##List of features
########################################################################################################################
features=['countOfFeatures','containsReference','containsDate','containsAmount','row_noOfCharacters','ratio_rowSection',
          'between_head_total','NumberAlphaRatio','row_distanceFromLeft','row_distanceFromTop']

########################################################################################################################
##Training and testing
def predict_on_test_data(model_path,df_test_path,write_file_path):
    global features
    model=jb.load(model_path)
    df_test=pd.read_csv(df_test_path)
    df_test=calculate_features(df_test)



    pred=model.predict(df_test[features])
    pred_prob=model.predict_proba(df_test[features])
    df_pred=pd.DataFrame(pred,columns=['is_remittance_predicted'])
    df_prob=pd.DataFrame(pred_prob)
    df_prob.columns=['prob_0','prob_1']
    df_test['is_remittance_predicted']=df_pred['is_remittance_predicted']
    df_test['is_remittance_prob_0']=df_prob['prob_0']
    df_test['is_remittance_prob_1']=df_prob['prob_1']
    df_test.to_csv(write_file_path+"/is_remittance_pred.csv",index=False,encoding="utf-8")
    return df_test







