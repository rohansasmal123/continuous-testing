from __future__ import unicode_literals
import pandas as pd
import  numpy as np
import re
import sys
import random
from pathlib import Path
import random

######################################################################################################################
def date_search(s):
    curr_year=pd.datetime.now().year
    s=str(s)
    pattern = re.compile(
        "Jan[\s+|\,|\'|\-|\/|\d]|Feb[\s+|\,|\'|\-|\/|\d]|Mar[\s+|\,|\'|\-|\/|\d]|Apr[\s+|\,|\'|\-|\/|\d]|May[\s+|\,|\'|\-|\/|\d]|Jun[\s+|\,|\'|\-|\/|\d]|June[\s+|\,|\'|\-|\/|\d]|Jul[\s+|\,|\'|\-|\/|\d]|Jul[\s+|\,|\'|\-|\/|\d]|Aug[\s+|\,|\'|\-|\/|\d]|Sep[\s+|\,|\'|\-|\/|\d]|Oct[\s+|\,|\'|\-|\/|\d]|Nov[\s+|\,|\'|\-|\/|\d]|Dec[\s+|\,|\'|\-|\/|\d]|December|January|February|March|April|August|September|October|November|December?\s+\d{1,2}[,/.]\s+\d{4}([0-3]?[0-9][.|/][0-1]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][.|/][0-3]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|\d{1,2}[\-|\,|\/|\.]{1}\d{1,2}[\-|\,|\/|\.]{1}\d{2,4}",
        re.IGNORECASE)

    if pattern.search(s) is not None:
        return 1
    return 0
########################################################################################################################

#amount_search() returns 1 if input string is amount else returns 0
########################################################################################################################
def amount_search(s):
    s=str(s)
    if('$' in s):
        return 1
    s=s.replace(", ",",")
    digits = re.findall(r"\-?\d+\.\s*\d+\-?|\-?\d{1,2}\,{1}\s*\d{1,3}\.{1}\s*\d{1,2}\-?", s, flags=re.MULTILINE)
    if(len(digits)>0):
        return 1
    return 0
########################################################################################################################

def alphanumeric_check(s):
    s=str(s)
    if(bool(re.search('\d',s))):
        return 0
    return 1

########################################################################################################################


#reference_no_date_capture() returns 1 if input string is date, returns 2 if input string is invoice/reference no

def reference_no_date_capture(s):


    curr_year=pd.datetime.now().year
    containsReference=0
    s=str(s)

    pattern = re.compile(
        "Jan[\s+|\,|\'|\-|\/|\d]|Feb[\s+|\,|\'|\-|\/|\d]|Mar[\s+|\,|\'|\-|\/|\d]|Apr[\s+|\,|\'|\-|\/|\d]|May[\s+|\,|\'|\-|\/|\d]|Jun[\s+|\,|\'|\-|\/|\d]|June[\s+|\,|\'|\-|\/|\d]|Jul[\s+|\,|\'|\-|\/|\d]|Jul[\s+|\,|\'|\-|\/|\d]|Aug[\s+|\,|\'|\-|\/|\d]|Sep[\s+|\,|\'|\-|\/|\d]|Oct[\s+|\,|\'|\-|\/|\d]|Nov[\s+|\,|\'|\-|\/|\d]|Dec[\s+|\,|\'|\-|\/|\d]|December|January|February|March|April|August|September|October|November|December?\s+\d{1,2}[,/.]\s+\d{4}([0-3]?[0-9][.|/][0-1]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][.|/][0-3]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|\d{1,2}[\-|\,|\/|\.]{1}\d{1,2}[\-|\,|\/|\.]{1}\d{2,4}",
        re.IGNORECASE)

    if pattern.search(s) is not None:
        containsReference=1

    digits = re.findall(r"[0-9]+", s, flags=re.MULTILINE)
    for j in digits:
        if (len(j) == 6):
            mm = int(j[0:2])
            dd = int(j[2:4])
            yy = int(j[4:6])
            if (((mm > 0) and (mm <= 12) and (dd > 0) and (dd <= 31) and (yy >= 16) and (yy <= curr_year%100)) or (
                                        (mm > 0) and (mm <= 31) and (dd > 0) and (dd <= 12) and (yy >= 16) and (
                        yy <= curr_year%100))):
                containsReference=1
                continue
        if (len(j) == 8):
            mm = int(j[0:2])
            dd = int(j[2:4])
            yyyy = int(j[4:8])
            if (((mm > 0) and (mm <= 12) and (dd > 0) and (dd <= 31) and (yyyy >= 2016) and (yyyy <= curr_year)) or (
                                        (mm > 0) and (mm <= 31) and (dd > 0) and (dd <= 12) and (yyyy >= 2016) and (
                                yyyy <= curr_year))):
                containsReference=1
                continue
        if (len(j)>=5):
            containsReference=2
            continue
    return containsReference


#
# Capture the indices of the heading words
# given_headingPredictionIsntFound_then_indicesHeadingWordsArrayIsntPopulated
#
def capture_indices_heading_words(page_containing_rows):
    lst_index=[]
    for i in range(0,page_containing_rows.shape[0]):
        if(page_containing_rows['is_heading_predicted'].values[i]==1):
            page_containing_rows['row_string_pipe'].values[i]=str(page_containing_rows['row_string_pipe'].values[i])
            s_head=page_containing_rows['row_string_pipe'].values[i].split('|')
            for j in range(0,len(s_head)):
                if(re.search('[a-z0-9]*(invoice number|inv no|inv nu|invno|invnu|invoice no|invoice|inv|ref)[a-z0-9]*',str(s_head[j]).lower())==None):
                    continue
                else:
                    lst_index.append(j)
            break
    return lst_index

########################################################################################################################


########################################################################################################################
##capture_invoice_no_heading_present() captures invoice numbers from pages where heading is present

########################################################################################################################

def capture_invoice_no_heading_present(page_containing_rows,lst_index):
    for i in range(0, page_containing_rows.shape[0]):
        if (page_containing_rows['is_remittance_predicted'].values[i] == 1):
            s_remit = page_containing_rows['row_string_pipe'].values[i].split('|')
            for k in lst_index:
                if (k < len(s_remit)):
                    if (date_search(s_remit[k]) == 1):
                        # print("1: ",s_remit[k])
                        continue
                    if (amount_search(s_remit[k]) == 1):
                        # print("2: ",s_remit[k])
                        continue
                    if (alphanumeric_check(s_remit[k]) == 1):
                        # print("3: ",s_remit[k])
                        continue
                    # print("4: ",s_remit[k])

                    page_containing_rows['invoice_number_captured'].values[i] = s_remit[k]
                    # print(s_remit[k])
                    break
    return page_containing_rows

########################################################################################################################

##capture_invoice_no_heading_present() captures invoice_numbers from
# 1> pages where heading is absent
# 2> heading present but invoice number not captured because of index mismatch


def capture_invoice_no_heading_absent(page_containing_rows):
    for i in range(0,page_containing_rows.shape[0]):
        if((page_containing_rows['is_remittance_predicted'].values[i]==1) and (page_containing_rows['invoice_number_captured'].values[i]==None)):
            s_remit=page_containing_rows['row_string_pipe'].values[i].split('|')
            for k in s_remit:
                    flag=reference_no_date_capture(k)
                    if(flag==1):
                        continue
                    elif(flag==2):
                        page_containing_rows['invoice_number_captured'].values[i]=k
                        break
    return page_containing_rows


########################################################################################################################
##invoice_capture() is the main function which calls helper methods "capture_invoice_no_heading_present()" and
##capture_invoice_no_heading_absent()

########################################################################################################################
def invoice_capture(x):
    lst_index=[]
    #
    # Capture the indices of the heading words
    #
    lst_index=capture_indices_heading_words(x)
    #
    # Use the captured indices to extract the invoice numbers from the predicted line items
    #
    if(len(lst_index)>0):
        x=capture_invoice_no_heading_present(x,lst_index)

    # For cases where there are no headings or invoice no is not captured, extract the invoice numbers
    # from the predicted line items
    #
    x=capture_invoice_no_heading_absent(x)

    return x
#######################################################################################################################



##########################################################################################
##Regex to capture regex words
##########################################################################################

def credit_feature(x):
    s=str(x['row_string'])
    s=s.lower()
    if(re.findall(r"^(credit|cred)|(cred|credit)$|(\s+(cred|credit)\s+)|^(shortage|short)|(short|shortage)$|(\s+(short|shortage)\s+)",s,flags=re.MULTILINE)):
        return 1
    else:
        return 0


#############################################################################################




def check_amount(s):
    s=str(s)
    if ('$' in s):
        s = s.replace('$', ' ')

    s = s.replace(", ", ",")
    digits = re.findall(r"\-?\d+\.\s*\d+\-?|\-?\d{1,2}\,{1}\s*\d{1,3}\.{1}\s*\d{1,2}\-?", s,
                        flags=re.MULTILINE)
    return digits


def amount_capture_util(x):
    for i in range(0,len(x)):
        s=str(x['row_string'].values[i])
        if('total' in s):
            continue
        cred_val=0.0
        if(x['credit'].values[i]==1):
            cred_digit=check_amount(s)
            if(len(cred_digit)==0):
                continue
            else:
                for d in cred_digit:

                    d = str(d)
                    d = d.replace(',', '')
                    d = d.replace(' ', '')
                    if ('-' in d):
                        d = d.replace('-', '')
                        neg_val_cred = -float(d)
                        x['amount_captured'].values[i] = neg_val_cred
                        break
                    cred_val = max(cred_val, float(d))
                    x['amount_captured'].values[i]=-cred_val
                    continue




        digits=check_amount(s)

        ref=""
        lst=[]

        if((x['is_remittance_predicted'].values[i]==1) & (len(digits)==0) & (x['credit'].values[i]==0)):

            if((i+1<len(x)) and (x['is_remittance_predicted'].values[i+1]==0)  ):
                i=i+1
                s_new=str(x['row_string'].values[i])
                digits_new=check_amount(s_new)

                if(len(digits_new)>0):
                    val = 0.0
                    neg_val = 0.0
                    min_val=sys.maxsize
                    for digit in digits_new:
                        digit = str(digit)
                        digit = (digit.replace(',', ''))
                        digit = digit.replace(' ', '')
                        if ('-' in digit):
                            digit = digit.replace('-', '')
                            neg_val = -float(digit)
                            continue
                        val = max(val, float(digit))
                        min_val=min(min_val,float(digit))


                    if (neg_val != 0):
                        x['amount_captured'].values[i] = float(val + neg_val)


                    elif(min_val!=val):

                        x['amount_captured'].values[i]= float(val-min_val)


                    else:
                        x['amount_captured'].values[i] = float(val)


        elif((x['is_remittance_predicted'].values[i]==1) & (len(digits)>0) & (x['credit'].values[i]==0)):

            val = 0.0
            neg_val = 0.0
            min_val=sys.maxsize

            for digit in digits:
                digit = str(digit)
                digit = (digit.replace(',', ''))
                digit = digit.replace(' ', '')
                if (('-' in digit)):
                    digit = digit.replace('-', '')
                    neg_val = -float(digit)
                    continue

                val = max(val, float(digit))

                min_val=min(min_val,float(digit))

            if (neg_val != 0):
                x['amount_captured'].values[i]=float(val+neg_val)


            elif(min_val!=val):

                x['amount_captured'].values[i]=float(val-min_val)
            else:
                x['amount_captured'].values[i]=float(val)

    return x

def invoice_validate(x):
    if (x['invoice_number_captured'] == None):
        return 0.0
    else:
        return x['amount_captured']


def monitoring_litm_data(data_path,write_file_path):
    data=pd.read_csv(data_path)
    data = data[data['page_pageType'] == 'REMITTANCE_PAGE']
    #data['is_remittance_sum'] = data.groupby('check_checkNumber')['is_remittance_predicted'].transform('sum')
    data['invoice_number_captured'] = None
    g = data.groupby(['check_checkNumber', 'batch_id', 'page_pageNumber']).apply(invoice_capture)
    g['amount_captured'] = 0.0
    g['reference_number_captured'] = None
    g['row_string_pipe'] = g['row_string_pipe'].astype(str)
    g['row_string_pipe'] = g['row_string_pipe'].apply(lambda x: x.lower())
    g['credit'] = g.apply(lambda x: credit_feature(x), axis=1)

    check_group = g.groupby(['check_checkNumber','batch_id' ,'page_pageNumber'])
    filtered_data = check_group.apply(amount_capture_util)
    filtered_data.reset_index(inplace=True, drop=True)



    filtered_data['amount_captured'] = filtered_data.apply(lambda x: invoice_validate(x), axis=1)

    grouped_checks = filtered_data.groupby(by=['check_checkNumber', 'batch_id','page_pageNumber'])

    remittance_only_cheques_match = grouped_checks.filter(lambda x: x['amount_captured'].sum() == x['check_checkAmount'].values[0])
   # print(remittance_only_cheques_match['check_checkNumber'].nunique()/data['check_checkNumber'].nunique())
    remittance_only_cheques_match=remittance_only_cheques_match[['check_checkNumber','check_checkAmount','page_pageNumber','batch_id','row_string_pipe','is_heading_predicted','is_total_predicted','is_remittance_predicted','invoice_number_captured','amount_captured']]

    remittance_only_cheques_match.to_csv(write_file_path+"/correctly_closed_checks.csv",index=False,encoding="utf-8")
    filtered_data.to_csv(write_file_path+"/amount_captured_data.csv",index=False,encoding="utf-8")


    return data