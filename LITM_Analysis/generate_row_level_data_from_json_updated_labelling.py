import sys
import os
import glob
import json
import re


from flatten_json import flatten
from collections import defaultdict
import pandas as pd

def read_jsons(path, file_pattern, encoding='utf8'):
    os.chdir(path)
    for filename in glob.iglob(file_pattern, recursive=True):
        try:
            with open(filename, encoding=encoding) as json_data:
                yield json.load(json_data), filename
        except Exception as e:
            print(str(e), filename)


def check_data(data):
    if 'checkNumber' not in list(data['joasisMLCheckContext'].keys()):
        return True
    elif data['joasisMLCheckContext']['checkNumber'] is "":
        return True
    return False

## if flag ==0, then row_string is formed which are space separated
## else row_string is formed which is pipe separated

def create_row_string(merged_word_context_list,flag):
    if(flag==0):
        merged_string = ' '.join([merged_word['value'].replace('\r', '') for merged_word in merged_word_context_list])
    else:
        merged_string=' | '.join([merged_word['value'].replace('\r', '') for merged_word in merged_word_context_list])

    return merged_string


def prefix_dict(old_dict, prefix):
    new_dict = {prefix + key: old_dict[key] for key in old_dict.keys()}
    return new_dict


def get_section_dict(section_context_list, row_no):
    cum_rows = 0
    for section in section_context_list:
        cum_rows += section['noOfRows']
        if row_no <= cum_rows:
            return section

def get_spaces_between_words(mergedWordContext_List):
    left_coordinates = []
    right_coordinates = []
    top_coordinate = []
    bottom_coordinate = []
    for word_count in range(0,len(mergedWordContext_List)):
        merged_word = mergedWordContext_List[word_count]
        left_coordinates.append(int(merged_word["joasisItemCoordinates"]["left"]))
        right_coordinates.append(int(merged_word["joasisItemCoordinates"]["right"]))
        top_coordinate.append(int(merged_word["joasisItemCoordinates"]["top"]))
        bottom_coordinate.append(int(merged_word["joasisItemCoordinates"]["bottom"]))
    return left_coordinates,right_coordinates,int(sum(top_coordinate)/len(top_coordinate)),int(sum(bottom_coordinate)/len(bottom_coordinate))

def get_row_dict(row):
    row_dict = defaultdict()
    row_dict = {key: row[key] for key in row.keys() if key != 'joasisMLMergedWordContexts'}
    left_coordinates,right_coordinates,top_avg,bottom_avg = get_spaces_between_words(row['joasisMLMergedWordContexts'])
    row_dict['JosasisLRCoordinates_top'] = top_avg
    row_dict['JosasisLRCoordinates_bottom'] = bottom_avg
    row_dict['mergedWords_LeftCoordinates'] = str(left_coordinates)
    row_dict['mergedWords_RightCoordinates'] = str(right_coordinates)
    row_dict['row_string'] = create_row_string(row['joasisMLMergedWordContexts'],0)
    row_dict['row_string_pipe']= create_row_string(row['joasisMLMergedWordContexts'],1)
    row_dict['rowNumber'] = row_dict['rowNumber'] - 1
    row_dict.update(get_tag_count(row['joasisMLMergedWordContexts']))
    return flatten_prefix(row_dict, 'row_')


# Prefix then Flatten
def flatten_prefix(dictionary, prefix):
    return flatten(prefix_dict(dictionary, prefix))


# Get word tag counts
def get_tag_count(merged_word_list):
    tag_count_dict = defaultdict(int)
    for merged_word in merged_word_list:
        tag_count_dict[merged_word['tag']] += 1
    return tag_count_dict



########################################################################################################################
def create_row_df_util(final_dict_list):
    all_rows = pd.DataFrame(final_dict_list)
    all_rows['image_file'] = all_rows['file'].apply(lambda x: x[:x.find('ML')]).astype(str) + "images/" + all_rows[
        'page_pageNumber'].astype(str) + ".png"
    all_rows['check_checkNumber'] = all_rows['check_checkNumber'].astype(str)
    all_rows['batch_id'] = all_rows['file'].apply(lambda x: re.search('\d+', x).group(0))
    all_rows['unique_row_identifier'] = all_rows['check_checkNumber'].astype(str) + "-" + all_rows['page_pageNumber'].astype(str) \
                                         + "-" +  all_rows['row_rowNumber'].astype(str)+ "-"+ all_rows['batch_id'].astype(str)

    all_rows['unique_page_identifier'] = all_rows['check_checkNumber'].astype(str) + "-" + all_rows['page_pageNumber'].astype(str)

    return all_rows

def create_row_df(path, file_pattern="**/**.json"):
    data_generator = read_jsons(path, file_pattern)
    final_dict_list = []
    for data, filename in data_generator:
        if check_data(data):
            print("No check number")
            continue
        for row in data['joasisMLRowContexts']:
            final_dict = defaultdict()
            row_dict = get_row_dict(row)
            section_dict = get_section_dict(data['joasisMLSectionContexts'], row['rowNumber'])
            final_dict.update(row_dict)
            final_dict.update(flatten_prefix(section_dict, 'section_'))
            final_dict.update(flatten_prefix(data['joasisMLCheckContext'], 'check_'))
            final_dict.update(flatten_prefix(data['joasisMLPageContext'], 'page_'))
            final_dict['check_checkNumber'] = (final_dict['check_checkNumber'])
            final_dict['file'] = filename
            final_dict_list.append(final_dict)

    return final_dict_list

########################################################################################################################

def create_merged_word_df(path, file_pattern):
        data_generator = read_jsons(path, file_pattern)
        final_dict_list = []
        i = 0
        for data, filename in data_generator:
            if check_data(data):
                print("No check number")
                continue
            for merged_word in data['joasisMLMergedWordContexts']:
                final_dict = defaultdict()
                final_dict.update(flatten_prefix(merged_word, 'MergedWord_'))
                final_dict['MergedWord_value'] = final_dict['MergedWord_value'].replace('\r', '')
                final_dict['MergedWord_resultedValue'] = final_dict['MergedWord_resultedValue'].replace('\r', '')
                final_dict.update(flatten_prefix(data['joasisMLCheckContext'], 'check_'))
                final_dict.update(flatten_prefix(data['joasisMLPageContext'], 'page_'))
                final_dict['file'] = filename
                final_dict_list.append(final_dict)

        all_rows = pd.DataFrame(final_dict_list)
        print(len(final_dict_list))
        all_rows['MergedWord_rowNumber'] = all_rows['MergedWord_rowNumber'] - 1
        all_rows['check_checkNumber'] = all_rows['check_checkNumber'].astype(float)
        all_rows['batch_id'] = all_rows['file'].apply(lambda x: re.search('\d+', x).group(0))

        all_rows['unique_row_identifier'] = all_rows['check_checkNumber'].astype(str) + "-" + all_rows['page_pageNumber'].astype(str)\
                                            + "-" + all_rows['MergedWord_rowNumber'].astype(str) + "-" \
                                            + all_rows['batch_id'].astype(str)
        all_rows['unique_page_identifier'] = all_rows['check_checkNumber'].astype(str) + "-" + all_rows['page_pageNumber'].astype(str)

        return all_rows
########################################################################################################################
def generate_row_level_data_util(path,write_path):
    final_dict_list=create_row_df(path)
    data=create_row_df_util(final_dict_list)
    data=data.rename(columns={'row_row_string':'row_string','row_row_string_pipe':'row_string_pipe'})
    data.to_csv(write_path,index=False)
    return data
########################################################################################################################

