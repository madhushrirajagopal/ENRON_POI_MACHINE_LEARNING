#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from pprint import pprint
from feature_format import featureFormat, targetFeatureSplit
from sklearn import preprocessing

def valid_value(value):
    if value == 'NaN':
        value = 0
    return value

def remove_outliers(data_dict, outliers):
    for item in outliers:
        data_dict.pop(item, 0)

def calc_cash_received(data_dict, fields):
    # Mutate dictionary vs. making a copy
    for key, value in data_dict.iteritems():
        for field in fields:
            cash_received = sum([valid_value(value[field]) for field in fields])
        total_value = valid_value(value['total_payments']) +\
                      valid_value(value['total_stock_value'])
        if total_value != 0:
            cash_received_pctg = float(cash_received) / total_value
        else:
            cash_received_pctg = 'NaN'
        # Add new fields
        value['cash_received'] = cash_received
        value['cash_received_pctg'] = cash_received_pctg
    return data_dict

def calc_poi_messages_pctg(data_dict):
    # Mutate dictionary vs. making a copy
    for key, value in data_dict.iteritems():
        total_messages = valid_value(value['from_messages']) +\
                         valid_value(value['to_messages'])
        poi_messages = valid_value(value['from_poi_to_this_person']) +\
                       valid_value(value['from_this_person_to_poi'])
        if total_messages != 0:
            poi_messages_pctg = float(poi_messages) / total_messages
        else:
            poi_messages_pctg = 'NaN'
        # Add new field
        value['poi_messages_pctg'] = poi_messages_pctg
    return data_dict

if __name__ == '__main__':
    # Load the dictionary containing the dataset
    data_dict = pickle.load(open("../project/final_project_dataset.pkl", "r"))
    outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
    # Remove outliers
    remove_outliers(data_dict, outliers)
    # Fields to be considered as cash received
    cash_fields = ['salary','bonus','exercised_stock_options','loan_advances']
    # Add cash received fields
    calc_cash_received(data_dict, cash_fields)
    # Add POI message %
    calc_poi_messages_pctg(data_dict)
    print "Done"
    #pprint(data_dict)