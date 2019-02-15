#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from pprint import pprint
from feature_format import featureFormat, targetFeatureSplit
from feature_update import valid_value, remove_outliers, calc_cash_received, calc_poi_messages_pctg 
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing

def select_k_best(data_dict, features_list, k):
    # Create dataset from feature list
    data = featureFormat(data_dict, features_list)
    # Split dataset into labels and features
    labels, features = targetFeatureSplit(data)
    # Create Min/Max Scaler
    scaler = preprocessing.MinMaxScaler()
    # Scale Features
    features = scaler.fit_transform(features)
    # Create k_best feature selection
    k_best = SelectKBest(k=k)
    # Fit k_best
    k_best.fit(features, labels)
    # Get k_best scores
    scores = k_best.scores_
    # Create list with features and scores
    unsorted_pairs = zip(features_list[1:], scores)
    # Sort list
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    # Create dict
    if k == 'all':
        k_best_features = dict(sorted_pairs)
    else:
        k_best_features = dict(sorted_pairs[:k])
    return k_best_features

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
    #pprint(data_dict)
    # Create features list
    features_list = ['poi',
                     'bonus',
                     'cash_received',
                     'cash_received_pctg',
                     'deferral_payments',
                     'deferred_income',
                     'director_fees',
                     'exercised_stock_options',
                     'expenses',
                     'loan_advances',
                     'long_term_incentive',
                     'other',
                     'restricted_stock',
                     'restricted_stock_deferred',
                     'salary',
                     'total_payments',
                     'total_stock_value',
                     'from_messages',
                     'to_messages',
                     'from_poi_to_this_person',
                     'from_this_person_to_poi',
                     'shared_receipt_with_poi',
                     'poi_messages_pctg']

    # Run k_best for all
    k = 'all'
    k_best_features = select_k_best(data_dict, features_list, k)
    # Print k_best features
    print "{0} best features: {1}\n".format(k, k_best_features.keys())
    pprint(k_best_features)