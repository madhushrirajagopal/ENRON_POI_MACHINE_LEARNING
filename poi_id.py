
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#Data exploration to see summary statistics

import pandas as pd 
import pandasql
from copy import copy

def df(data_dict):
    df = pd.DataFrame()
    for key, value in data_dict.iteritems():
        new_df = value.copy()
        new_df['name'] = key
        new_df['count'] = 1
        df = df.append(new_df, ignore_index = True)
    return df

def number_of_values(data_dict):
    val = dict.fromkeys(data_dict.keys(),0)
    for item in val:
        value = data_dict[item]
        for key in value:
            if value[key] != 'NaN':
               val[item] += 1
    return val

# def  number_valid_values(data_dict):       

#     number_of_values = dict.fromkeys(data_dict.keys(), 0)
#     for item in number_of_values:
#         value = data_dict[item]
#     for key in value:
#         if value[key] != 'NaN':
#            number_of_values[item] += 1
#            return number_of_values

#Data Pre processing 

from pprint import pprint 
from sklearn import preprocessing 

#Function for replacing NaN's with 0 

def replace_missing(value):
    if value == 'NaN':
        value = 0
    return value

#Function for removing items in the outliers list 

def remove_outliers(data_dict,outliers):
    for item in outliers: 
        data_dict.pop(item,0)

#Function for creating the new feature to add all types of income received and the percentage of the income 

def total_income(data_dict, incomes):
    for key,value in data_dict.iteritems():
        for income in incomes:
            total_income = sum([replace_missing(value[income]) for income in incomes])

        #Calculating the total values from total payments and total stock value for calculating income percentage 
        total_value = replace_missing(value['total_payments']) +\
                      replace_missing(value['total_stock_value'])
        if total_value != 0:
               income_percentage = float(total_income) / total_value
        else: 
               income_percentage = 'NaN'
        # Creating this new features into the data dictionary 
        value['total_income'] = total_income
        value['income_percentage'] = income_percentage
    return data_dict

#Function for creating the new feature to calculate the percentage of messages sent and received by/to POI's for all individuals in the data

def poi_messages_percentage(data_dict):

    for key, value in data_dict.iteritems():
        total_messages = replace_missing(value['from_messages']) +\
        replace_missing(value['to_messages'])
        poi_messages = replace_missing(value['from_poi_to_this_person']) +\
        replace_missing(value['from_this_person_to_poi'])
        if total_messages != 0:
            poi_messages_percentage = float(poi_messages) / total_messages
        else:
            poi_messages_pctg = 'NaN'
        # Creating this new features into the data dictionary 
        value['poi_messages_percentage'] = poi_messages_percentage
    return data_dict 

#Before selecting the features, calling the above two functions in order to include the newly added features 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# creating df 
df = df(data_dict)

#print(len(data_dict))


#Identifying outliers by identifying features under the 5% quantile and above 95% quantile

def quantile_outliers(outlier, df):

    upper_limit = {}
    lower_limit = {}

    for item in outlier:
        check = df[item][df[item] != 'NaN']
        lower = check.quantile(.05)
        upper = check.quantile(.95)

        #Finding outliers under the lower 5% quantile
        outliers_lower = df[['name', item]][df[item] < lower]
        outliers_lower = outliers_lower[outliers_lower[item] != 'NaN']
        if len(outliers_lower) > 0:
            lower_limit[item] = zip(outliers_lower['name'])
        else:
            None
        #Finding outliers under the upper 95% quantile
        outliers_upper = df[['name', item]][df[item] > upper]
        outliers_upper = outliers_upper[outliers_upper[item] != 'NaN']
        if len(outliers_upper) > 0:
            upper_limit[item] = zip(outliers_upper['name'])
        else:
            None
    return upper_limit, lower_limit

outlier_list = [    'bonus',
'deferral_payments',
'deferred_income',
'director_fees',
'exercised_stock_options',
'expenses',
'loan_advances',
'long_term_incentive',
'restricted_stock',
'restricted_stock_deferred',
'salary',
'total_payments',
'total_stock_value'
]

upper_outlier, lower_outlier = quantile_outliers(outlier_list,df)

#pprint(upper_outlier)
#pprint(lower_outlier)


# valid_values = number_valid_values(data_dict)

# for key, value in valid_values.iteritems():
#     if value < 4:
#         #print (str(key), value)
#         #pprint(data_dict[key])
   

# Removing the outliers  from the data for further analysis

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
remove_outliers(data_dict, outliers)

### Task 3: Create new feature(s)

incomeType = ['salary', 'bonus', 'exercised_stock_options','loan_advances']
total_income(data_dict, incomeType)
poi_messages_percentage(data_dict)

### Store to my_dataset for easy export below.
my_dataset = data_dict



# Performing k best for selecting the features to be used in the algorithms with highest k scores 

from sklearn.feature_selection import SelectKBest

#Function for getting the K scores of the input features list 

def select_k_best(data_dict, features_list, k):

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    #Creating a Min/Max scaler
    scaler = preprocessing.MinMaxScaler()

    #Scale features
    features = scaler.fit_transform(features)

    #Using k best feature selection algorithm
    k_best =  SelectKBest(k=k)

    #Fit the k_best 
    k_best.fit(features, labels)

    #Finding out the k scores

    scores = k_best.scores_

    #Creating a list that has the features with the k scores
    feature_score_list = zip(features_list[1:],scores)

    #Sort the list to see the features with high scores
    sorted_score_list = list(reversed(sorted(feature_score_list, key=lambda x: x[1])))

    #Creating a dictionary of the k best features 
    if k =='all':
        k_best = dict(sorted_score_list)
    else:
        k_best = dict(sorted_score_list[:k])
        return k_best

#Determine the k scores for all the features for final feature selection 

features_list = ['poi',
'bonus',
'total_income',
'income_percentage',
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
'poi_messages_percentage']

k = 'all'
k_best = select_k_best(data_dict, features_list, k)

pprint(k_best)


features_list = [    'poi','total_income', 'total_payments', 
'total_stock_value']




### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



# Create Min/Max Scaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
# Scale Features
features = scaler.fit_transform(features)


# Try a variety of classifiers.

# GaussianNB

from sklearn.naive_bayes import GaussianNB
gb_clf = GaussianNB()


# DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
dc_clf = DecisionTreeClassifier(random_state = 45)


# KNeighborsClassifier


from sklearn.neighbors import KNeighborsClassifier

#clf = KNeighborsClassifier()

#Modified / Tuned algorithm 

clf = KNeighborsClassifier(algorithm = 'auto',
 leaf_size = 30,
 metric = 'minkowski',
 metric_params=None,  
 n_neighbors = 5,
 p = 2,
 weights = 'uniform')

# Ada Boost Classifier


from sklearn.ensemble import AdaBoostClassifier
ab_clf = AdaBoostClassifier(random_state = 45)

# Random Forest Classifier


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state = 45)

#Validation 

    # Split data using Stratified Shuffle Split

from sklearn.cross_validation import StratifiedShuffleSplit
cv_split = StratifiedShuffleSplit(labels, n_iter = 1, test_size = 0.3, random_state = 45)
for train_idx, test_idx in cv_split: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append(features[ii])
        labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])


# Cross-validation for K Neighbors Classifier



from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import numpy as np 

param_grid = {'n_neighbors': np.arange(2,10),'weights': ['uniform', 'distance'],
'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],'metric': ['minkowski','euclidean','manhattan'],
'leaf_size': [5, 10, 15, 20, 25, 30]}


knc_clf = GridSearchCV(clf,param_grid,cv = StratifiedKFold(labels_train, n_folds = 10),verbose = 0)
knc_clf = knc_clf.fit(features_train, labels_train)

print knc_clf.best_estimator_ 

# Prediction for K Neighbors Classifier

knc_pred = knc_clf.predict(features_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(labels_test, knc_pred)

print accuracy




dump_classifier_and_data(clf, my_dataset, features_list)