import sys
import pickle
sys.path.append("../tools/")

from copy import copy
from pprint import pprint
import pandas as pd
import pandasql

def create_df(dict_in):
    # Create empty dataframe
    df = pd.DataFrame()
    for key, value in dict_in.iteritems():
        d2 = value.copy()
        # Add name
        d2['name'] = key
        # Add cnt field
        d2['cnt'] = 1
        # Append to dataframe
        df = df.append(d2, ignore_index=True)
    return df

def check_missing_values(dict_in):       
    # Initiate dictionary to count missing values for each feature
    check = dict.fromkeys(dict_in.itervalues().next().keys(), 0)
    for key, value in dict_in.iteritems():
        for item in check:
            if value[item] == 'NaN':
                #print i
                check[item] += 1
    return check

def check_valid_values(dict_in):       
    # Initiate dictionary to count valid values for each person
    check = dict.fromkeys(dict_in.keys(), 0)
    for item in check:
        value = dict_in[item]
        for key in value:
            if value[key] != 'NaN':
                check[item] += 1
    return check

def poi_cnt(df):
    # Pandasql query to summarize poi counts               
    q = """
    select poi,sum(cnt) as cnt
    from df
    group by poi
    """
    # Execute SQL command against df
    cnt = pandasql.sqldf(q, locals())
    return cnt

if __name__ == '__main__':
    # Load the dictionary containing the dataset
    data_dict = pickle.load(open("../project/final_project_dataset.pkl", "r") )
    print "Total number of data points equals %i" % (len(data_dict))

    # Create df
    df = create_df(data_dict)
    #pprint(df.head())
    
    # Create missing values dict for each feature
    missing_values = check_missing_values(data_dict)
    print "Total number of features equals %i" % (len(missing_values))
    print "Each feature and count of missing values:"          
    pprint(missing_values)

    # Create valid values dict for each person
    valid_values = check_valid_values(data_dict)
    #print "Each person and count of valid values:"
    #pprint(valid_values)
    
    # Save data out to a csv for review
    file_out = "../project/enron_data.csv"
    df.to_csv(file_out)

    # POI counts
    cnt = poi_cnt(df)
    print "POI count summary:"
    pprint(cnt)

    pprint(df['name'][df['poi'] == True])
