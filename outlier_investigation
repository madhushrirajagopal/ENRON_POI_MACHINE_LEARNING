import sys
import pickle
sys.path.append("../tools/")

from pprint import pprint
import pandas as pd

from data_exploration import create_df, check_valid_values

def check_quantile_outliers(outlier_check, df):
    # Initiate empty dicts
    outliers_lb = {}
    outliers_ub = {}

    for item in outlier_check:
        a = df[item][df[item] != 'NaN']
        lb = a.quantile(.05)
        ub = a.quantile(.95)

        # Lower quantile
        dfl = df[['name', item]][df[item] < lb]
        dfl = dfl[dfl[item] != 'NaN']
        if len(dfl) > 0:
            outliers_lb[item] = zip(dfl['name'])
        else:
            None
            #print "No lower quantile outliers for %s" % str(item)

        # Upper quantile
        dfu = df[['name', item]][df[item] > ub]
        dfu = dfu[dfu[item] != 'NaN']
        if len(dfu) > 0:
            outliers_ub[item] = zip(dfu['name'])
        else:
            None
            #print "No upper quantile outliers for %s" % str(item)
    return outliers_lb, outliers_ub

if __name__ == '__main__':
    ### Load the dictionary containing the dataset
    data_dict = pickle.load(open("../project/final_project_dataset.pkl", "r") )
    # Create list of fields to check for quantile outliers
    outlier_check = ['bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses', 'loan_advances', 'long_term_incentive', 'restricted_stock', 'restricted_stock_deferred', 'salary', 'total_payments', 'total_stock_value']

    # Create df
    df = create_df(data_dict)
    
    outliers_lb, outliers_ub = check_quantile_outliers(outlier_check, df)
    
    print "Lower quantile outlier results:"
    pprint(outliers_lb)
    print "Upper quantile outlier results:"
    pprint(outliers_ub)

    # Data for grand total row
    print "Grand total row data:"
    pprint(data_dict['TOTAL'])

    # Create valid values dict for each person
    valid_values = check_valid_values(data_dict)

    for key, value in valid_values.iteritems():
        if value < 4:
            print "Valid value count for %s is %i" % (str(key), value)
            pprint(data_dict[key])
            