import numpy as np
import pandas as pd

def cont_outliers(df):
    """
    Apply outlier handling to continuous variables.
    - Exponential variables: capped at 90% percentile
    - Age: IQR method
    - avg_review_score: [clarification needed]
    """
    # Variables with exponential distribution - cap at 90% percentile
    exp_vars = ['total_orders', 'total_spend_usd', 'avg_order_value_usd', 
                'days_since_last_purchase', 'reviews_given', 'wishlist_items']
    
    for var in exp_vars:
        p975 = df[var].quantile(0.975)
        df[var] = np.where(df[var] > p975, p975, df[var])
    
    # Age: IQR method
    Q1 = df['age'].quantile(0.25)
    Q3 = df['age'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['age'] = np.where(df['age'] < lower_bound, lower_bound, df['age'])
    df['age'] = np.where(df['age'] > upper_bound, upper_bound, df['age'])
    
    return df