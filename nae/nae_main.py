import pandas as pd
from faker import Faker
import random
import numpy as np
import sys
# from forex_python.converter import CurrencyRates

fake = Faker()
def generate_fake_cost_data(num_schools):
    data = {
        'School': [fake.random_element(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']) for _ in range(num_schools)],
        'Region': [fake.random_element(['Region1', 'Region2']) for _ in range(num_schools)],
        'Teachers': [random.randint(50, 110) for _ in range(num_schools)],
        'Total Teacher Cost': [random.randint(40000000, 70000000) for _ in range(num_schools)],
        'Non-Teaching Staff': [random.randint(4, 6) for _ in range(num_schools)],
        'Non-Teaching Staff Cost': [random.randint(45000, 60000) for _ in range(num_schools)],
        'Extra-Curricular Cost': [random.randint(3300000, 12500000) for _ in range(num_schools)],
        'Test Scores': [random.randint(70, 90) for _ in range(num_schools)],
        'Students': [random.randint(650, 1200) for _ in range(num_schools)]
    }
    return pd.DataFrame(data)

def staff_cost_data_create():
    return generate_fake_cost_data(8)

def gen_fake_anova_data(num_students, reject_null_condition=True, normal_distribution=True):
    data = {
        'Student Name': [fake.name() for _ in range(num_students)],
        'Native-Language': [fake.random_element(['UK-English', 'USA-English', 'Chinese']) for _ in range(num_students)],
        'Writing-Score': []
    }

    for lang in data['Native-Language']:
        if reject_null_condition:
            if normal_distribution:
                if lang in ['UK-English', 'USA-English']:
                    data['Writing-Score'].append(int(np.random.normal(83, 3)))
                else:
                    data['Writing-Score'].append(int(np.random.normal(80, 6)))
            else:
                data['Writing-Score'].append(int(np.random.exponential(scale=10, size=1)))
        else:
            data['Writing-Score'].append(int(np.random.normal(82, 5)))

    return pd.DataFrame(data)

def student_data_create(reject_null_condition=True, normal_distribution=True):
    return gen_fake_anova_data(1000, reject_null_condition, normal_distribution)

def currency_to_usd(df):
    #todo: use currency converter lib

    # create an instance of CurrencyRates class
    # c = CurrencyRates()
    # get the exchange rate from RMB to USD
    # exchange_rate = c.get_rate('CNY', 'USD')
    # jenny_exchange_rate = 6.6792
    # # Multiply each value in the column by the exchange rate
    # for col in list_of_cols:
    #     df[col] = df[col] / jenny_exchange_rate
    region1 = ['A', 'B', 'C', 'D']
    region2 = ['E', 'F', 'G', 'H']
    for col in df.columns:
        if '_cost' in col:
            for school in region1:
                mask1 = df['school'] == school
                df.loc[mask1, col] = df.loc[mask1, col] * 1.013
            for school in region2:
                mask2 = df['school'] == school
                df.loc[mask2, col] = df.loc[mask2, col] / 1.023
    return df
