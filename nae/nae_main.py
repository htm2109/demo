import pandas as pd
from faker import Faker
import random
import numpy as np
from forex_python.converter import CurrencyRates
import sys

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
                    data['Writing-Score'].append(int(np.random.normal(78, 6)))
            else:
                data['Writing-Score'].append(int(np.random.exponential(scale=10, size=1)))
        else:
            data['Writing-Score'].append(int(np.random.normal(82, 5)))

    return pd.DataFrame(data)

def student_data_create(reject_null_condition=True, normal_distribution=True):
    return gen_fake_anova_data(1000, reject_null_condition, normal_distribution)

def generate_random_values():
    return fake.random_int(min=0, max=100)

def finance_data_create():
    # Create a DataFrame
    rows = ["Pre-Nursery", "Nursery", "Reception", "Early Years", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5",
            "Year 6", "Primary", "Year 7", "Year 8", "Year 9", "Year 10", "Year 11", "Year 12", "Year 13", "Other",
            "Secondary", "Total FTE"]
    columns = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "FY Avg", "In Year Growth",
               "In Year Growth %"]

    data = [[generate_random_values() for _ in range(len(columns))] for _ in range(len(rows))]

    df = pd.DataFrame(data, columns=columns, index=rows)
    return df


def currency_to_usd(df, target_currency):
    c = CurrencyRates()
    regions = {
        'A': 'USD',
        'B': 'USD',
        'C': 'USD',
        'D': 'USD',
        'E': 'CNY',
        'F': 'CNY',
        'G': 'CNY',
        'H': 'HKD',
    }
    for col in df.columns:
        if '_cost' in col:
            for school, source_currency in regions.items():
                mask = df['school'] == school
                rate = c.get_rate(source_currency, target_currency)
                if school in ['E', 'F', 'G', 'H']:
                    df.loc[mask, col] = df.loc[mask, col] * rate
                else:
                    df.loc[mask, col] = df.loc[mask, col] / rate
    return df


def gen_fake_survey_data():
    fake = Faker()

    data = []
    for _ in range(15000):
        school = random.choice(['School A', 'School B', 'School C', 'School D', 'School E', 'School F'
                                'School G', 'School H', 'School I', 'School J', 'School K'])
        year = 'FY23'
        parent_number = fake.uuid4()
        language = random.choice(['English', 'Chinese simplified', 'Chinese Traditional', 'French', 'Korean'])
        broading = random.choice(['Day School', 'Boarding'])
        phase = random.choice(['Early / Primary', 'Primary', 'Secondary'])
        nps_score = random.randint(1, 10)
        nps_follow_up = fake.sentence()
        nps_follow_up_translated = fake.sentence()
        where_school_needs_to_develop_translated = fake.sentence()
        value_most_about_school_translated = fake.sentence()
        multiple_children_feedback_translated = fake.sentence()

        row = [school, year, parent_number, language, broading, phase, nps_score, nps_follow_up, nps_follow_up_translated,
               where_school_needs_to_develop_translated, value_most_about_school_translated, multiple_children_feedback_translated]
        data.append(row)

    columns = ['school', 'year', 'Parent number', 'Language', 'Broading', 'Phase', 'NPS Score', 'nps_follow_up', 'nps_follow_up_translated',
               'where_school_needs_to_develop_translated', 'value_most_about_school_translated', 'multiple_children_feedback_translated']
    df = pd.DataFrame(data, columns=columns)
    return df
