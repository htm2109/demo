import sys
import time
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from faker import Faker
import random
from random import choices
import pickle
from nae.enrollment.constants import constants as c

def _total_time(start_time):
    # Calculate the runtime
    end_time = time.time()
    runtime = end_time - start_time

    # Calculate the components of time
    hours = int(runtime / 3600)
    minutes = int((runtime % 3600) / 60)
    seconds = int(runtime % 60)
    # Print the runtime in the desired format
    print(f"Training runtime: hours: [{hours}], minutes: [{minutes}], seconds: [{seconds}]")


def plotter__feature_importances(model):
    # todo: likely better to version these by model and not date of data
    feature_importances_names = model.best_estimator_['classifier'].feature_names_in_

    feature_importances_vals = model.best_estimator_['classifier'].feature_importances_
    indices = np.argsort(feature_importances_vals)[::-1]

    top_num = 15
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_num), feature_importances_vals[indices][:top_num], align='center')
    plt.xticks(range(top_num), feature_importances_names[indices][:top_num], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(c.filenamer('../paper_viz/feature_importances.png'))
    plt.show()


def plotter__example_tree(model, tree_index, new_tree=True):
    from tools import helpers__model_train as hmt
    if new_tree:
        X_train = joblib.load(c.filenamer(f'../data/xtrain_Enquiry__2023.10.30.pkl')). \
            pipe(hmt.raw_preprocess). \
            pipe(hmt.features_create). \
            pipe(hmt.field_filter, 'Enquiry'). \
            reset_index(drop=True)
        y_train = joblib.load(c.filenamer(f'../data/ytrain_Enquiry__2023.10.30.pkl'))

        clf = tree.DecisionTreeClassifier(max_depth=3)  # set hyperparameter
        clf.fit(X_train, y_train)

    # plot tree
        plt.figure(figsize=(18, 10))  # set plot size (denoted in inches)
        tree.plot_tree(clf, fontsize=10, feature_names=clf.feature_names_in_.tolist())
        plt.savefig(c.filenamer('../paper_viz/tree.png'))
    else:
        rf_tree = model.best_estimator_['classifier'].estimators_[tree_index]
        feature_names = model.best_estimator_['classifier'].feature_names_in_

        tree.plot_tree(rf_tree, feature_names=feature_names.tolist(), filled=True)
        plt.savefig(c.filenamer('../paper_viz/tree.png'), dpi=100)
        plt.show()


def batch_data_scorer(model, save=True):
    # todo: likely better to version these by model and not date of data
    # todo: just filter the data in the previous step, ie in the function that calls this one
    df = joblib.load(c.filenamer(f'../data/xtest_Enquiry__2023.10.30.pkl'))
    df = df.assign(
        starting_probability=lambda x: model.predict_proba(x)[:, 1],
        year_group=lambda x: x['Year/Grade'].map(c.big_sf_year_group).fillna('MISSING'),
        school=lambda x: x['School of Interest'].map(c.sf_schools_map).fillna('MISSING')
    )
    df = df[['Opportunity Name', 'Stage', 'school', 'year_group', 'Created Date', 'Start Date', 'starting_probability']]
    if save:
        df.to_excel(c.filenamer('../paper_viz/historical_sf_probs_combined.xlsx'), index=False)


def score_current_pipeline():
    output = pd.DataFrame()
    total_FTEs = 0
    for stage in c.ordered_pipeline:
        model = joblib.load(c.filenamer(f'../data/lr_{stage}__2023.10.30.pkl'))
        df = joblib.load(c.filenamer(f'../data/current_pipeline__2023.10.30.pkl')). \
            query(f'`Previous Stage Before Lost/Denied` == "{stage}"'). \
            assign(
            starting_probability=lambda x: model.predict_proba(x)[:, 1]
        )
        print(df.shape[0])
        print(df['starting_probability'].mean())
        print(f'Expected FTEs of those currently in {stage} stage: {df["starting_probability"].sum()}')
        total_FTEs += df["starting_probability"].sum()
        output = pd.concat([output, df], ignore_index=True)
    output = output.assign(
        year_group=lambda x: x['Year/Grade'].map(c.big_sf_year_group).fillna('MISSING'),
        school=lambda x: x['School of Interest'].map(c.sf_schools_map).fillna('MISSING')
    )
    output = output[['Opportunity Name', 'Stage', 'school', 'year_group', 'Created Date', 'Start Date', 'starting_probability']]
    output.to_excel(c.filenamer('../paper_viz/cp_sf_probs_combined.xlsx'), index=False)
    # output = output.pivot(index='year_group', columns='School of Interest', values='starting_probability').reset_index()
    # output.to_excel(c.filenamer('../paper_viz/cp_probs_school_pivot.xlsx'), index=False)
    print(f'\nTotal FTEs expected in current pipeline: {total_FTEs}')


def stage_count():
    for stage in c.ordered_pipeline:
        y = joblib.load(c.filenamer(f'../data/ytrain_{stage}__2023.10.30.pkl'))
        print(y.shape[0])
        print(y.sum())

    cp = joblib.load(c.filenamer(f'../data/current_pipeline__2023.10.30.pkl'))
    print(cp.shape[0])


def best_parameters():
    for stage in c.ordered_pipeline:
        model = joblib.load(c.filenamer(f'../data/lr_{stage}__2023.10.30.pkl'))
        print(model.best_score_)
        print(model.best_params_)


def gen_fake_sf_data(num_records, data_date, load_new_fake_data=True):
    if load_new_fake_data:
        fake = Faker()
        random.seed(42)
        rows = []
        for _ in range(num_records):
            row = {
                "Opportunity ID": fake.uuid4(),
                "Opportunity Name": fake.name(),
                "Child Full Name": fake.name(),
                "Child Preferred Name": fake.first_name(),
                "Child Birthdate": fake.date_of_birth(minimum_age=2, maximum_age=18).strftime("%d/%m/%Y"),
                "Child Current Age": round(random.uniform(2, 18), 1),
                "Child Gender": random.choice(["Male", "Female", "Other", "Unknown"]),
                "Account Record Type": random.choice(["Family", "External Relationships"]),
                "Parent Country of Residence": random.choice(["China", "Hong Kong SAR", "South Korea", "United Kingdom", "United States of America", "Other"]),
                "Country": random.choice(["China", "Hong Kong SAR", "South Korea", "United Kingdom", "United States of America", "Other"]),
                "Relocation City": random.choice(["UK", "Singapore", "Australia", "US", "Other"]),
                "Parent Nationality": random.choice(["American", "British", "Chinese", "Chinese - Hong Kong", "South Korean", "Other"]),
                "Parent Preferred Language": random.choice(["Chinese", "English", "German", "Korean", "Other"]),
                "Billing Country": random.choice(["China", "Hong Kong SAR", "South Korea", "United Kingdom", "United States of America", "Other"]),
                "Opportunity Record Type": fake.word(),
                "Stage": random.choice(["Enquiry", "Visit", "Application", "Acceptance", "Enrolled", "Started", "Denied", "Lost"]),
                "Created Date": fake.date_time_this_decade().strftime("%d/%m/%Y"),
                "Stage Duration": random.randint(0, 365),
                "Last Stage Change Date": fake.date_time_this_decade().strftime("%d/%m/%Y"),
                "Account: Created Date": fake.date_time_this_decade().strftime("%d/%m/%Y"),
                "Days between Enquiry and Enrolled": random.randint(0, 365),
                "Days between Enquiry and Visit": random.randint(0, 365),
                "Days between Visit and Application": random.randint(0, 365),
                "Days between Application and Acceptance": random.randint(0, 365),
                "Days between Application and Enrolled": random.randint(0, 365),
                "Days between Acceptance and Enrolled": random.randint(0, 365),
                "Days between Enrolled and Started": random.randint(0, 365),
                "Won": random.choice([0, 1]),
                "Age": random.randint(0, 100),
                "Days since last Activity": random.randint(0, 365),
                "School of Interest": random.choice(["A", "B", "C", "D", "E", "F", "G", "H"]),
                "Opportunity Owner": fake.name(),
                "Year/Grade": random.choice(["Pre-Nursery", "Nursery", "Reception", "Year 1", "Year 2", "Year 3",
                                                "Year 4", "Year 5", "Year 6", "Year 7", "Year 8", "Year 9",
                                                "Year 10", "Year 11", "Year 12", "Year 13", "Other"]),
                "Enrolment Month": fake.month(),
                "Fiscal Year": random.choice([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2050, "Other"]),
                "Lead Source": random.choice(["Agent", "Call", "Direct", "Email", "Events", "External Relationships",
                                                "Offline", "Online", "Organic Social", "Paid_Referral", "Paid Social",
                                                "Referral", "Walk-In", "Word of mouth", "Other"]),
                "Staff Child": random.choice([0, 1]),
                "Probability (%)": random.randint(0, 100),
                "Previous Stage Before Lost/Denied": random.choice(["Enquiry", "Visit", "Application", "Acceptance", "Enrolled", "Started"]),
                "Child Native Language": random.choice(["Chinese", "English", "German", "Korean", "Cantonese", "Japanese", "Other"]),
                "Child Student ID": fake.uuid4(),
                "Child Nationality": random.choice(["American", "Australian", "British", "Canadian", "Chinese", "South Korean", "Other"]),
                "Full Account ID": fake.uuid4(),
                "Account Name": fake.company(),
                "Lead Method": fake.word(),
                "Lead Submission Date": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Parent Mobile": fake.phone_number(),
                "Enquiry Start Date": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Visit Start Date": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Opportunity First Visit Date": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Application Start Date": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Acceptance Start Date": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Enrolled Start Date": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Start Date": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Last Activity": fake.word(),
                "Last Activity Date RG": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Account: Last Activity": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Last Activity Date RG.1": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Last Modified Date": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Account: Last Modified Date": fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M"),
                "Notes": fake.text()
            }
            for col in ['Child Birthdate', 'Gender', 'Probability (%)', 'Lead Source', 'Fiscal Year', 'Child Nationality']:
                if random.random() < 0.1:
                    row[col] = np.nan
            # Define corresponding coverage percentages
            coverage_percentages = {
                "Enquiry Start Date": 1.0,
                "Visit Start Date": 1.0,
                "Opportunity First Visit Date": 0.8,
                "Application Start Date": 0.6,
                "Acceptance Start Date": 0.4,
                "Enrolled Start Date": 0.2,
                "Start Date": 1.0,
            }

            generated_dates = {
                field: (
                    fake.date_time_this_decade().strftime("%d/%m/%Y %H:%M")
                    if choices([True, False], weights=[coverage_percentages[field], 1 - coverage_percentages[field]])[0]
                    else ""
                )
                for field in coverage_percentages.keys()
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        pickle.dump(df, open(c.filenamer(f'../data/fake_data_{data_date}.pkl'), 'wb'))
    else:
        with open(c.filenamer(f'../data/fake_data_{data_date}.pkl'), 'rb') as f:
            df = pickle.load(f)
    return df
