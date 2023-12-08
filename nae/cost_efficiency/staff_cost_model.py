import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f
from nae import nae_main as nae
import joblib
import pickle
import statsmodels.api as sm
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import constants as c
import seaborn as sns
import os
import sys

pd.set_option('display.max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

def plotter(df):
    plt.figure(figsize=(8, 8))
    scatter = sns.scatterplot(data=df, x='predicted', y='test_scores', hue='school', palette='viridis', s=100)
    scatter.axes.set_aspect('equal')
    max_value = df['test_scores'].max()
    plt.plot([0, max_value+50], [0, max_value+50], linestyle='--', color='gray')
    # plt.fill_between([0, max_value], [0 - 50, max_value - 50], [0 + 50, max_value + 50], color='gray', alpha=0.3)
    for i, row in df.iterrows():
        annotation_text = f"{row['school']} ({row['predicted']:.2f}, {row['test_scores']:.2f})"
        plt.annotate(annotation_text,
                     (row['predicted'], row['test_scores']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    plt.xlabel('Predicted')
    plt.ylabel('Actual Test Scores')
    plt.title('Predicted vs. Actual Test Scores by School')
    plt.legend(bbox_to_anchor=(.95, .25), loc='lower right')
    plt.tight_layout()
    plt.ylim(50, 100)
    plt.xlim(50, 100)
    plt.savefig(c.filenamer(f'nae/cost_efficiency/outputs/staff_cost_plot.png'))
    plt.show()

def col_cleaner(df):
    df.columns = df.columns.str.\
        lower().str. \
        replace(' ', '_').str. \
        replace('-', '_')
    return df

def calculator(df):
    return df.assign(
        staff_headcount=lambda x: x['teachers'] + x['non_teaching_staff'],
        total_school_cost=lambda x: x['total_teacher_cost'] + x['non_teaching_staff_cost'] + x['extra_curricular_cost']
    )

def currency_converter(df):
    return df.pipe(nae.currency_to_usd)

def data_create(load_new_df=True):
    if load_new_df:
        df = nae.staff_cost_data_create().\
                pipe(col_cleaner). \
                pipe(calculator).\
                pipe(currency_converter)
        df.pipe(joblib.dump, c.filenamer(f'nae/cost_efficiency/data/{MODEL}_staff_costs.pkl'))
    else:
        df = joblib.load(c.filenamer(f'nae/cost_efficiency/data/{MODEL}_staff_costs.pkl'))
    return df

def model(df):
    print(df)
    # sys.exit()
    X_columns = ['teachers', 'students', 'non_teaching_staff', 'extra_curricular_cost']

    X = df[X_columns]
    y = df['test_scores']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    coefficients = model.coef_
    intercept = model.intercept_

    equation = f"test_scores = {intercept:.2f} + "
    for i, (col, coef) in enumerate(zip(X_columns, coefficients)):
        equation += f"{coef:.10f}*{col}"
        if i < len(X_columns) - 1:
            equation += " + "
    f_statistic = (r2 / len(X_columns)) / ((1 - r2) / (len(y) - len(X_columns) - 1))
    p_value = 1 - f.cdf(f_statistic, len(X_columns), len(y) - len(X_columns) - 1)
    print("Equation:", equation)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print(f"F-statistic: {f_statistic:.2f}")
    print(f"Prob (F-statistic): {p_value:.4f}")

    df['predicted'] = model.predict(X)

    df['diff'] = df['test_scores'] - df['predicted']
    df = df[['school'] + X_columns + ['test_scores', 'predicted', 'diff']]
    print(df)
    c.saver(df, 'nae/cost_efficiency/outputs/staff_cost_preds.xlsx')
    # sys.exit()
    #TODO: use dummy for school in question (maybe school specific effects that are not captured)
    plotter(df)


if __name__ == '__main__':
    MODEL = 'DEMO'
    df = data_create(load_new_df=False)
    model(df)