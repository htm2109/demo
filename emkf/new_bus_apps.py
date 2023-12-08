import pandas as pd
import datetime as dt
import seaborn as sns
from itertools import product
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from kauffman.data import bfs
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# CLEAN DATA
df_whole = bfs(['BA_BA'], obs_level='us') \
    .sort_values('time') \
    [['naics', 'time', 'BA_BA']] \
    .assign(month=lambda x: x.time.dt.month) \
    .dropna()
df_whole.index = df_whole.time
df_whole.index = df_whole.index.to_period('M')
df = df_whole.query('time < "2020-01-01"')

# EXPLORATORY ANALYSIS
df.plot('time', 'BA_BA')  # Regular plot
df.plot('month', 'BA_BA', kind='scatter')  # Monthly patterns (annual seasonality)
sns.boxplot(x='month', y='BA_BA', data=df)  # Boxplot of annual seasonality
plot_acf(df.BA_BA.dropna())  # Autocorrelation plot

############## FUNCTIONS ################
import numpy as np
from sklearn.metrics import mean_squared_error as mse


def graph_preds(predictions_df, pred='predicted_mean', true='BA_BA', lower='lower', upper='upper',
                label_true='Actual Business \nApplications', label_pred='Predicted Business \nApplications'):
    fig, ax = plt.subplots()
    index = predictions_df.index.to_timestamp()
    plt.xticks(rotation=45)
    plt.grid()

    ax.plot(index, predictions_df[true], label=label_true)
    ax.plot(index, predictions_df[pred], label=label_pred)
    ax.fill_between(
        index, predictions_df[lower], predictions_df[upper],
        label='Prediction Interval', color='b', alpha=.1
    )
    ax.legend(loc='center', bbox_to_anchor=(1.25, .5),
              fancybox=True, shadow=True, ncol=1)


def run_arima(time_cutoff, order, trend, seasonal_order=(0, 0, 0, 0), graph=True):
    year_ahead = f'{int(time_cutoff[:4]) + 1}' + time_cutoff[4:]
    selected_df = df.query(f'time < "{time_cutoff}"')
    hold_out_df = df.query(f'"{time_cutoff}" <= time < "{year_ahead}"')

    # Fit model
    data = selected_df['BA_BA']
    model = ARIMA(endog=data, order=order, trend=trend, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # Get predictions for next year
    start_index, end_index = hold_out_df.index[0], hold_out_df.index[-1]
    predictions = model_fit.predict(start=start_index, end=end_index, freq='M')
    forecast_interval = model_fit.get_prediction(start=start_index, end=end_index, freq='M').conf_int()
    lower, upper = forecast_interval['lower BA_BA'], forecast_interval['upper BA_BA']

    # evaluate
    rmse = np.sqrt(mse(hold_out_df['BA_BA'], predictions))
    mae = np.mean(np.abs(hold_out_df['BA_BA'] - predictions))

    # prep output
    prediction_info = pd.DataFrame(predictions). \
        assign(lower=lower, upper=upper, BA_BA=hold_out_df['BA_BA'], rmse=rmse, mae=mae)

    # graph
    if graph:
        print(f'RMSE: {rmse}, MAE: {mae}')
        graph_preds(prediction_info)

    return prediction_info


def eval_all(order, trend_term, s_order, date_grid='default', graph=True):
    if date_grid == 'default':
        date_grid = [
            '2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01',
            '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01'
        ]

    try:
        results = pd.concat([
            run_arima(date, order, trend_term, s_order, graph=False)
            for date in date_grid
        ])

        mean_rmse, mean_mae = np.mean(results.rmse), np.mean(results.mae)

        if graph:
            print(f'Mean RMSE: {mean_rmse}, Mean MAE: {mean_mae}')
            graph_preds(results)

        return pd.DataFrame([[order, trend_term, s_order, mean_rmse, mean_mae]],
                            columns=['order', 'trend_term', 's_order', 'mean_rmse', 'mean_mae'])

    except:
        return pd.DataFrame()


def arima_cv(
        dates_grid=['2009-01-01', '2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01',
                    '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01'],
        trend_grid=['n', 't', 'ct', [1, 1, 1]],
        p_range=(0, 3), d_range=(0, 3), q_range=(0, 3),
        P_range=(0, 3), D_range=(0, 3), Q_range=(0, 3)
):
    # Get a list of all parameters we want to compare
    p_values, d_values, q_values = [list(range(*x)) for x in [p_range, d_range, q_range]]
    order_grid = list(product(p_values, d_values, q_values))

    P_values, D_values, Q_values, s_values = [list(range(*x)) for x in [P_range, D_range, Q_range]] + [[12]]
    s_order_grid = list(product(P_values, D_values, Q_values, s_values))

    # For each parameter set, train the model on "before cutoff" data, and then evaluate it on "1-2 years after cutoff" data
    results_df = pd.concat([
        eval_all(order, trend_term, s_order, dates_grid, graph=False)
        for order in order_grid
        for trend_term in trend_grid
        for s_order in s_order_grid
    ])

    return results_df


def dive_into_model(order=(1, 0, 1), trend='t', seasonal_order=(0, 0, 0, 12)):
    # fit model
    model = ARIMA(endog=df['BA_BA'], order=order, trend=trend, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # show in-sample fit
    print('In sample fit:')
    insample_intervals = model_fit.get_prediction().conf_int()
    lower, upper = insample_intervals['lower BA_BA'], insample_intervals['upper BA_BA']
    insample_preds = pd.DataFrame(model_fit.predict()). \
        assign(BA_BA=df.BA_BA, lower=lower, upper=upper)
    graph_preds(insample_preds[1:])

    rmse = np.sqrt(mse(df['BA_BA'][1:], insample_preds['predicted_mean'][1:]))
    mae = np.mean(np.abs(df['BA_BA'][1:] - insample_preds['predicted_mean'][1:]))
    print(f'RMSE: {rmse}, MAE: {mae}')

    # show year-by-year out-of-sample fit
    rmse_oos, mae_oos = eval_all(order, trend, seasonal_order)[['mean_rmse', 'mean_mae']].values[0]
    print(f'RMSE: {rmse_oos}, MAE: {mae_oos}')

    # evaluate error terms
    print('Residuals of model:')
    resids = df['BA_BA'] - insample_preds['predicted_mean']
    resids[1:].plot()  # Looks good--looks random, hovering around 0

    print('Autocorrelation of error terms:')
    plot_acf(resids[1:])  # not autocorrelated

    sns.distplot(resids[1:])  # normally distributed

    # show prediction for next year
    forecast_bounds = model_fit.get_prediction(start="2020-01", end="2021-12").conf_int()
    lower, upper = forecast_bounds['lower BA_BA'], forecast_bounds['upper BA_BA']
    forecast = model_fit.predict(start="2020-01", end="2021-12")
    graph_preds(df_whole[:'2021-12'].assign(predicted_mean=forecast, lower=lower, upper=upper))

    # Zoomed in to 2020-2021 values
    forecast_2020 = pd.DataFrame(forecast[:'2021-12']). \
        assign(BA_BA=df_whole.BA_BA, lower=lower, upper=upper)
    graph_preds(forecast_2020)


def get_yearbyyear(results):
    # To get results, just run to the "results" portion of eval_all function
    results['year'] = results.index.year
    for year in range(2010, 2021):
        results[f'predicted_mean_{year}'] = results.predicted_mean.where(results.year == year)
        results[f'lower_{year}'] = results.lower.where(results.year == year)
        results[f'upper_{year}'] = results.upper.where(results.year == year)

    fig, ax = plt.subplots()
    index = results.index.to_timestamp()
    plt.xticks(rotation=45)
    plt.grid()

    ax.plot(index, results["BA_BA"], label="True")

    for year in range(2010, 2021):
        ax.plot(index, results[f'predicted_mean_{year}'], color='orange')
        ax.fill_between(
            index, results[f'lower_{year}'], results[f'upper_{year}'],
            color='b', alpha=.1
        )

    legend_elements = [
        Line2D([0], [0], color='C0', label='Actual Business \nApplications'),
        Line2D([0], [0], color='orange', label='Predicted Business \nApplications'),
        Patch(facecolor='b', alpha=.1, label='Prediction Interval')
    ]
    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.25, .5),
              fancybox=True, shadow=True, ncol=1)


################### Get figures in brief #####################
def get_sd(df=df_whole):
    "This gets Figure A.1"

    df = df \
        [['time', 'BA_BA']] \
        .assign(year=lambda x: x.time.dt.year) \
        .groupby('year') \
        .agg(['std', 'mean']) \
        .reset_index()

    df.columns = ['year', 'BA_std', 'BA_mean']

    return df \
        .assign(cov=lambda x: x.BA_std / x.BA_mean)


def graph(df, index, ncol=1, legend=True):
    fig, ax = plt.subplots()
    plt.xticks(rotation=45)
    plt.grid()

    for c in df.columns:
        ax.plot(index, df[c], label=c)

    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncol)


def graph_BA():
    "This gets Figures 1 and 2"

    whole = df_whole[:"2022-01"][['BA_BA', 'time']] \
        .rename(columns={'BA_BA': 'New Business Applications'})
    df_2020 = whole["2020-01":'2021-12']

    graph(whole[['New Business Applications']], index=whole.time, legend=False)
    graph(df_2020[['New Business Applications']], index=df_2020.time, legend=False)


def graph_monthly_BA():
    "This gets Figures A.6 and A.7"

    df = df_whole \
        .assign(year=lambda x: x.time.dt.year, month=lambda x: x.time.dt.month) \
        .query('2004 < year < 2021') \
        .assign(
        year_avg=lambda x: x.groupby('year').transform('mean')['BA_BA'],
        frac_of_avg=lambda x: x.BA_BA / x.year_avg
    ) \
        [['frac_of_avg', 'month', 'year']].reset_index() \
        .pivot(index='month', columns='year', values='frac_of_avg')

    df2 = df \
        .assign(
        avg_2005_2019=lambda x: x[[y for y in range(2005, 2020)]] \
            .apply('mean', axis=1)
    ) \
        .rename(columns={'avg_2005_2019': 'Avg 2005-2019'}) \
        [[2020, 'Avg 2005-2019']]

    index = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'
    ]

    graph(df, index, ncol=2)
    graph(df2, index)