import sys
import joblib
import numpy as np
import pandas as pd
from nae.enrollment.constants import constants as c
from sklearn.model_selection import train_test_split


def _date_time(df):
    date_cols = c.date_columns + c.recent_date_columns
    return df.assign(
                **{
                    col: pd.to_datetime(df[col], format='%d/%m/%Y %H:%M', errors='coerce')
                    if col in ['Opportunity First Visit Date', 'Another_Date_Column', 'Yet_Another_Date_Column']
                    else pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
                    for col in date_cols
                },
                lead_date=lambda x: pd.to_datetime(x['Enquiry Start Date'].str.split().str[0], format='%d/%m/%Y')
            )


def _missing_dates_backfill(df):
    df = df.\
        assign(
            start_date=lambda x: np.where(x['Start Date'] < x['Created Date'], x['Created Date'] + pd.DateOffset(months=6), x['Start Date']),
            enrolled_start_date=lambda x: np.where((x['Won'] == 1) & (x['Enrolled Start Date'] != x['Enrolled Start Date']), x['Start Date'], x['Enrolled Start Date']),
            acceptance_start_date=lambda x: np.where((x['enrolled_start_date'] == x['enrolled_start_date']) & (x['Acceptance Start Date'] != x['Acceptance Start Date']), x['enrolled_start_date'], x['Acceptance Start Date']),
            application_start_date=lambda x: np.where((x['acceptance_start_date'] == x['acceptance_start_date']) & (x['Application Start Date'] != x['Application Start Date']), x['acceptance_start_date'], x['Application Start Date']),
            opportunity_first_visit_date=lambda x: np.where((x['application_start_date'] == x['application_start_date']) & (x['Opportunity First Visit Date'] != x['Opportunity First Visit Date']), x['application_start_date'], x['Opportunity First Visit Date']),
            lead_date=lambda x: np.where(x['lead_date'].isna(), x['Created Date'], x['lead_date'])
        )
    return df


def _wrong_dates_backfill(df):
    return df.\
        assign(
            lead_date=lambda x: np.where(x['Created Date'] < x['lead_date'], x['Created Date'], x['lead_date']),
            start_date=lambda x: np.where(x['Start Date'] < x['enrolled_start_date'], x['enrolled_start_date'], x['start_date']),
            opportunity_first_visit_date=lambda x: np.where(x['opportunity_first_visit_date'] < x['Created Date'], x['Created Date'], x['opportunity_first_visit_date']),
            application_start_date=lambda x: np.where(x['application_start_date'] < x['Opportunity First Visit Date'], x['Opportunity First Visit Date'], x['application_start_date']),
            acceptance_start_date=lambda x: np.where(x['acceptance_start_date'] < x['Application Start Date'], x['Application Start Date'], x['acceptance_start_date']),
            enrolled_start_date=lambda x: np.where(x['enrolled_start_date'] < x['Acceptance Start Date'], x['Acceptance Start Date'], x['enrolled_start_date']),
        )


def _kpi_dates(df):
    return df. \
        pipe(_missing_dates_backfill).\
        pipe(_wrong_dates_backfill)


def _prev_stage_b4_ld(x):
    stage_num = x[['Created Date', 'opportunity_first_visit_date', 'application_start_date', 'acceptance_start_date', 'enrolled_start_date']].count()

    if x['Stage'] == 'Started':
        return x['Stage']
    elif stage_num == 1:
        return 'Enquiry'
    elif stage_num == 2:
        return 'Visit'
    elif stage_num == 3:
        return 'Application'
    elif stage_num == 4:
        return 'Acceptance'
    elif stage_num == 5:
        return 'Enrolled'
    print('something wong'); sys.exit()


def _last_stage_change(df):
    return df.\
        assign(
            last_stage_change_date=lambda x: x['Last Stage Change Date'].combine_first(df[c.stage_dates].max(axis=1)),
            previous_stage_before_lost_denied=lambda x: x.apply(_prev_stage_b4_ld, axis=1),
        )


def _field_filter_update(df):
    return df.\
        drop(labels=
             [
                 'Start Date', 'Opportunity First Visit Date', 'Application Start Date', 'Acceptance Start Date',
                 'Enrolled Start Date', 'Last Stage Change Date', 'Previous Stage Before Lost/Denied'
             ],
            axis=1
        ).\
        rename(columns=
            {
                'lead_date': 'Lead Date',
                'start_date': 'Start Date',
                'opportunity_first_visit_date': 'Opportunity First Visit Date',
                'application_start_date': 'Application Start Date',
                'acceptance_start_date': 'Acceptance Start Date',
                'enrolled_start_date': 'Enrolled Start Date',
                'last_stage_change_date': 'Last Stage Change Date',
                'previous_stage_before_lost_denied': 'Previous Stage Before Lost/Denied'
            }
        )


def stage_fix(df):
    return df. \
        pipe(_date_time). \
        pipe(_kpi_dates). \
        pipe(_last_stage_change).\
        pipe(_field_filter_update)


def outcome(df):
    return df.assign(outcome=lambda x: np.where(x['Stage'] == 'Started', 1, 0))


def _current_pipline_filter(df, date):
    """Saves current pipeline separately."""
    df.\
        query(f'Stage in {c.ordered_pipeline}').\
        pipe(joblib.dump, c.filenamer(f'../data/current_pipeline__{date}.pkl'))

    return df.\
        query(f'Stage not in {c.ordered_pipeline}')


def _stage_filter(df, stage):
    if stage == 'Enquiry':
        return df
    elif stage == 'Visit':
        return df.query(f'`Previous Stage Before Lost/Denied` != "Enquiry"')
    elif stage == 'Application':
        return df.query(f'`Previous Stage Before Lost/Denied` not in ["Enquiry", "Visit"]')
    elif stage == 'Acceptance':
        return df.query(f'`Previous Stage Before Lost/Denied` not in ["Enquiry", "Visit", "Application"]')
    elif stage == 'Enrolled':
        return df.query(f'`Previous Stage Before Lost/Denied` not in ["Enquiry", "Visit", "Application", "Acceptance"]')
    else:
        print('something wong'); sys.exit()


def _stage_splitter(df, stage, filename):
        df_subset = df.pipe(_stage_filter, stage)
        y = df_subset['outcome']
        X = df_subset.drop(labels='outcome', axis=1)
        #todo: fix faker data issues
        print(y)
        print(X)
        sys.exit()
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        joblib.dump(X, c.filenamer(f'../data/x{filename}.pkl'))
        joblib.dump(y, c.filenamer(f'../data/y{filename}.pkl'))


def _test_stage_splitter(df, filename):
    y = df['outcome']
    X = df.drop(labels='outcome', axis=1)

    joblib.dump(X, c.filenamer(f'../data/x{filename}.pkl'))
    joblib.dump(y, c.filenamer(f'../data/y{filename}.pkl'))


def splitter(df, date):
    """
    Subsets the raw dataset according to
        (1) current pipeline/ther
        (2) stage
        (3) train/test
    """
    df_historical = df.\
        pipe(_current_pipline_filter, date)

    df_train = df_historical.query('`Start Date` < "2023-08-01"')
    df_test = df_historical.query('"2023-10-01" >= `Start Date` >= "2023-08-01"')
    for stage in c.ordered_pipeline:
        _stage_splitter(df_train, stage, f'train_{stage}__{date}')
        _stage_splitter(df_test, stage, f'test_{stage}__{date}')

    df_test.pipe(_test_stage_splitter, f'test__{date}')
