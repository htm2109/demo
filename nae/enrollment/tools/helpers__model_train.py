import sys
import datetime
import numpy as np
import pandas as pd
from nae.enrollment.constants import constants as c

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.1f' % x)


def _clean_cats(df):
    return df.assign(
        school=lambda x: x['School of Interest'].map(c.sf_schools_map),
        year_group=lambda x: x['Year/Grade'].map(c.big_sf_year_group)
    )


def _fill_missing(df):
    # Fill with mean
    df['Child Birthdate'].fillna(df['Child Birthdate'].mean(), inplace=True)

    # Columns to fill with 0
    columns_to_fill_with_0 = ['Staff Child', 'Probability (%)']
    df[columns_to_fill_with_0] = df[columns_to_fill_with_0].fillna(0)

    return df

def raw_preprocess(df):
    return df.\
        pipe(_clean_cats).\
        pipe(_fill_missing)

def _notes(df):
    return df.assign(notes=np.where(df['Notes'].isna(), 0, 1))

def _sep_starter(df):
    return df.assign(sep_starter=lambda y: np.where((8 <= y['Start Date'].dt.month) & (y['Start Date'].dt.month <= 9), 1, 0))

def _jan_starter(df):
    return df.assign(jan_starter=lambda y: np.where((1 <= y['Start Date'].dt.month) & (y['Start Date'].dt.month <= 2), 1, 0))

def _iy_starter(df):
    return df.assign(iy_starter=lambda y: np.where(y['Start Date'].dt.month.isin([8, 9, 1, 2]), 0, 1))

def _chinese_heritage_proxy(df):
    return df.assign(chinese_heritage_proxy=lambda x: np.where(
            (
                (x['Child Nationality'] == 'Chinese') |
                (x['Child Native Language'].isin(['Chinese', 'Chinese - Mandarin'])) |
                (x['Parent Nationality'] == 'Chinese') |
                (x['Parent Preferred Language'] == 'Chinese') |
                (x['Parent Preferred Language'].str.contains('Mandarin', case=False, regex=True))
            ),
            1,
            0
        )
   )


def _last_activity(df):
    return df.assign(last_activity_main=lambda x: x[c.recent_date_columns].max(axis=1))


def _time_covars(df):
    today = pd.to_datetime(datetime.date.today())
    return df.assign(
        created_date_age=lambda x: (pd.to_datetime(x['Created Date'], format='%d/%m/%Y') - pd.to_datetime(x['Child Birthdate'], format='%d/%m/%Y')) / pd.to_timedelta(365, unit='D'),
        pipeline_age=lambda x: (pd.to_datetime(x['Last Stage Change Date'], format='%d/%m/%Y') - pd.to_datetime(x['Created Date'], format='%d/%m/%Y')).dt.days.astype(float),
        days_since_last_stage_change=lambda x: ((today - pd.to_datetime(x['Last Stage Change Date'], format='%d/%m/%Y')).dt.days).astype(float),
        days_since_act=lambda x: ((today - pd.to_datetime(x['last_activity_main'], format='%d/%m/%Y')).dt.days).astype(float),
        days_today_to_start=lambda x: ((today - pd.to_datetime(x['Start Date'], format='%d/%m/%Y')).dt.days).astype(float),
        days_lead_enq=lambda x: ((pd.to_datetime(x['Created Date'], format='%d/%m/%Y') - pd.to_datetime(x['Lead Date'], format='%d/%m/%Y')).dt.days).astype(float),
        days_enq_to_start=lambda x: ((pd.to_datetime(x['Start Date'], format='%d/%m/%Y') - pd.to_datetime(x['Created Date'], format='%d/%m/%Y')).dt.days).astype(float),
        days_enq_vis=lambda x: ((pd.to_datetime(x['Opportunity First Visit Date'], format='%d/%m/%Y') - pd.to_datetime(x['Created Date'], format='%d/%m/%Y')).dt.days).astype(float),
        days_vis_app=lambda x: ((pd.to_datetime(x['Application Start Date'], format='%d/%m/%Y') - pd.to_datetime(x['Opportunity First Visit Date'], format='%d/%m/%Y')).dt.days).astype(float),
        days_app_acc=lambda x: ((pd.to_datetime(x['Acceptance Start Date'], format='%d/%m/%Y') - pd.to_datetime(x['Application Start Date'], format='%d/%m/%Y')).dt.days).astype(float),
        days_acc_enr=lambda x: ((pd.to_datetime(x['Enrolled Start Date'], format='%d/%m/%Y') - pd.to_datetime(x['Acceptance Start Date'], format='%d/%m/%Y')).dt.days).astype(float),
    )


def _gender(df):
    return df.assign(
        gender___Female=lambda x: np.where(x['Child Gender'] == 'Female', 1, 0),
        gender___Male=lambda x: np.where(x['Child Gender'] == 'Male', 1, 0),
        gender___Other=lambda x: np.where(x['Child Gender'] == 'Other', 1, 0),
        gender___Unknown=lambda x: np.where(x['Child Gender'] == 'Unknown', 1, 0),
        gender___MISSING=lambda x: np.where(x['Child Gender'] != x['Child Gender'], 1, 0)
    )


def _acc_rec_type(df):
    return df.assign(
        acc_rec__family=lambda x: np.where(x['Account Record Type'] == 'Family', 1, 0),
        acc_rec__ext_relation=lambda x: np.where(x['Account Record Type'] == 'External Relationships', 1, 0)
    )


def _parent_residence(df):
    df['Parent Country of Residence'] = np.where(
        (df['Parent Country of Residence'].isin(['China', 'Hong Kong SAR', 'South Korea', 'United Kingdom', 'United States of America'])) | (df['Parent Country of Residence'] != df['Parent Country of Residence']),
        df['Parent Country of Residence'],
        'Other'
    )
    return df.assign(
        parent_residence___China=lambda x: np.where(x['Parent Country of Residence'] == 'China', 1, 0),
        parent_residence___Hong_Kong_SAR=lambda x: np.where(x['Parent Country of Residence'] == 'Hong Kong SAR', 1, 0),
        parent_residence___South_Korea=lambda x: np.where(x['Parent Country of Residence'] == 'South Korea', 1, 0),
        parent_residence___UK=lambda x: np.where(x['Parent Country of Residence'] == 'United Kingdom', 1, 0),
        parent_residence___USA=lambda x: np.where(x['Parent Country of Residence'] == 'United States of America', 1, 0),
        parent_residence___MISSING=lambda x: np.where(x['Parent Country of Residence'] != x['Parent Country of Residence'], 1, 0),
        parent_residence___Other=lambda x: np.where(x['Parent Country of Residence'] == 'Other', 1, 0)
    )


def _country(df):
    df['Country'] = np.where(
        (df['Country'].isin(['China', 'Hong Kong SAR', 'South Korea', 'United Kingdom', 'United States of America'])) | (df['Country'] != df['Country']),
        df['Country'],
        'Other'
    )
    return df.assign(
        country___China=lambda x: np.where(x['Country'] == 'China', 1, 0),
        country___Hong_Kong_SAR=lambda x: np.where(x['Country'] == 'Hong Kong SAR', 1, 0),
        country___South_Korea=lambda x: np.where(x['Country'] == 'South Korea', 1, 0),
        country___UK=lambda x: np.where(x['Country'] == 'United Kingdom', 1, 0),
        country___USA = lambda x: np.where(x['Country'] == 'United States of America', 1, 0),
        country___MISSING=lambda x: np.where(x['Country'] != x['Country'], 1, 0),
        country___Other=lambda x: np.where(x['Country'] == 'Other', 1, 0)
    )


def _relocation_city(df):
    df['Relocation City'] = np.where(
        (df['Relocation City'].isin(['UK', 'Singapore', 'Australia', 'US'])) | (df['Relocation City'] != df['Relocation City']),
        df['Relocation City'],
        'Other'
    )
    return df.assign(
        relocation_city___UK=lambda x: np.where(x['Relocation City'] == 'UK', 1, 0),
        relocation_city___Singapore=lambda x: np.where(x['Relocation City'] == 'Singapore', 1, 0),
        relocation_city___Australia=lambda x: np.where(x['Relocation City'] == 'Australia', 1, 0),
        relocation_city___US=lambda x: np.where(x['Relocation City'] == 'US', 1, 0),
        relocation_city___MISSING=lambda x: np.where(x['Relocation City'] != x['Relocation City'], 1, 0),
        relocation_city___Other=lambda x: np.where(x['Relocation City'] == 'Other', 1, 0)
    )


def _parent_nationality(df):
    df['Parent Nationality'] = np.where(
        (df['Parent Nationality'].isin(['Chinese', 'Chinese - Hong Kong', 'South Korean', 'British', 'American'])) | (df['Parent Nationality'] != df['Parent Nationality']),
        df['Parent Nationality'],
        'Other'
    )
    return df.assign(
        parent_nationality___American=lambda x: np.where(x['Parent Nationality'] == 'American', 1, 0),
        parent_nationality___British=lambda x: np.where(x['Parent Nationality'] == 'British', 1, 0),
        parent_nationality___Chinese=lambda x: np.where(x['Parent Nationality'] == 'Chinese', 1, 0),
        parent_nationality___Chinese_Hong_Kong=lambda x: np.where(x['Parent Nationality'] == 'Chinese - Hong Kong', 1, 0),
        parent_nationality___South_Korean=lambda x: np.where(x['Parent Nationality'] == 'South Korean', 1, 0),
        parent_nationality___MISSING=lambda x: np.where(x['Parent Nationality'] != x['Parent Nationality'], 1, 0),
        parent_nationality___Other=lambda x: np.where(x['Parent Nationality'] == 'Other', 1, 0)
    )


def _parent_language(df):
    df['Parent Preferred Language'] = np.where(df['Parent Preferred Language'].str.contains('Chinese'), 'Chinese', df['Parent Preferred Language'])
    df['Parent Preferred Language'] = np.where(
        (df['Parent Preferred Language'].isin(['Chinese', 'English', 'German', 'Korean'])) | (df['Parent Preferred Language'] != df['Parent Preferred Language']),
        df['Parent Preferred Language'],
        'Other'
    )
    return df.assign(
        parent_language___Chinese=lambda x: np.where(x['Parent Preferred Language'] == 'Chinese', 1, 0),
        parent_language___English=lambda x: np.where(x['Parent Preferred Language'] == 'English', 1, 0),
        parent_language___German=lambda x: np.where(x['Parent Preferred Language'] == 'German', 1, 0),
        parent_language___Korean=lambda x: np.where(x['Parent Preferred Language'] == 'Korean', 1, 0),
        parent_language___MISSING=lambda x: np.where(x['Parent Preferred Language'] != x['Parent Preferred Language'], 1, 0),
        parent_language___Other=lambda x: np.where(x['Parent Preferred Language'] == 'Other', 1, 0),
    )

def _billing_country(df):
    df['Billing Country'] = np.where(
        (df['Billing Country'].isin(['China', 'Hong Kong SAR', 'South Korea', 'United Kingdom', 'United States of America'])) | (df['Billing Country'] != df['Billing Country']),
        df['Billing Country'],
        'Other'
    )
    return df.assign(
        billing_country___China=lambda x: np.where(x['Billing Country'] == 'China', 1, 0),
        billing_country___Hong_Kong_SAR=lambda x: np.where(x['Billing Country'] == 'Hong Kong SAR', 1, 0),
        billing_country___South_Korea=lambda x: np.where(x['Billing Country'] == 'South Korea', 1, 0),
        billing_country___UK=lambda x: np.where(x['Billing Country'] == 'United Kingdom', 1, 0),
        billing_country___USA = lambda x: np.where(x['Billing Country'] == 'United States of America', 1, 0),
        billing_country___MISSING=lambda x: np.where(x['Billing Country'] != x['Billing Country'], 1, 0),
        billing_country___Other=lambda x: np.where(x['Billing Country'] == 'Other', 1, 0)
    )

def _fiscal_year(df):
    df['Fiscal Year'] = np.where(
        df['Fiscal Year'].isin([2023, 2022, 2021, 2020, 2019, 2050]),
        df['Fiscal Year'],
        'Other'
    )
    return df.assign(
        fyear___2023=lambda x: np.where(x['Fiscal Year'] == 2023, 1, 0),
        fyear___2022=lambda x: np.where(x['Fiscal Year'] == 2022, 1, 0),
        fyear___2021=lambda x: np.where(x['Fiscal Year'] == 2021, 1, 0),
        fyear___2020=lambda x: np.where(x['Fiscal Year'] == 2020, 1, 0),
        fyear___2019=lambda x: np.where(x['Fiscal Year'] == 2019, 1, 0),
        fyear___2050=lambda x: np.where(x['Fiscal Year'] == 2050, 1, 0),
        fyear___Other=lambda x: np.where(x['Fiscal Year'] == 'Other', 1, 0)
    )


def _lead_source(df):
    df['Lead Source'] = df['Lead Source'].replace(
        {
            'Event': 'Events',
            'Organic': 'Organic Social',
        }
    )
    return df.assign(
        lead_source___Agent=lambda x: np.where(x['Lead Source'] == 'Agent', 1, 0),
        lead_source___Call=lambda x: np.where(x['Lead Source'] == 'Call', 1, 0),
        lead_source___Direct=lambda x: np.where(x['Lead Source'] == 'Direct', 1, 0),
        lead_source___Email=lambda x: np.where(x['Lead Source'] == 'Email', 1, 0),
        lead_source___Events=lambda x: np.where(x['Lead Source'] == 'Events', 1, 0),
        lead_source___External_Relationships=lambda x: np.where(x['Lead Source'] == 'External Relationships', 1, 0),
        lead_source___Offline=lambda x: np.where(x['Lead Source'] == 'Offline', 1, 0),
        lead_source___Online=lambda x: np.where(x['Lead Source'] == 'Online', 1, 0),
        lead_source___Organic_Social=lambda x: np.where(x['Lead Source'] == 'Organic Social', 1, 0),
        lead_source___Paid_Referral=lambda x: np.where(x['Lead Source'] == 'Paid_Referral', 1, 0),
        lead_source___Paid_Social=lambda x: np.where(x['Lead Source'] == 'Paid Social', 1, 0),
        lead_source___Referral=lambda x: np.where(x['Lead Source'] == 'Referral', 1, 0),
        lead_source___Walk_In=lambda x: np.where(x['Lead Source'] == 'Walk-In', 1, 0),
        lead_source___Word_of_mouth=lambda x: np.where(x['Lead Source'] == 'Word of mouth', 1, 0),
        lead_source___MISSING=lambda x: np.where(x['Lead Source'] != x['Lead Source'], 1, 0),
        lead_source___Other=lambda x: np.where(x['Lead Source'] == 'Other', 1, 0)
    )


def _child_native_language(df):
    df['Child Native Language'] = np.where(df['Child Native Language'] == 'Chinese - Mandarin', 'Chinese', df['Child Native Language'])
    df['Child Native Language'] = np.where(df['Child Native Language'] == 'Chinese - Cantonese', 'Cantonese', df['Child Native Language'])
    df['Child Native Language'] = np.where(
        (df['Child Native Language'].isin(['Chinese', 'English', 'German', 'Korean', 'Cantonese', 'Japanese'])) | (df['Child Native Language'] != df['Child Native Language']),
        df['Child Native Language'],
        'Other'
    )
    return df.assign(
        language___Chinese=lambda x: np.where(x['Child Native Language'] == 'Chinese', 1, 0),
        language___English=lambda x: np.where(x['Child Native Language'] == 'English', 1, 0),
        language___German=lambda x: np.where(x['Child Native Language'] == 'German', 1, 0),
        language___Korean=lambda x: np.where(x['Child Native Language'] == 'Korean', 1, 0),
        language___Cantonese=lambda x: np.where(x['Child Native Language'] == 'Cantonese', 1, 0),
        language___Japanese=lambda x: np.where(x['Child Native Language'] == 'Japanese', 1, 0),
        language___MISSING=lambda x: np.where(x['Child Native Language'] != x['Child Native Language'], 1, 0),
        language___Other=lambda x: np.where(x['Child Native Language'] == 'Other', 1, 0),
    )


def _child_nationality(df):
    df['Child Nationality'] = np.where(
        (df['Child Nationality'].isin(['American', 'Chinese', 'South Korean', 'British', 'Canadian', 'Australian'])) | (df['Child Nationality'] != df['Child Nationality']),
        df['Child Nationality'],
        'Other'
    )
    return df.assign(
        nationality___American=lambda x: np.where(x['Child Nationality'] == 'American', 1, 0),
        nationality___Australian=lambda x: np.where(x['Child Nationality'] == 'Australian', 1, 0),
        nationality___British=lambda x: np.where(x['Child Nationality'] == 'British', 1, 0),
        nationality___Canadian=lambda x: np.where(x['Child Nationality'] == 'Canadian', 1, 0),
        nationality___Chinese=lambda x: np.where(x['Child Nationality'] == 'Chinese', 1, 0),
        nationality___South_Korean=lambda x: np.where(x['Child Nationality'] == 'South Korean', 1, 0),
        nationality___MISSING=lambda x: np.where(x['Child Nationality'] != x['Child Nationality'], 1, 0),
        nationality___Other=lambda x: np.where(x['Child Nationality'] == 'Other', 1, 0)
    )


def _school(df):
    return df.assign(
        school___bsg=lambda x: np.where(x['school'] == 'bsg', 1, 0),
        school___chengdu=lambda x: np.where(x['school'] == 'chengdu', 1, 0),
        school___hongkong=lambda x: np.where(x['school'] == 'hongkong', 1, 0),
        school___naisgz=lambda x: np.where(x['school'] == 'naisgz', 1, 0),
        school___nanjing=lambda x: np.where(x['school'] == 'nanjing', 1, 0),
        school___pudong=lambda x: np.where(x['school'] == 'pudong', 1, 0),
        school___puxi=lambda x: np.where(x['school'] == 'puxi', 1, 0),
        school___sanlitun=lambda x: np.where(x['school'] == 'sanlitun', 1, 0),
        school___shunyi=lambda x: np.where(x['school'] == 'shunyi', 1, 0),
        school___MISSING=lambda x: np.where(x['school'] != x['school'], 1, 0)
    )


def _year_group(df):
    return df.assign(
        year_group___Pre_Nursery = lambda x: np.where(x['year_group'] == 'Pre-Nursery', 1, 0),
        year_group___Nursery=lambda x: np.where(x['year_group'] == 'Nursery', 1, 0),
        year_group___Reception=lambda x: np.where(x['year_group'] == 'Reception', 1, 0),
        year_group___Year_1=lambda x: np.where(x['year_group'] == 'Year 1', 1, 0),
        year_group___Year_2=lambda x: np.where(x['year_group'] == 'Year 2', 1, 0),
        year_group___Year_3=lambda x: np.where(x['year_group'] == 'Year 3', 1, 0),
        year_group___Year_4=lambda x: np.where(x['year_group'] == 'Year 4', 1, 0),
        year_group___Year_5 = lambda x: np.where(x['year_group'] == 'Year 5', 1, 0),
        year_group___Year_6=lambda x: np.where(x['year_group'] == 'Year 6', 1, 0),
        year_group___Year_7=lambda x: np.where(x['year_group'] == 'Year 7', 1, 0),
        year_group___Year_8 = lambda x: np.where(x['year_group'] == 'Year 8', 1, 0),
        year_group___Year_9=lambda x: np.where(x['year_group'] == 'Year 9', 1, 0),
        year_group___Year_10 = lambda x: np.where(x['year_group'] == 'Year 10', 1, 0),
        year_group___Year_11 = lambda x: np.where(x['year_group'] == 'Year 11', 1, 0),
        year_group___Year_12 = lambda x: np.where(x['year_group'] == 'Year 12', 1, 0),
        year_group___Year_13 = lambda x: np.where(x['year_group'] == 'Year 13', 1, 0),
        year_group___Other=lambda x: np.where(x['year_group'] == 'Other', 1, 0),
        year_group___MISSING = lambda x: np.where(x['year_group'] != x['year_group'], 1, 0)
    )


def features_create(df):
    return df. \
        pipe(_notes).\
        pipe(_jan_starter).\
        pipe(_sep_starter).\
        pipe(_iy_starter). \
        pipe(_chinese_heritage_proxy). \
        pipe(_last_activity).\
        pipe(_time_covars).\
        pipe(_gender).\
        pipe(_acc_rec_type).\
        pipe(_parent_residence). \
        pipe(_country). \
        pipe(_relocation_city). \
        pipe(_parent_nationality). \
        pipe(_parent_language). \
        pipe(_billing_country). \
        pipe(_fiscal_year). \
        pipe(_lead_source). \
        pipe(_child_native_language). \
        pipe(_child_nationality). \
        pipe(_school). \
        pipe(_year_group)


def field_filter(df, stage):
    # todo: what other columns or features would be predictive?

    return df[c.common_columns_to_keep + c.stage_columns(stage)]
