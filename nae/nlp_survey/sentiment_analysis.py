import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer
from nae import nae_main as nae
from nae.nlp_survey import survey_helpers as sh
import constants as c
import joblib
import sys
import time

# start the timer
start_time = time.time()

pd.set_option('display.max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

def add_q_prefix_to_column(df, column, prefix):
    df[column] = df[column].apply(lambda x: f"{prefix}{x}" if pd.notnull(x) else None)
    return df

def col_cleaner(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = add_q_prefix_to_column(df, 'where_school_needs_to_develop_translated', 'School needs to develop:  ')
    df = add_q_prefix_to_column(df, 'value_most_about_school_translated', 'I value most about the school: ')
    df = add_q_prefix_to_column(df, 'nps_score', 'From 1-10, I rate the school a ')
    return df

def subset(df):
    return df[['school', 'phase', 'year', 'nps_score'] + sh.translated_responses]

def add_nps_prefix_with_value(col):
    df[col] = df['nps_score'] + '. ' + df[col]
    return df

def data_create(load_new_df=True):
    if load_new_df:
        df = nae.gen_fake_survey_data().\
                pipe(col_cleaner)
        df.pipe(joblib.dump, c.filenamer(f'nae/nlp_survey/data/fake_survey_data.pkl'))
    else:
        df = joblib.load(c.filenamer(f'nae/nlp_survey/data/fake_survey_data.pkl'))
    df.\
        pipe(col_cleaner).\
        pipe(subset)
    return df

def sentiment_calculator(col):
    classifier = pipeline("sentiment-analysis",
                          model="distilbert-base-uncased-finetuned-sst-2-english",
                          tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english"))
    df[f'label_{col}'] = df[col].apply(lambda x: classifier(x)[0]['label'] if pd.notnull(x) and len(classifier.tokenizer.encode(x)) <= 512 else None)
    df[f'score_{col}'] = df[col].apply(lambda x: classifier(x)[0]['score'] if pd.notnull(x) and len(classifier.tokenizer.encode(x)) <= 512 else None)
    return df

if __name__ == '__main__':
    df = data_create(load_new_df=True)

    for col in sh.translated_responses:
        df = add_nps_prefix_with_value(col)
        df = sentiment_calculator(col)

    print(df.head())
    print(df['label_nps_follow_up_translated'].value_counts())
    print(df['label_value_most_about_school_translated'].value_counts())
    print(df['label_where_school_needs_to_develop_translated'].value_counts())
    print(df['label_multiple_children_feedback_translated'].value_counts())
    print(df['school'].value_counts())

    end_time = time.time()
    runtime = end_time - start_time
    hours = int(runtime / 3600)
    minutes = int((runtime % 3600) / 60)
    seconds = int(runtime % 60)
    print(f"Script runtime: hours: [{hours}], minutes: [{minutes}], seconds: [{seconds}]")
    sys.exit()

