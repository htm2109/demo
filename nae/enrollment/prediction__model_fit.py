import sys
import time
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV

from tools import helpers__preprocess as hpp, helpers__model_train as hmt, helpers as h
# from constants import constants as c, prediction__constants as pc
from nae.enrollment.constants import constants as c, prediction__constants as pc

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def training_data_prep(date):
    df = h.gen_fake_sf_data(load_new_fake_data=True, num_records=40000, data_date=date)
    print(df.head())
    sys.exit()
    pd.read_csv(c.filenamer('../data/sf_export6.csv'), encoding='latin-1'). \
        pipe(hpp.stage_fix). \
        pipe(hpp.outcome).\
        pipe(hpp.splitter, date)


def scratch__pipeline_test(X_train, stage):
    print(stage)
    X_train.\
        pipe(hmt.raw_preprocess).\
        pipe(hmt.features_create).\
        pipe(hmt.field_filter, stage).\
        reset_index(drop=True).\
        pipe(lambda x: print(x.info(verbose=True, show_counts=True)))

    input('\n\nNext stage?')


def model_train(stage, date, pipeline_test=False):
    print(f'Model train for {stage}...\n')
    X_train = joblib.load(c.filenamer(f'../data/xtrain_{stage}__{date}.pkl'))
    y_train = joblib.load(c.filenamer(f'../data/ytrain_{stage}__{date}.pkl'))

    if pipeline_test:
        scratch__pipeline_test(X_train, stage)
    else:
        pipeline = Pipeline(
            [
                ('raw_preprocess', FunctionTransformer(hmt.raw_preprocess, validate=False)),
                ('feature_create', FunctionTransformer(hmt.features_create, validate=False)),
                ('feature_drop', FunctionTransformer(hmt.field_filter, kw_args={'stage': stage}, validate=False)),
                ('classifier', RandomForestClassifier())
            ]
        )

        random_search = RandomizedSearchCV(pipeline, pc.param_grid_randomized, n_iter=10, scoring='roc_auc', verbose=10, n_jobs=-1, cv=3)
        random_search.fit(X_train, y_train)

        print(random_search.best_params_)
        print(random_search.best_score_)
        joblib.dump(random_search, c.filenamer(f'../data/lr_{stage}__{date}.pkl'))


def model_validate():
    model = joblib.load(c.filenamer(f'../data/lr_Enquiry__2023.10.30.pkl'))

    ## for plotting feature importances
    h.plotter__feature_importances(model)

    # ## for plotting the tree
    h.plotter__example_tree(model, 0)

    ## count of rows in each stage
    h.stage_count()

    # ## best parameters
    h.best_parameters()

    # scoring current pipeline
    h.score_current_pipeline()

    ## batch scoring
    h.batch_data_scorer(model, save=True)

    # for scoring test or other data
    # h.score_test()


def main(data_date):
    training_data_prep(data_date)
    start_time = time.time()
    for stage in c.ordered_pipeline:
        # model_train(stage, data_date, pipeline_test=True)
        model_train(stage, data_date)
    h._total_time(start_time)

    model_validate()


if __name__ == '__main__':
    data_date = '9_20_23'
    main(data_date)
