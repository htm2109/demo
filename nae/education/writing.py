import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from nae import nae_main as nae
import joblib
import constants as c
import sys
from scipy.stats import f_oneway
from statistics import mean, stdev
from statsmodels.stats.multicomp import pairwise_tukeyhsd

pd.set_option('display.max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

def col_cleaner(df):
    df.columns = df.columns.str.\
        lower().str. \
        replace(' ', '_').str. \
        replace('-', '_')
    return df

def data_create(load_new_df=True):
    if load_new_df:
        df = nae.student_data_create().\
                pipe(col_cleaner)
        df.pipe(joblib.dump, c.filenamer(f'nae/education/data/{MODEL}_staff_costs.pkl'))
    else:
        df = joblib.load(c.filenamer(f'nae/education/data/{MODEL}_staff_costs.pkl'))
    return df

def investigator(df):
    print(df.head())

    # Descriptive Statistics
    print("\nDescriptive Statistics:")
    print(df.groupby('native_language')['writing_score'].describe())

    # Histograms with side-by-side bars
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='writing_score', hue='native_language', bins=20, kde=True, multiple="dodge")
    plt.title('Distribution of Writing Scores by Native Language')
    plt.xlabel('Writing Score')
    plt.ylabel('Frequency')
    # plt.show()
    return df

def check_normality(df):
    print("\nShapiro-Wilk p-values tell us whether the data is likely normally distributed. A higher p-value (> 0.05) suggests normality.")
    print("Skewness measures the asymmetry of the distribution. Positive skewness indicates a right-skewed distribution, and negative skewness indicates a left-skewed distribution.")
    print("Kurtosis measures the tail heaviness of the distribution. Positive kurtosis indicates heavy tails, and negative kurtosis indicates light tails. \n ")

    # Shapiro-Wilk test for normality
    for lang in df['native_language'].unique():
        subset = df[df['native_language'] == lang]['writing_score']
        stat_sw, p_value_sw = stats.shapiro(subset)
        # Calculate skewness and kurtosis
        skewness = stats.skew(subset)
        kurtosis = stats.kurtosis(subset)

        print(f"Statistics for {lang}:")
        print(f"  Shapiro-Wilk test: Statistic={stat_sw}, p-value={p_value_sw}")
        print(f"  Skewness: {skewness}")
        print(f"  Kurtosis: {kurtosis}\n")
    return df

def one_way_anova(df):
    print("\nOne-way ANOVA:")
    k = len(df['native_language'].unique())
    N = len(df['writing_score'])

    # Calculate degrees of freedom
    df_between = k - 1
    df_within = N - k

    # Significance level (alpha)
    alpha = 0.05

    # Critical value from the F-distribution table
    critical_value = stats.f.ppf(1 - alpha, df_between, df_within)

    # Perform one-way ANOVA
    anova_results = stats.f_oneway(
        *[df[df['native_language'] == lang]['writing_score'] for lang in df['native_language'].unique()])

    # Print results
    print(f"F-statistic: {anova_results.statistic}")
    print(f"Degrees of Freedom (Between): {df_between}")
    print(f"Degrees of Freedom (Within): {df_within}")
    print(f"Critical Value: {critical_value}")
    print(f"P-value: {anova_results.pvalue}")

    # sys.exit()

    # Interpretation
    if anova_results.pvalue < alpha:
        print("\nThe p-value is less than 0.05, suggesting that there are significant differences in writing scores across native languages.")
    else:
        print("\nThe p-value is greater than 0.05, suggesting that there are no significant differences in writing scores across native languages.")
    if anova_results.statistic > critical_value:
        print("Moreover, the F-statistic is greater than the critical value, reinforcing the evidence of significant differences.\n")
    else:
        print("However, the F-statistic is not greater than the critical value, so additional caution in interpretation may be warranted.\n")
    return df

def tukey_adhoc(df):
    # Perform Tukey's HSD post hoc test
    tukey_results = pairwise_tukeyhsd(df['writing_score'], df['native_language'])
    print("Tukey's HSD Post Hoc Test:\n" + str(tukey_results) + "\n")
    return df

def effect_size(df):
    def cohens_d(group1, group2):
        diff = mean(group1) - mean(group2)
        pooled_std = ((stdev(group1) ** 2 + stdev(group2) ** 2) / 2) ** 0.5
        return diff / pooled_std

    # Calculate Cohen's d for each pair of groups
    languages = df['native_language'].unique()
    for i in range(len(languages)):
        for j in range(i + 1, len(languages)):
            lang1 = languages[i]
            lang2 = languages[j]
            d = cohens_d(df[df['native_language'] == lang1]['writing_score'],
                         df[df['native_language'] == lang2]['writing_score'])
            print(f"Cohen's d between {lang1} and {lang2}: {d}")
    return df

def box_whisker(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='native_language', y='writing_score')
    plt.title('Boxplot of Writing Scores by Native Language')
    plt.xlabel('Native Language')
    plt.ylabel('Writing Score')
    # plt.show()
    return df

if __name__ == '__main__':
    MODEL = 'DEMO'
    df = data_create(load_new_df=False)
    df = investigator(df)
    df = check_normality(df)
    df = one_way_anova(df)
    df = tukey_adhoc(df)
    df = effect_size(df)
    df = box_whisker(df)
    sys.exit()


