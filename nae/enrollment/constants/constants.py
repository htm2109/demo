import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
def filenamer(path):
    return os.path.join(ROOT_DIR, path)


sf_path = '/Users/haydenmurray/PycharmProjects/sf/raw_data/'

sf_schools = [
    'Nord Anglia International School Hong Kong',
    'The British School of Beijing, Shunyi',
    'The British International School Shanghai, Puxi',
    'The British School of Guangzhou',
    'The British School of Beijing, Sanlitun',
    'Léman Chengdu International School',
    'Nord Anglia International School Shanghai, Pudong',
    'British School of Nanjing',
    'Nord Anglia International School of Guangzhou',
]

sf_schools_map = {
    'Nord Anglia International School Hong Kong': 'hongkong',
    'The British School of Beijing, Shunyi': 'shunyi',
    'The British International School Shanghai, Puxi': 'puxi',
    'The British School of Beijing, Sanlitun': 'sanlitun',
    'The British School of Guangzhou': 'bsg',
    'Léman Chengdu International School': 'chengdu',
    'Nord Anglia International School Shanghai, Pudong': 'pudong',
    'British School of Nanjing': 'nanjing',
    'Nord Anglia International School of Guangzhou': 'naisgz'
}

ordered_pipeline = ['Enquiry', 'Visit', 'Application', 'Acceptance', 'Enrolled']  # , 'Started']

categorical_columns = [
    'Child Gender', 'Parent Country of Residence', 'Country','Relocation City', 'Opportunity Record Type',
    'Account Record Type', 'Parent Nationality', 'Parent Preferred Language', 'Billing Country', 'Stage',
    'Opportunity Owner', 'Year/Grade', 'Fiscal Year', 'Lead Source', 'Previous Stage Before Lost/Denied',
    'Child Native Language', 'Child Nationality', 'Lead Method', 'School of Interest', 'school', 'year_group'
]

columns_to_drop = ['Opportunity ID', 'Opportunity Name', 'Child Full Name', 'Child Preferred Name', 'Child Birthdate',
     'Child Current Age', 'Child Gender', 'Account Record Type', 'Parent Country of Residence', 'Country',
     'Relocation City', 'Parent Nationality', 'Parent Preferred Language', 'Billing Country', 'Opportunity Record Type',
     'Stage', 'Created Date', 'Stage Duration', 'Last Stage Change Date', 'Account: Created Date',
     'Days between Enquiry and Enrolled', 'Days between Enquiry and Visit', 'Days between Visit and Application',
     'Days between Application and Acceptance', 'Days between Application and Enrolled',
     'Days between Acceptance and Enrolled', 'Days between Enrolled and Started', 'Won', 'Age',
     'Days since last Activity', 'School of Interest', 'Opportunity Owner', 'Year/Grade', 'Enrolment Month',
     'Fiscal Year', 'Lead Source', 'Staff Child', 'Previous Stage Before Lost/Denied',
     'Child Native Language', 'Child Student ID', 'Child Nationality', 'Full Account ID', 'Account Name', 'Lead Method',
     'Lead Submission Date', 'Parent Mobile', 'Enquiry Start Date', 'Visit Start Date', 'Opportunity First Visit Date',
     'Application Start Date', 'Acceptance Start Date', 'Enrolled Start Date', 'Start Date', 'Last Activity',
     'Last Activity Date RG', 'Account: Last Activity', 'Last Activity Date RG.1', 'Last Modified Date',
     'Account: Last Modified Date',
      'school', 'year_group', 'Lead Date', 'Last Activity (MAIN)']


# 'last_activity_main', 'pipeline_age',
# latter is a date, former has nans
# 'days_today_to_start', 'days_enq_to_start', 'days_enr_start', ...leakage
#     'days_since_last_stage_change', 'days_since_act', # todo what are these?, I think days_since... needs to go probably, things that are stage dependent need to be thrown out if bad
common_columns_to_keep = [
    'notes', 'jan_starter', 'sep_starter', 'iy_starter', 'chinese_heritage_proxy', 'created_date_age', 'days_enq_to_start', 'pipeline_age', 'days_lead_enq',
    'gender___Female', 'gender___Male', 'gender___Other', 'gender___Unknown', 'gender___MISSING', 'acc_rec__family',
    'acc_rec__ext_relation', 'parent_residence___China', 'parent_residence___Hong_Kong_SAR',
    'parent_residence___South_Korea', 'parent_residence___UK', 'parent_residence___USA', 'parent_residence___MISSING',
    'parent_residence___Other', 'country___China', 'country___Hong_Kong_SAR', 'country___South_Korea', 'country___UK',
    'country___USA', 'country___MISSING', 'country___Other', 'relocation_city___UK', 'relocation_city___Singapore',
    'relocation_city___Australia', 'relocation_city___US', 'relocation_city___MISSING', 'relocation_city___Other',
    'parent_nationality___American', 'parent_nationality___British', 'parent_nationality___Chinese',
    'parent_nationality___Chinese_Hong_Kong', 'parent_nationality___South_Korean', 'parent_nationality___MISSING',
    'parent_nationality___Other', 'parent_language___Chinese', 'parent_language___English', 'parent_language___German',
    'parent_language___Korean', 'parent_language___MISSING', 'parent_language___Other', 'billing_country___China',
    'billing_country___Hong_Kong_SAR', 'billing_country___South_Korea', 'billing_country___UK', 'billing_country___USA',
    'billing_country___MISSING', 'billing_country___Other', 'fyear___2023', 'fyear___2022', 'fyear___2021',
    'fyear___2020', 'fyear___2019', 'fyear___2050', 'fyear___Other', 'lead_source___Agent', 'lead_source___Call',
    'lead_source___Direct', 'lead_source___Email', 'lead_source___Events', 'lead_source___External_Relationships',
    'lead_source___Offline', 'lead_source___Online', 'lead_source___Organic_Social', 'lead_source___Paid_Referral',
    'lead_source___Paid_Social', 'lead_source___Referral', 'lead_source___Walk_In', 'lead_source___Word_of_mouth',
    'lead_source___MISSING', 'lead_source___Other', 'language___Chinese', 'language___English', 'language___German',
    'language___Korean', 'language___Cantonese', 'language___Japanese', 'language___MISSING', 'language___Other',
    'nationality___American', 'nationality___Australian', 'nationality___British', 'nationality___Canadian',
    'nationality___Chinese', 'nationality___South_Korean', 'nationality___MISSING', 'nationality___Other',
    'school___bsg', 'school___chengdu', 'school___hongkong', 'school___naisgz', 'school___nanjing', 'school___pudong',
    'school___puxi', 'school___sanlitun', 'school___shunyi', 'school___MISSING', 'year_group___Pre_Nursery',
    'year_group___Nursery', 'year_group___Reception', 'year_group___Year_1', 'year_group___Year_2',
    'year_group___Year_3', 'year_group___Year_4', 'year_group___Year_5', 'year_group___Year_6', 'year_group___Year_7',
    'year_group___Year_8', 'year_group___Year_9', 'year_group___Year_10', 'year_group___Year_11',
    'year_group___Year_12', 'year_group___Year_13', 'year_group___Other', 'year_group___MISSING'
]


def stage_columns(stage):
    if stage == 'Enquiry':
        return []
    elif stage == 'Visit':
        return ['days_enq_vis']
    elif stage == 'Application':
        return ['days_enq_vis', 'days_vis_app']
    elif stage == 'Acceptance':
        return ['days_enq_vis', 'days_vis_app', 'days_app_acc']
    elif stage == 'Enrolled':
        return ['days_enq_vis', 'days_vis_app', 'days_app_acc', 'days_acc_enr']
    else:
        print('something wong'); sys.exit()


date_columns = [
    'Child Birthdate', 'Last Stage Change Date', 'Created Date', 'Opportunity First Visit Date',
    'Application Start Date', 'Acceptance Start Date', 'Enrolled Start Date', 'Start Date'
]

recent_date_columns = [
    'Last Activity', 'Last Activity Date RG', 'Account: Last Activity', 'Last Activity Date RG.1', 'Last Modified Date',
    'Account: Last Modified Date'
]

stage_dates = [
    'Created Date', 'Opportunity First Visit Date', 'Application Start Date', 'Acceptance Start Date',
    'Enrolled Start Date', 'Start Date'
]


big_sf_year_group = {
    'BSI-Reception': 'Reception',
    'NHK-Nursery': 'Nursery',
    'BSI-Year 06': 'Year 6',
    'NHK-Year1': 'Year 1',
    'BPD-11': 'Year 11',
    'NHK-Year6': 'Year 6',
    'BPD-8': 'Year 8',
    'BSG-Year 1': 'Year 1',
    'BSI-Year 07': 'Year 7',
    'NHK-Year4': 'Year 4',
    'BPD-3': 'Year 3',
    'BSG-Year 2': 'Year 2',
    'BSI-Teddies': 'Pre-Nursery',
    'NHK-Year8': 'Year 8',
    'BPD-2': 'Year 2',
    'BSG-Year 3': 'Year 3',
    'BSI-Year 02': 'Year 2',
    'NHK-Year3': 'Year 3',
    'BPX-Year9': 'Year 9',
    'BPD-1': 'Year 1',
    'BSG-Nursery': 'Nursery',
    'NHK-Year7': 'Year 7',
    'BSI-Year 05': 'Year 5',
    'BPX-PreNursery': 'Pre-Nursery',
    'BPD-10': 'Year 10',
    'BSG-PreNursery': 'Pre-Nursery',
    'BSI-Year 10': 'Year 10',
    'NHK-Reception': 'Reception',
    'BPX-Reception': 'Reception',
    'BPD-6': 'Year 6',
    'BPX-Year13': 'Year 13',
    'NHK-Year5': 'Year 5',
    'BPD-R': 'Reception',
    'BSI-Year 01': 'Year 1',
    'BPX-Year1': 'Year 1',
    'BPD-PN': 'Pre-Nursery',
    'BSG-Toddler': 'Pre-Nursery',
    'BPX-Year12': 'Year 12',
    'BPD-7': 'Year 7',
    'BPX-Year2': 'Year 2',
    'BSI-Year 08': 'Year 8',
    'BPX-Year8': 'Year 8',
    'BSG-Reception': 'Reception',
    'BPD-N': 'Nursery',
    'NHK-Year2': 'Year 2',
    'BPD-4': 'Year 4',
    'BSG-Year 6': 'Year 6',
    'BPX-Year7': 'Year 7',
    'BSI-Year 09': 'Year 9',
    'BSI-Year 03': 'Year 3',
    'BPX-Year5': 'Year 5',
    'BSI-Year 04': 'Year 4',
    'BPX-Year6': 'Year 6',
    'BSI-Nursery': 'Nursery',
    'NHK-Year10': 'Year 10',
    'BPX-Year3': 'Year 3',
    'BPX-Cubs': 'Pre-Nursery',
    'NHK-Year9': 'Year 9',
    'BPX-Year10': 'Year 10',
    'BPD-5': 'Year 5',
    'BSI-Year 11': 'Year 11',
    'BPD-12': 'Year 12',
    'NHK-Year12': 'Year 12',
    'BPD-9': 'Year 9',
    'BSI-GK 2': 'Other',
    'BSI-Year 12': 'Year 12',
    'NHK-Year11': 'Year 11',
    'BPX-Nursery': 'Nursery',
    'BSI-Year 13': 'Year 13',
    'BPX-Year4': 'Year 4',
    'BSI-GK 1': 'Other',
    'BSI-GK 3': 'Other',
    'BSI-GK 4': 'Other',
    'NHK-Year13': 'Year 13',
    'BPX-Year11': 'Year 11',
    'LCI-Nursery': 'Nursery',
    'LCI-Year 7': 'Year 7',
    'LCI-Reception': 'Reception',
    'LCI-Year 3': 'Year 3',
    'LCI-Year 8': 'Year 8',
    'LCI-Year 9': 'Year 9',
    'LCI-Year 4': 'Year 4',
    'LCI-Year 1': 'Year 1',
    'LCI-Year 6': 'Year 6',
    'LCI-Year 10': 'Year 10',
    'LCI-Year 5': 'Year 5',
    'LCI-Year 2': 'Year 2',
    'LCI-Year 12': 'Year 12',
    'LCI-Year 11': 'Year 11',
    'LCI-PreNursery': 'Pre-Nursery',
    'NGM-PreNursery': 'Pre-Nursery',
    'NGM-Year 3': 'Year 3',
    'BSG-Year 7': 'Year 7',
    'BSG-Year 8': 'Year 8',
    'BSG-Year 4': 'Year 4',
    'BSG-Year 5': 'Year 5',
    'BSG-Year 10': 'Year 10',
    'BSG-Year 12': 'Year 12',
    'BSG-Year 9': 'Year 9',
    'NGM-Toddler': 'Pre-Nursery',
    'LCI-Year 13': 'Year 13',
    'NGM-Year 1': 'Year 1',
    'NGM-Year 2': 'Year 2',
    'NGM-Nursery': 'Nursery',
    'BSG-Year 11': 'Year 11',
    'BNJ-Year 2': 'Year 2',
    'BNJ-Year 3': 'Year 3',
    'BNJ-Nursery': 'Nursery',
    'BNJ-PreNursery': 'Pre-Nursery',
    'BNJ-Year 11': 'Year 11',
    'BNJ-Year 8': 'Year 8',
    'BNJ-Year 1': 'Year 1',
    'BNJ-Year 6': 'Year 6',
    'BNJ-Year 12': 'Year 12',
    'BNJ-Reception': 'Reception',
    'BNJ-Year 5': 'Year 5',
    'BSN-Year 2': 'Year 2',
    'BNJ-Year 4': 'Year 4',
    'BSN-Tadpoles': 'Pre-Nursery',
    'BSN-Reception': 'Reception',
    'BSN-Year 1': 'Year 1',
    'NGM-Reception': 'Reception',
    'BNJ-Year 7': 'Year 7',
    'BSN-Nursery': 'Nursery',
    'BSN-Year 4': 'Year 4',
    'NGM-Year 5': 'Year 5',
    'BNJ-Year 10': 'Year 10',
    'BSN-Year 5': 'Year 5',
    'BSN-Year 3': 'Year 3',
    'BNJ-Year 9': 'Year 9',
    'BSN-PreNursery': 'Pre-Nursery',
    'BSN-Year 6': 'Year 6',
    'NGM-Year 4': 'Year 4',
    'BSG-Year 13': 'Year 13',
    'NGM-Year 7': 'Year 7',
    'BPD-13': 'Year 13',
    'NGM-Year 6': 'Year 6',
    'BNJ-Year 13': 'Year 13',
    'Tadpoles': 'Pre-Nursery',
    'Toddler ': 'Pre-Nursery',
    'GK 1': 'Other',
    'GK 2': 'Other',
    'GK 3': 'Other',
    'Cubs': 'Pre-Nursery',
    'GK 4': 'Other',
}
