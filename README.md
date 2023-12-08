# Overview
This repository includes examples of projects from a few past work projects. The purpose is to demonstrate the Python code used to conduct research, forecast, and perform other data science tasks. Each of the folders in this directory include subfolders for the respective data and outputs. More detailed descriptions of each of the folders are provided below:


# emkf
EVAs are key performance indicators. 

Schools are usually communicating and paying close attention to people in these stages. 
The pipeline typically goes in this order, but not always:

[Every potential new starter starts out in SF in the Leads dataset, which can be summarized as unqualified opportunities. They move to an enquiry in the opportunities dataset once it is determined they are qualified]

-> Enquiry -> Visit -> Application -> Acceptance -> Enrolled 

-> [At any point, the potential new starter can be moved to one of these stages: Started, Lost, Denied] 

- Enquiry Date = 'Created Date'
- Visit Date = 'Opportunity First Visit Date'
- Application Date = 'Application Start Date'
- Acceptance Date = 'Acceptance Start Date'
- Enrolled Date = 'Enrolled Start Date'
- Start Date = 'Start Date'

***We want to predict 'Started'***

# nae 
<pre>
assign(id=lambda x: x['opportunity_name'].astype(str) +\
                    x['child_student_id'].astype(str) +\
                    x['full_account_id'].astype(str) +\
                    x['created_date'].astype(str)).\
    drop_duplicates('id')
</pre>

Untangled pipeline - EVAs with a start date this academic year  
Tangled pipeline - EVAs for any start date

# Files
***sf_th_hm_ft.py*** 

- Hayden's development on the decision tree

***sf_pipeline_trends.py***

- Some code to look at different trends in the pipeline

***constants.py***

- This includes variables that may be used in more than one script 

***raw_data/***

- This includes the most recent raw data export of the pipeline from SalesForce

# Variables

***opportunity_id*** - unique ID to the opportunity, not necessarily the person or even the EVA. The same 'enquiry' might be listed twice and show up as two different rows with two different values for opportunity_id.

***days_since_last_activity*** - Days since last email, phone call, or any other interaction between the school and the opportunity




