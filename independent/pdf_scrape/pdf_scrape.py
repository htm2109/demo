import pandas as pd
import constants as c
import requests
import shutil
import os
import sys
import time
import json
import camelot
from io import BytesIO

start_time = time.time()

pd.set_option('display.max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

PDF_URL = 'https://indicators.kauffman.org/wp-content/uploads/sites/2/2022/03/2021-Early-State-Entrepreneurship-National-Report.pdf'

def fetch_pdf():
    response = requests.get(PDF_URL)
    return BytesIO(response.content)

def extract_tables(pdf_content):
    temp_pdf_path = c.filenamer('independent/pdf_scrape/temp_pdf.pdf')
    with open(temp_pdf_path, 'wb') as temp_pdf_file:
        temp_pdf_file.write(pdf_content.getvalue())
    tables = camelot.read_pdf(temp_pdf_path, flavor='stream', pages='10')
    os.remove(temp_pdf_path)
    return tables

def saver(tables):
    combined_df = pd.concat([table.df for table in tables], ignore_index=True)
    header_row = combined_df[combined_df.iloc[:, 0].str.strip() == 'YEAR'].index[0]
    combined_df.columns = combined_df.iloc[header_row]
    combined_df = combined_df.iloc[header_row + 1:].reset_index(drop=True)
    combined_df = combined_df[combined_df.iloc[:, 0].astype(str).str.isnumeric()].reset_index(drop=True)
    combined_csv_path = c.filenamer('independent/pdf_scrape/tables/combined_tables.csv')
    combined_df.to_csv(combined_csv_path, index=False)

def main():
    pdf_content = fetch_pdf()
    tables = extract_tables(pdf_content)
    saver(tables)

if __name__ == "__main__":
    main()

    sys.exit(c.timer(start_time))
