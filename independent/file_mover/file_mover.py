import pandas as pd
import constants as c
import requests
import shutil
import os
import sys
import time
import json

start_time = time.time()

pd.set_option('display.max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

API_BASE_URL = 'https://jsonplaceholder.typicode.com'

def fetch_large_data(file_path):
    response = requests.get(f'{API_BASE_URL}/posts')
    data = response.json()

    chunk_size = 1000
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        write_data_to_json(chunk, file_path)

    print('Large JSON file generated and saved to loc1.')

def write_data_to_json(data, file_path):
    with open(file_path, 'a') as json_file:
        for entry in data:
            json_str = json.dumps(entry)  # Convert dictionary to JSON string
            json_file.write(json_str)
            json_file.write('\n')

def move_files_between_locations(move_to_loc2=True):
    source_location = c.filenamer('independent/file_mover/loc1/')
    destination_location = c.filenamer('independent/file_mover/loc2/') if move_to_loc2 else c.filenamer('independent/file_mover/loc1/')

    os.makedirs(destination_location, exist_ok=True)

    for file_name in os.listdir(source_location):
        source_path = os.path.join(source_location, file_name)
        destination_path = os.path.join(destination_location, file_name)
        shutil.move(source_path, destination_path)

    print(f'Files moved {"from loc1 to loc2" if move_to_loc2 else "from loc2 to loc1"}')

def main(gen_new_large_data, move_to_loc2):
    destination_directory = c.filenamer('independent/file_mover/loc1/') if not move_to_loc2 else c.filenamer('independent/file_mover/loc2/')
    destination_file_path = os.path.join(destination_directory, 'large_data.json')

    if gen_new_large_data:
        fetch_large_data(destination_file_path)

    move_files_between_locations(move_to_loc2)

if __name__ == "__main__":
    main(gen_new_large_data=True, move_to_loc2=True)
    print('Files moved successfully')

    sys.exit(c.timer(start_time))
