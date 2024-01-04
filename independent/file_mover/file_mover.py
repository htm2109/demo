import pandas as pd
import constants as c
from faker import Faker
import msgpack
import shutil
import os
import sys
import time

start_time = time.time()

pd.set_option('display.max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

fake = Faker()

def generate_large_data(file_path, num_rows, chunk_size=1000):
    data = []

    for _ in range(num_rows):
        row = {
            "Name": fake.name(),
            "Email": fake.email(),
            "Phone": fake.phone_number(),
            "Address": fake.address()
        }
        data.append(row)

        if len(data) == chunk_size:
            write_data_to_msgpack(data, file_path)
            data = []

    if data:
        write_data_to_msgpack(data, file_path)

    print('Large MessagePack file generated and saved to loc1.')

def write_data_to_msgpack(data, file_path):
    with open(file_path, 'ab') as msgpack_file:
        packed_data = msgpack.packb(data)
        msgpack_file.write(packed_data)

def move_files_between_locations(move_to_loc2=True):
    source_location = c.filenamer('independent/file_mover/loc1/')
    destination_location = c.filenamer('independent/file_mover/loc2/') if move_to_loc2 else c.filenamer('independent/file_mover/loc1/')

    for file_name in os.listdir(source_location):
        source_path = os.path.join(source_location, file_name)
        destination_path = os.path.join(destination_location, file_name)
        shutil.move(source_path, destination_path)

    print(f'Files moved {"from loc1 to loc2" if move_to_loc2 else "from loc2 to loc1"}')

def main(gen_new_large_data, move_to_loc2):
    if gen_new_large_data:
        generate_large_data(os.path.join(c.filenamer('independent/file_mover/loc1/'), 'large_data.msgpack'), num_rows=100000)
    move_files_between_locations(move_to_loc2)

if __name__ == "__main__":
    main(gen_new_large_data=True, move_to_loc2=True)
    print('Files moved successfully')

    sys.exit(c.timer(start_time))
