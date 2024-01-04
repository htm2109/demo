import os
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
def filenamer(path):
    return os.path.join(ROOT_DIR, path)

def saver(df, file):
    df.to_excel(f'{ROOT_DIR}/{file}', index=False)

def timer(start_time):
    end_time = time.time()
    runtime = end_time - start_time
    hours = int(runtime / 3600)
    minutes = int((runtime % 3600) / 60)
    seconds = int(runtime % 60)
    print(f"\nScript runtime: hours: [{hours}], minutes: [{minutes}], seconds: [{seconds}]")

