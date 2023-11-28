import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
def filenamer(path):
    return os.path.join(ROOT_DIR, path)

def saver(df, file):
    df.to_excel(f'{ROOT_DIR}/{file}', index=False)


