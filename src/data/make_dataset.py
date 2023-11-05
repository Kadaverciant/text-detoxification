import requests
from zipfile import ZipFile
import os
import pandas as pd

def get_dataset(path, filename, url):
    response = requests.get(url)
    zip_path = os.path.join(path, filename)
    with open(zip_path, "wb") as file:
        file.write(response.content)
    return zip_path   

def unzip_dataset(zip_path, destination_path):
    with ZipFile(zip_path, "r") as file:
        file.extractall(destination_path)

def create_dataset(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.rename(columns={'lenght_diff': 'length_diff'})
    return df

def filter_dataset(df, toxicity_threshold, non_toxicity_threshold):
    df_ref_tox = df[(df['ref_tox'] > toxicity_threshold) & (df['trn_tox'] < non_toxicity_threshold)]
    df_ref_tox = df_ref_tox.rename(columns={'reference': 'toxic', 'translation': 'non-toxic', 'ref_tox':'toxic_metric', 'trn_tox':'non-toxic_metric'})
    df_trn_tox = df[(df['ref_tox'] < non_toxicity_threshold) & (df['trn_tox'] > toxicity_threshold)]
    df_trn_tox = df_trn_tox.rename(columns={'reference': 'non-toxic', 'translation': 'toxic', 'ref_tox':'non-toxic_metric', 'trn_tox':'toxic_metric'})
    df_trn_tox = df_trn_tox[['toxic', 'non-toxic', 'similarity', 'length_diff', 'toxic_metric', 'non-toxic_metric']]
    df_united = pd.concat([df_ref_tox, df_trn_tox])
    df_final = df_united[['toxic', 'non-toxic']]
    return df_final

def create_train_val_test(df, seed):
    df.to_csv("./data/raw/suitable.csv")
    train = df.sample(10000, random_state=seed)
    val = df.drop(train.index).sample(1000, random_state=seed)
    test = df.drop(train.index).drop(val.index).sample(1000, random_state=seed)
    train.to_csv("./data/raw/train.csv")
    val.to_csv("./data/raw/val.csv")
    test.to_csv("./data/raw/test.csv")


URL = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
PATH = os.path.abspath("data/raw/")
SEED = 177013
archive_path = get_dataset(PATH, "archive.zip", URL)
unzip_dataset(archive_path, PATH)
dataset = create_dataset(os.path.abspath("data/raw/filtered.tsv"))
filtered_dataset = filter_dataset(dataset, 0.95, 0.05)
create_train_val_test(filtered_dataset, SEED)
