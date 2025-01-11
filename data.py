from datasets import load_dataset, Dataset
import pandas as pd
import csv
import sys
import os
import logging
import pdb

log = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)

def load_news_commentary_de_en():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path,"data/news-commentary-v18/news-commentary-v18.de-en.tsv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        dataset = []
        id = 0
        for row in tsv_reader:
            example = {}
            id += 1
            try:
                example['en'] = row[1]
                example['de'] = row[0]
                dataset.append(example)
            except IndexError:
                print(id,": ",row)
        # dataset = Dataset.from_dict(dataset)
        df = pd.DataFrame(dataset)
        dataset = Dataset.from_pandas(df)
        dataset = dataset.train_test_split(test_size = 0.1, seed = 2024)
    return dataset


def load_news_commentary_de_ja():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path,"data/news-commentary-v18/news-commentary-v18.de-ja.tsv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        dataset = {'de':[],'ja':[]}
        id = 0
        for row in tsv_reader:
            id += 1
            try:
                dataset['ja'].append(row[1])
                dataset['de'].append(row[0])
            except IndexError:
                print(id,": ",row)
        dataset = Dataset.from_dict(dataset)
        dataset = dataset.train_test_split(test_size = 0.1, seed = 2024)
    return dataset

def load_news_commentary_de_zh():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path,"data/news-commentary-v18/news-commentary-v18.de-zh.tsv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        dataset = {'de':[],'zh':[]}
        id = 0
        for row in tsv_reader:
            id += 1
            try:
                dataset['zh'].append(row[1])
                dataset['de'].append(row[0])
            except IndexError:
                print(id,": ",row)
        dataset = Dataset.from_dict(dataset)
        dataset = dataset.train_test_split(test_size = 0.1, seed = 2024)
    return dataset

def load_news_commentary_en_ja():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path,"data/news-commentary-v18/news-commentary-v18.en-ja.tsv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        dataset = {'en':[],'ja':[]}
        id = 0
        for row in tsv_reader:
            id += 1
            try:
                dataset['ja'].append(row[1])
                dataset['en'].append(row[0])
            except IndexError:
                print(id,": ",row)
        dataset = Dataset.from_dict(dataset)
        dataset = dataset.train_test_split(test_size = 0.1, seed = 2024)
    return dataset

def load_news_commentary_en_zh():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path,"data/news-commentary-v18/news-commentary-v18.en-zh.tsv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        dataset = {'en':[],'zh':[]}
        id = 0
        for row in tsv_reader:
            id += 1
            try:
                dataset['zh'].append(row[1])
                dataset['en'].append(row[0])
            except IndexError:
                print(id,": ",row)
        dataset = Dataset.from_dict(dataset)
        dataset = dataset.train_test_split(test_size = 0.1, seed = 2024)
    return dataset

def load_news_commentary_ja_zh():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path,"data/news-commentary-v18/news-commentary-v18.ja-zh.tsv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        dataset = {'zh':[],'ja':[]}
        id = 0
        for row in tsv_reader:
            id += 1
            try:
                dataset['ja'].append(row[0])
                dataset['zh'].append(row[1])
            except IndexError:
                print(id,": ",row)
        dataset = Dataset.from_dict(dataset)
        dataset = dataset.train_test_split(test_size = 0.1, seed = 2024)
    return dataset

def load_Tatoeba_en_ja():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path,"data/tatoeba/tatoeba_jp_en_sentence.tsv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        dataset = {'en':[],'ja':[]}
        id = 0
        for row in tsv_reader:
            id += 1
            try:
                dataset['ja'].append(row[1])
                dataset['en'].append(row[3])
            except IndexError:
                print(id,": ",row)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.train_test_split(test_size = 0.1, seed = 2024)
    return dataset

def load_Tatoeba_en_zh():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path,"data/tatoeba/tatoeba_en_zh_sentence.tsv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        dataset = {'en':[],'zh':[]}
        id = 0
        for row in tsv_reader:
            id += 1
            try:
                dataset['en'].append(row[3])
                dataset['zh'].append(row[1])                
            except IndexError:
                print(id,": ",row)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.train_test_split(test_size = 0.1, seed = 2024)
    return dataset

def load_Tatoeba_en_de():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path,"data/tatoeba/tatoeba_de_en_sentence.tsv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        dataset = {'en':[],'de':[]}
        id = 0
        for row in tsv_reader:
            id += 1
            try:
                dataset['de'].append(row[1])
                dataset['en'].append(row[3])
            except IndexError:
                print(id,": ",row)

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.train_test_split(test_size = 0.1, seed = 2024)
    return dataset


def SeqToSeqEncode(example, input_lang, target_lang, tokenizer, max_length=None):
    inputs = tokenizer(
        example[input_lang],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    outputs = tokenizer(
        example[target_lang],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    results = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"],
        "decoder_attention_mask": outputs["attention_mask"],
    }

    return results

DATASET_MAP = {
    "news_commentary_de_en":load_news_commentary_de_en,
    "news_commentary_de_zh":load_news_commentary_de_zh,
    "news_commentary_de_ja":load_news_commentary_de_ja,
    "news_commentary_en_zh":load_news_commentary_en_zh,
    "news_commentary_en_ja":load_news_commentary_en_ja,
    "news_commentary_ja_zh":load_news_commentary_ja_zh,
    "tatoeba_en_ja":load_Tatoeba_en_ja,
    "tatoeba_en_zh":load_Tatoeba_en_zh,
    "tatoeba_de_en":load_Tatoeba_en_de
}

if __name__ =="__main__":
    dataset = load_Tatoeba_en_ja()
    print(dataset)