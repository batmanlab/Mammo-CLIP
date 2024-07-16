import argparse
import ast
import os
import random
import sys
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.marian import MarianMTModel
from sklearn.model_selection import GroupKFold

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_report_into_segment_concat_aug(report_list):
    """clean up raw reports into sentences"""
    if len(report_list) == 0:
        return
    else:
        report_aug = []
        report_list = ast.literal_eval(report_list)
        for report in report_list:
            report_aug.append(report.lower())

        return report_aug


def _split_report_into_segment_concat(report):
    """clean up raw reports into sentences"""
    if pd.isnull(report):
        return
    else:
        report = report.replace('\n', ' ').replace('. ', '.').replace('.', '. ')
        reports = report.split(". ")
        study_sent = []
        for sent in reports:
            if len(sent) == 0:
                continue

            sent = sent.replace("\ufffd\ufffd", " ")
            tokens = nltk.wordpunct_tokenize(sent.lower())

            if len(tokens) <= 1:
                continue

            # filter tokens for current sentence
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)

            if len(included_tokens) > 2:  # only include relative long sentences
                study_sent.append(" ".join(included_tokens))
        concatenated_string = ""

        for sentence in study_sent:
            concatenated_string += sentence.strip() + ' '

        #         print(concatenated_string)
        return concatenated_string.strip()


class TextDataset(Dataset):
    def __init__(self, tokenizer, df, text_data_list=None):
        self.df, self.text_num_list, self.text_data_list = None, None, None
        self.tokenizer = tokenizer

        if df.shape[0] > 0:
            self.df = df
            self.text_data_list = []
            self.text_num_list = []

            for idx in range(len(self.df)):

                if "text" in self.df:
                    text_list = ast.literal_eval(self.df["text"][idx])

                else:
                    raise NotImplementedError

                self.text_num_list.append(len(text_list))
                self.text_data_list.extend(text_list)

        if text_data_list:
            self.text_data_list = text_data_list

    def __len__(self):
        return len(self.text_data_list)

    def __getitem__(self, index):
        return self.text_data_list[index]

    def collate_fn(self, instances):
        tokens = self.tokenizer(instances, return_tensors="pt", padding=True)
        return tokens


def convert_df_to_folds(out_data_path, csv_path):
    new_df = pd.read_csv(out_data_path / csv_path)
    gkf = GroupKFold(n_splits=4)
    new_df['group'] = new_df['patient_id'].astype(str)
    new_df['fold'] = -1
    for fold_number, (train_index, test_index) in enumerate(gkf.split(new_df, groups=new_df['group'])):
        new_df.loc[test_index, 'fold'] = fold_number

    new_df.drop(columns=['group'], inplace=True)
    new_df.to_csv(out_data_path / csv_path, index=False)


class BackTranslation:
    def __init__(self, lang="de"):
        self.lang = lang
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.en_lang_tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
        self.lang_en_tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang}-en")

        self.en_lang_translator = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}").to(self.device)
        self.lang_en_translator = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{lang}-en").to(self.device)

    def do_back_translation(self, df, out_data_path, csv_path, batch_size, temperature, **generate_kwargs):
        assert len(temperature) <= 2
        temp1, temp2 = (temperature[0], temperature[0]) if len(temperature) == 1 else temperature

        pandas_text_dataset = TextDataset(self.en_lang_tokenizer, df)
        dataloader = DataLoader(
            pandas_text_dataset,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            batch_size=batch_size,
            collate_fn=pandas_text_dataset.collate_fn,
        )

        text_num_list = pandas_text_dataset.text_num_list
        lang_out_list = []

        for batch in tqdm(dataloader):
            lang_out = self.en_lang_translator.generate(**batch.to(self.device), temperature=temp1, **generate_kwargs)
            for out in lang_out:
                lang_out_list.append(
                    self.en_lang_tokenizer.decode(out).replace("<pad>", "").replace("</s>", "").strip())

        with open(out_data_path / f"{self.lang}.txt", "w") as fout:
            fout.write("\n".join(lang_out_list))
            fout.write("\n")

        lang_text_dataset = TextDataset(self.lang_en_tokenizer, df=df, text_data_list=lang_out_list)
        dataloader = DataLoader(
            lang_text_dataset, shuffle=False, drop_last=False, num_workers=4, batch_size=batch_size,
            collate_fn=lang_text_dataset.collate_fn
        )

        en_out_list = []
        for batch in tqdm(dataloader):
            en_out = self.lang_en_translator.generate(**batch.to(self.device), temperature=temp2, **generate_kwargs)
            for out in en_out:
                en_out_list.append(self.lang_en_tokenizer.decode(out).replace("<pad>", "").replace("</s>", "").strip())

        text_augment_list = []
        start = 0
        for text_num in text_num_list:
            text_augment_list.append(en_out_list[start: start + text_num])
            start += text_num

        pandas_text_dataset.df["text_augment"] = text_augment_list
        pandas_text_dataset.df.to_csv(out_data_path / csv_path, index=False)

        pandas_text_dataset.df = pd.read_csv(dataset_path / csv_path)
        pandas_text_dataset.df["text_augment"] = pandas_text_dataset.df["text_augment"].apply(
            _split_report_into_segment_concat_aug)
        if "text" not in pandas_text_dataset.df:
            pandas_text_dataset.df["text_original"] = pandas_text_dataset.text_data_list
        pandas_text_dataset.df.to_csv(out_data_path / csv_path, index=False)


def process_df(df, csv_path):
    unique_patient_count = df['patient_id'].nunique()
    print(f"Number of unique patients: {unique_patient_count}")

    df["FINDINGS"] = df["FINDINGS"].fillna(" ")
    df["IMPRESSION"] = df["IMPRESSION"].fillna(" ")
    df['IMPRESSION'] = df['IMPRESSION'].str.replace('BI-RADS', 'BIRADS')

    df["IMPRESSION_1"] = df["IMPRESSION"].apply(_split_report_into_segment_concat)
    df["FINDINGS_1"] = df["FINDINGS"].apply(_split_report_into_segment_concat)

    grouped_df = df.groupby(['patient_id', 'laterality'])

    patient_id_arr = []
    lat_arr = []
    image_id_arr = []
    view_arr = []
    cc_image_id_arr = []
    mlo_image_id_arr = []
    text_arr = []
    finding_arr = []
    impression_arr = []
    fold_arr = []

    idx = 0
    for group_identifier, group_df in grouped_df:
        patient_id, laterality = group_identifier
        fold = group_df['fold'].tolist()
        image_id = group_df['image_id'].tolist()
        views = list(set(group_df['view'].tolist()))
        cc_image_paths = list(set(group_df[group_df['view'] == 'CC']['image_id'].tolist()))
        mlo_image_paths = list(set(group_df[group_df['view'] == 'MLO']['image_id'].tolist()))
        text = []
        if group_df['FINDINGS_1'] is not None:
            text += group_df['FINDINGS_1'].tolist()
        if group_df['IMPRESSION_1'] is not None:
            text += group_df['IMPRESSION_1'].tolist()

        unique_text = []
        seen = set()
        for sublist in text:
            # Convert each sublist to a tuple to make it hashable
            if sublist is not None:
                tuple_sublist = tuple(sublist)
                if tuple_sublist not in seen:
                    unique_text.append(sublist)
                    seen.add(tuple_sublist)

        patient_id_arr.append(patient_id)
        fold_arr.append(fold[0])
        lat_arr.append(laterality)
        image_id_arr.append(image_id)
        view_arr.append(views)
        cc_image_id_arr.append(cc_image_paths)
        mlo_image_id_arr.append(mlo_image_paths)
        text_arr.append(unique_text)

        finding_arr.append(list(set(group_df['FINDINGS_1'].tolist()))[0])
        impression_arr.append(list(set(group_df['IMPRESSION_1'].tolist()))[0])

        idx += 1

    new_df = pd.DataFrame({
        'patient_id': patient_id_arr,
        "laterality": lat_arr,
        'image': image_id_arr,
        'view': view_arr,
        'CC': cc_image_id_arr,
        'MLO': mlo_image_id_arr,
        'text': text_arr,
        'findings': finding_arr,
        'impressions': impression_arr,
    })

    new_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", type=str,
                        default="/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/src/codebase/data_csv")
    parser.add_argument("--csv-path", type=str,
                        default="upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv")
    parser.add_argument("--dataset", type=str,
                        default="rsna")
    parser.add_argument("--lang", type=str, default="it")
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--column", type=str, default="findings")

    args = parser.parse_args()
    args.temperature = [args.temperature]

    dataset_path = Path(args.dataset_path)
    df = pd.read_csv(dataset_path / args.csv_path)
    print(df.shape)
    print(df.head(10))

    output_csv = "clip_pretrain_100.csv"
    process_df(df, dataset_path / output_csv)
    df = pd.read_csv(dataset_path / output_csv)
    print(df.shape)
    print(df.head(10))

    backtranslation = BackTranslation(lang=args.lang)
    set_random_seed(args.seed)
    out_data_path = args.dataset_path
    backtranslation.do_back_translation(
        df, Path(out_data_path), output_csv, batch_size=args.bsz, num_beams=args.num_beams,
        temperature=args.temperature,
        do_sample=True
    )
    convert_df_to_folds(Path(out_data_path), output_csv)
    print(args.dataset_path)


