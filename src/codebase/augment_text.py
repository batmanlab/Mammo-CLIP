import argparse
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.marian import MarianMTModel

warnings.filterwarnings("ignore")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TextDataset(Dataset):
    def __init__(self, tokenizer, df, col_name, text_data_list=None):

        self.text_data_list = []
        self.tokenizer = tokenizer
        self.df = df
        self.df = self.df.fillna("")
        if text_data_list:
            self.text_data_list = text_data_list
        else:
            self.text_data_list = []
            for idx in range(len(self.df)):
                if col_name in self.df:
                    text_list = self.df[col_name][idx]
                else:
                    raise NotImplementedError

                self.text_data_list.append(text_list)

    def __len__(self):
        return len(self.text_data_list)

    def __getitem__(self, index):
        return self.text_data_list[index]

    def collate_fn(self, instances):
        tokens = self.tokenizer(instances, return_tensors="pt", padding=True)
        return tokens, instances


class BackTranslation:
    def __init__(self, lang="de"):
        self.lang = lang
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.en_lang_tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
        self.lang_en_tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang}-en")

        self.en_lang_translator = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}").to(self.device)
        self.lang_en_translator = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{lang}-en").to(self.device)

    def do_back_translation(self, df, col_name, batch_size, temperature, **generate_kwargs):
        assert len(temperature) <= 2
        temp1, temp2 = (temperature[0], temperature[0]) if len(temperature) == 1 else temperature

        pandas_text_dataset = TextDataset(self.en_lang_tokenizer, df=df, col_name=col_name)

        dataloader = DataLoader(
            pandas_text_dataset,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            batch_size=batch_size,
            collate_fn=pandas_text_dataset.collate_fn,
        )

        lang_out_list = []

        idx = 0
        for batch in tqdm(dataloader):
            tokens, instances = batch
            # print(f"{idx} - {instances}")
            idx += 1
            lang_out = self.en_lang_translator.generate(**tokens.to(self.device), temperature=temp1, **generate_kwargs)
            for out in lang_out:
                lang_out_list.append(
                    self.en_lang_tokenizer.decode(out).replace("<pad>", "").replace("</s>", "").strip())

        lang_text_dataset = TextDataset(self.lang_en_tokenizer, df, text_data_list=lang_out_list, col_name=col_name)
        dataloader = DataLoader(
            lang_text_dataset, shuffle=False, drop_last=False, num_workers=4, batch_size=batch_size,
            collate_fn=lang_text_dataset.collate_fn
        )

        text_augment_list = []
        for batch in tqdm(dataloader):
            tokens, instances = batch
            en_out = self.lang_en_translator.generate(**tokens.to(self.device), temperature=temp2, **generate_kwargs)
            for out in en_out:
                text_augment_list.append(
                    self.lang_en_tokenizer.decode(out).replace("<pad>", "").replace("</s>", "").strip())

        return text_augment_list, lang_out_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", type=str,
                        default="/ocean/projects/asc170022p/shg121/PhD/Mammo-CLIP/src/codebase/data_csv")
    parser.add_argument("--csv-path", type=str,
                        default="upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv")
    parser.add_argument("--dataset", type=str,
                        default="upmc")
    parser.add_argument("--lang", type=str, default="it")
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5)

    args = parser.parse_args()
    args.temperature = [args.temperature]

    dataset_path = Path(args.dataset_path)
    df = pd.read_csv(dataset_path / args.csv_path)
    print(df.shape)

    print(df.head(10))
    print(df.columns)
    backtranslation = BackTranslation(lang=args.lang)
    set_random_seed(args.seed)
    col_name = "FINDINGS"
    text_augment_list, lang_out_list = backtranslation.do_back_translation(
        df, batch_size=args.bsz, col_name=col_name, num_beams=args.num_beams, temperature=args.temperature,
        do_sample=True
    )
    print(len(text_augment_list), len(lang_out_list))
    print(text_augment_list[:5], lang_out_list[:5])
    df[f"{col_name}_back_translated"] = text_augment_list
    pickle.dump(text_augment_list,
                open(Path(args.dataset_path) / f"{args.dataset}_{col_name}_text_augment_list.pkl", "wb"))
    pickle.dump(lang_out_list,
                open(Path(args.dataset_path) / f"{args.dataset}_{col_name}_lang_out_list.pkl", "wb"))

    col_name = "IMPRESSION"
    text_augment_list, lang_out_list = backtranslation.do_back_translation(
        df, batch_size=args.bsz, col_name=col_name, num_beams=args.num_beams, temperature=args.temperature,
        do_sample=True
    )
    print(len(text_augment_list), len(lang_out_list))
    print(text_augment_list[:5], lang_out_list[:5])
    df[f"{col_name}_back_translated"] = text_augment_list
    pickle.dump(text_augment_list,
                open(Path(args.dataset_path) / f"{args.dataset}_{col_name}_text_augment_list.pkl", "wb"))
    pickle.dump(lang_out_list,
                open(Path(args.dataset_path) / f"{args.dataset}_{col_name}_lang_out_list.pkl", "wb"))

    print(args.dataset_path)

    print(df.head(10))
    df.to_csv(dataset_path / args.csv_path, index=False)
