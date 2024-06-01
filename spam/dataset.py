import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset

from util import get_special_token


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, label="Label", max_length=None, pad_token='<|endoftext|>', encoding="gpt2"):
        self.data = pd.read_csv(csv_file)
        self.label = label
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data['Text']
        ]
        if max_length is not None:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        else:
            self.max_length = self._longest_encoded_length()

        self.encoded_texts = [
            encoded_text + [get_special_token(pad_token, encoding)] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.data.iloc[idx][self.label]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            if len(encoded_text) > max_length:
                max_length = len(encoded_text)
        return max_length


train_dataset = SpamDataset(
    csv_file="spam/train.csv",
    tokenizer=tiktoken.get_encoding("gpt2")
)

val_dataset = SpamDataset(
    csv_file="spam/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tiktoken.get_encoding("gpt2")
)

test_dataset = SpamDataset(
    csv_file="spam/test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tiktoken.get_encoding("gpt2")
)

if __name__ == "__main__":
    print(train_dataset.max_length)
