import os
import urllib.request

from util.download import download_and_load_file
from .dataset import create_dataloader_v1

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
text_data = download_and_load_file(file_path, url)

TEXT_DATA = text_data
