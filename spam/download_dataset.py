import urllib.request
import zipfile
import os
from pathlib import Path

spam_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
spam_zip_path = "sms_spam_collection.zip"
spam_extracted_path = "sms_spam_collection"
spam_data_file_path = Path(spam_extracted_path) / "SMSSpamCollection.tsv"


def download_and_extract(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"File {data_file_path} already exists!")
        return

    with urllib.request.urlopen(url) as response, open(zip_path, 'wb') as out_file:
        with open(zip_path, 'wb') as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

    original_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_path, data_file_path)
    print(f"File {data_file_path} has been downloaded!")


if __name__ == "__main__":
    download_and_extract(spam_url, spam_zip_path, spam_extracted_path, spam_data_file_path)

