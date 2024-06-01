from pathlib import Path

spam_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
spam_zip_path = "sms_spam_collection.zip"
spam_extracted_path = "sms_spam_collection"
spam_data_file_path = Path(spam_extracted_path) / "SMSSpamCollection.tsv"
