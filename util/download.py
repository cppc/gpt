import json
import os
import urllib.request
from tqdm import tqdm


def download_file(url, fn):
    with urllib.request.urlopen(url) as response:
        file_size = int(response.headers.get("Content-Length", 0))

        if os.path.exists(fn):
            if file_size == os.path.getsize(fn):
                return

        block_size = 1024

        with tqdm(total=file_size,
                  unit="iB", unit_scale=True,
                  desc=os.path.basename(url)) as pb:
            with open(fn, "wb") as f:
                buf = response.read(block_size)
                while buf:
                    f.write(buf)
                    pb.update(len(buf))
                    buf = response.read(block_size)


def download_all(base, names, dir, label):
    os.makedirs(dir, exist_ok=True)
    for fn in names:
        furl = os.path.join(base, label, fn)
        fpath = os.path.join(dir, fn)
        download_file(furl, fpath)


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text_data = f.read()

    return text_data
