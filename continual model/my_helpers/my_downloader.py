import requests
from bs4 import BeautifulSoup
import re
import os
from tqdm import tqdm
import pandas as pd
import scipy
import zipfile
import io
import numpy as np



base_url = "https://ninapro.hevs.ch/files/DB1/Preprocessed/"
def download_NinaPro(url):
    resp = requests.get(base_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    pattern = re.compile(r".*\.zip$")
    urls = soup.find_all("a", href=pattern)

    extract_dir = os.getcwd() + "/data"

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    files = []
    print("Downloading Files")
    for url in tqdm(urls): # every subject
        resp = io.BytesIO(requests.get(base_url + url["href"]).content)
        with zipfile.ZipFile(resp, 'r') as z:
            data = []
            for file_name in z.namelist(): # every file within subject's folder
                with z.open(file_name) as file:
                    raw_read = scipy.io.loadmat(file)
                    glove_df = pd.DataFrame(np.asarray(raw_read["glove"]), columns=[f"glove_{i}" for i in range(np.asarray(raw_read["glove"]).shape[1])])
                    emg_df = pd.DataFrame(np.asarray(raw_read["emg"]), columns=[f"emg_{i}" for i in range(np.asarray(raw_read["emg"]).shape[1])])
                    keys_to_remove = ['emg', 'glove', 'subject', 'exercise', '__header__', '__globals__', '__version__']
                    for key in keys_to_remove:
                        raw_read.pop(key, None)
                    for key in raw_read:
                        raw_read[key] = np.squeeze(raw_read[key])
                    mat = pd.DataFrame(raw_read)
                    mat["file_name"] = file_name
                    final_data = pd.concat([emg_df, glove_df, mat], axis=1)
                    data.append(final_data)
            files.append(pd.concat(data, axis = 0))

    print("Writing Files")
    for index, f_ in tqdm(enumerate(files)):
        # ! ISSUE: this actually indexes wrong --> the files are out of order of the subject number
        out_path = os.path.join("./data", f"subject_{index}.csv")
        f_.to_csv(out_path, index=False)

def read_data():
