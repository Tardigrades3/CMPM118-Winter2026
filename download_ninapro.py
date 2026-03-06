import urllib.request
import zipfile
import os
import sys
import shutil

URL = "https://nina-pro-dataset.s3.us-east-2.amazonaws.com/db1/NinaProData.zip"

ZIP_NAME = "NinaProData.zip"
TARGET_DIR = "NinaProData"


def download_file(url, output_path):
    print("Downloading dataset...")
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as e:
        print("Download failed:", e)
        sys.exit(1)


def extract_zip(zip_path, extract_to="."):
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                raw_name = member.filename

                normalized = raw_name.replace("\\", "/")

                parts = [p for p in normalized.split("/") if p not in ("", ".", "..")]
                if not parts:
                    continue

                out_path = os.path.join(extract_to, *parts)

                if raw_name.endswith("/") or raw_name.endswith("\\"):
                    os.makedirs(out_path, exist_ok=True)
                    continue

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                with zf.open(member) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    except Exception as e:
        print("Extraction failed:", e)
        sys.exit(1)


def main():
    if os.path.exists(TARGET_DIR):
        print(f"'{TARGET_DIR}' already exists. Skipping download.")
        return

    download_file(URL, ZIP_NAME)
    extract_zip(ZIP_NAME)
    os.remove(ZIP_NAME)

    print("Done.")
    print(f"Dataset available in ./{TARGET_DIR}")


if __name__ == "__main__":
    main()