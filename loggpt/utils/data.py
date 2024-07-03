from utils.s3_connection import get_s3_bucket
from pathlib import Path
import zipfile
from loguru import logger
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, TensorDataset

def download_and_extract(data_path, bucket, file_key):
    logger.info("Preparing training data...")

    data_folder = Path.cwd().parent  # get the current directory
    data_folders = data_folder.joinpath(*data_path)

    s3_bucket = get_s3_bucket(bucket)

    logger.info(f"downloading dataset {file_key}")
    dest_file_name = str(data_folders / Path(file_key).name)
    s3_bucket.download_file(file_key, dest_file_name)


    logger.info(f"extracting dataset {dest_file_name}")
    try:
        with zipfile.ZipFile(dest_file_name, "r") as zip_ref:
            logger.info("opened zip file")
            dest_dir = Path(data_folders)
            dest_dir.mkdir(exist_ok=True)
            logger.info("attempting to extract all")
            zip_ref.extractall(dest_dir)
            logger.info("extracted all")
    except:
        logger.info(f"Cannot extract {dest_file_name} to {dest_dir}")

    return dest_dir