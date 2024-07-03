"""Hello world workflow"""

from train_task import train_model, Result
from typing import List

from model.model import LogGPT
import numpy

def segmentation_workflow(data_path: List[str] = ["datasets"],
    s3_bucket: str = "",
    file_key: str = "",
    dataset:str = "Thunderbird") -> numpy.float64:
    
    model = train_model(data_path=data_path,
                s3_bucket=s3_bucket,
                file_key=file_key,
                dataset=dataset)

    return model


if __name__ == "__main__":
    model = segmentation_workflow()
    
