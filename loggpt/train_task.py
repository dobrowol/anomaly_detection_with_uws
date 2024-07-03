"""Hello world task."""


from loguru import logger
from utils import utils
from utils.data import download_and_extract
from args import bgl_config, hdfs_config, thunderbird_config, openstack_config, nokia_config
from typing import List
from model import model, initGPT
from model.model import LogGPT
import mlflow 
from best_hyperparameters import bayes_for_voting_experts, bayes_for_sliding_window_on_separate_files
from dataclasses import dataclass
from typing import NamedTuple, Tuple
from scipy.stats import ttest_rel
import numpy

class Result(NamedTuple):
    f1: str
    window: int
    threshold: int

@dataclass
class Args:
    dest_dir: str
    iterations: int

def train_model(
    data_path: List[str],
    s3_bucket: str,
    file_key: str,
    dataset: str
) -> numpy.float64:
    # mlflow.pytorch.autolog()

    if dataset == 'Thunderbird':
        parser = thunderbird_config.get_args()
    elif dataset == 'BGL':
        parser = bgl_config.get_args()
    elif dataset == 'HDFS':
        parser = hdfs_config.get_args()
    elif dataset == 'OpenStack':
        parser = openstack_config.get_args()
    else:
        print("Only support HDFS, Thunderbird and BGL dataset!")
    args, unknown = parser.parse_known_args()
    options = vars(args)
    print(options)
    utils.set_seed(options['seed'])

    dest_dir = download_and_extract(data_path, s3_bucket, file_key)
    args = Args(dest_dir=dest_dir, iterations=10)

    res = bayes_for_voting_experts(args, options, segmentation_column='PRID_ThreadID')


    return res[0]
