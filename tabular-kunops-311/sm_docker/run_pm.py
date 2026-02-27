import os
import time
import mimetypes
import json
import shutil
import pickle
import traceback
import argparse
import warnings
import boto3
from botocore.exceptions import ClientError
import papermill as pm
from papermill.exceptions import PapermillExecutionError
import pprint
import sys

import run_pm_utils as utils
import conf

pp = pprint.PrettyPrinter(width=41, compact=True, indent=4)
warnings.filterwarnings('ignore')


# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker Training Configuration")
    # --- 관리용 파라미터 ---
    parser.add_argument('--project_hashkey', type=str, default='')
    parser.add_argument('--experiment_hashkey', type=str, default='')
    parser.add_argument('--table_name', type=str, default='')
    parser.add_argument('--dataset_table_name', type=str, default='')
    parser.add_argument('--username', type=str, default='')
    parser.add_argument('--job_type', type=str, default='training')
    parser.add_argument('--task_token', type=str, default='')
    # --- LightGBM hyperparameters ---
    parser.add_argument('--objective', type=str, default='binary')
    parser.add_argument('--metric', type=str, default='binary_logloss')
    parser.add_argument('--num_leaves', type=str, default='31')
    parser.add_argument('--learning_rate', type=str, default='0.1')
    parser.add_argument('--n_estimators', type=str, default='100')
    parser.add_argument('--max_depth', type=str, default='10')
    parser.add_argument('--random_state', type=str, default='42')
    parser.add_argument('--verbose', type=str, default='0')
    # --- train/val split ---
    parser.add_argument('--val_ratio', type=str, default='0.2')
    parser.add_argument('--random_state_split', type=str, default='42')
    args, unknown = parser.parse_known_args()

    print('=== Hyperparameters ===')
    pp.pprint(vars(args))
    if unknown:
        print('=== Unknown args ===')
        pp.pprint(unknown)
    return args


def upload_file_to_s3(local_path: str, bucket: str, prefix: str) -> str:
    """
    Upload a local file to S3 under s3://{bucket}/{prefix}/<basename>.
    Returns the S3 URI.
    """
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    s3 = boto3.client("s3")

    filename = os.path.basename(local_path)
    prefix = prefix.strip("/")
    key = f"{prefix}/{filename}" if prefix else filename

    content_type, _ = mimetypes.guess_type(local_path)
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type

    try:
        s3.upload_file(local_path, bucket, key, ExtraArgs=extra_args or None)
    except ClientError as e:
        raise RuntimeError(f"S3 upload failed: {e}") from e

    return f"s3://{bucket}/{key}"
    

def run_papermill(input_nb, output_dir, params=None):
    os.chdir(output_dir)
    output_nb = input_nb.replace('.ipynb', '_output.ipynb')
    try:
        pm.execute_notebook(
            input_nb,
            output_nb,
            parameters=params or dict(),
            kernel_name=conf.kernel_name,
            report_mode=True
        )
        bucket_name = "retail-mlops-edu-2026"
        s3_prefix = "edu-2w/kunops/output"
        s3_uri = upload_file_to_s3(output_nb, bucket_name, s3_prefix)
        print(f"Uploaded to: {s3_uri}")
    except PapermillExecutionError as e:
        pp.pprint(e)
        pass


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    try:
        args = parse_args()

        output_dir = "."
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        input_nb = 'train_titanic_lightgbm.ipynb'

        # hyperparameters를 papermill parameters로 전달
        params = {k: v for k, v in vars(args).items() if v}
        run_papermill(input_nb, output_dir, params)
    except Exception as e:
        traceback.print_exc()
        print(e)