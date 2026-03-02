import os
import mimetypes
import traceback
import argparse
import warnings
import boto3
from botocore.exceptions import ClientError
import papermill as pm
from papermill.exceptions import PapermillExecutionError
import pprint

import run_pm_utils as utils
import conf

pp = pprint.PrettyPrinter(width=41, compact=True, indent=4)
warnings.filterwarnings('ignore')


# ----------------------------
# SageMaker-style Local Directories
# ----------------------------
SM_BASE = '/opt/ml'
SM_INPUT_DIR = os.path.join(SM_BASE, 'input', 'data')
SM_CHANNEL_TRAIN = os.path.join(SM_INPUT_DIR, 'train')
SM_CHANNEL_VAL = os.path.join(SM_INPUT_DIR, 'val')
SM_MODEL_DIR = os.path.join(SM_BASE, 'model')
SM_OUTPUT_DIR = os.path.join(SM_BASE, 'output', 'data')
SM_CODE_DIR = os.path.join(SM_BASE, 'code')


# ----------------------------
# Argument Parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker Training Configuration")
    # --- S3 설정 ---
    parser.add_argument('--s3_bucket', type=str, default='retail-mlops-edu-2026')
    parser.add_argument('--s3_user', type=str, default='kunops')
    parser.add_argument('--s3_notebook', type=str, default='',
                        help='실행할 노트북의 S3 URI (예: s3://bucket/path/notebook.ipynb)')
    parser.add_argument('--s3_input_prefix', type=str, default='',
                        help='S3 input prefix (default: edu-2w/{s3_user}/input)')
    parser.add_argument('--s3_output_prefix', type=str, default='',
                        help='S3 output prefix (default: edu-2w/{s3_user}/output)')
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

    # S3 prefix 기본값 설정
    if not args.s3_input_prefix:
        args.s3_input_prefix = f"edu-2w/{args.s3_user}/input"
    if not args.s3_output_prefix:
        args.s3_output_prefix = f"edu-2w/{args.s3_user}/output"

    print('=== Configuration ===')
    pp.pprint(vars(args))
    if unknown:
        print('=== Unknown args ===')
        pp.pprint(unknown)
    return args


# ----------------------------
# SageMaker Directory Setup
# ----------------------------
def setup_sm_directories():
    """SageMaker 스타일 로컬 디렉토리 생성 및 환경변수 설정"""
    for d in [SM_CHANNEL_TRAIN, SM_CHANNEL_VAL, SM_MODEL_DIR, SM_OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

    os.environ['SM_CHANNEL_TRAIN'] = SM_CHANNEL_TRAIN
    os.environ['SM_CHANNEL_VAL'] = SM_CHANNEL_VAL
    os.environ['SM_MODEL_DIR'] = SM_MODEL_DIR
    os.environ['SM_OUTPUT_DATA_DIR'] = SM_OUTPUT_DIR

    print(f"  SM_CHANNEL_TRAIN:   {SM_CHANNEL_TRAIN}")
    print(f"  SM_CHANNEL_VAL:     {SM_CHANNEL_VAL}")
    print(f"  SM_MODEL_DIR:       {SM_MODEL_DIR}")
    print(f"  SM_OUTPUT_DATA_DIR: {SM_OUTPUT_DIR}")


# ----------------------------
# S3 I/O Functions
# ----------------------------
def download_s3_notebook(s3_uri):
    """S3에서 노트북 파일을 다운로드하여 SM_CODE_DIR에 저장"""
    # s3://bucket/prefix/notebook.ipynb → bucket, key 파싱
    s3_path = s3_uri.replace("s3://", "")
    bucket = s3_path.split("/", 1)[0]
    key = s3_path.split("/", 1)[1]
    filename = os.path.basename(key)

    os.makedirs(SM_CODE_DIR, exist_ok=True)
    local_path = os.path.join(SM_CODE_DIR, filename)

    print(f"  📥 {s3_uri} → {local_path}")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    print(f"  ✅ Downloaded: {filename}")
    return filename


def download_s3_input(bucket, s3_input_prefix):
    """S3에서 입력 데이터를 로컬 SageMaker 디렉토리로 다운로드"""
    print(f"  📥 s3://{bucket}/{s3_input_prefix}/ → {SM_CHANNEL_TRAIN}/")
    utils.download_s3_files_to_directory(bucket, s3_input_prefix, SM_CHANNEL_TRAIN)
    # 다운로드 결과 확인
    files = os.listdir(SM_CHANNEL_TRAIN)
    print(f"  📂 Downloaded {len(files)} file(s): {files}")


def upload_s3_output(bucket, s3_output_prefix, output_nb=None):
    """로컬 출력물(model, output, notebook)을 S3로 업로드"""
    # 모델 업로드
    if os.path.isdir(SM_MODEL_DIR) and os.listdir(SM_MODEL_DIR):
        model_prefix = s3_output_prefix.rsplit('/', 1)[0] + '/model'
        print(f"  📤 Model → s3://{bucket}/{model_prefix}/")
        utils.upload_directory_to_s3(SM_MODEL_DIR, bucket, model_prefix)

    # 평가 결과/아티팩트 업로드
    if os.path.isdir(SM_OUTPUT_DIR) and os.listdir(SM_OUTPUT_DIR):
        print(f"  📤 Output → s3://{bucket}/{s3_output_prefix}/")
        utils.upload_directory_to_s3(SM_OUTPUT_DIR, bucket, s3_output_prefix)

    # 실행된 노트북 업로드
    if output_nb and os.path.isfile(output_nb):
        print(f"  📤 Notebook → s3://{bucket}/{s3_output_prefix}/")
        upload_file_to_s3(output_nb, bucket, s3_output_prefix)


def upload_file_to_s3(local_path, bucket, prefix):
    """단일 파일을 S3에 업로드"""
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

    s3_uri = f"s3://{bucket}/{key}"
    print(f"  ✅ Uploaded: {s3_uri}")
    return s3_uri


# ----------------------------
# Papermill Execution
# ----------------------------
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
        print(f"  ✅ Notebook executed: {output_nb}")
        return output_nb
    except PapermillExecutionError as e:
        pp.pprint(e)
        return None


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    try:
        args = parse_args()

        # 1. SageMaker 디렉토리 구성 + 환경변수 설정
        print("\n" + "=" * 50)
        print("📁 [1/4] Setting up SageMaker directories")
        print("=" * 50)
        setup_sm_directories()

        # 2. S3 → Local: 입력 데이터 다운로드
        print("\n" + "=" * 50)
        print("📥 [2/4] Downloading input from S3")
        print("=" * 50)
        download_s3_input(args.s3_bucket, args.s3_input_prefix)

        # 3. 노트북 준비 + 실행
        print("\n" + "=" * 50)
        print("🚀 [3/4] Running Papermill")
        print("=" * 50)

        # S3 노트북 지정 시 다운로드, 아니면 로컬 기본 노트북 사용
        if args.s3_notebook:
            input_nb = download_s3_notebook(args.s3_notebook)
        else:
            input_nb = 'train_titanic_lightgbm.ipynb'

        # S3 설정 키를 제외한 hyperparameters만 papermill로 전달
        s3_keys = {'s3_bucket', 's3_user', 's3_notebook', 's3_input_prefix', 's3_output_prefix'}
        params = {k: v for k, v in vars(args).items() if v and k not in s3_keys}
        # 노트북에서 사용하는 S3 변수명으로 전달
        params['S3_BUCKET'] = args.s3_bucket
        params['S3_USER'] = args.s3_user

        output_nb = run_papermill(input_nb, SM_CODE_DIR, params)

        # 4. Local → S3: 출력물 업로드
        print("\n" + "=" * 50)
        print("📤 [4/4] Uploading outputs to S3")
        print("=" * 50)
        upload_s3_output(args.s3_bucket, args.s3_output_prefix, output_nb)

        print("\n" + "=" * 50)
        print("✅ Training pipeline completed successfully!")
        print("=" * 50)

    except Exception as e:
        traceback.print_exc()
        print(e)
