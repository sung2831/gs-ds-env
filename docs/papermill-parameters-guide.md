# Papermill Parameters 동작 가이드

## 개요

SageMaker Estimator에서 전달한 `hyperparameters`가 어떻게 노트북에 적용되는지 설명합니다.

---

## 전체 흐름

```
Estimator(hyperparameters={...})
    │
    │  SageMaker가 CLI 인자로 변환 (모든 값을 string으로 전달)
    ▼
run_pm.py --num_leaves 50 --learning_rate 0.05
    │
    │  parse_args()로 수신 → params dict 구성
    ▼
pm.execute_notebook(parameters=params)
    │
    │  Papermill이 노트북에서 "parameters" 태그 셀을 찾아
    │  바로 아래에 "injected-parameters" 셀을 자동 삽입
    ▼
노트북 실행 (위에서 아래로 순차 실행)
```

---

## 단계별 상세

### 1단계: Estimator에서 hyperparameters 전달

```python
estimator = Estimator(
    image_uri="...",
    hyperparameters={
        "num_leaves": "50",        # 반드시 string
        "learning_rate": "0.05",
        "objective": "binary",
    },
)
estimator.fit({"training": train_s3_path})
```

> SageMaker는 모든 hyperparameter 값을 **string**으로 변환하여 컨테이너에 전달합니다.

### 2단계: run_pm.py에서 CLI 인자로 수신

SageMaker가 컨테이너 내부에서 실행하는 명령:

```bash
python run_pm.py --num_leaves 50 --learning_rate 0.05 --objective binary
```

`run_pm.py`의 `parse_args()`가 이를 파싱하여 dict로 구성:

```python
params = {"num_leaves": "50", "learning_rate": "0.05", "objective": "binary"}
```

### 3단계: Papermill이 노트북에 파라미터 주입

```python
pm.execute_notebook(
    "train_titanic_lightgbm.ipynb",       # 원본 노트북
    "train_titanic_lightgbm_output.ipynb", # 실행 결과 노트북
    parameters=params,                     # 주입할 파라미터
    kernel_name="conda_tabular-kunops-311",
)
```

Papermill 내부 동작:

1. 노트북 JSON을 읽음
2. `"tags": ["parameters"]`가 있는 셀을 찾음
3. 해당 셀 **바로 아래**에 `injected-parameters` 셀을 자동 생성
4. `params` dict를 Python 코드로 변환하여 삽입
5. 노트북 전체를 위에서 아래로 실행

---

## 노트북 셀 실행 순서

### 원본 노트북 (실행 전)

```
Cell 6  [tags: parameters]        ← default 값 정의
────────────────────────────────
  objective = "binary"
  num_leaves = 31                   ← int (default)
  learning_rate = 0.1               ← float (default)
  ...

Cell 7  [타입 변환]
────────────────────────────────
  num_leaves = int(num_leaves)
  learning_rate = float(learning_rate)
  hyperparameters = {...}
```

### 실행 후 output 노트북 (Papermill이 셀을 삽입한 상태)

```
Cell 6  [tags: parameters]        ← 1번째로 실행 (default 값 설정)
────────────────────────────────
  objective = "binary"
  num_leaves = 31
  learning_rate = 0.1

Cell ?  [tags: injected-parameters] ← 2번째로 실행 (override!) ★ 자동 생성됨
────────────────────────────────
  # Parameters
  num_leaves = "50"                  ← string으로 덮어씀
  learning_rate = "0.05"             ← string으로 덮어씀

Cell 7  [타입 변환]               ← 3번째로 실행 (string → 원래 타입)
────────────────────────────────
  num_leaves = int(num_leaves)       ← "50" → 50
  learning_rate = float(learning_rate) ← "0.05" → 0.05
```

---

## 핵심 포인트

### parameters 태그 설정

노트북의 셀 메타데이터에 `parameters` 태그가 있어야 Papermill이 인식합니다:

```json
{
  "cell_type": "code",
  "metadata": {
    "tags": ["parameters"]
  },
  "source": "num_leaves = 31\nlearning_rate = 0.1\n..."
}
```

### injected-parameters 셀 특징

| 항목 | 설명 |
|------|------|
| 생성 위치 | `parameters` 태그 셀 바로 다음 |
| 생성 시점 | `pm.execute_notebook()` 호출 시 |
| 존재 위치 | **output 노트북에만 존재** (원본은 수정되지 않음) |
| 태그 | `["injected-parameters"]` |
| 내용 형식 | `# Parameters\nkey = value\n...` |

### 타입 변환이 필요한 이유

SageMaker가 모든 값을 string으로 전달하므로:

```python
# injected-parameters 셀에서는 이렇게 들어옴
num_leaves = "50"      # string
learning_rate = "0.05" # string

# 타입 변환 셀에서 복원
num_leaves = int(num_leaves)       # 50 (int)
learning_rate = float(learning_rate) # 0.05 (float)
```

### override 동작

- Estimator에서 **값을 넘긴 경우** → injected-parameters 셀이 default를 덮어씀
- Estimator에서 **값을 안 넘긴 경우** → parameters 셀의 default 값 그대로 사용

---

## 파일 구조

```
tabular-kunops-311/
├── sm_docker/
│   ├── run_pm.py                      # Papermill 실행기 (parse_args → pm.execute_notebook)
│   ├── train_titanic_lightgbm.ipynb   # 학습 노트북 (parameters 태그 셀 포함)
│   └── conf.py                        # kernel_name 등 설정
└── test/
    └── run_training_estimator_test.ipynb  # Estimator 테스트 (hyperparameters 전달)
```
