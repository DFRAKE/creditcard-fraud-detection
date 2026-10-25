# 신용카드 사기 탐지 딥러닝 모델

TensorFlow/Keras를 활용한 신용카드 사기 거래 이진 분류 프로젝트입니다.

## 프로젝트 개요

실제 신용카드 거래 데이터(284,807건)를 바탕으로 정상 거래와 사기 거래를 분류하는 딥러닝 모델을 구현했습니다.  
데이터의 극심한 불균형(정상 99.83% / 사기 0.17%)을 `class_weight`로 처리한 것이 핵심입니다.

## 모델 성능

| 지표 | 정상 (0) | 사기 (1) |
|------|----------|----------|
| Precision | 1.00 | 0.08 |
| Recall | 0.98 | **0.91** |
| F1-score | 0.99 | 0.14 |

- **전체 정확도**: 98%
- **사기 탐지율(Recall)**: 91% — 사기 거래 98건 중 89건 탐지

> 사기 탐지 도메인에서는 Recall이 핵심 지표입니다. 사기를 놓치는 비용(FN)이 오탐(FP)보다 훨씬 크기 때문입니다.

## 모델 구조

```
Input (30) → Dense(64, ReLU) → Dropout(0.3)
           → Dense(32, ReLU) → Dropout(0.3)
           → Dense(16, ReLU)
           → Dense(1, Sigmoid)
```

- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 20 / Batch size: 512
- 불균형 처리: `compute_class_weight` (사기 클래스에 577배 가중치)

## 결과 시각화

| Loss 변화 | Accuracy 변화 |
|-----------|---------------|
| ![loss](loss_graph.png) | ![accuracy](accuracy_graph.png) |

| 클래스 분포 | 혼동 행렬 |
|-------------|-----------|
| ![dist](class_distribution.png) | ![cm](confusion_matrix.png) |

| 피처-Class 상관계수 | Amount 분포 (정상 vs 사기) |
|---------------------|---------------------------|
| ![corr](eda_correlation.png) | ![amount](eda_amount_dist.png) |

## 실행 방법

### 1. 데이터 다운로드

[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 에서 `creditcard.csv`를 받아 프로젝트 폴더에 넣어주세요.

### 2. 환경 설정

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### 3. 노트북 실행

```bash
jupyter notebook 신용카드_사기탐지.ipynb
```

## 데이터셋

- 출처: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 총 284,807건의 유럽 신용카드 거래 (2013년 9월, 이틀치)
- 피처: V1~V28 (PCA 익명화), Time, Amount (총 30개)
- 타겟: Class (0=정상, 1=사기)

## 기술 스택

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 성능 해석과 현재 한계

위 성능 표는 기존 실험에서 README에 기록한 결과입니다. 현재 공개 노트북은 실행 출력이 비어 있으므로, 이 표의 수치를 저장된 셀 출력에서 다시 확인할 수는 없습니다.

- 사기 클래스 **Recall 0.91**은 **Precision 0.08 · F1 0.14**와 함께 해석해야 합니다. 높은 재현율과 많은 오탐 사이의 절충이 있으며, 운영 가능한 탐지 시스템의 성능이 검증되었다는 의미는 아닙니다.
- 실제 서비스의 미탐·오탐 비용은 업무 조건에 따라 다릅니다. 이 실험은 비용을 수치화하거나 최적 임계값을 검증하지 않았습니다.
- 현재 코드는 `StandardScaler.fit_transform`을 전체 데이터의 Time·Amount에 적용한 뒤 학습/테스트 데이터를 분리합니다. 테스트 데이터의 통계가 전처리에 반영되므로, 데이터 누수를 방지하는 순서로 변경 후 재평가해야 합니다.
- 학습/검증/테스트를 먼저 분리하고 학습 데이터에만 전처리기를 fit한 뒤, 검증·테스트 데이터에는 transform을 적용하는 방식이 필요합니다. [scikit-learn 공식 가이드](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- 새로운 거래 예시 2건은 코드에 작성한 합성 입력입니다. 별도 실거래 검증 데이터나 일반화 성능의 증거로 해석하지 않습니다.

## 실행 환경 보완

기존 환경 설정 명령에 더해, 로컬 노트북 실행에는 Jupyter가 필요합니다.

```bash
python3 -m pip install notebook
python3 -m notebook "신용카드_사기탐지.ipynb"
```

- 노트북 메타데이터에는 Python 3.11.0, `tf_env` 커널이 기록되어 있습니다. 라이브러리 버전은 고정되어 있지 않습니다.
- `tf_env`가 없는 환경에서는 설치한 Python 환경의 커널을 선택합니다.
- macOS의 AppleGothic 폰트를 지정한 코드가 있습니다. 다른 운영체제에서는 설치된 한글 폰트로 조정해야 할 수 있습니다.
- 원본 거래 CSV는 저장소에 포함되어 있지 않습니다. 위 데이터 출처에서 받아 저장소 루트에 둡니다.

## 재현성과 평가 개선 계획

1. 데이터 누수를 방지하도록 분리·전처리 순서를 수정하고 재학습합니다.
2. 실행 환경과 데이터 분할·임계값·학습 조건을 기록합니다.
3. 재평가한 Precision·Recall·F1, PR-AUC와 혼동 행렬을 실행 출력·결과 파일로 함께 보존합니다.
4. 검증 데이터로 임계값을 비교하고 최종 테스트 데이터는 별도로 평가합니다.
5. 단순 기준 모델과 비교하여 클래스 가중치와 모델 선택의 효과를 설명합니다.

위 항목은 후속 개선 계획입니다. 이 README 보강으로 모델 성능이 개선되거나 재검증된 것은 아닙니다.
