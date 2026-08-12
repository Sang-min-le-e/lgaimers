# 프로젝트: KBO 투구 제구 성공 확률 예측 (LG Aimers 대회)

투구 직전 시점 정보로 해당 투구의 "제구 성공" 확률(0~1)을 예측하는 이진 분류 대회.
평가지표는 Brier Skill Score. 규칙/컬럼 설명은 `data_description.md` 참고.

**세션 시작 시 가장 먼저 `IMPROVEMENT_NOTES.md`를 읽을 것.** 지금까지 논의하며 결정한
방향, 이유, 실험 결과, 미해결 이슈가 전부 거기 있음. 이 파일은 요약/포인터 용도.

## 파일 구조

- `[Baseline_Train]...ipynb` (루트): 모델 학습 노트북. 실행하면 `./model/rf.pkl` 생성.
  현재 파이프라인: asof_* Bayesian smoothing → 파생 피처(투수-타자 능력 차이, 상황
  압박) → OneHot 인코딩 → RandomForest (recency weighting 적용).
- `[Baseline_Inference]...ipynb` (루트): 위 모델로 로컬 추론 테스트하는 노트북.
- `baseline_submit/`: 실제 제출 패키지
  - `script.py`: 평가 서버가 실행하는 추론 코드. **주의**: 노트북에서 정의한 커스텀
    transformer 클래스(`AsofRateSmoother`, `DerivedFeatureBuilder`)를 여기에도
    동일하게 복사해서 넣어둠 — joblib pickle이 클래스 정의를 요구하기 때문. 노트북에서
    이 클래스들을 바꾸면 반드시 `script.py`도 같이 바꿔야 함 (자동 동기화 안 됨).
  - `requirements.txt`: `scikit-learn==1.6.1`로 고정 (로컬 Python 3.9.6에서 설치 가능한
    최신 버전. 평가 서버는 Python 3.11이라 더 새 버전도 설치는 되지만, pickle 버전
    호환성 때문에 **학습에 실제로 쓴 버전과 반드시 일치**시켜야 함 — 아래 미해결 이슈 참고)
  - `model/rf.pkl`: 제출용 모델. 루트 `./model/rf.pkl`을 노트북 재실행 후 복사해서 씀.
    **git에 커밋 안 함** (아래 gitignore 참고, 재실행으로 재생성).
- `data/`: `train.csv`(2019~2024), `test.csv`(5행 샘플), `trackman_history.csv`(개인
  매칭 불가 — pitcher_id/trackman_id 겹치는 값 0개, join 불가능. 리그 트렌드용으로만 검토됨).
  **폴더 전체가 gitignore 대상** — 대회 페이지에서 직접 다운로드해서 채워 넣어야 함.

## git 관련

- `data/`, 루트 `model/`, `baseline_submit/model/`, `submit/`, `submit.zip`, `output/`은
  전부 gitignore 처리됨 (재생성 가능하거나 우리가 수정하지 않는 파일들).
- 새 환경에서 clone하면 `data/` 아래 4개 파일과 `model/rf.pkl`, `baseline_submit/model/rf.pkl`이
  없는 상태로 시작함 — 데이터는 대회 페이지에서 재다운로드, 모델은 Train 노트북 재실행 +
  `cp model/rf.pkl baseline_submit/model/rf.pkl`로 채워야 함.

## 현재 상태 / 미해결 이슈

1. **CV 검증에서 val=2023 fold가 계속 0점** — overfitting 아니고 트렌드 추정 실패로 진단
   완료 (자세한 내용 `IMPROVEMENT_NOTES.md`). recency weighting 시도했지만 효과 미미.
   다음 시도 후보: 시즌별 선형회귀로 다음 시즌 예상 성공률을 구해 예측 평균을 보정하는
   calibration shift. **다음 세션 1순위 작업.**
2. 노트북 재실행 후 `model/rf.pkl` → `baseline_submit/model/rf.pkl` 복사 필요 (자동 아님).

## 작업 규칙

- `test.csv`의 다른 행을 이용한 통계(누적/빈도/target encoding/rolling)는 대회 규칙상 금지.
- 새 피처가 test.csv에 이미 있는 컬럼 조합이면 노트북(학습 pipeline)만 수정하면 됨.
  외부 데이터(`trackman_history.csv`) 등 pipeline 밖 로직을 쓰면 `script.py`의
  `build_features()`도 반드시 같이 수정해야 함.
- 개선/실험할 때마다 CV 점수(fold별 + mean/std)를 사용자에게 확인해서
  `IMPROVEMENT_NOTES.md`의 "CV 점수 히스토리" 표에 기록하는 게 이 프로젝트의 관례임.
