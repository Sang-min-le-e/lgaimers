# 개선 노트

KBO 투구 제구 성공 확률 예측 대회 — 논의하면서 결정한 방향/이유를 기록. 새로운 아이디어나 결정이 생기면 이 파일에 이어서 추가.

## 검증(Validation) 전략

**현황 파악**: `train.csv`의 시즌별 `control_success` 비율

| 시즌 | 행 수 | 성공률 |
|---|---|---|
| 2019 | 237,413 | 0.565 |
| 2020 | 244,087 | 0.533 |
| 2021 | 247,088 | 0.533 |
| 2022 | 247,472 | 0.529 |
| 2023 | 245,525 | 0.500 |
| 2024 | 253,507 | 0.486 |

6년간 거의 단조 감소. 공인구/판정 기준 변화 등 구조적 시즌 트렌드로 추정.

**결정한 검증 방식**:
1. **랜덤 K-fold 금지** — 시즌 트렌드가 뚜렷해서 랜덤 스플릿은 미래 정보 누수로 검증 점수를 과대평가함.
2. **Walk-forward(rolling-origin) CV**: train 2019~2021→val 2022, train 2019~2022→val 2023, train 2019~2023→val 2024. 여러 fold 평균/분산으로 "미래 시즌 예측" 상황을 흉내냄. RF 학습이 1분대라 3~4 fold 도는 데 부담 없음.
3. **역할 분리**: 피처/하이퍼파라미터 튜닝 의사결정은 walk-forward CV 평균 기준. "제출할지 말지" 최종 판단은 2024 holdout(가장 최근 시즌) 기준. 실제 평가(2025, 245,789행)가 한 시즌 크기와 거의 일치해서 시즌 단위 holdout이 실전과 잘 맞음.
4. **cold-start 구간 별도 확인**: `asof_pitcher_n` 낮은(신인/이적생) 행들만 따로 val 성능 체크. 2025에도 신인/이적생 존재 → 이 구간 성능이 실제 점수를 좌우할 가능성.

**액션 아이템**:
- [x] Train 노트북에 walk-forward CV 루프 구현 (2026-08-12)
- [ ] fold별 Brier Skill Score + cold-start 구간 Brier Skill Score 로깅

### 적용 내역 (2026-08-12)

`[Baseline_Train]...ipynb`의 `## 4. 모델 정의와 학습` + `## 5. 검증` 두 섹션을
`## 4. 모델 학습과 검증 — Walk-forward CV` 하나로 통합하고, 단일 split을
walk-forward CV로 교체. 이후 `## 6. 전체 데이터로 재학습 & 모델 저장`은
`## 5.`로 번호만 당김 (내용 변경 없음, 여전히 전체 데이터로 최종 재학습).

**Before**: `train["season"] == 2024`로 한 번만 split → fit → 검증 (baseline 그대로)

**After**:
- `make_model()`: fold마다 새 파이프라인(RF, max_depth=10, min_samples_leaf=200 — 하이퍼파라미터는 그대로)을 만드는 함수로 분리
- `brier_skill_score()`: 기존 인라인 계산식을 함수로 분리해 fold마다 재사용
- `FOLDS`: `(2019~2021→2022)`, `(2019~2022→2023)`, `(2019~2023→2024)` 3개 fold. 마지막 fold는 기존 baseline과 동일한 split이라 이전 결과(Validation Score 415.57)와 비교 가능
- fold별 점수 출력 + 평균/표준편차(`mean_score`, `std_score`)로 안정성 확인
- 마지막에 `model = make_model()`로 학습 전 파이프라인만 재정의 — 다음 섹션(전체 재학습)에서 그대로 사용

**아직 안 한 것**: cold-start(`asof_pitcher_n` 낮은 행) 구간 별도 breakdown은 다음 단계.

## 피처 엔지니어링: asof_* 결측치 처리

**배경**: `train.csv` 결측치는 전부 `asof_*` 컬럼에만 있음 (다른 44개 수치형 컬럼은 결측 0).
- `asof_pitcher_success_rate` 등 5개(→ `asof_pitcher_n`): 0.05% 결측, 정확히 `asof_pitcher_n==0`(792행)일 때만 결측
- `asof_batter_success_rate` 등 2개(→ `asof_batter_n`): 0.06% 결측, `asof_batter_n==0`(830행)
- `asof_pitcher_fastball_rate` 등 3개(→ `asof_pitcher_pitchmix_n`): 0.05% 결측, `asof_pitcher_pitchmix_n==0`(792행)
- `asof_pitcher_prev1/3/5_game_*` 6개: 1.98%(29,185행) 결측, prev1/3/5가 항상 같이 결측(중첩 구조) — 첫 등판 이후 두 번째 완결 경기 전까지 값이 없음

**중요 발견**: cold-start(`asof_pitcher_n==0`, 792행)의 `control_success` 평균은 0.552로
전체 평균(0.524)보다 오히려 높음. 표본이 작아 우연일 수 있지만, 무작정 median으로
채우면 이 패턴이 사라지므로 별도 플래그로 모델이 직접 학습하게 함.

**결정한 처리 방식** (기존 test.csv 컬럼만 사용 → `script.py` 수정 불필요, Train 노트북만 수정):
1. **Bayesian smoothing**: `(n*rate + k*prior) / (n+k)`, `k=50`. `n=0`이면 자동으로
   prior(학습 데이터 평균)로 수렴. `AsofRateSmoother` 커스텀 sklearn transformer로 구현,
   `fit()`에서 prior 계산 → walk-forward CV의 각 fold가 자기 학습 데이터로만 prior를
   계산하므로 검증 데이터 누수 없음.
2. **cold-start 플래그**: `is_pitcher_cold_start`, `is_batter_cold_start` 2개 컬럼 추가.
3. **prev1/3/5_game 결측**: 같은 지표의 스무딩된 누적값(`asof_pitcher_success_rate` /
   `asof_pitcher_middle_rate`)으로 fallback.

**적용 위치**: `[Baseline_Train]...ipynb`
- Import 셀: `from sklearn.base import BaseEstimator, TransformerMixin` 추가
- `## 3. 전처리 정의`: `AsofRateSmoother` 클래스 정의 + `NUM_COLS_EXT`(플래그 2개 추가) + `preprocessor`
- `## 4. 모델 학습과 검증`: `make_model()`의 `Pipeline` 맨 앞에 `("asof_smooth", AsofRateSmoother())` 단계 추가

20만 행 샘플로 로컬 실행 검증 완료(에러 없음, 스무딩 후 결측 0개, cold-start 플래그 정상 생성).
전체 데이터 walk-forward CV 결과는 실행 후 추가 기록 예정.

**액션 아이템**:
- [x] 전체 데이터로 walk-forward CV 실행 (2026-08-12) — baseline 대비 소폭 상승 확인. 정확한 fold별 점수는 아래 "CV 점수 히스토리" 참고
- [ ] `SMOOTHING_K=50` 값 튜닝 여지 있음 (지금은 임의로 고정)

## 피처 엔지니어링: 파생 피처 + 원-핫 인코딩 (2026-08-12)

asof_* 스무딩에 이어서 세 가지를 한 번에 적용하고 CV로 효과를 확인하기로 함.

1. **투수-타자 능력 차이 피처**: `pitcher_batter_success_diff`,
   `pitcher_batter_middle_diff` — 스무딩된 `asof_pitcher_*_rate`와 `asof_batter_*_rate`의
   차이. RF가 얕은 트리(depth=10)라 이런 상호작용을 스스로 찾으려면 여러 split이
   필요한데, 명시적으로 만들어주면 더 적은 split으로 패턴을 잡을 수 있음.
2. **상황 압박 파생 피처**: `count_pressure`(볼-스트라이크 차), `is_full_count`,
   `is_two_strike_pressure`, `is_scoring_position`(2·3루 주자), `same_hand_matchup`(좌우 매치업)
3. **범주형 인코딩: OrdinalEncoder → OneHotEncoder**: `base_state`/`game_type`/`top_bottom`은
   명목형인데 기존 Ordinal은 임의 순서를 부여했음. 카디널리티 작아서(8/2/2, 총 12컬럼)
   원-핫 비용 부담 없음.

**적용 위치**: `## 3. 전처리 정의`에 `DerivedFeatureBuilder` 클래스 추가(파이프라인에서
`AsofRateSmoother` 다음 단계로 실행, 스무딩된 rate를 사용하므로), `preprocessor`의
`cat` 인코더를 `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`로 교체.
`## 4.`의 `make_model()`에 `("derived", DerivedFeatureBuilder())` 단계 추가.
20만 행 샘플로 에러 없이 동작 확인 완료. **script.py 수정 불필요** (기존 컬럼 조합만 사용).

**액션 아이템**:
- [ ] 전체 데이터로 walk-forward CV 실행 → asof_* 스무딩만 했을 때 대비 점수 변화 확인

## CV 점수 히스토리

실행할 때마다 fold별 점수를 여기에 기록. (mean / std, fold별: [2022, 2023, 2024])

| 단계 | CV Score (mean) | fold별 | 비고 |
|---|---|---|---|
| baseline (단일 split, 2024만 검증) | 415.57 | - | walk-forward 적용 전 |
| walk-forward CV 적용 (스무딩 전) | 8XX | 20XX, 0, 6XX | 사용자 확인 |
| + asof_* 스무딩 | 8XX | 20XX, 0, 6XX | "조금 오른 양상" |
| + 파생 피처 3종 (OneHot 포함) | 884.4 | [2176.6, 0.0, 476.7] | 재현 스크립트로 정밀 확인 (val=2022/2023/2024) |
| + recency weighting (half_life=2) | 910.9 | [2188.9, 0.0, 543.7] | 효과 미미. 아래 "recency weighting 시도" 참고 |
| RF → HGB 교체 (기본 파라미터, max_iter=100/lr=0.1/leaf=200) | 937.03 | [2213.53, 0, 597.54] | **실제 제출 점수 810점** (2026-08-14) |
| + HGB 그리드서치 (lr=0.03, max_leaf_nodes=63, min_samples_leaf=400) | 981.79 | [2254.29, 0, 691.08] | CV상 최적이었으나 **실제 제출 767점으로 하락** (810점 대비 -43). 되돌림 — 아래 "교훈" 참고 |
| + TrendCalibrator (season 선형회귀 + logit shift, anchor=마지막 학습 시즌) | 908.5 | [2075.08, 0, 650.35] | mean은 기본 HGB(937.03)보다 낮지만 fold=2024(대리 지표)는 개선(597.54→650.35). **실제 제출 862.9점** (810점 대비 +52.9) — fold=2024 우선 판단이 실전과 일치 |
| + TrendCalibrator 반기 단위 업그레이드 (season→season+0.5*half) | 952.9 | [2195.25, 0, 663.54] | 3-fold 전부 개선 방향이었으나 **실제 제출 851점으로 하락** (862.9 대비 -11.9). HGB 그리드서치와 같은 CV/실전 불일치 패턴. 되돌림 — 아래 "반기 단위 TrendCalibrator 실제 제출 결과" 참고 |
| + game_type(R/F)별 분리 트렌드 (season 단위 위에 적용) | 827.4 | [2074.83, 0, 407.23] | CV부터 기존보다 하락 — F 표본이 적어 추세선이 노이즈에 휘둘림. 실제 제출 안 함, 코드 미반영. 아래 "game_type(R/F) 분리 시도" 참고 |
| HGB → CatBoost 교체 (season TrendCalibrator 유지) | 956.1 | [2165.0, 0, 703.3] | CV상 최고(RF/HGB/LightGBM/XGBoost/CatBoost 중)였으나 **실제 제출 862.2점으로 862.9와 사실상 동률(-0.7)**. CV 개선(mean +47.6)이 실전에 전혀 반영 안 됨 — 아래 "CV proxy 신뢰도에 대한 종합 진단" 참고 |
| hgb_native (OneHot → OrdinalEncoder + HGB `categorical_features`) | 900.00 | [2020.52, 0, 679.46] | fold=2022 -54.6 악화, fold=2024 +29.1 개선. 채택 안 함 — 아래 "네이티브 범주형 처리 실험" 참고 |
| catboost_native (원본 문자열 + CatBoost `cat_features`, OneHot/Ordinal 안 씀) | 970.55 | [2179.51, 0, 732.12] | mean·fold=2022·fold=2024 전부 CV 역대 최고(HGB+OneHot 대비 mean +62.1, fold=2024 +81.8)였으나 **실제 제출 855점으로 862.9 대비 -7.9, 862.2(CatBoost+OneHot) 대비도 -7.2 하락**. CV proxy가 가장 크게 개선됐는데 실전은 가장 크게 하락한 사례 — 되돌림. 아래 "네이티브 범주형 처리 실험", "CV proxy 신뢰도에 대한 종합 진단" 참고 |
| **hgb_ohe + team_id 범주형 수정** (HGB, CAT_COLS에 pitcher_team_id/batter_team_id 추가) | **934.02** | **[2122.82, 0, 679.24]** | fold=2022(+47.7)·fold=2024(+28.9) **동시에 개선**. **실제 제출 875점 — 새 최고 기록** (862.9 대비 +12.1) — 아래 "team_id 범주형 처리 누락 수정" 참고 |

## fold=2023 점수가 0으로 나오는 이유 진단 (2026-08-12)

**증상**: walk-forward CV 3개 fold 중 val=2023만 계속 0점. 처음엔 overfitting 의심.

**진단**: fold별 TRAIN vs VAL 점수를 직접 비교(재현 스크립트로 train도 예측/채점)해서 확인.

| val 시즌 | TRAIN 점수 | VAL 점수 | VAL 예측평균 | VAL 실제평균 | 차이 |
|---|---|---|---|---|---|
| 2022 | 2302.9 | 2176.6 | 0.5305 | 0.5289 | +0.0016 |
| 2023 | 2395.8 | 0.0 | 0.5216 | 0.5000 | +0.0216 |
| 2024 | 2173.0 | 476.7 | 0.5003 | 0.4861 | +0.0142 |

**결론**: **overfitting이 아니라 트렌드 추정 실패.** TRAIN 점수는 세 fold 모두 2173~2396으로
안정적(과적합이라면 fold마다 크게 달라야 함). 문제는 RF가 `season`을 정수 피처로만
다뤄서 훈련 구간 밖의 하락 추세를 못 뻗어(extrapolate)나가 예측 평균이 실제보다 항상
높게 나온다는 것. 과대예측 폭이 클수록(2023이 최대) 점수가 더 크게 깎임. Brier Skill
Score 공식이 `100000 * (1 - brier/기준선)`이고 기준선(`r(1-r)`)이 항상 0.24~0.25로
거의 고정이라, Brier의 아주 작은 절대 차이(0.01 수준)도 점수로는 수천 점 차이로
증폭되어 보임 — fold 간 점수 차이가 커 보이는 건 실제 예측력 차이보다 메트릭 특성 때문인
부분이 큼.

## 2025 실전 성능이 CV 평균과 다를 수 있다는 논의 (2026-08-12)

- 22/23/24 fold 단순 평균을 2025 예상 점수로 신뢰하기 어려움. **2022 fold는 학습 데이터가
  2019~2021(3개 시즌)뿐이라 최종 제출 모델(2019~2024, 6개 시즌)과 조건이 달라 대표성이
  낮음.**
- **2024 fold가 2025 추정에 가장 적절한 대리 지표**: 학습 시즌 수(5개)와 "1년 앞 예측"
  구조가 최종 제출(2019~2024 학습 → 2025 예측)과 가장 유사함.
- 핵심 불확실성은 "2024→2025 하락폭이 얼마나 클까"이고, 이건 데이터만으로는 사전에
  알 수 없음.

## recency weighting 시도 (2026-08-12)

**아이디어**: `RandomForestClassifier.fit()`에 `sample_weight`를 줘서 오래된 시즌
영향력을 줄이고 최근 시즌 비중을 높임. `season_sample_weight(seasons, half_life=2)`
— 2년마다 가중치 절반, 기준점은 학습 데이터의 마지막 시즌.

**적용 위치**: `## 4.`(CV 루프의 `fold_model.fit(..., clf__sample_weight=...)`) +
`## 5.`(최종 재학습 `model.fit(..., clf__sample_weight=...)`). `make_model()`은
그대로, `season_sample_weight()` 함수만 추가.

**결과**: 효과가 미미함. mean 884.4 → 910.9 (+26), val=2023은 여전히 0.0 (예측평균
0.5216→0.5209로 겨우 -0.0007 이동). **원인**: recency weighting은 "가장 최근 관측
시즌의 값에 가깝게" 앵커링할 뿐, 그 이후의 추가 하락을 미래로 외삽하진 못함. 2023 fold의
경우 가장 최근 학습 시즌(2022)의 실제값(0.529)조차 2023 실제값(0.500)보다 0.029 높아서,
가중치를 아무리 줘도 이 격차는 못 좁힘.

**참고로 확인한 것**: 시즌별 성공률에 단순 선형회귀를 적용하면 실제값과 매우 잘
맞음(예: train 2019~2023만으로 2024를 외삽하면 0.4918, 실제 0.4861 — 오차 0.0057).
recency weighting보다 **명시적 트렌드 외삽 기반 보정(calibration shift)**이 더 유망해
보이지만, 아직 구현하지 않음 — 제출/버전 이슈 처리하느라 잠시 보류.

**액션 아이템**:
- [ ] 트렌드 외삽 기반 calibration shift 구현 검토 (다음 세션 후보 1순위)
- [ ] `half_life` 값을 더 작게(예: 0.5~1) 해서 recency weighting 한계 추가 확인

## 제출 시도 & 발견한 버그 2건 (2026-08-12)

### 버그 1: `script.py`가 커스텀 클래스를 못 읽어서 로드 실패 (제출 전 미리 발견, 수정 완료)

`AsofRateSmoother`, `DerivedFeatureBuilder`는 노트북에서 정의한 커스텀 클래스인데,
`joblib.dump()`로 저장된 pickle은 클래스 이름/모듈 경로만 저장하고 로드하는 쪽에 그
클래스가 실제로 정의돼 있어야 복원됨. `script.py`에 이 클래스들이 없으면
`AttributeError: Can't get attribute 'AsofRateSmoother' on <module '__main__'>`로
실패 — 별도 프로세스에서 재현해서 확인.

**수정**: `baseline_submit/script.py` 상단에 두 클래스를 노트북과 동일하게 복사해서
추가. 임시 모델로 별도 프로세스 로드 + `test.csv` 5행 추론까지 정상 동작 확인 완료.
**주의**: 노트북에서 두 클래스 로직을 바꾸면 `script.py`에도 반드시 동일하게 반영해야
함 (자동 동기화 안 됨).

### 버그 2: 실제 제출 시 발생 — scikit-learn 버전 불일치 (원인 파악, 수정은 사용자 확인 대기 중)

**증상**: 제출 후 서버에서 `AttributeError: Can't get attribute '_RemainderColsList'
on <module 'sklearn.compose._column_transformer'>` 발생.

**원인**: `requirements.txt`엔 `scikit-learn==1.8.0`이라고 적혀 있는데, 실제 로컬
학습 환경(Train 노트북을 실행한 이 환경)엔 **scikit-learn 1.6.1**이 설치돼 있음
(joblib 1.5.3, pandas 2.3.3은 requirements.txt와 정확히 일치 — sklearn만 다름).
`joblib.dump()`로 저장된 sklearn 객체(`ColumnTransformer` 등)는 pickle 기반이라
저장 버전과 로드 버전이 정확히 같아야 하는데, 서버는 requirements.txt대로 1.8.0을
설치해서 내부 구현(`_RemainderColsList` 등 비공개 클래스)이 안 맞아 터짐.

**더 근본적인 원인**: 로컬 Python이 **3.9.6**인데 scikit-learn 1.7.0은 Python≥3.10,
1.8.0은 Python≥3.11을 요구해서 로컬에선 애초에 1.8.0 설치 자체가 불가능
(`pip install scikit-learn==1.8.0` 시도 결과 확인). 에러 메시지 속 서버 경로가
`python3.11/dist-packages/...`인 걸 보면 **평가 서버는 Python 3.11**이라 1.8.0 설치가
가능했던 것.

**검토한 선택지**:
- **A (추천, 미적용)**: `requirements.txt`를 `scikit-learn==1.6.1`로 수정. 재학습 불필요,
  지금 모델 그대로 재제출 가능. 1.6.1은 Python 버전 하한만 요구해서 서버(3.11)에서도
  설치 문제없음.
- B: 로컬 Python을 3.11+로 올리고 1.8.0으로 재학습. 훨씬 번거롭고 지금 쓰는 기능엔
  버전 이점이 없어서 비추천.

**상태**: A안대로 `requirements.txt`를 `scikit-learn==1.6.1`로 수정 완료 (2026-08-12).

## 모델 교체: RandomForest → HistGradientBoosting (2026-08-14)

**배경**: `make_model(model_type=...)`으로 동일한 전처리·walk-forward CV·recency
weighting 위에서 RF와 `HistGradientBoostingClassifier`(HGB)를 바로 비교할 수 있게
구현됨(`[Baseline_Train]...ipynb` `## 4.`). HGB는 log-loss를 직접 최적화하며 순차적으로
잔차를 학습해 확률 보정이 RF보다 좋은 경향이 있고, sklearn 내장이라 `requirements.txt`에
새 의존성이 필요 없음.

**결과**: HGB로 실제 제출 → **평가 점수 810점**. (walk-forward CV local mean은 기본
파라미터 기준 937.03 — recency weighting 적용 RF 910.9보다 높음. CV와 실제 평가 점수의
절대값 차이는 fold 구성이 다르니 그러려니 하되, HGB가 RF보다 낫다는 방향은 일치.)

**제출 중 발견한 버그 3 (수정 완료)**: `HistGradientBoostingClassifier`는 fit 후
`_feature_subsample_rng`라는 numpy `Generator(PCG64)` 객체를 내부 상태로 들고 있는데
(predict에는 안 쓰이는 fit 전용 상태), 이게 pickle에 그대로 들어가면 채점 서버의 numpy
버전이 학습 환경과 다를 때 `"not a known BitGenerator module"` 에러로 unpickle이 깨짐.
`## 5.` 저장 직전에 `del clf._feature_subsample_rng`로 제거해서 해결 (predict 결과에는
영향 없음 확인).

## HGB 하이퍼파라미터 그리드서치 (2026-08-14)

`## 4-1.`에 그리드서치 셀 추가 (`learning_rate` × `max_leaf_nodes` × `min_samples_leaf`,
27개 조합 × walk-forward 3-fold, `max_iter=100`은 고정). 백그라운드 재현 스크립트로
27개 조합 전체 실행 완료 (총 ~1067초).

**CV 최적 조합**: `learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=400` →
mean=981.79 (fold=[2254.29, 0, 691.08]). 기본 파라미터(937.03) 대비 +44.79.

**❌ 실제 제출 결과 — CV와 반대로 하락 (2026-08-14)**: 이 조합으로 재학습해 제출했더니
평가 점수가 **767점**(기본 파라미터 810점보다 -43점) 나옴. `## 4.`의 `make_model()`
HGB 분기를 기본 파라미터(`learning_rate=0.1, min_samples_leaf=200, max_leaf_nodes`
미지정=31)로 되돌리고, `## 4-1.` 그리드서치 셀은 노트북에서 삭제함 (아래 "교훈" 및 결과
테이블은 기록으로 남겨둠).

**교훈 — 이 walk-forward CV로 하이퍼파라미터 미세 튜닝은 신뢰하기 어려움**:
- fold=2023이 모든 조합에서 항상 0점이라, 27개 조합의 순위를 사실상 fold=2022/2024
  단 2개만으로 매긴 셈. 표본이 너무 적어 조합 간 CV 점수 차이가 진짜 일반화 성능 차이가
  아니라 노이즈일 가능성이 큼 (27번 비교하면 그중 하나는 우연히 높게 나옴 — 다중비교 문제).
- fold=2024(2025 예측의 가장 적절한 대리 지표로 판단했던 fold, 위 "2025 실전 성능이
  CV 평균과 다를 수 있다는 논의" 참고)도 CV상으로는 597.54→691.08로 개선됐는데 실제는
  반대로 갔음 — "2024 fold가 2025의 좋은 proxy"라는 가정 자체도 이번 사례로 약해짐.
- 가설: `max_leaf_nodes`를 63으로 늘려 모델 용량을 키운 게, 이 프로젝트의 핵심 약점인
  "미래로 갈수록 하락하는 추세를 못 뻗어나감(extrapolation 실패)"을 CV 관측 시즌
  (2022/2024)에서는 완화하는 것처럼 보였지만, 실제로는 과거 시즌의 세부 패턴에 더
  밀착(암기)한 것이라 2025로의 외삽은 오히려 더 나빠졌을 수 있음. 검증되지 않은 가설.
- **결론**: 이 CV로 모델 종류(RF vs HGB처럼 큰 구조적 차이)를 비교하는 건 유효했지만,
  같은 모델 안에서 하이퍼파라미터 미세 조정 순위를 매기는 데는 신뢰도가 부족함. 다음에
  하이퍼파라미터 탐색을 다시 시도한다면 (a) fold=2023 문제를 먼저 해결하거나,
  (b) CV 점수 차이가 fold std 대비 충분히 크지 않으면 채택하지 않는 등 더 보수적인
  기준이 필요.

| learning_rate | max_leaf_nodes | min_samples_leaf | mean_score |
|---|---|---|---|
| 0.03 | 63 | 400 | 981.79 |
| 0.03 | 63 | 200 | 977.27 |
| 0.03 | 31 | 200 | 975.20 |
| 0.03 | 31 | 400 | 974.51 |
| 0.03 | 63 | 100 | 973.86 |
| 0.1 (기존값) | 31 (기존값) | 200 (기존값) | 937.03 |

**주의 — 탐색 경계 효과**: 상위 5개 조합이 전부 `learning_rate=0.03`(탐색 범위 최솟값)에
몰려 있고, 상위권은 `max_leaf_nodes=63`(탐색 범위 최댓값)도 선호함. 즉 최적값이 그리드
경계 밖(`learning_rate < 0.03`, `max_leaf_nodes > 63`)에 있을 가능성이 있음 —
`learning_rate`를 낮추면 보통 `max_iter`를 늘려야 성능이 유지되므로, 다음 라운드는
`learning_rate ∈ {0.01, 0.02, 0.03}` × `max_iter ∈ {100, 200, 300}` ×
`max_leaf_nodes ∈ {63, 127}` 정도로 범위를 넓혀서 재탐색해볼 만함 (다음 세션 후보).

**참고**: `min_samples_leaf`는 세 값 모두에서 큰 값(400)이 조금씩 더 나은 경향 —
데이터가 140만 행으로 크기 때문에 리프를 더 크게 잡아도 과소적합으로 이어지지 않는 듯.

**fold=2023이 모든 조합에서 0점인 건 여전함** — HGB 하이퍼파라미터로는 해결 안 되는
문제. 원인은 기존에 진단한 "트렌드 추정 실패"(RF/HGB 공통, 위 "fold=2023 점수가 0으로
나오는 이유 진단" 참고)와 동일한 것으로 보임 — 트렌드 외삽 기반 calibration shift가
여전히 유효한 다음 액션 아이템.

## 트렌드 외삽 기반 calibration shift 구현 (2026-08-18)

액션 아이템 1순위였던 `TrendCalibrator` 구현 완료. `[Baseline_Train]...ipynb`
`## 3-1.`에 새 섹션 추가:

- **클래스**: `TrendCalibrator(BaseEstimator, ClassifierMixin)` — `make_model()`이
  만드는 파이프라인(asof_smooth/derived/pre/clf) 전체를 감싸는 메타 추정기.
  `fit()`에서 (a) 내부 파이프라인을 평소처럼 학습하고, (b) 학습 데이터의 시즌별
  실제 성공률에 1차 선형회귀(`np.polyfit`)를 적합해 추세선을 구하고, (c) 내부
  파이프라인 자신의 학습 데이터 평균 예측 확률(`train_pred_mean_`)을 보정
  기준점으로 저장. `predict_proba()`에서는 각 행의 `season`을 추세선에 대입한
  `target_rate`와 `train_pred_mean_`의 차이를 logit 공간에서 구해 모든 예측에
  동일하게 shift(순위 보존, 수준만 이동).
- **적용 위치**: `make_model(calibrate=USE_TREND_CALIBRATION)` — 파이프라인을
  만든 뒤 `calibrate=True`면 `TrendCalibrator`로 감싼다. walk-forward CV와 최종
  재학습 셀 둘 다 이 플래그를 그대로 따라감. 최종 재학습 셀은 `_feature_subsample_rng`
  삭제 시 `model.pipeline.named_steps["clf"]`로 한 단계 더 들어가도록 수정.
- **script.py**: `TrendCalibrator` 클래스(+ `logit`/`sigmoid` 헬퍼)를
  `AsofRateSmoother`/`DerivedFeatureBuilder`와 동일하게 복사해서 추가 완료.
  `build_features()`는 수정 불필요 (`season`이 이미 test.csv 컬럼).
- **`season` 컬럼 확인**: `test.csv` 5행 샘플의 `season`이 전부 2025로 고정되어
  있음을 확인 — 실전에서도 각 행이 자기 `season`(2025)을 추세선에 대입하는 구조가
  그대로 맞아떨어짐.

**아직 안 한 것**: 노트북 재실행으로 walk-forward CV 점수(보정 전/후 비교) 확인 —
사용자가 직접 실행해서 fold별 점수를 알려주면 위 "CV 점수 히스토리" 표에 추가 기록.
CV 결과가 좋으면 `model/rf.pkl` → `baseline_submit/model/rf.pkl`(또는 실제 폴더명
`submit/model/rf.pkl` — 아래 "파일 구조 메모" 참고) 복사 후 재제출 검토.

**미해결 리스크**: 이 보정은 "전체 예측 평균을 추세선 쪽으로 옮기는" 수준의 단순
intercept 보정이라, HGB 하이퍼파라미터 그리드서치 사례(CV 981.79 → 실제 767점,
"교훈" 섹션 참고)처럼 CV에서 좋아 보여도 실제 2025 성능은 다르게 나올 가능성이
있음. 특히 이번엔 fold=2023의 0점 문제 자체가 해소되는지가 이 보정이 진짜 효과가
있는지의 핵심 신호이므로, CV 결과를 볼 때 이 지점을 우선 확인할 것.

### 버그 발견 & 수정: 보정 기준점을 잘못 잡아 fold=2024 점수 급락 (2026-08-18)

**증상**: 사용자가 노트북을 재실행해서 CV를 돌렸더니 fold=2024 점수가 597.54(보정
전 HGB 기본값)에서 **361로 급락**. calibration을 켰는데 오히려 나빠짐.

**원인 진단** (`/tmp/verify_calibration.py` 재현 스크립트로 확인, 저장 안 함 — 1회성):
최초 구현은 보정 기준점(`train_pred_mean_`)을 "학습 전체 기간(2019~2023)에 대한
파이프라인 raw 예측 평균"으로 잡았음(fold=2024 기준 0.5313). 그런데 RF/HGB 같은
트리 기반 모델은 학습 범위 밖의 `season` 값(예: 2024)에 대해 진짜로 extrapolate하지
못하고, 마지막 분기점(2023)과 같은 leaf로 취급함 — 그 결과 `season=2024`(학습 범위
밖) 행의 raw 예측은 이미 "마지막 학습 시즌(2023) 수준"으로 저절로 수렴해 있었음
(실측: raw 예측 평균 0.4980 ≈ 2023 실제 평균 0.4999, 거의 일치). 기준점을 이보다 훨씬
높은 전체 기간 평균(0.5313)으로 잡으니, 이미 낮아져 있는 raw 예측을 한 번 더 크게
끌어내리는 과잉보정(overshoot)이 발생 — 예측 평균이 0.459까지 떨어져 실제 2024
평균(0.4861)을 오히려 지나쳐버림.

**수정**: 기준점을 "마지막 학습 시즌 자체의 행들에 대한 raw 예측 평균"
(`anchor_pred_mean_`)으로 변경. `TrendCalibrator.fit()`이 `season == last_season`인
행만 걸러 그 raw 예측 평균을 구함. 노트북 `## 3-1.` 셀 + `submit/script.py`의
`TrendCalibrator` 둘 다 반영 완료.

**재현 스크립트로 검증한 fold별 효과** (수정된 버전, HGB 기본 파라미터):

| val 시즌 | raw(보정 전) | calibrated(수정 후) | 비고 |
|---|---|---|---|
| 2022 | 2213.53 | 2075.08 | **악화** — 3개 시즌(2019~2021)뿐인 선형회귀가 과도하게 하락 추세를 잡아 target_rate가 실제보다 낮게 나옴(대표성 낮은 fold라는 기존 진단과 일치) |
| 2023 | 0.00 | 0.00 | 변화 없음 — 예측 평균 오차는 크게 줄었지만(±0.0204 → ±0.0041) Brier 절대값이 여전히 점수를 0으로 깎는 구간 |
| 2024 | 597.54 | 650.35 | **개선** — "2025 예측의 가장 적절한 대리 지표"로 이미 판단했던 fold(위 "2025 실전 성능이 CV 평균과 다를 수 있다는 논의" 참고) |

3-fold 단순 평균으로는 (2075.08+0+650.35)/3 ≈ 908.5로 보정 전 mean(937.03)보다 낮음.
**하지만 fold=2022는 학습 시즌 수가 적어(3개) 대표성이 낮다고 이미 결론 내린 fold라,
mean 대신 fold=2024를 우선 신호로 보는 기존 방침(위 "2025 실전 성능이 CV 평균과
다를 수 있다는 논의" 참고)을 따르면 이 수정은 순방향임.** 다만 이건 판단이지 확정이
아니므로, 사용자가 실제 노트북에서 3-fold 전체 출력을 다시 확인하고 판단할 것.

**✅ 실제 제출 결과 (2026-08-18): 862.9점.** 이전 HGB 기본(810점) 대비 **+52.9**.
fold=2024를 대리 지표로 우선한 판단이 실전에서도 맞아떨어짐 — HGB 그리드서치
사례(CV 좋아졌지만 실전 -43점)와 반대로, 이번엔 CV 신호와 실전 결과 방향이 일치.
`subit_sangmin.zip`(`script.py` + `requirements.txt` + `model/rf.pkl`)으로 제출.

## Isotonic 사후보정 + season/month 세밀화 시도 — 둘 다 기각 (2026-08-18)

862.9점 달성 후 "더 극적인 개선"을 위해 두 방향을 프로토타입(`/tmp/verify_calibration2.py`,
저장 안 함 — 1회성)으로 검증. **둘 다 노트북에 반영하지 않음.**

**시도 1 — season+month 결합 축으로 추세선 세밀화**: `time_idx = season*12+game_month`로
표본수 가중회귀. 효과 거의 없음 — fold=2022 2075.08→2046.34(악화), fold=2024
650.35→658.33(+8, 노이즈 수준). 노트북 반영 안 함.

**시도 2 — Isotonic 사후보정 (leave-one-fold-out)**: 각 fold의 val 예측을 채점할 때
"다른 두 fold"의 val 예측+실제값을 pooling해서 isotonic curve를 적합, 이 fold에 적용.
**오히려 크게 악화**: fold=2022 2046.34→776.44, fold=2024 658.33→458.87 (mean
901.56→411.77).

**원인**: isotonic regression은 "이 구간 예측이면 실제로 이 정도 나오더라"를 학습
데이터 pool의 평균에 맞춰 적합함. leave-one-fold-out으로 pooling한 다른 fold들은
자기 시즌 평균 성공률이 다름(예: 2022 예측 평가 시 pool=2023(0.500)+2024(0.486),
2022 실제=0.529) — isotonic이 이 시즌 간 평균 차이를 다시 끌어들여서, TrendCalibrator가
이미 제거한 시즌 레벨 편향을 도로 주입하는 꼴이 됨. "구간별 보정"을 의도했는데
실제로는 "다른 시즌 평균으로 재오염"이 발생.

**결론 / 교훈**: 시즌 간 평균이 다른 다중 연도 데이터를 pooling해서 어떤 형태로든
사후보정을 적합할 때는(isotonic이든 다른 재보정 기법이든) **먼저 시즌 레벨 효과를
제거한 잔차(residual) 공간에서** 학습해야 함. 이걸 안 하면 "정교한 보정"을 시도하다
트렌드 보정으로 얻은 이득을 다시 깎아먹을 수 있음 — 다음에 유사한 사후보정을 시도할
때는 이 함정을 먼저 고려할 것. 잔차 공간에서 다시 시도하는 건 구현 복잡도 대비 효과가
불확실해서 보류, 862.9점 버전(단순 season 레벨 트렌드 보정만)을 현재 최선으로 유지.

## trackman_history 활용 재검토: SSL/pseudo-labeling도 불가 — 완전히 기각 (2026-08-18)

"라벨이 없으면 SSL로 pseudo-label 만들어서 쓰면 안 되나?" 질문에 `data_description.md`를
다시 확인해서 확정적으로 기각함. 삼중으로 막혀 있음:

1. **애초에 라벨 대리 신호가 없음**: `trackman_history.csv` 컬럼은 `rel_speed`,
   `spin_rate`, `induced_vert_break`, `horz_break`, `extension`, `rel_height`,
   `rel_side`, `zone_speed`뿐 — 전부 투구 물리 특성이고 스트라이크존 통과 위치/판정
   결과 같은 "제구 성공" 관련 정보가 전혀 없음. Pseudo-label을 만들 재료 자체가 없음.
2. **join key 없음** (기존 확인 사항): `pitcher_id`/`pitcher_trackman_id` 겹치는 값 0개.
3. **규칙상 명시적 금지** (`data_description.md` "6) 사용 금지 정보" 섹션):
   > 현재 투구의 실제 위치 또는 코스 정보 / **현재 투구의 Trackman 측정값**
   row 단위로 trackman 측정값(혹은 그 파생값)을 피처로 쓰는 것 자체가 join 가능 여부와
   무관하게 규칙 위반. `trackman_history`는 오직 시즌/월 단위 **리그 집계** 형태로만
   허용됨(위 "trackman_history 리그 트렌드 피처 검토" 참고 — 그마저도 season과 거의
   공선이라 효과 약함).

**결론**: trackman_history 활용은 이 세 가지 이유로 완전히 닫힌 방향. 다음 세션에서
재검토 불필요.

## 다음 논의 후보 (로드맵)

- [x] 피처 엔지니어링: `trackman_history.csv` 활용 — **완전히 기각** (2026-08-18).
  join key 없음 + 라벨 대리 신호 없음 + 규칙상 row 단위 사용 명시적 금지, 삼중으로
  막힘. SSL/pseudo-labeling으로도 못 뚫음. 위 "trackman_history 활용 재검토" 섹션 참고.
  **다음 세션에서 재검토 불필요.**
- [x] 피처 엔지니어링: `asof_*` 결측치(표본 0) 처리 — 위 섹션 참고
- [x] 기존 컬럼 조합 파생 피처 (카운트 압박, 매치업 등) — 위 섹션 참고
- [x] recency weighting 시도 — 효과 미미, 트렌드 외삽 기반 보정이 더 유망 (위 섹션 참고)
- [x] **1순위**: 트렌드 외삽 기반 calibration shift 구현 — `TrendCalibrator` 추가 완료,
  버그(보정 기준점 오류) 수정 완료. **실제 제출 862.9점** (810점 대비 +52.9, 위 섹션 참고)
- [x] `requirements.txt` scikit-learn 버전 수정 (1.8.0 → 1.6.1)
- [x] 모델 선택: RF → HGB 비교 — HGB 채택, 실제 제출 810점 (위 섹션 참고)
- [x] 모델 선택: HGB → LightGBM/XGBoost/CatBoost 비교 — **CatBoost 채택** (2026-08-19,
  CV mean 908.5→956.1). 위 "모델 교체: HistGradientBoosting → CatBoost" 참고.
  아직 실제 제출 전 — 노트북 재실행 필요.
- [x] HGB 하이퍼파라미터 그리드서치 — **실제 제출에서 810→767점 하락, 기본 파라미터로 되돌림** (위 섹션 "교훈" 참고). `## 4-1.` 셀은 노트북에서 삭제, 결과는 기록으로만 남김.
- [x] Isotonic 사후보정 + season/month 세밀화 — **둘 다 기각** (2026-08-18, 위 섹션 참고).
  Isotonic은 시즌 간 평균 재오염으로 오히려 악화, season+month 세밀화는 효과 미미.
- [ ] 그리드 경계 밖 재탐색은 보류 — 이 CV로 미세 튜닝 자체의 신뢰도 문제부터 해결 필요
- [x] 앙상블/스태킹 — CatBoost 비교와 같이 20개 조합 테스트, 대부분 CatBoost 단독보다
  못했고 최고 조합도 마진 대비 pickle/버전 관리 복잡도가 커서 보류 (2026-08-19,
  위 "모델 교체" 섹션 참고)
- [x] fold=2023이 모든 시도에서 계속 0점인 근본 원인 파고들기 — **부분 해결** (2026-08-19).
  regime(룰 변경 시점) 가설은 기각, 대신 반기 단위 시간축으로 TrendCalibrator를
  업그레이드해서 raw 미보정 오차를 절반으로 줄임(+0.0205→+0.0099). 여전히 0점이지만
  Brier 문턱 효과로 보임 — 위 "regime 가설 검증 + 반기 단위 TrendCalibrator 업그레이드"
  섹션 참고. 노트북 재실행으로 CV 재확인 및 재제출 검토가 다음 액션.

## regime 가설 검증 + 반기 단위 TrendCalibrator 업그레이드 (2026-08-19)

**출발점**: "다음 세션 후보 1순위"였던 fold=2023 근본 원인을 더 파고들기 위해, 사용자가
KBO 룰 변경 시점 기준 regime 가설을 제안: 2019~2021(구체제) / 2022~2023(스트라이크존
정상화) / 2024~(ABS). 실제 이 연도 구분이 맞는지 WebSearch로 확인 — **정확함**:
스트라이크존 정상화는 2022시즌부터([한국일보](https://m.hankookilbo.com/News/Read/A2022032308560000897)),
ABS는 2024시즌부터 전면 도입([스포츠Q](https://www.sportsq.co.kr/news/articleView.html?idxno=463319)).

**regime 가설 자체는 기각**: raw(보정 전) 예측을 regime별로 나눠 TRAIN/VAL 오차를
찍어보니, 오차가 regime 경계와 무관했음:

| fold(val) | 상황 | raw 오차 |
|---|---|---|
| val=2022 | regime A→B **경계** (완전 새 regime) | +0.0023 (양호) |
| val=2023 | regime B **내부** (2022→2023, "같은 regime") | **+0.0205 (최악)** |
| val=2024 | regime B→C **경계** (ABS 도입) | +0.0119 (중간) |

즉 "새 regime으로 넘어가는 게 더 어렵다"는 직관과 반대로 최악의 오차는 오히려
같은 regime 내부에서 발생. 연도별 실제 낙폭(2019→20: -0.032, 20→21: 0.000,
21→22: -0.004, **22→23: -0.029**, 23→24: -0.014)도 큰 낙폭 두 개가 전부 regime
내부에서 일어남 — regime 카테고리를 피처/캘리브레이션 축으로 추가해도 fold=2023을
설명하지 못함.

**대신 발견한 것 — 반기 단위로 쪼개면 "급락"이 사라진다**: `game_month`로 시즌을
전반기(3~6월)/후반기(7~10월)로 나눠보니, 2022년 성공률 하락(H1=0.5407→H2=0.5167,
-0.024)이 **이미 2022년 안에서 진행 중**이었음. 시즌 평균(2022=0.5289)은 이 하락을
앞선 H1의 높은 값과 섞어서 실제보다 높게 잡히게 만드는 착시였던 것. 반기 경계에서
실제 낙폭(H2 2022→H1 2023: -0.0212)은 시즌 평균 기준 낙폭(-0.0289)보다 훨씬 작음.
(참고: 2023시즌 KBO 주심 판정 정확도가 91.3%라는 기사로 미루어, 2022년에 시행된
룰이 실제 판정 관행에 완전히 반영되는 데 한 시즌 더 걸렸을 가능성 — 규칙 시행
연도와 통계적 효과 발현 시점이 어긋날 수 있다는 교훈.)

**적용**: `TrendCalibrator`의 시간축을 `season`(시즌 단위)에서
`season + 0.5*(game_month>=7)`(반기 단위)로 교체. `game_month`는 이미 test.csv
컬럼이라 새 피처 아님, `script.py` `build_features()` 수정 불필요. 대신
`TrendCalibrator` 클래스 정의는 노트북 `## 3-1.`/`submit/script.py` 둘 다 동일하게
수정 완료 (버그 수정 이력 포함해서 docstring도 갱신).

**재현 스크립트로 확인한 CV 효과**:

| | season 단위 (기존, 862.9점 제출 버전) | 반기 단위 (신규) |
|---|---|---|
| val=2022 | 2075.08 | **2195.25** |
| val=2023 | 0.00 (raw 오차 +0.0205) | 0.00 (raw 오차 +0.0099로 절반↓, 여전히 baseline 못 넘음) |
| val=2024 | 650.35 | **663.54** |
| **mean** | 908.5 | **952.9** |

3-fold 전부 방향이 개선(fold=2023은 Brier 문턱 효과로 여전히 0점이지만 내부
미보정 오차는 크게 줄었음). 특히 2025 예측의 대리 지표로 삼아온 fold=2024가
꾸준히 좋아졌다는 점에서 채택. HGB 그리드서치처럼 여러 조합을 CV로 비교해서
고른 게 아니라 단일 원인 진단(반기 내 하락 패턴)에 기반한 변경이라 과적합
위험은 상대적으로 낮다고 판단하지만, 확정은 아님 — 노트북 재실행 후 3-fold
전체 출력을 사용자가 직접 확인할 것.

**전략적 함의 (미해결)**: 최종 제출 모델(2019~2024 학습 → 2025 예측)은 ABS
regime의 딱 1개 시즌(2024)만 보고 그 다음 해를 예측해야 하는데, 이건 구조적으로
fold=2023(스트라이크존 regime 1년차 2022만 보고 2년차 2023 예측)과 유사한 상황 —
그 fold가 가장 나쁜 케이스였음. "regime 2년차에 추가로 크게 떨어지는" 패턴이
2025에도 반복될 위험이 있다는 뜻이지만, 실제 관측 사례가 (2022→2023) 단 1건뿐이라
이 패턴에 근거해 캘리브레이션을 추가로 공격적으로 조정하는 건 보류 (n=1 근거로
HGB 그리드서치처럼 작은 표본에 과적합할 위험).

**액션 아이템**:
- [ ] 노트북 재실행 → 3-fold 전체 출력 확인, `model/rf.pkl` → `submit/model/rf.pkl`
  복사 후 재제출 검토
- [ ] fold=2023이 반기 단위로도 여전히 0점인 것 — Brier metric의 문턱 효과(baseline
  0.25를 살짝 못 넘음)로 보이나, 근본적으로 "다음 반기의 정확한 낙폭"은 4~5개
  표본만으로는 예측 불가능한 노이즈에 가까울 수 있음. 추가로 깊게 팔 가치가
  있는지는 불확실 — 다른 방향(모델 비교/앙상블)과 기대값 저울질 필요

## 반기 단위 TrendCalibrator 실제 제출 결과 — 851점으로 하락, 되돌림 (2026-08-19)

CV에서 mean 908.5→952.9로 전 fold 개선을 보였던 반기 단위 TrendCalibrator를 실제
제출한 결과 **851점** (season 단위 862.9점 대비 **-11.9**). HGB 그리드서치 때와
동일한 패턴 — fold=2024 proxy가 좋아졌는데 실전은 나빠짐.

**메커니즘 분석**: 최종 모델(전체 2019~2024 재학습)에서 두 버전이 2025에 실제로
예측하는 target rate를 비교하면, 전체 평균은 거의 동일함(season 0.4747 vs
반기 blended ~0.4739). 다른 건 "누구를 얼마나 끌어내리는지"의 재분배 — 반기
버전은 H1 행은 덜 끌어내리고(anchor 대비 -0.0062) H2 행은 더 끌어내림(-0.0134).
전체 수준은 안 바뀌었는데 행별 재분배가 손해를 본 셈. **교훈**: fold=2024는
"2025 예측"의 대리 지표로 딱 1번(2024→2025)만 검증 가능한 극소표본이라, 이 proxy가
좋아졌다고 실전이 좋아진다는 보장이 없다 — 이미 HGB 그리드서치에서 배운 교훈이
캘리브레이션 세분화에도 그대로 적용됨. 두 번 연속 같은 패턴이 나온 이상, "CV에서
좋아 보이는 세분화"를 실제 제출 없이 신뢰하는 건 위험하다는 게 이제 훨씬 강한 증거로
확립됨.

## game_type(R/F) 분리 시도 — 실제 평가셋 구성에 대한 중요한 발견 (2026-08-19)

fold=2023을 더 파려고 `game_type` 컬럼을 조사하다가 예상 밖의 구조를 발견함.

**`game_type`은 "한국시리즈"가 아니라 R(1군 정규시즌, 전체의 ~89%)/F(2군, 추정
퓨처스리그, ~11%) 구분**. 나무위키로 확인한 퓨처스리그 특징: 낮경기만, 10회부터
승부치기, 무엇보다 **"투수는 투구폼 교정·신 변화구 연습"** 목적으로 등판하는
경우가 많음 — 즉 안정적인 제구가 아니라 실험이 목적인 리그. 실제로 F의 season별
성공률이 극심하게 요동침(0.689/0.588/0.704/0.709/0.473/0.459)에 반해 R은 훨씬
매끄럽게 단조 감소(0.550/0.527/0.513/0.504/0.503/0.490). 웹서치로 확인: ABS가
퓨처스리그에는 2020년 8월부터 일부 구장 시범 운영으로 먼저 들어갔고(1군은 2024년
전면 도입), 그 시범 운영 기간 동안 존 파라미터가 계속 바뀌었을 것으로 추정 — F가
안정된 하나의 regime이 아니라 실험장이었다는 뜻.

**"R만 학습/검증하면 나아지지 않을까" 실험 (4가지 조합, R-only로 val 채점)**:

| 조합 | mean | fold=[2022,2023,2024] |
|---|---|---|
| B. 학습 R+F 그대로 / 검증만 R로 제한 | 457.0 | [354.4, 379.6, 637.1] |
| C. 학습도 R만 / 검증 R만 | 379.6 | [342.3, **207.4**, 589.2] |
| D. 학습 R+F / 캘리브레이션만 R 서브셋 / 검증 R만 | 439.9 | [364.8, 328.7, 626.1] |

C(F를 아예 학습에서 제외)가 가장 나쁨 — F를 버리면 학습 행 수가 10~12% 줄면서
row-level 변별력(resolution)이 나빠지는 손해가, "깨끗한 R 트렌드"로 얻는 이득보다
큼. **F를 학습에서 빼는 건 손해.**

**그런데 이 R-only 검증 실험 자체가 잘못된 가정 위에 서 있었음이 드러남**: 실제
2025 평가 행 수(245,789)를 season별 R-only 행 수(21만~22만대)와 R+F 합계
행 수(24만대, 특히 2023=245,525)에 각각 대조해보면, **R+F 합계 쪽이 압도적으로
가깝다.** `test.csv`의 5개 샘플이 전부 R이었던 건 우연(F 비중 11%가 5번 연속
안 뽑힐 확률 약 56%)일 가능성이 큼 — **실제 평가셋은 R-only가 아니라 학습
데이터와 같은 비율로 R+F가 섞여 있을 것으로 추정.** 즉 지금까지 써온 R+F 혼합
walk-forward CV가 맞는 population을 보고 있었던 것이고, B/C/D는 애초에 잘못된
가정(R-only 평가)을 검증한 것이었음.

**그래서 시도한 것 — game_type별로 "따로" 추세선을 적합**(F를 버리지 않고 유지,
단 R행은 R추세, F행은 F추세로 각자 보정): `GameTypeTrendCalibrator`. **결과가 더
나쁨** — mean 827.4 [2074.83, 0, 407.23], 특히 fold=2024가 650.35→407.23으로
크게 하락. 원인: F의 season별 값 자체가 표본이 적어(연 2.5~3만행) 너무 노이즈가
커서, F만의 회귀선을 따로 적합하면 그 노이즈에 그대로 휘둘림(예: fold=2024 학습
구간에서 F slope=-0.031로 과도하게 가파르게 잡힘 — 2023년의 이상치성 폭락 하나에
회귀선이 끌려감).

**결론**: game_type 축을 어떤 식으로 써봐도(제외/분리 추정 모두) season 단일
트렌드보다 나은 조합을 못 찾음. 코드에는 반영하지 않음. 다만 "F가 왜 저렇게 튀는지"
설명이 되는 진짜 도메인 지식(퓨처스리그의 실험적 성격, ABS 시범운영 구간의 파라미터
변동)을 확보했다는 점은 소득 — 나중에 다른 형태로 활용할 여지는 남아있음(아래
"다음 논의 후보" 참고).

## season 단위 TrendCalibrator(862.9점)로 되돌림 (2026-08-19)

반기 단위, game_type 분리 두 실험 모두 실전/CV에서 기존 대비 나빠서 노트북
`## 3-1.`/`TrendCalibrator` 클래스와 `submit/script.py`를 season 단위 버전
(862.9점 실제 제출 버전)으로 되돌림. **`model/rf.pkl`, `submit/model/rf.pkl`은
아직 반기 단위 코드로 학습된 상태(851점 버전)라 stale함** — 재제출하려면 노트북을
다시 실행해서(`## 4.`/`## 5.`) season 단위 코드로 재학습한 모델을 새로 저장해야 함.

## KBO 나무위키 도메인 조사 — 존 크기 자체가 매년 조정되고 있었음 (2026-08-19)

나무위키 "자동 투구 판정 시스템" 문서에서 확인한 사실: ABS 스트라이크존 파라미터가
1군 도입(2024) 이후에도 매년 계속 조정되고 있음.

| 시즌 | 존 상단 기준(신장 대비) | 존 하단 기준 |
|---|---|---|
| 2024 | 56.35% | 27.64% |
| 2025 | 55.75% | 27.04% |
| 2026 | 53.5% | 27.0% |

2024→2025로 존 상단이 약 1cm 더 낮아짐(존이 작아짐 → 제구 성공이 더 어려워짐) —
**이건 데이터 안에서 유추하는 게 아니라 KBO가 공식 발표한 사실.** "2025년에도
하락이 계속될 것"이라는 방향성은 이 자료로 뒷받침되지만, 정확한 하락 폭까지
이 수치로 역산해서 calibration에 반영하는 건 아직 안 함 — 매핑 방법이 불분명하고
(zone 면적 축소율과 control_success 하락률의 관계를 모름), 지금까지 캘리브레이션을
세밀화하는 시도가 두 번 다 실전에서 역효과였다는 점을 고려하면 섣불리 이 숫자를
보정 공식에 꽂아 넣는 것도 위험. 다만 "제구 성공"의 정의 자체가 스트라이크존
경계와 관련 있다면(데이터 설명서 재확인 필요), 존 파라미터가 test.csv에 없는
정보이므로 새로운 외부 지식으로 활용할 여지는 있음 — 아래 로드맵 참고.

## 모델 교체: HistGradientBoosting → CatBoost (2026-08-19)

캘리브레이션 세분화(반기 단위, game_type 분리)가 두 번 다 CV/실전에서 역효과였던
반면, **RF→HGB 교체는 CV·실전 방향이 일치했던 유일한 성공 사례**였다는 점에 착안해
"모델 계열 자체를 바꾸는" 구조적 레버를 다시 시도. `LightGBM`/`XGBoost`/`CatBoost`를
설치(로컬 venv, 각각 4.7.0/3.4.1/1.2.10)하고 재현 스크립트(`/tmp/.../model_compare.py`,
저장 안 함)로 season TrendCalibrator 위에서 5개 모델을 동일 조건(walk-forward
3-fold, recency weighting half_life=2, iterations/n_estimators=100,
learning_rate=0.1) 비교:

| 모델 | CV mean | fold=[2022,2023,2024] |
|---|---|---|
| RF | 887.3 | [2048.5, 0, 613.5] |
| HGB (기존) | 908.5 | [2075.1, 0, 650.4] |
| LightGBM | 909.1 | [2097.9, 0, 629.4] |
| XGBoost | 847.7 | [2027.8, 0, 515.2] |
| **CatBoost** | **956.1** | **[2165.0, 0, 703.3]** |

**CatBoost가 mean·fold=2024 둘 다에서 가장 좋음** (HGB 대비 mean +47.6, fold=2024
+53.0). 하이퍼파라미터를 튜닝한 게 아니라 알고리즘 자체만 비교한 결과라, 27개
조합을 비교해 노이즈에 낚였던 HGB 그리드서치 사례와는 성격이 다름 — RF→HGB
교체와 같은 급의 "구조적" 비교.

**앙상블도 같이 확인**: RF+HGB+LightGBM+XGBoost+CatBoost 중 2~3개 조합 20가지를
raw predict_proba의 logit 평균 → season TrendCalibrator 순서로 테스트. 대부분
CatBoost 단독(956.1)보다 낮았고, 최고 조합(LGBM+CatBoost, mean 962.45)도 CatBoost
단독 대비 마진이 작은 반면 모델 2~3개를 pickle/버전까지 관리해야 하는 복잡도가
커짐 — 오늘 이미 pickle/버전 불일치 버그를 3번 겪은 전례(위 "제출 시도 & 발견한
버그" 섹션들)를 고려해 앙상블은 채택하지 않고 **CatBoost 단독**으로 결정.

**적용**: 노트북 `## 1.`에 `from catboost import CatBoostClassifier` 추가,
`## 4.`의 `make_model()`에 `model_type="catboost"` 분기 추가 후 `MODEL_TYPE`
기본값을 `"catboost"`로 변경. `preprocessor`(OneHotEncoder 등)는 그대로 공유하므로
CatBoost 고유의 categorical 처리(ordered boosting 등)는 아직 활용 안 함 — 추가
개선 여지로 남음. `submit/requirements.txt`에 `catboost==1.2.10` 추가.

**버전/pickle 호환성 사전 검증 (2026-08-19)**: PyPI에서 `catboost==1.2.10`이
`cp311`(평가 서버 Python 버전) wheel을 제공하는지 확인 완료. 로컬에서 fit →
`joblib.dump()` → 별도 프로세스에서 `submit/script.py`를 통째로 복사해 실행하는
end-to-end 테스트로 `test.csv` 5행 추론까지 정상 동작 확인. `script.py` 자체는 수정
불필요 — `CatBoostClassifier`는 라이브러리 클래스라 `requirements.txt`로 설치만
되면 joblib이 알아서 임포트함(`AsofRateSmoother` 등 커스텀 클래스와 다름). HGB의
`_feature_subsample_rng` 같은 pickle 안 되는 내부 상태도 CatBoost에는 없음을 확인.

**✅ 실제 제출 결과 (2026-08-19): 862.2점.** season TrendCalibrator+HGB(862.9점)
대비 **-0.7, 사실상 동률** — CV에서 mean +47.6, fold=2024 +53.0이나 개선됐던 것에
비하면 사실상 무의미한 변화. 되돌리지는 않음(HGB와 성능이 동등하고 구조적으로
더 나은 모델이라는 근거는 여전히 있음) — 아래 "CV proxy 신뢰도에 대한 종합 진단"
참고.

## CV proxy(특히 fold=2024) 신뢰도에 대한 종합 진단 (2026-08-19)

오늘 세션에서 CV(특히 "2025의 가장 적절한 대리 지표"로 삼아온 fold=2024)가 크게
개선된 3가지 시도를 실제 제출까지 확인했는데, **3번 다 실전에서 그 개선이 재현되지
않음**:

| 시도 | fold=2024 CV 변화 | 실전 점수 변화 |
|---|---|---|
| HGB 하이퍼파라미터 그리드서치 (2026-08-14) | 597.5→691.1 (+93.5) | 810→767 (**-43**) |
| 반기 단위 TrendCalibrator (2026-08-19) | 650.4→663.5 (+13.2) | 862.9→851 (**-11.9**) |
| HGB→CatBoost (2026-08-19) | 650.4→703.3 (+53.0) | 862.9→862.2 (**-0.7**, 동률) |
| CatBoost 네이티브 범주형 처리 (2026-08-20) | 650.4→732.1 (**+81.8, 역대 최고**) | 862.9→855 (**-7.9, 역대 최대 하락**) |

**대조적으로 유일하게 성공한 사례** — TrendCalibrator 최초 도입(버그 수정판)은
CV fold=2024 개선폭(+52.8)과 실전 개선폭(+52.9)이 거의 정확히 일치했음.

**잠정 결론**: "진짜 로직 오류를 고치는 변화"(TrendCalibrator의 잘못된 anchor 수정
등)는 CV와 실전이 일치하지만, "이미 합리적으로 작동하는 파이프라인을 더 정교하게
다듬는 변화"(하이퍼파라미터 튜닝, 캘리브레이션 시간축 세분화, 심지어 모델 계열
교체까지)는 CV에서 큰 개선을 보여도 실전에 전혀 반영되지 않는 패턴이 3번 연속
재현됨. 가설: 모델들 간 CV 점수 차이(RF~887, HGB~908, CatBoost~956)의 상당 부분이
"우리가 가진 6개 시즌의 특정 잡음 패턴(2019년 이례적 고점, F리그 오염, 존 정상화의
지연 효과 등)에 얼마나 잘 들어맞았는가"를 반영하는 것이지 "다음 시즌 실제 낙폭을
더 잘 예측하는 능력"을 반영하는 게 아닐 수 있음 — 매 시즌 실제 낙폭 자체가 6개
데이터 포인트만으로는 근본적으로 예측 불가능한 성분(외부 규정 세부 조정, 팬덤/판정
관행 변화 속도 등)을 가진 것으로 보임.

**시사점**: 이 시점부터는 "CV가 개선됐다"는 신호만으로 실전 개선을 기대하기 어려움.
로직 버그가 명백한 경우가 아니라면, 추가적인 미세 조정 시도의 기대값이 낮을 수
있음 — 862~863점대가 이 접근법(GBM + season trend calibration)의 사실상의 성능
천장일 가능성을 염두에 두고 다음 방향을 판단할 것.

## HGB vs CatBoost 이론 비교 + 코드 롤백 (2026-08-19, 오늘 세션 마무리)

CatBoost가 862.2점(HGB 862.9점과 동률)에 그친 이유를 이론적으로 짚어봄.

**이론적 차이**: HGB는 leaf-wise 트리 + 표준 gradient boosting. CatBoost는
**ordered boosting**(잔차 계산 시 타깃 누수/prediction shift 방지 — 노이즈 많은
데이터에서 일반화에 유리하다고 알려짐)과 **대칭(oblivious) 트리**(구조적으로 더
강한 정규화)를 씀. 확률 보정 품질도 이론상 CatBoost가 유리한 경우가 많음.

**그런데 이 이론적 차이가 실측에 안 나타남**: CV에서는 CatBoost가 크게 좋았지만
(RF→HGB 교체 때와 달리) 실전에는 반영이 안 됨. 해석: 지금 이 문제의 병목은
"어떤 GBM을 쓰는가"가 아니라 **"6개 시즌만으로 다음 시즌 낙폭을 맞히는 것" 자체의
근본적 불확실성**으로 보임 — 이 앞에서는 모델 알고리즘의 이론적 우열이 무의미해짐.

**아직 안 써본 진짜 차별점**: 지금 파이프라인은 범주형 컬럼(`game_type` 등)을
OneHotEncoder로 미리 인코딩해서 HGB/CatBoost 둘 다에 넣고 있음 — **양쪽 다 네이티브
범주형 처리 능력을 안 쓰고 있음**(HGB는 `categorical_features` 파라미터, CatBoost는
`cat_features` + ordered target statistics). 특히 `game_type`(R/F)이 성공률
0.51 vs 0.60로 실제 차이가 크다는 걸 오늘 확인했으므로, 이 정보를 다른 피처와의
상호작용까지 살려서 인코딩하면(OHE보다 표현력 높음) 추가 개선 여지가 있을 수
있음 — **다음 세션 후보**. CatBoost의 ordered target statistics 쪽이 이론상 더
정교하지만, HGB의 native categorical_features도 시도 안 해본 옵션.

**최종 결정**: 코드를 HGB(season TrendCalibrator, 862.9점 검증판)로 롤백.
`submit/requirements.txt`에서 `catboost==1.2.10` 제거. 노트북 `## 4.`의
`make_model()`에서 CatBoost 분기 제거, `MODEL_TYPE` 기본값 `"hgb"`로 복귀.
**`model/rf.pkl`, `submit/model/rf.pkl`은 아직 CatBoost로 학습된 상태(862.2점
버전)라 stale함** — 다음에 HGB로 재제출하려면 노트북 `## 4.`/`## 5.` 재실행 필요.

## 네이티브 범주형 처리 실험 (2026-08-20)

**배경**: 지난 세션 "HGB vs CatBoost 이론 비교" 결론에서 "아직 안 써본 진짜 차별점"으로
남겨뒀던 것 — HGB/CatBoost 둘 다 그동안 `game_type` 등 범주형 3개(`top_bottom`,
`game_type`, `base_state`)를 OneHotEncoder로 미리 펼쳐서 넣고 있었고, 두 모델의
네이티브 범주형 처리 능력(HGB `categorical_features`, CatBoost `cat_features` ordered
target statistics)은 시도한 적이 없었음. `game_type`(R/F) 성공률 차이가 0.51 vs
0.60로 크다는 걸 이미 확인해뒀던 터라, 다른 피처와의 상호작용까지 살려서 인코딩하면
개선 여지가 있을 거라는 가설.

**재현 스크립트**(`/tmp/.../native_categorical_compare.py`, 저장 안 함 — 1회성)로
walk-forward CV 3가지 변형 비교 (season TrendCalibrator + recency weighting
half_life=2 유지, 다른 조건 전부 동일):

| 변형 | mean | fold=[2022, 2023, 2024] |
|---|---|---|
| `hgb_ohe` (현재 프로덕션 재현) | 908.48 | [2075.08, 0, 650.35] |
| `hgb_native` (OrdinalEncoder + HGB `categorical_features`) | 900.00 | [2020.52, 0, 679.46] |
| `catboost_native` (원본 문자열 그대로 + CatBoost `cat_features`) | **970.55** | **[2179.51, 0, 732.12]** |

`hgb_ohe`가 기존 로그(908.5, [2075.08, 0, 650.35])와 정확히 일치해 재현 스크립트
자체는 신뢰할 수 있음을 먼저 확인.

- **hgb_native**: mean은 오히려 -8.48 하락(fold=2022가 -54.6 악화). fold=2024만
  +29.1 개선. 외부 의존성 추가 없이 시도할 수 있는 저위험 옵션이었지만 이득이
  작고 방향이 엇갈려 채택 안 함.
- **catboost_native**: mean·fold=2022·fold=2024 **전부** 개선. 특히 fold=2024=732.12는
  지금까지 나온 모든 시도(그리드서치, 캘리브레이션 세분화, 이전 CatBoost+OneHot
  포함) 중 역대 최고. CatBoost+OneHot(956.1/703.3, 위 표 참고)보다도 더 좋음 —
  네이티브 처리가 OneHot보다 실제로 더 많은 정보를 살리는 것으로 보임.

**리스크 — CV proxy 신뢰도 문제와의 충돌**: 위 "CV proxy(특히 fold=2024) 신뢰도에
대한 종합 진단"에서 이미 확립한 패턴은, "이미 합리적으로 작동하는 파이프라인을 더
정교하게 다듬는 변화"는 CV에서 크게 좋아 보여도 실전에 반영 안 된 사례가 3연속
(그리드서치 CV+93.5→실전-43, 반기 캘리브레이션 CV+13.2→실전-11.9, HGB→CatBoost
CV+53.0→실전-0.7 동률)이었다는 것. `catboost_native`도 성격상 이 세 사례와 비슷함
(파이프라인 정교화, 큰 CV 개선). 특히 CatBoost는 이미 한 번(OneHot 버전) 이 패턴으로
실패한 전례가 있어 재현 위험이 더 큼.

**결정**: 사용자 판단으로 `catboost_native` 채택, 재제출 진행. 노트북
(`[Baseline_Train]...ipynb`)과 `submit/script.py`/`requirements.txt`에 반영:
- `## 1.` (임포트): `from catboost import CatBoostClassifier` 추가
- `## 3.`: `preprocessor_native_cat` 신규 추가 — `CAT_COLS`는 `"passthrough"`로
  원본 문자열 그대로 통과, `NUM_COLS_EXT`만 median impute. 기존 OneHot
  `preprocessor`는 rf/hgb용으로 그대로 유지.
- `## 4.`: `make_model()`이 `model_type`별로 `pre`(전처리)까지 같이 선택하도록
  변경 (`rf`/`hgb`는 `preprocessor`, `catboost_native`는 `preprocessor_native_cat`).
  `catboost_native` 분기에서 `CatBoostClassifier(cat_features=list(range(len(CAT_COLS))))`
  — `preprocessor_native_cat`이 `set_output(transform="pandas")`라 컬럼 순서(CAT_COLS
  가 항상 앞쪽)가 고정되므로 위치 인덱스로 지정 가능. `MODEL_TYPE` 기본값을
  `"catboost_native"`로 변경.
- `submit/requirements.txt`: `catboost==1.2.10` 추가 (이전에 되돌리며 제거했던 걸 복원).
- **`submit/script.py`는 수정 불필요** — `CatBoostClassifier`는 라이브러리 클래스라
  `requirements.txt`로 설치만 되면 joblib이 알아서 임포트함 (커스텀 클래스가
  아니므로 클래스 정의를 script.py에 복사할 필요 없음). `"passthrough"`도 sklearn
  내장 문자열 sentinel이라 별도 클래스 불필요.

**사전 검증 (2026-08-20)**: `submit/script.py`의 클래스만 써서 5만 행 샘플로
`catboost_native` 파이프라인 fit → pickle → **별도 프로세스**에서 `submit/script.py`
클래스만으로 unpickle → `data/test.csv` 5행 추론까지 end-to-end 테스트 완료
(이전 HGB/CatBoost 도입 때와 동일한 절차). 에러 없음, 예측값 정상 범위
(`[0.428, 0.443, 0.452, 0.475, 0.463]`).

**✅ 실제 제출 결과 (2026-08-20): 855점.** 862.9점(HGB+OneHot) 대비 **-7.9**,
862.2점(CatBoost+OneHot) 대비도 **-7.2** — 네이티브 처리가 OneHot보다도 못함.
CV 개선폭(mean +62.1, fold=2024 +81.8, 역대 최고)이 가장 컸는데 실전 하락폭도
가장 컸던, 지금까지 중 가장 심한 CV/실전 역방향 사례. **"CV proxy 신뢰도에 대한
종합 진단" 섹션에 4번째 실패 사례로 추가 기록.** 노트북/`submit/requirements.txt`
모두 HGB(season TrendCalibrator)로 되돌림 — 아래 "team_id 범주형 처리 누락 수정"
섹션 참고 (되돌리면서 같이 적용한 새 개선).

**교훈**: 네이티브 범주형 처리(더 표현력 높은 인코딩)가 OneHot보다 나을 거라는
가설은 이번 프로젝트/데이터에서는 틀렸음. `game_type`(F리그)처럼 연도별로 성공률이
크게 요동치는 범주에 대해, 표현력이 높은 인코딩일수록 2019~2024 특유의 잡음
패턴에 더 밀착(암기)해서 2025 외삽이 오히려 나빠질 수 있다는 가설이 이제 더
힘을 얻음 — HGB 그리드서치 실패 때 세웠던 가설과 같은 계열.

## team_id 범주형 처리 누락 수정 (2026-08-20)

**배경**: catboost_native가 실전에서 실패(855점)한 뒤, "파이프라인을 더 정교하게
다듬는" 방향이 아니라 "실제 결함을 고치는" 방향의 다른 레버를 찾다가 발견함.
`CAT_COLS = ["top_bottom", "game_type", "base_state"]` 세 개뿐이라
`pitcher_team_id`/`batter_team_id`가 `NUM_COLS`에 들어가 있었음 — 즉 13개
범주(값 12~25)인 팀 ID가 **순서형 숫자로 median impute만 되고 원-핫이 안 된 채**
모델에 들어가고 있었음. 몇 세션 전 `base_state`/`game_type`/`top_bottom`에서
고쳤던 "OrdinalEncoder가 명목형에 임의 순서를 부여하는 문제"와 정확히 같은 종류의
결함 — 새로 만든 피처가 아니라 **기존 파이프라인의 놓친 부분**이라는 점에서, 지금까지
4연패한 "파이프라인 정교화" 시도들과는 성격이 다름(유일한 성공 사례인 TrendCalibrator
anchor 버그 수정과 같은 계열).

참고로 `pitcher_id`/`batter_id`(792/830개 고카디널리티 raw ID)도 같은 문제지만
원-핫 하기엔 카디널리티가 너무 크고, asof_* 피처들이 이미 개인별 능력치를 요약해주고
있어서 raw ID 자체는 드랍 후보로 같이 검토함.

**재현 스크립트**(`/tmp/.../team_id_categorical_compare.py`, 저장 안 함 — 1회성)로
walk-forward CV 3가지 비교. 기준은 catboost_native가 아니라 **실전 최고인 HGB+OneHot
+season TrendCalibrator**(recency weighting half_life=2 유지):

| 변형 | mean | fold=[2022, 2023, 2024] |
|---|---|---|
| `hgb_ohe_base` (기존, 재현 확인용) | 908.48 | [2075.08, 0, 650.35] |
| `hgb_ohe_teamid` (team_id를 CAT_COLS에 추가, 5개 원-핫) | **934.02** | **[2122.82, 0, 679.24]** |
| `hgb_ohe_teamid_noid` (+ raw pitcher_id/batter_id까지 드랍) | 916.12 | [2075.28, 0, 673.08] |

`hgb_ohe_teamid`가 가장 좋음 — **fold=2022(+47.7)와 fold=2024(+28.9)가 동시에
개선**됨. 지금까지 시도 대부분은 한쪽 fold가 좋아지면 다른 쪽이 나빠지는 트레이드오프
패턴이었는데(예: hgb_native), 이번엔 방향이 일치함. raw player_id까지 드랍하는 건
오히려 손해(934.02→916.12) — 개인 식별자 자체가 어느 정도 약한 신호(예: 낮은 ID일수록
오래전 데뷔 = 베테랑일 가능성)를 담고 있는 것으로 추정, 그대로 유지.

**주의 — 과신 금물**: catboost_native도 fold=2022(+104.4)·fold=2024(+81.8) 둘 다
CV에서 개선됐지만 실전은 하락했음(위 표 참고). "두 fold 동시 개선"이 실전 성공을
보장하는 신호는 아님이 이미 한 번 반증됨. 다만 이번 건은 알고리즘 교체가 아니라
명백한 인코딩 누락 수정이라는 점에서 성격상 다르다고 판단해 채택함 — 이것도 확정이
아니라 판단이므로, 실제 제출 결과가 나오면 이 절에 추가 기록할 것.

**적용**: 노트북 `## 1.`(임포트 셀)의 `CAT_COLS`에 `"pitcher_team_id"`,
`"batter_team_id"` 추가. `NUM_COLS`/`preprocessor`/`preprocessor_native_cat`은
전부 `CAT_COLS`를 참조해서 만들어지므로 별도 수정 없이 자동으로 반영됨.
`catboost_native`는 위에서 실전 실패가 확인됐으므로 이 기회에 같이 되돌림 —
`MODEL_TYPE` 기본값을 `"hgb"`로 복귀, `make_model()`의 `catboost_native` 분기와
`preprocessor_native_cat`, `from catboost import CatBoostClassifier` 임포트 전부
제거(HGB→CatBoost 첫 롤백 때와 동일한 방식). `submit/requirements.txt`에서
`catboost==1.2.10` 제거. **`submit/script.py`는 수정 불필요** (컬럼 조합만 바뀌었고
전처리 로직 자체는 pickle 안에 있음).

**✅ 실제 제출 결과 (2026-08-20): 875점.** 862.9점(HGB+OneHot, 기존 최고) 대비
**+12.1**, 새 최고 기록. CV fold=2022/2024 동시 개선이 실전에서도 순방향으로
확인됨 — catboost_native(두 fold 동시 개선했지만 실전 하락)와 달리, 이번엔
"명백한 인코딩 결함 수정"이 CV·실전 방향 일치라는 이 프로젝트의 유일한 성공
패턴(TrendCalibrator anchor 버그 수정)을 그대로 재현함. **가설 강화**: CV가
실전과 일치하는 변화는 "버그/결함 수정" 계열이고, 어긋나는 변화는 "이미 돌아가는
걸 더 정교하게 다듬는" 계열이라는 구분이 이제 5개 사례(성공 2, 실패 3)로 더 뚜렷해짐.

## 데이터 감사 — team_id류 결함이 더 있는지 전수 확인 (2026-08-20)

875점 달성 후, team_id처럼 "실제 결함"이 더 있는지 전체 47개 피처의 dtype/카디널리티를
훑어봄(`df[c].dtype`, `nunique()`, `min/max`, 결측 여부). 발견/확인 사항:

- **`pitcher_hand`/`batter_hand`**: 값 {1,2}인 이진 범주형인데 `CAT_COLS`에 없음 —
  team_id와 같은 종류처럼 보였으나, 재현 스크립트로 CV 비교하니 **점수가 소수점까지
  완전히 동일**(추가 전/후 `[2118.86, 0, 675.72]`로 동일). 이진 변수는 원-핫이든
  raw 숫자든 트리가 잡을 수 있는 분할 지점이 하나뿐이라 인코딩 방식이 무관함 —
  카디널리티 3 이상인 범주형(team_id처럼)에서만 원-핫이 실제로 의미 있다는 게
  확인됨. **반영 안 함.**
- **`pitcher_team_id == batter_team_id`**: 147만 행 전부 0건 — 자기 팀 매치업 같은
  데이터 이상 없음.
- **`home_win_expectancy` + `away_win_expectancy`**: 항상 100(평균 100.0001, std
  0.02) — 사실상 중복 변수지만 트리 모델엔 무해해서 손댈 이유 없음.
- **`li`(레버리지 지수)**: `control_success`와 거의 무관(사분위별 0.515~0.529로
  평평) — 압박 상황 자체는 제구 성공에 예측력이 약함.
- **`asof_pitcher_n` 등 `asof_*_n` 컬럼**: **시즌마다 리셋되지 않고 2019년부터
  누적**되는 값이었음(시즌별 max: 2019=3141, 2020=5883, 2021=8669, 2022=11643,
  2023=13636, 2024=15449). 버그는 아니지만 이 사실이 `AsofRateSmoother`의
  `SMOOTHING_K=50`이 왜 사실상 cold-start(n=0)에만 유의미하게 작동하는지 설명해줌
  — 중앙값 n≈2661 대비 K=50은 프라이어 가중치가 ~1.8%에 불과해 대부분의 행에서는
  스무딩이 거의 raw rate 그대로임(의도한 설계와 일치, cold-start 전용 보정이라는
  원래 목적 그대로).

**결론**: `run_top/bot/total_before` 정합성(항상 일치), `base_state` 인코딩 등도
같이 확인했고 추가로 손볼 만한 뚜렷한 결함은 못 찾음 — team_id 급의 "명백한
결함"은 이번 라운드로 거의 다 훑은 것으로 판단. 다음에 또 이런 감사를 하게 되면
위 확인된 항목들은 재검토 불필요.

## 파일 구조 메모

- `notebooks/`에 있던 두 노트북(Train/Inference)이 루트로 이동됨. Train 노트북 실행 결과로 루트 `./model/rf.pkl`이 새로 생성됨 — `baseline_submit/model/rf.pkl`과는 별개 파일이므로 제출 zip 만들 때 어느 걸 쓸지 주의.
