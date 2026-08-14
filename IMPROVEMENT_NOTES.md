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

## 다음 논의 후보 (로드맵)

- [ ] 피처 엔지니어링: `trackman_history.csv` 활용 — **주의**: `pitcher_id`(main, 정수)와
  `pitcher_trackman_id`(trackman, 정수) 겹치는 값 0개, `pitcher_team_id`(정수)와
  `pitcher_team`(문자열 팀코드)도 형식이 달라 매핑 테이블 없이는 개인/팀 단위 join 불가.
  season/month 단위 리그 전체 트렌드 피처 정도만 현실적.
- [x] 피처 엔지니어링: `asof_*` 결측치(표본 0) 처리 — 위 섹션 참고
- [x] 기존 컬럼 조합 파생 피처 (카운트 압박, 매치업 등) — 위 섹션 참고
- [x] recency weighting 시도 — 효과 미미, 트렌드 외삽 기반 보정이 더 유망 (위 섹션 참고)
- [ ] **1순위**: 트렌드 외삽 기반 calibration shift 구현 — 시즌별 선형회귀로 다음 시즌
  예상 성공률을 구해서 모델 예측 평균을 그쪽으로 보정. recency weighting보다 직접적인 해법일 가능성.
- [x] `requirements.txt` scikit-learn 버전 수정 (1.8.0 → 1.6.1)
- [x] 모델 선택: RF → HGB 비교 — HGB 채택, 실제 제출 810점 (위 섹션 참고)
- [ ] 모델 선택: HGB → LightGBM/XGBoost/CatBoost 비교 (여유 있으면)
- [x] HGB 하이퍼파라미터 그리드서치 — **실제 제출에서 810→767점 하락, 기본 파라미터로 되돌림** (위 섹션 "교훈" 참고). `## 4-1.` 셀은 노트북에서 삭제, 결과는 기록으로만 남김.
- [ ] 그리드 경계 밖 재탐색은 보류 — 이 CV로 미세 튜닝 자체의 신뢰도 문제부터 해결 필요
- [ ] 앙상블/스태킹 (여유 있으면)

## 파일 구조 메모

- `notebooks/`에 있던 두 노트북(Train/Inference)이 루트로 이동됨. Train 노트북 실행 결과로 루트 `./model/rf.pkl`이 새로 생성됨 — `baseline_submit/model/rf.pkl`과는 별개 파일이므로 제출 zip 만들 때 어느 걸 쓸지 주의.
