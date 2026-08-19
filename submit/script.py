# script.py
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

ID_COL = "row_id"
TARGET_COL = "control_success"


# =======================
# 학습 노트북에서 정의한 커스텀 전처리 클래스 (그대로 복사)
#
# joblib.load()는 pickle 기반이라, 모델 파일 안에 저장된 커스텀 클래스
# 인스턴스를 복원하려면 로드하는 쪽(this script)에도 동일한 이름의 클래스가
# 정의되어 있어야 한다. 노트북에서 클래스를 바꾸면 여기도 반드시 같이 바꿀 것.
# =======================

class AsofRateSmoother(BaseEstimator, TransformerMixin):
    """표본수(n)가 작을수록 prior(학습 데이터 평균) 쪽으로, 클수록 실제 관측값
    쪽으로 끌어당기는 (n*rate + k*prior) / (n+k) 형태의 empirical Bayes smoothing.
    n=0(결측)이면 그대로 prior 값이 된다.
    """

    RATE_GROUPS = [
        ("asof_pitcher_n", ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                             "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
                             "asof_pitcher_strike_rate"]),
        ("asof_batter_n", ["asof_batter_success_rate", "asof_batter_middle_rate"]),
        ("asof_pitcher_pitchmix_n", ["asof_pitcher_fastball_rate",
                                      "asof_pitcher_breaking_rate",
                                      "asof_pitcher_offspeed_rate"]),
    ]
    PREV_GAME_FALLBACK = [
        ("asof_pitcher_prev1_game_success_rate", "asof_pitcher_success_rate"),
        ("asof_pitcher_prev3_game_success_rate", "asof_pitcher_success_rate"),
        ("asof_pitcher_prev5_game_success_rate", "asof_pitcher_success_rate"),
        ("asof_pitcher_prev1_game_middle_rate", "asof_pitcher_middle_rate"),
        ("asof_pitcher_prev3_game_middle_rate", "asof_pitcher_middle_rate"),
        ("asof_pitcher_prev5_game_middle_rate", "asof_pitcher_middle_rate"),
    ]
    SMOOTHING_K = 50

    def fit(self, X, y=None):
        self.priors_ = {
            c: X[c].mean()
            for _, rate_cols in self.RATE_GROUPS for c in rate_cols
        }
        return self

    def transform(self, X):
        X = X.copy()
        X["is_pitcher_cold_start"] = (X["asof_pitcher_n"] == 0).astype(int)
        X["is_batter_cold_start"] = (X["asof_batter_n"] == 0).astype(int)

        for n_col, rate_cols in self.RATE_GROUPS:
            n = X[n_col]
            for c in rate_cols:
                raw = X[c].fillna(0)
                X[c] = (n * raw + self.SMOOTHING_K * self.priors_[c]) / (n + self.SMOOTHING_K)

        for prev_col, fallback_col in self.PREV_GAME_FALLBACK:
            X[prev_col] = X[prev_col].fillna(X[fallback_col])

        return X


class DerivedFeatureBuilder(BaseEstimator, TransformerMixin):
    """투수-타자 능력 차이, 카운트/주자 압박, 좌우 매치업 등 도메인 지식 기반
    상호작용을 명시적으로 만든다. row별 계산이라 fit에서 저장할 상태가 없다.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["pitcher_batter_success_diff"] = (
            X["asof_pitcher_success_rate"] - X["asof_batter_success_rate"])
        X["pitcher_batter_middle_diff"] = (
            X["asof_pitcher_middle_rate"] - X["asof_batter_middle_rate"])

        X["count_pressure"] = X["balls_before"] - X["strikes_before"]
        X["is_full_count"] = ((X["balls_before"] == 3) & (X["strikes_before"] == 2)).astype(int)
        X["is_two_strike_pressure"] = (X["strikes_before"] == 2).astype(int)
        X["is_scoring_position"] = ((X["runner_on_2b"] == 1) | (X["runner_on_3b"] == 1)).astype(int)

        X["same_hand_matchup"] = (X["pitcher_hand"] == X["batter_hand"]).astype(int)
        return X


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TrendCalibrator(BaseEstimator, ClassifierMixin):
    """시즌별 성공률의 선형 하락 추세를 학습 구간 밖(다음 시즌)으로 외삽해서,
    감싼 파이프라인의 예측 확률 평균 수준을 그 추세선 쪽으로 이동시키는 사후 보정.

    보정 기준점은 "마지막 학습 시즌 자체에 대한 raw 예측 평균"이다 (전체 기간
    평균이 아님). 트리 모델은 학습 범위 밖 season 값을 extrapolate하지 못하고
    마지막 분기점과 같은 leaf로 취급하므로, 학습 범위 밖 시즌의 raw 예측은 이미
    마지막 학습 시즌 수준으로 수렴해 있다 — 기준점을 전체 기간 평균으로 잡으면
    과잉보정(overshoot)하게 된다.

    [2026-08-19] 반기 단위 시간축, game_type(R/F)별 추세선 실험 모두 CV/실전에서
    이 season 단일 버전(실제 제출 862.9점)보다 못해 되돌림 — IMPROVEMENT_NOTES.md 참고.
    """

    def __init__(self, pipeline, season_col="season"):
        self.pipeline = pipeline
        self.season_col = season_col

    def fit(self, X, y, **fit_params):
        self.pipeline.fit(X, y, **fit_params)

        seasons = X[self.season_col]
        last_season = seasons.max()
        raw_train_pred = self.pipeline.predict_proba(X)[:, 1]
        self.anchor_pred_mean_ = raw_train_pred[(seasons == last_season).values].mean()

        season_means = y.groupby(seasons).mean()
        if season_means.shape[0] < 2:
            self.slope_, self.intercept_ = 0.0, float(season_means.mean())
        else:
            self.slope_, self.intercept_ = np.polyfit(
                season_means.index.values, season_means.values, 1)
        return self

    def _target_rate(self, X):
        return self.slope_ * X[self.season_col].values + self.intercept_

    def predict_proba(self, X):
        raw = self.pipeline.predict_proba(X)[:, 1]
        delta = logit(self._target_rate(X)) - logit(self.anchor_pred_mean_)
        calibrated = sigmoid(logit(raw) + delta)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


# =======================
# 학습 때 사용한 전처리 (그대로)
# =======================

def build_features(df):
    """모델 입력 추출 — 학습 때와 동일하게 row_id만 빼고 전부 사용.

    범주형 인코딩(top_bottom, game_type, base_state)과 결측 대치는
    모델 파일 안의 파이프라인이 함께 수행하므로 여기서는 컬럼만 고른다.
    """
    return df.drop(columns=[ID_COL])


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR = "./data"            # test.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"          # rf.pkl 위치
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "rf.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    model = joblib.load(MODEL_PATH)
    print(f" OK. n_features={getattr(model, 'n_features_in_', '?')}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 전처리 (학습과 동일) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test)
    print(f" features={X.shape[1]}")

    # ---- 예측 (제구 성공 확률) ----
    print("Inference model...")
    preds = model.predict_proba(X)[:, 1] if len(X) else []
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"✅ Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
