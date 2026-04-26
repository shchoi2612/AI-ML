"""
regime.py — K-Means Regime Detector
Week 2 K-Means 클러스터링 직접 응용:
변동성 피처로 시장 상태를 Bull / Sideways / Bear 3분류
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


REGIME_FEATURES = ["RV_21d", "Vol_Zscore", "VIX_Level", "MA200_Ratio"]
REGIME_NAMES = {0: "Bull", 1: "Sideways", 2: "Bear"}


def fit_regime(features: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """
    K-Means로 시장 Regime 학습.
    반환: (fitted KMeans, fitted StandardScaler)
    """
    X = features[REGIME_FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(X_scaled)
    return km, scaler


def predict_regime(features: pd.DataFrame,
                   km: KMeans, scaler: StandardScaler) -> pd.Series:
    """피처 데이터프레임에 Regime 레이블 부여."""
    X = features[REGIME_FEATURES].copy()
    mask = X.notna().all(axis=1)
    labels = pd.Series(index=features.index, dtype=object)
    X_scaled = scaler.transform(X[mask])
    raw_labels = km.predict(X_scaled)
    # 클러스터 → Bull/Bear/Sideways 매핑 (VIX 기준: 낮을수록 Bull)
    centers = scaler.inverse_transform(km.cluster_centers_)
    vix_col = REGIME_FEATURES.index("VIX_Level")
    vix_means = centers[:, vix_col]
    order = np.argsort(vix_means)          # 낮은 VIX = Bull, 높은 VIX = Bear
    mapping = {order[0]: "Bull", order[1]: "Sideways", order[2]: "Bear"}
    labels[mask] = pd.Series(raw_labels, index=features.index[mask]).map(mapping)
    return labels


def run_regime_detection(features: pd.DataFrame) -> tuple:
    """
    전체 파이프라인 실행.
    Returns: (regime_series, km, scaler, silhouette_score)
    """
    from sklearn.metrics import silhouette_score
    km, scaler = fit_regime(features)
    regimes = predict_regime(features, km, scaler)
    X = features[REGIME_FEATURES].dropna()
    X_scaled = scaler.transform(X)
    sil = silhouette_score(X_scaled, km.predict(X_scaled))
    return regimes, km, scaler, sil


if __name__ == "__main__":
    from data import build_dataset
    from features import build_features
    ds = build_dataset()
    feat = build_features(ds["prices"], ds["returns"])
    regimes, km, scaler, sil = run_regime_detection(feat)
    print("Silhouette score:", round(sil, 4))
    print(regimes.value_counts())
