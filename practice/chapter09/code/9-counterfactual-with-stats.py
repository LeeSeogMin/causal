"""
제9장: Interrupted Time Series 반사실 예측 + 통계적 검정(부트스트랩 CI, 단측 t-검정)
===================================================================================

목적:
- 정책 개입 시점(intervention)을 기준으로 반사실(counterfactual) 시나리오를 구성하고
  개입 효과(policy effect)를 추정한다.
- 부트스트랩(재표집)으로 평균 효과의 95% 신뢰구간을 계산한다.
- 단측 일표본 t-검정(H1: 효과 > 0)으로 정책 효과의 유의성을 평가한다.

데이터:
- 기본적으로 `practice/chapter09/data/9-1-preprocessed-data.csv`가 있으면 이를 사용한다.
- 파일이 없으면 재현 가능한(시드 고정) 합성 시계열을 생성하여 예제를 수행한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_PATH = Path("practice/chapter09/data/9-1-preprocessed-data.csv")


@dataclass(frozen=True)
class CounterfactualResult:
    with_policy: np.ndarray
    without_policy: np.ndarray
    policy_effect: np.ndarray
    average_effect: float
    ci_95: Tuple[float, float]
    t_statistic: float
    p_value: float
    is_significant: bool


def _load_or_simulate_series() -> np.ndarray:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        if "demand" not in df.columns:
            raise ValueError(f"Expected column 'demand' in {DATA_PATH.as_posix()}")
        series = df["demand"].to_numpy(dtype=float)
        series = series[~np.isnan(series)]
        if len(series) < 200:
            raise ValueError(f"Not enough observations in {DATA_PATH.as_posix()}: {len(series)}")
        return series

    rng = np.random.default_rng(42)
    n = 24 * 365  # 1년(시간 단위) 예제
    t = np.arange(n)
    base = 60000.0
    seasonal = 4000.0 * np.sin(2 * np.pi * t / (24 * 365))
    daily = 2500.0 * np.sin(2 * np.pi * (t % 24) / 24)
    noise = rng.normal(0.0, 800.0, size=n)
    series = base + seasonal + daily + noise
    return series


def _fit_pretrend_counterfactual(series: np.ndarray, intervention_idx: int) -> np.ndarray:
    """
    개입 전 구간으로 단순 선형 추세를 학습한 뒤, 개입 후에도 동일한 추세가 유지된다는
    반사실 시나리오를 구성한다(교육용 단순화).
    """
    y = series.astype(float)
    x = np.arange(len(y), dtype=float)
    x_pre = x[:intervention_idx]
    y_pre = y[:intervention_idx]

    x_mean = float(np.mean(x_pre))
    y_mean = float(np.mean(y_pre))
    denom = float(np.sum((x_pre - x_mean) ** 2))
    if denom == 0.0:
        slope = 0.0
    else:
        slope = float(np.sum((x_pre - x_mean) * (y_pre - y_mean)) / denom)
    intercept = y_mean - slope * x_mean

    return intercept + slope * x


def _one_sided_one_sample_ttest_greater(sample: np.ndarray) -> Tuple[float, float]:
    """
    H0: mean(sample) <= 0, H1: mean(sample) > 0
    SciPy가 있으면 정확한 p-value, 없으면 정규근사(표본이 큰 경우)로 계산한다.
    """
    x = np.asarray(sample, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n < 2:
        return float("nan"), float("nan")

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd == 0.0:
        t_stat = float("inf") if mean > 0 else float("-inf")
        p_value = 0.0 if mean > 0 else 1.0
        return t_stat, p_value

    t_stat = mean / (sd / math.sqrt(n))

    try:
        from scipy import stats  # type: ignore

        p_value = float(stats.t.sf(t_stat, df=n - 1))
        return t_stat, p_value
    except Exception:
        # 정규근사: p = 1 - Phi(t)
        p_value = 0.5 * math.erfc(t_stat / math.sqrt(2))
        return t_stat, p_value


def run_counterfactual_with_stats(
    intervention_ratio: float = 0.6,
    evaluation_horizon: int = 24 * 30,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> CounterfactualResult:
    series = _load_or_simulate_series()
    n = len(series)
    intervention_idx = int(n * intervention_ratio)
    intervention_idx = max(50, min(intervention_idx, n - evaluation_horizon - 1))

    without_policy = _fit_pretrend_counterfactual(series, intervention_idx)

    with_policy = series.copy()
    effect_size = 600.0
    with_policy[intervention_idx:] = with_policy[intervention_idx:] + effect_size

    start = intervention_idx
    end = intervention_idx + evaluation_horizon
    policy_effect = with_policy[start:end] - without_policy[start:end]
    avg_effect = float(np.mean(policy_effect))

    rng = np.random.default_rng(seed)
    bootstrap_effects = []
    for _ in range(int(bootstrap_n)):
        indices = rng.integers(0, policy_effect.size, size=policy_effect.size)
        bootstrap_effects.append(float(np.mean(policy_effect[indices])))

    ci_lower = float(np.percentile(bootstrap_effects, 2.5))
    ci_upper = float(np.percentile(bootstrap_effects, 97.5))

    t_stat, p_value = _one_sided_one_sample_ttest_greater(policy_effect)

    return CounterfactualResult(
        with_policy=with_policy[start:end],
        without_policy=without_policy[start:end],
        policy_effect=policy_effect,
        average_effect=avg_effect,
        ci_95=(ci_lower, ci_upper),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        is_significant=bool(p_value < 0.05) if not math.isnan(p_value) else False,
    )


def main() -> None:
    result = run_counterfactual_with_stats()
    print("=" * 80)
    print("제9장: Counterfactual + Bootstrap CI + One-sided t-test")
    print("=" * 80)
    print(f"- 평균 정책 효과: {result.average_effect:,.2f}")
    print(f"- 95% CI (bootstrap): [{result.ci_95[0]:,.2f}, {result.ci_95[1]:,.2f}]")
    print(f"- t-statistic: {result.t_statistic:,.3f}")
    print(f"- p-value (one-sided, H1: effect > 0): {result.p_value:,.6f}")
    print(f"- 유의 여부(p<0.05): {result.is_significant}")


if __name__ == "__main__":
    main()

