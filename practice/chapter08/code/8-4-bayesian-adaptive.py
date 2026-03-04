#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제8장: Bayesian Adaptive Design 구현 (CSV 기반)
Beta-Binomial 사후분포를 이용해 조기 중단 판단
"""

import numpy as np
import pandas as pd
import pathlib
from scipy.stats import beta

print("=" * 80)
print("Bayesian Adaptive Design 시뮬레이션 (CSV 데이터)")
print("=" * 80)

# 0단계: 데이터 로드
print("\n[0단계] 데이터 로드")
print("-" * 80)

import pathlib
base_dir = pathlib.Path(__file__).resolve().parents[1]
data_path = base_dir / 'data' / '8-4-bayesian-adaptive.csv'
df = pd.read_csv(data_path)
print(f"  데이터 파일: {data_path}")
print(f"  총 행 수: {len(df)}")
print(f"  시나리오 수: {df['scenario'].nunique()}개")
print()

# 시나리오 정의 (중간 분석 시점)
interim_points = [50, 100, 150, 200]

results_summary = []

for scenario_id in df['scenario'].unique():
    print(f"\n[시나리오 {scenario_id}] 분석 시작")
    print("-" * 80)
    df_s = df[df['scenario'] == scenario_id]
    p_control_true = df_s['p_control_true'].iloc[0]
    p_treatment_true = df_s['p_treatment_true'].iloc[0]
    print(f"  실제 성공률: 대조군 {p_control_true:.2f}, 처치군 {p_treatment_true:.2f}")

    stopped_early = False
    stop_reason = None
    final_n = None
    final_post_prob = None
    interim_results = []

    for n in interim_points:
        df_n = df_s[df_s['participant_id'] <= n]
        # 대조군 / 처치군 성공/실패 카운트
        ctrl = df_n[df_n['treatment'] == 0]
        treat = df_n[df_n['treatment'] == 1]
        successes_c = ctrl['outcome'].sum()
        failures_c = len(ctrl) - successes_c
        successes_t = treat['outcome'].sum()
        failures_t = len(treat) - successes_t
        # 사후분포 파라미터 (Beta(1+success, 1+failure))
        alpha_c, beta_c = 1 + successes_c, 1 + failures_c
        alpha_t, beta_t = 1 + successes_t, 1 + failures_t
        # 사후 확률 P(p_t > p_c) via Monte Carlo
        samples = 10000
        p1 = np.random.beta(alpha_t, beta_t, samples)
        p2 = np.random.beta(alpha_c, beta_c, samples)
        post_prob = np.mean(p1 > p2)
        decision = "continue"
        if post_prob >= 0.99:
            stopped_early = True
            stop_reason = "efficacy"
            decision = "stop_efficacy"
        elif post_prob <= 0.05:
            stopped_early = True
            stop_reason = "futility"
            decision = "stop_futility"
        interim_results.append({"n": n, "post_prob": post_prob, "decision": decision})
        print(f"  n={n:3d}: P(p_t > p_c) = {post_prob:.4f}, 결정: {decision}")
        if stopped_early:
            break
    # 최종값 설정
    final_n = interim_results[-1]["n"]
    final_post_prob = interim_results[-1]["post_prob"]
    results_summary.append({
        "scenario": scenario_id,
        "final_n": final_n,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "final_post_prob": final_post_prob
    })

# 비교 표 출력
print("\n[4단계] Bayesian vs Frequentist GSD 비교")
print("-" * 80)
import pandas as pd
comp_df = pd.DataFrame(results_summary)
print(comp_df.to_string(index=False))

print("\n" + "=" * 80)
print("Bayesian Adaptive Design 분석 요약")
print("=" * 80)
for r in results_summary:
    print(f"시나리오 {r['scenario']}: 최종 표본 크기 {r['final_n']}, 조기 중단 {r['stopped_early']}, 이유 {r['stop_reason']}, 사후확률 {r['final_post_prob']:.4f}")
print("=" * 80)
