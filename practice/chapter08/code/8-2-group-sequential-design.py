#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제8장: Group Sequential Design (그룹 순차 설계)
O'Brien-Fleming 및 Pocock 경계 계산 및 시뮬레이션
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import pathlib

print("=" * 80)
print("Group Sequential Design 시뮬레이션")
print("=" * 80)

# 시뮬레이션 설정
print("\n[0단계] 시뮬레이션 설정")
print("-" * 80)
np.random.seed(42)
K = 3  # 중간 분석 횟수
n_max = 200
print(f"  중간 분석 횟수: {K}")
print(f"  최대 표본 크기: {n_max}")
print()

# 1. 경계 계산 함수

def obrien_fleming_boundary(K, alpha=0.05):
    info_fractions = np.arange(1, K + 1) / K
    c = norm.ppf(1 - alpha / 2)
    boundaries = c / np.sqrt(info_fractions)
    p_bounds = 2 * (1 - norm.cdf(boundaries))
    return boundaries, p_bounds

def pocock_boundary(K, alpha=0.05):
    # Pocock 은 일정한 z값 사용
    z = norm.ppf(1 - alpha / (2 * K))
    boundaries = np.full(K, z)
    p_bounds = np.full(K, alpha / K)
    return boundaries, p_bounds

K = 3
print("[1단계] O'Brien-Fleming 경계")
print("-" * 80)
ob_boundaries, ob_p = obrien_fleming_boundary(K)
for i, (z, p) in enumerate(zip(ob_boundaries, ob_p), 1):
    print(f"  Stage {i}: z = {z:.3f}, p = {p:.4f}")

print("\n[2단계] Pocock 경계")
print("-" * 80)
pc_boundaries, pc_p = pocock_boundary(K)
for i, (z, p) in enumerate(zip(pc_boundaries, pc_p), 1):
    print(f"  Stage {i}: z = {z:.3f}, p = {p:.4f}")

# 2. 시뮬레이션: Null 가정 (p_control = p_treatment = 0.40)
print("\n[3단계] Null 가정 시뮬레이션 (조기 중단 확률)")
print("-" * 80)
np.random.seed(42)
N_SIM = 5000
early_stop_ob = 0
early_stop_pc = 0

for _ in range(N_SIM):
    # 1:1 할당, 총 n_max = 200
    outcomes_control = np.random.binomial(1, 0.40, 100)
    outcomes_treat = np.random.binomial(1, 0.40, 100)
    # 누적 z 통계 계산 (간단히 차이 비율)
    for k, n in enumerate([66, 133, 200]):  # 대략적인 interim n
        diff = outcomes_treat[:n//2].mean() - outcomes_control[:n//2].mean()
        se = np.sqrt(0.40 * 0.60 / (n//2) * 2)
        z = diff / se
        if abs(z) >= ob_boundaries[k]:
            early_stop_ob += 1
            break
        if abs(z) >= pc_boundaries[k]:
            early_stop_pc += 1
            break

print(f"  O'Brien-Fleming 조기 중단 확률 (Null): {early_stop_ob / N_SIM:.4f}")
print(f"  Pocock 조기 중단 확률 (Null): {early_stop_pc / N_SIM:.4f}")

print("\n[4단계] 결과 요약 테이블")
print("-" * 80)
summary = pd.DataFrame({
    'Stage': [1, 2, 3],
    'OB_z': np.round(ob_boundaries, 3),
    'OB_p': np.round(ob_p, 4),
    'PC_z': np.round(pc_boundaries, 3),
    'PC_p': np.round(pc_p, 4)
})
print(summary.to_string(index=False))
