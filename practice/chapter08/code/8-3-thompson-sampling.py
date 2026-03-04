#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제8장: Thompson Sampling (CSV 기반)
3-armed bandit 시뮬레이션 결과를 CSV에서 읽어 분석
"""

import numpy as np
import pandas as pd

print("=" * 80)
print("Thompson Sampling 결과 분석 (CSV)")
print("=" * 80)

# 0단계: 데이터 로드
print("\n[0단계] 데이터 로드")
print("-" * 80)

import pathlib
base_dir = pathlib.Path(__file__).resolve().parents[1]
data_path = base_dir / 'data' / '8-3-thompson-sampling.csv'
df = pd.read_csv(data_path)
print(f"  데이터 파일: {data_path}")
print(f"  총 시행 횟수: {len(df)}")
print(f"  Arm 수: {df['arm_selected'].nunique()}")
print()

# 1. 기본 설정 확인
print("\n[1단계] 설정 확인")
print("-" * 80)
true_probs = df.groupby('arm_selected')['true_prob'].first().values
n_trials = len(df)
for i, prob in enumerate(true_probs):
    optimal = " (최적)" if prob == max(true_probs) else ""
    print(f"Arm {i} 실제 성공 확률: {prob:.2f}{optimal}")
print(f"총 시행 횟수: {n_trials}")

# 2. Arm 선택 횟수 및 보상
print("\n[2단계] Arm 선택 횟수 및 보상")
print("-" * 80)
arm_counts = df['arm_selected'].value_counts().sort_index()
total_reward = df['reward'].sum()
print("Arm 선택 횟수:")
for arm, cnt in arm_counts.items():
    print(f"  Arm {arm}: {cnt} ({cnt/n_trials*100:.1f}%)")
print(f"\n총 보상: {total_reward}/{n_trials} = {total_reward/n_trials:.3f}")
print(f"최적 전략 보상 (항상 최적 arm 선택): {int(max(true_probs) * n_trials)} (Arm {np.argmax(true_probs)})")
print(f"Regret: {max(true_probs) * n_trials - total_reward:.1f}")

# 3. 사후 평균 계산 (최종)
print("\n[3단계] 사후 평균 (최종)")
print("-" * 80)
for arm in range(len(true_probs)):
    df_arm = df[df['arm_selected'] == arm]
    if len(df_arm) > 0:
        successes = df_arm['reward'].sum()
        posterior_mean = (1 + successes) / (2 + len(df_arm))
        diff = posterior_mean - true_probs[arm]
        print(f"  Arm {arm}: {posterior_mean:.3f} (실제: {true_probs[arm]:.3f}, 차이: {diff:+.3f})")
    else:
        print(f"  Arm {arm}: N/A (선택되지 않음)")

# 4. 알고리즘 비교 (Uniform, Epsilon-Greedy)
print("\n[4단계] 알고리즘 비교")
print("-" * 80)
# Uniform (동일 선택)
uniform_reward = sum(true_probs) / len(true_probs) * n_trials
uniform_regret = max(true_probs) * n_trials - uniform_reward
# Epsilon-Greedy 시뮬레이션 (epsilon=0.10)
np.random.seed(42)
epsilon = 0.10
arm_rewards = np.zeros(len(true_probs))
arm_counts_eg = np.zeros(len(true_probs))
for _ in range(n_trials):
    if np.random.rand() < epsilon:
        arm = np.random.randint(len(true_probs))
    else:
        arm = np.argmax(arm_rewards / (arm_counts_eg + 1e-5))
    reward = np.random.binomial(1, true_probs[arm])
    arm_rewards[arm] += reward
    arm_counts_eg[arm] += 1
epsilon_greedy_reward = arm_rewards.sum()
epsilon_greedy_regret = max(true_probs) * n_trials - epsilon_greedy_reward

# 결과 테이블
import pandas as pd
comp = pd.DataFrame({
    'Algorithm': ['Uniform', 'Epsilon-Greedy', 'Thompson Sampling', 'Optimal'],
    'Total Reward': [uniform_reward, epsilon_greedy_reward, total_reward, max(true_probs) * n_trials],
    'Regret': [uniform_regret, epsilon_greedy_regret, max(true_probs) * n_trials - total_reward, 0]
})
print(comp.to_string(index=False))

print("\n" + "=" * 80)
print("Thompson Sampling 분석 요약")
print("=" * 80)
print(f"최적 arm (Arm {np.argmax(true_probs)}) 선택 비율: {arm_counts[np.argmax(true_probs)]/n_trials:.1%}")
print(f"총 보상: {total_reward} (최적 대비 {total_reward/(max(true_probs)*n_trials):.1%})")
print(f"Regret: {max(true_probs) * n_trials - total_reward:.1f}")
print("=" * 80)
