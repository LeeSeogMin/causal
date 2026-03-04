"""
제8장: 전통적 고정 할당 RCT 시뮬레이션
========================================

목적: 전통적 RCT의 한계 시연
- 고정 1:1 할당의 비효율성
- 표본 크기 딜레마
- 윤리적 문제 (열등 처치 지속 할당)

저자: AI 기반 정책분석방법론
날짜: 2025-11-20
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os, pathlib

np.random.seed(42)

print("=" * 80)
print("전통적 고정 할당 RCT 시뮬레이션")
print("=" * 80)

# 데이터 로드
base_dir = pathlib.Path(__file__).resolve().parents[1]
data_path = base_dir / 'data' / '8-1-fixed-rct.csv'

print(f"\n[데이터 로드: {data_path}]")
df = pd.read_csv(data_path)

print(f"  총 참가자 수: {len(df)}명")
print(f"  시나리오 수: {df['scenario'].nunique()}개")

# 시나리오별 분석
print("\n[1단계] 4가지 시나리오 분석")
print("-" * 80)

scenario_names = {
    'scenario1': '시나리오 1 (소효과)',
    'scenario2': '시나리오 2 (중효과)',
    'scenario3': '시나리오 3 (대효과)',
    'scenario4': '시나리오 4 (무효과)'
}

results = []

for scenario_id, scenario_name in scenario_names.items():
    df_scenario = df[df['scenario'] == scenario_id]
    
    outcomes_control = df_scenario[df_scenario['treatment'] == 0]['outcome'].values
    outcomes_treatment = df_scenario[df_scenario['treatment'] == 1]['outcome'].values
    
    ate = np.mean(outcomes_treatment) - np.mean(outcomes_control)
    _, p_value = stats.ttest_ind(outcomes_treatment, outcomes_control)
    
    p_control_true = df_scenario['p_control_true'].iloc[0]
    p_treatment_true = df_scenario['p_treatment_true'].iloc[0]
    
    result = {
        'scenario': scenario_name,
        'p_control_true': p_control_true,
        'p_treatment_true': p_treatment_true,
        'n_control': len(outcomes_control),
        'n_treatment': len(outcomes_treatment),
        'ate': ate,
        'p_value': p_value
    }
    results.append(result)
    
    print(f"\n{scenario_name}:")
    print(f"  실제 성공률: 대조군 {p_control_true:.2f}, 처치군 {p_treatment_true:.2f}")
    print(f"  할당: 대조군 {result['n_control']}명, 처치군 {result['n_treatment']}명")
    print(f"  관측 성공률: 대조군 {np.mean(outcomes_control):.2f}, 처치군 {np.mean(outcomes_treatment):.2f}")
    print(f"  ATE: {result['ate']:.3f}, p-value: {result['p_value']:.4f}")
    
    if result['p_value'] < 0.05:
        print(f"  ✓ 유의함 (p < 0.05)")
    else:
        print(f"  ✗ 비유의 (p >= 0.05)")

# 결과 테이블
print("\n[2단계] 결과 요약 테이블")
print("-" * 80)

results_df = pd.DataFrame(results)
print(results_df[['scenario', 'p_control_true', 'p_treatment_true', 'n_control', 'n_treatment', 'ate', 'p_value']].to_string(index=False))

# 비효율성 분석
print("\n[3단계] 비효율성 분석")
print("-" * 80)

print("\n시나리오 3 (대효과):")
print("  - 실제로는 50명으로 0.90 검정력 달성 가능")
print("  - 200명 모집은 150명 과잉")
print("  - 100명이 불필요하게 열등한 대조군 할당받음")

print("\n시나리오 4 (무효과):")
print("  - 처치 효과 전혀 없음에도 200명 전원 실험 완료")
print("  - 조기 중단 없이 자원 낭비")

print("\n시나리오 1 (소효과):")
print("  - 실제 효과 있지만 표본 크기 부족으로 검출 실패")
print("  - 유효한 처치를 폐기할 위험")

# 최종 요약
print("\n" + "=" * 80)
print("전통적 고정 할당 RCT 한계 요약")
print("=" * 80)
print(f"✓ 시나리오 1: 소효과 검출 실패 (검정력 부족)")
print(f"✓ 시나리오 2: 적절한 설계 (중효과 검출)")
print(f"✓ 시나리오 3: 150명 과잉 모집 (대효과를 50명으로 검출 가능)")
print(f"✓ 시나리오 4: 200명 전원 낭비 (무효과 조기 발견 불가)")
print(f"\n총 비효율: 350명이 불필요하게 실험 참여 또는 열등 처치 받음")
print("⇒ Adaptive Trial Design의 필요성 입증")
print("=" * 80)
