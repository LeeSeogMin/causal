#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제8장: 한국 디지털 치료기기 임상시험 사례 (CSV 기반)
불면증 치료 앱의 5가지 콘텐츠 전략 비교 (Thompson Sampling RAR)
"""

import numpy as np
import pandas as pd
import pathlib

print("=" * 80)
print("한국 디지털 치료기기 임상시험: 불면증 치료 앱")
print("=" * 80)

# 0단계: 데이터 로드
print("\n[0단계] 데이터 로드")
print("-" * 80)

import pathlib
base_dir = pathlib.Path(__file__).resolve().parents[1]
data_path = base_dir / 'data' / '8-5-korean-dtx.csv'
df = pd.read_csv(data_path)
print(f"  데이터 파일: {data_path}")
print(f"  총 참가자 수: {len(df)}명")
print(f"  전략 수: {df['strategy'].nunique()}개")
print()

# 1. 시험 설정 정보 출력
arm_names = df['strategy_name'].unique()
true_rates = df.groupby('strategy')['true_rate'].first().values
print("\n[1단계] 시험 설정")
print("-" * 80)
print(f"총 참가자 수: {len(df)}명")
print(f"콘텐츠 전략 수: {len(arm_names)}개")
print("\n실제 성공률 (불면증 개선율):")
for i, (name, rate) in enumerate(zip(arm_names, true_rates)):
    print(f"  {i+1}. {name}: {rate:.1%}")

# 2. 중간 분석 (각 시점별 할당 비율)
print("\n[2단계] 중간 분석 (할당 비율)")
print("-" * 80)
interim_points = [200, 400, 600, 800]
for n in interim_points:
    df_n = df[df['participant_id'] <= n]
    print(f"\nn = {n}명:")
    print("  할당 비율:")
    for strategy_id, name in enumerate(arm_names):
        cnt = len(df_n[df_n['strategy'] == strategy_id])
        ratio = cnt / n
        print(f"    {name}: {ratio:.1%} ({cnt}명)")

# 3. 최종 결과 요약
print("\n[3단계] 최종 결과 요약")
print("-" * 80)
summary = []
for strategy_id, name in enumerate(arm_names):
    df_s = df[df['strategy'] == strategy_id]
    participants = len(df_s)
    successes = df_s['success'].sum()
    # Beta(1+success, 1+failure) 사후 평균
    posterior = (1 + successes) / (2 + participants)
    summary.append({
        '전략': name,
        '참가자 수': participants,
        '성공자 수': successes,
        '할당 비율': f"{participants/len(df):.1%}",
        '사후 평균': f"{posterior:.3f}",
        '실제 성공률': f"{true_rates[strategy_id]:.3f}"
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# 4. 성과 분석 (고정 할당 대비 개선)
print("\n[4단계] 성과 분석")
print("-" * 80)

total_successes = df['success'].sum()
N = len(df)
expected_fixed = sum(true_rates) / len(true_rates) * N
expected_optimal = max(true_rates) * N
improvement = (total_successes - expected_fixed) / expected_fixed * 100
efficiency = (total_successes - expected_fixed) / (expected_optimal - expected_fixed) * 100
print(f"총 성공자: {total_successes}명")
print(f"고정 할당 (1:1:1:1:1) 예상 성공자: {expected_fixed:.1f}명")
print(f"최적 할당 (모두 최고 전략) 예상 성공자: {expected_optimal:.1f}명")
print(f"고정 할당 대비 개선: +{improvement:.1f}%")
print(f"최적 대비 효율성: {efficiency:.1f}%")

print("\n" + "=" * 80)
print("디지털 치료기기 임상시험 분석 요약")
print("=" * 80)
print(f"✓ 총 성공자: {total_successes}명 (고정 대비 +{improvement:.1f}%)")
print(f"✓ 최고 전략 ({arm_names[np.argmax(true_rates)]}) 할당: {df[df['strategy']==np.argmax(true_rates)]['participant_id'].count()}명 ({df[df['strategy']==np.argmax(true_rates)]['participant_id'].count()/N:.1%})")
print(f"✓ 최저 전략 ({arm_names[np.argmin(true_rates)]}) 할당: {df[df['strategy']==np.argmin(true_rates)]['participant_id'].count()}명 ({df[df['strategy']==np.argmin(true_rates)]['participant_id'].count()/N:.1%})")
print("✓ Thompson Sampling이 효율성과 윤리성을 동시에 개선")
print("=" * 80)
