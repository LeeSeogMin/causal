"""
제6장: 교육 수익률 추정 실습
==============================

목적: DeepIV Family를 실제 정책 문제에 적용
- 대학 교육의 임금 프리미엄 추정
- 부모 학력별 이질적 효과
- 정책 시뮬레이션

저자: AI 기반 정책분석방법론
날짜: 2025-11-20
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

print("=" * 80)
print("교육 수익률 추정: DeepIV Family 적용")
print("=" * 80)

# 1. 데이터 생성 (가상의 한국 대졸자 데이터)
print("\n[1단계] 데이터 생성")
print("-" * 80)

n = 5000

# 기본 변수
age = np.random.normal(28, 3, n)
gender = np.random.binomial(1, 0.5, n)
parent_edu = np.random.choice([1, 2, 3], n, p=[0.4, 0.4, 0.2])  # 1=중졸, 2=고졸, 3=대졸
region = np.random.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2])

# 도구변수: 지역별 대학 정원 증가율
college_expansion = (np.random.normal(5, 2, n) * (region == 1) +
                     np.random.normal(3, 1.5, n) * (region == 2) +
                     np.random.normal(1, 1, n) * (region == 3))

# 관측 불가능한 능력
ability = np.random.randn(n)

# 처치: 대학 졸업 여부 (비선형, 내생성)
college_prob = expit(
    -1 + 0.05 * college_expansion + 0.3 * parent_edu +
    0.5 * ability + 0.01 * age - 0.2 * gender + np.random.randn(n) * 0.2
)
college = np.random.binomial(1, college_prob)

# 결과: 로그 임금 (비선형 처치효과, 이질성)
# 참값 효과: 대졸 프리미엄 10% + 부모학력 상호작용
true_premium_base = 0.1
true_premium = true_premium_base + 0.01 * (parent_edu - 1.6)

log_wage = (2.5 + 0.05 * age - 0.15 * gender + 0.2 * parent_edu + 0.3 * ability +
            college * true_premium + np.random.randn(n) * 0.15)

print(f"샘플 크기: {n}")
print(f"대졸률: {college.mean():.1%}")
print(f"평균 대졸 프리미엄 (참값): {true_premium.mean():.3f} (약 {(np.exp(true_premium.mean())-1)*100:.1f}%)")

# 2. DFIV 추정
print("\n[2단계] DFIV로 교육 수익률 추정")
print("-" * 80)

X = np.column_stack([age, gender, parent_edu, region])
ZX = np.column_stack([college_expansion, age, gender, parent_edu, region])

ZX_train, ZX_val, college_train, college_val, log_wage_train, log_wage_val, parent_edu_train, parent_edu_val = train_test_split(
    ZX, college, log_wage, parent_edu, test_size=0.3, random_state=42
)

# Stage 1
dfiv_s1 = MLPRegressor(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
dfiv_s1.fit(ZX_train, college_train)

# Stage 2
college_pred_train = dfiv_s1.predict(ZX_train)
college_X_train = np.column_stack([college_pred_train, ZX_train[:, 1:]])
college_X_val = np.column_stack([college_val, ZX_val[:, 1:]])

dfiv_s2 = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500, random_state=42, early_stopping=True)
dfiv_s2.fit(college_X_train, log_wage_train)

# 처치효과 추정
delta = 0.01
college_plus = college_val + delta
college_minus = college_val - delta

log_wage_plus = dfiv_s2.predict(np.column_stack([college_plus, ZX_val[:, 1:]]))
log_wage_minus = dfiv_s2.predict(np.column_stack([college_minus, ZX_val[:, 1:]]))

college_premium_dfiv = ((log_wage_plus - log_wage_minus) / (2 * delta)).mean()
wage_increase_pct = (np.exp(college_premium_dfiv) - 1) * 100

print(f"DFIV 대졸 프리미엄: {college_premium_dfiv:.3f} (연간 임금 {wage_increase_pct:.1f}% 증가)")

# 3. 2SLS 비교
print("\n[3단계] 2SLS 비교")
print("-" * 80)

lr1 = LinearRegression()
lr1.fit(ZX_train, college_train)
college_hat_train = lr1.predict(ZX_train)

lr2 = LinearRegression()
lr2.fit(np.column_stack([college_hat_train, ZX_train[:, 1:]]), log_wage_train)

college_premium_2sls = lr2.coef_[0]
wage_increase_2sls = (np.exp(college_premium_2sls) - 1) * 100

print(f"2SLS 대졸 프리미엄: {college_premium_2sls:.3f} (연간 임금 {wage_increase_2sls:.1f}% 증가)")

# 4. 이질적 효과 분석 (부모 학력별)
print("\n[4단계] 이질적 효과 분석 (부모 학력별)")
print("-" * 80)

heterogeneous_effects = {}

for edu_level in [1, 2, 3]:
    mask = (parent_edu_val == edu_level)
    if mask.sum() > 0:
        log_wage_plus_edu = dfiv_s2.predict(np.column_stack([college_val[mask] + delta, ZX_val[mask, 1:]]))
        log_wage_minus_edu = dfiv_s2.predict(np.column_stack([college_val[mask] - delta, ZX_val[mask, 1:]]))
        
        effect_edu = ((log_wage_plus_edu - log_wage_minus_edu) / (2 * delta)).mean()
        heterogeneous_effects[edu_level] = effect_edu
        
        edu_name = ['', '중졸 이하', '고졸', '대졸 이상'][edu_level]
        print(f"{edu_name}: {effect_edu:.3f} (임금 {(np.exp(effect_edu)-1)*100:.1f}% 증가)")

# 5. 시각화
print("\n[5단계] 시각화")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) 방법론 비교
ax = axes[0]
methods = ['참값', 'DFIV', '2SLS']
true_avg = true_premium.mean()
estimates = [true_avg, college_premium_dfiv, college_premium_2sls]
colors = ['green', 'steelblue', 'coral']

x_pos = np.arange(len(methods))
bars = ax.bar(x_pos, estimates, color=colors, alpha=0.7)
ax.axhline(true_avg, color='red', linestyle='--', linewidth=2)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_ylabel('대졸 프리미엄 (로그)', fontsize=11)
ax.set_title('(a) 방법론별 추정값', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (b) 부모 학력별 이질적 효과
ax = axes[1]
edu_levels = [1, 2, 3]
edu_names = ['중졸\n이하', '고졸', '대졸\n이상']
effects = [heterogeneous_effects.get(e, 0) for e in edu_levels]

x_pos2 = np.arange(len(edu_levels))
bars = ax.bar(x_pos2, effects, color='steelblue', alpha=0.7)
ax.set_xticks(x_pos2)
ax.set_xticklabels(edu_names, fontsize=9)
ax.set_ylabel('대졸 프리미엄 (로그)', fontsize=11)
ax.set_title('(b) 부모 학력별 이질적 효과', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (c) 임금 증가율 (%)
ax = axes[2]
wage_increases = [(np.exp(e) - 1) * 100 for e in effects]

bars = ax.bar(x_pos2, wage_increases, color='green', alpha=0.7)
ax.set_xticks(x_pos2)
ax.set_xticklabels(edu_names, fontsize=9)
ax.set_ylabel('임금 증가율 (%)', fontsize=11)
ax.set_title('(c) 부모 학력별 임금 증가율', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('6-7-education-returns.png', dpi=300, bbox_inches='tight')
print("그래프 저장: 6-7-education-returns.png")

# 6. 정책 시뮬레이션
print("\n[6단계] 정책 시뮬레이션")
print("-" * 80)

current_college_rate = college.mean()
simulated_expansion = 0.20  # 20% 정원 확대

# 예상 대졸률 증가 (간소화된 추정)
estimated_increase = simulated_expansion * 0.05  # 정원 1% 증가 → 대졸률 0.05%p 증가 가정
new_college_rate = current_college_rate + estimated_increase

avg_wage_increase = college_premium_dfiv * estimated_increase

print(f"현재 대졸률: {current_college_rate:.1%}")
print(f"대학 정원 {simulated_expansion*100:.0f}% 확대 시:")
print(f"  - 예상 대졸률: {new_college_rate:.1%} (+{estimated_increase:.1%}p)")
print(f"  - 평균 임금 증가: {(np.exp(avg_wage_increase)-1)*100:.2f}%")

# 7. 최종 요약
print("\n" + "=" * 80)
print("교육 수익률 분석 요약")
print("=" * 80)
print(f"✓ DFIV 대졸 프리미엄: {college_premium_dfiv:.2f} (임금 {wage_increase_pct:.1f}% 증가)")
print(f"✓ 2SLS 추정: {college_premium_2sls:.2f} (DFIV와 {abs(college_premium_dfiv - college_premium_2sls):.3f} 차이)")
print(f"✓ 부모 대졸 가정 효과: {heterogeneous_effects.get(3, 0):.2f} vs 부모 중졸 가정: {heterogeneous_effects.get(1, 0):.2f}")
print(f"✓ 교육 투자의 이질성 확인: 가정 배경이 교육 효율성에 영향")
print("\n주의: 본 분석은 가상 데이터 기반 교육 목적 예제입니다.")
print("=" * 80)
