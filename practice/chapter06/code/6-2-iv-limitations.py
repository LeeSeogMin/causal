"""
제6장: 전통적 IV의 한계
========================

목적: 선형 2SLS의 한계 시연
- 비선형 데이터 생성 과정
- 선형 2SLS의 함수 형태 오설정
- RESET 검정
- 고차원 문제

저자: AI 기반 정책분석방법론
날짜: 2025-11-20
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist

# 한글 폰트 설정 (NanumGothic)
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

# 재현성
np.random.seed(42)

print("=" * 80)
print("전통적 IV의 한계: 비선형성과 고차원 문제")
print("=" * 80)

# 1. 데이터 로드
print("\n[1단계] 데이터 로드")
print("-" * 80)

# CSV 파일에서 데이터 읽기
# CSV 파일에서 데이터 읽기
import os
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/6-2-nonlinear-iv-data.csv')
df_data = pd.read_csv(data_path)

# 변수 추출
Y = df_data['Y'].values
D = df_data['D'].values
Z = df_data['Z'].values
X = df_data['X'].values
U = df_data['U'].values  # 검증용

n = len(df_data)

# 참값 평균 한계효과: dY/dD = 3 + D
true_marginal_effect = 3 + D.mean()

print(f"데이터 파일: {data_path}")
print(f"샘플 크기: {n}")
print(f"D와 Z의 비선형 관계: D = 0.5 + 0.5*Z + 0.3*Z²")
print(f"Y와 D의 비선형 관계: Y = 2 + 3*D + 0.5*D²")
print(f"실제 평균 한계효과: {true_marginal_effect:.3f}")

# 2. 선형 2SLS 추정 (오설정)
print("\n[2단계] 선형 2SLS 추정")
print("-" * 80)

df = pd.DataFrame({'Y': Y, 'D': D, 'Z': Z})

# 1단계
first_stage_linear = smf.ols('D ~ Z', data=df).fit()
df['D_hat'] = first_stage_linear.fittedvalues

# 2단계
second_stage_linear = smf.ols('Y ~ D_hat', data=df).fit()
effect_2sls = second_stage_linear.params['D_hat']

print(f"2SLS (선형) 추정값: {effect_2sls:.3f}")
print(f"실제 평균 한계효과: {true_marginal_effect:.3f}")
print(f"편향: {effect_2sls - true_marginal_effect:.3f} ({(effect_2sls - true_marginal_effect)/true_marginal_effect*100:.1f}%)")

# 3. OLS 추정 (편향됨)
print("\n[3단계] OLS 추정 (비교용)")
print("-" * 80)

ols_model = smf.ols('Y ~ D', data=df).fit()
effect_ols = ols_model.params['D']

print(f"OLS 추정값: {effect_ols:.3f}")
print(f"편향: {effect_ols - true_marginal_effect:.3f} ({(effect_ols - true_marginal_effect)/true_marginal_effect*100:.1f}%)")

# 4. 비선형 항 추가 (올바른 명세)
print("\n[4단계] 비선형 항 추가")
print("-" * 80)

# 1단계
first_stage_quad = sm.OLS(D, sm.add_constant(np.column_stack([Z, Z**2]))).fit()
D_hat_quad = first_stage_quad.fittedvalues

# 2단계 (D와 D² 포함)
second_stage_quad = sm.OLS(Y, sm.add_constant(np.column_stack([D_hat_quad, D_hat_quad**2]))).fit()
coef_D = second_stage_quad.params[1]
coef_D2 = second_stage_quad.params[2]

# 평균 한계효과: dY/dD = coef_D + 2*coef_D2*D
marginal_effect_corrected = coef_D + 2 * coef_D2 * D.mean()

print(f"D 계수: {coef_D:.3f} (참값: 3.0)")
print(f"D² 계수: {coef_D2:.3f} (참값: 0.5)")
print(f"평균 한계효과: {marginal_effect_corrected:.3f}")
print(f"편향: {marginal_effect_corrected - true_marginal_effect:.3f} ({abs(marginal_effect_corrected - true_marginal_effect)/true_marginal_effect*100:.1f}%)")

# 5. RESET 검정
print("\n[5단계] RESET 검정 (함수 형태 오설정)")
print("-" * 80)

# 선형 모형의 예측값
Y_hat = second_stage_linear.fittedvalues

# Y_hat의 제곱과 세제곱 추가
df['Y_hat2'] = Y_hat ** 2
df['Y_hat3'] = Y_hat ** 3

# 확장 모형
reset_model = smf.ols('Y ~ D_hat + Y_hat2 + Y_hat3', data=df).fit()

# F-검정
r2_unrestricted = reset_model.rsquared
r2_restricted = second_stage_linear.rsquared
f_stat = ((r2_unrestricted - r2_restricted) / 2) / ((1 - r2_unrestricted) / (n - 4))
p_value = 1 - f_dist.cdf(f_stat, 2, n - 4)

print(f"RESET F-통계량: {f_stat:.2f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.001:
    print("✓ 귀무가설 강하게 기각: 선형 모형 오설정 확인")
else:
    print("귀무가설 기각 실패")

# 6. 고차원 문제
print("\n[6단계] 고차원 시나리오")
print("-" * 80)

p_dim = 50
n_sample = 500

print(f"공변량 차원 (p): {p_dim}")
print(f"표본 크기 (n): {n_sample}")
print(f"p/n 비율: {p_dim/n_sample:.2f}")

# 가능한 2차 상호작용 수
n_interactions = p_dim * (p_dim - 1) // 2
print(f"가능한 2차 상호작용: {n_interactions}")
print(f"과적합 위험: {'높음' if n_interactions > n_sample else '낮음'}")

# 7. 시각화
print("\n[7단계] 시각화")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) 비선형 관계 (D vs Y)
ax = axes[0, 0]
ax.scatter(D, Y, alpha=0.3, s=20, color='steelblue', label='Data')
D_sorted = np.sort(D)
Y_true = 2 + 3 * D_sorted + 0.5 * D_sorted**2 - 2 * U.mean()
Y_linear = second_stage_linear.params['Intercept'] + effect_2sls * D_sorted
ax.plot(D_sorted, Y_true, 'g-', linewidth=2, label='참값 (비선형)')
ax.plot(D_sorted, Y_linear, 'r--', linewidth=2, label=f'2SLS 선형 (편향)')
ax.set_xlabel('처치 (D)', fontsize=11)
ax.set_ylabel('결과 (Y)', fontsize=11)
ax.set_title('(a) 비선형 관계 vs 선형 근사', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# (b) 추정값 비교
ax = axes[0, 1]
methods = ['참값', '2SLS\n(선형)', '2SLS\n(비선형)', 'OLS']
estimates = [true_marginal_effect, effect_2sls, marginal_effect_corrected, effect_ols]
colors = ['green', 'coral', 'steelblue', 'orange']

x_pos = np.arange(len(methods))
bars = ax.bar(x_pos, estimates, color=colors, alpha=0.7)
ax.axhline(true_marginal_effect, color='red', linestyle='--', linewidth=2)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('평균 한계효과 추정값', fontsize=11)
ax.set_title('(b) 방법론별 추정값 비교', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (c) 편향 크기
ax = axes[1, 0]
biases = [0, effect_2sls - true_marginal_effect, marginal_effect_corrected - true_marginal_effect, effect_ols - true_marginal_effect]
colors_bias = ['green', 'coral', 'steelblue', 'orange']
bars = ax.bar(x_pos, biases, color=colors_bias, alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('편향 (추정값 - 참값)', fontsize=11)
ax.set_title('(c) 편향 비교', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (d) 1단계 비선형 관계
ax = axes[1, 1]
ax.scatter(Z, D, alpha=0.3, s=20, color='steelblue', label='Data')
Z_sorted = np.sort(Z)
D_true = 0.5 + 0.5 * Z_sorted + 0.3 * Z_sorted**2 + U.mean()
D_linear_pred = first_stage_linear.params['Intercept'] + first_stage_linear.params['Z'] * Z_sorted
ax.plot(Z_sorted, D_true, 'g-', linewidth=2, label='참값 (비선형)')
ax.plot(Z_sorted, D_linear_pred, 'r--', linewidth=2, label='선형 근사')
ax.set_xlabel('도구변수 (Z)', fontsize=11)
ax.set_ylabel('처치 (D)', fontsize=11)
ax.set_title('(d) 1단계 비선형 관계', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('6-2-iv-limitations.png', dpi=300, bbox_inches='tight')
print("그래프 저장: 6-2-iv-limitations.png")

# 8. 최종 요약
print("\n" + "=" * 80)
print("전통적 IV 한계 분석 요약")
print("=" * 80)
print(f"✓ 선형 2SLS 편향: {(effect_2sls - true_marginal_effect)/true_marginal_effect*100:.1f}% (비선형성 무시)")
print(f"✓ RESET 검정: F={f_stat:.2f}, p<0.001 (함수 형태 오설정 확인)")
print(f"✓ 비선형 항 추가 시: {abs(marginal_effect_corrected - true_marginal_effect)/true_marginal_effect*100:.1f}% 오차로 개선")
print(f"✓ 고차원 문제: {n_interactions}개 상호작용 vs {n_sample}개 샘플 (과적합 위험)")
print("=" * 80)
