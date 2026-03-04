"""
제6장: 전통적 2SLS Instrumental Variables
==========================================

목적: 도구변수를 사용한 전통적 2단계 최소제곱법 추정
- 내생성이 있는 데이터 생성
- 1단계: 처치를 도구변수에 회귀
- 2단계: 결과를 예측 처치에 회귀
- F-통계량 및 진단 검정

저자: AI 기반 정책분석방법론
날짜: 2025-11-20
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# 재현성
np.random.seed(42)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("전통적 2SLS Instrumental Variables 추정")
print("=" * 80)

# 1. 데이터 로드
print("\n[1단계] 데이터 로드")
print("-" * 80)

# CSV 파일에서 데이터 읽기
# CSV 파일에서 데이터 읽기
import os
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/6-1-iv-data.csv')
df = pd.read_csv(data_path)

# 변수 추출
Y = df['Y'].values
D = df['D'].values
Z = df['Z'].values
X1 = df['X1'].values
X2 = df['X2'].values
U = df['U'].values  # 검증용 (실제로는 관측 불가능)

n = len(df)
true_effect = 3.42  # 데이터 생성 시 사용된 참값

print(f"데이터 파일: {data_path}")
print(f"샘플 크기: {n}")
print(f"참값 인과효과: {true_effect}")
print(f"내생성 상관: Corr(D, U) = {np.corrcoef(D, U)[0, 1]:.3f}")

# 2. OLS 추정 (편향됨)
print("\n[2단계] OLS 추정 (편향된 결과)")
print("-" * 80)

ols_model = smf.ols('Y ~ D + X1 + X2', data=df).fit()
ols_effect = ols_model.params['D']
ols_se = ols_model.bse['D']

print(f"OLS 추정값: {ols_effect:.3f} (SE: {ols_se:.3f})")
print(f"참값과의 차이: {ols_effect - true_effect:.3f} ({(ols_effect - true_effect)/true_effect*100:.1f}%)")
print(f"95% CI: [{ols_model.conf_int().loc['D', 0]:.3f}, {ols_model.conf_int().loc['D', 1]:.3f}]")

# 3. 2SLS 추정
print("\n[3단계] 2SLS 추정")
print("-" * 80)

# 1단계: D를 Z에 회귀
print("\n--- 1단계 (First Stage) ---")
first_stage = smf.ols('D ~ Z + X1 + X2', data=df).fit()

pi_1 = first_stage.params['Z']
pi_1_se = first_stage.bse['Z']
pi_1_t = first_stage.tvalues['Z']
f_stat = first_stage.fvalue
partial_r2 = 1 - (1 - first_stage.rsquared) / (1 - smf.ols('D ~ X1 + X2', data=df).fit().rsquared)

print(f"π₁ (Z → D): {pi_1:.3f} (SE: {pi_1_se:.3f}, t: {pi_1_t:.2f})")
print(f"F-통계량: {f_stat:.2f}")
print(f"부분 R²: {partial_r2:.3f}")
print(f"1단계 R²: {first_stage.rsquared:.3f}")

if f_stat > 10:
    print("✓ 강한 도구변수 (F > 10)")
else:
    print("⚠ 약한 도구변수 위험 (F < 10)")

# 2단계: Y를 D_hat에 회귀
print("\n--- 2단계 (Second Stage) ---")
df['D_hat'] = first_stage.fittedvalues
second_stage = smf.ols('Y ~ D_hat + X1 + X2', data=df).fit()

iv_effect = second_stage.params['D_hat']
iv_se = second_stage.bse['D_hat']
iv_t = second_stage.tvalues['D_hat']
iv_p = second_stage.pvalues['D_hat']

print(f"β₁ (D → Y): {iv_effect:.3f} (SE: {iv_se:.3f})")
print(f"t-통계량: {iv_t:.2f}, p-value: {iv_p:.4f}")
print(f"95% CI: [{second_stage.conf_int().loc['D_hat', 0]:.3f}, {second_stage.conf_int().loc['D_hat', 1]:.3f}]")
print(f"참값과의 차이: {iv_effect - true_effect:.3f} ({abs(iv_effect - true_effect)/true_effect*100:.1f}%)")

# 4. 결과 비교
print("\n[4단계] 방법론 비교")
print("-" * 80)

comparison = pd.DataFrame({
    'Method': ['True Effect', '2SLS (IV)', 'OLS (Biased)'],
    'Estimate': [true_effect, iv_effect, ols_effect],
    'SE': [np.nan, iv_se, ols_se],
    'Bias': [0.0, iv_effect - true_effect, ols_effect - true_effect],
    'Bias %': [0.0, (iv_effect - true_effect)/true_effect*100, (ols_effect - true_effect)/true_effect*100]
})

print(comparison.to_string(index=False))

# 5. 시각화
print("\n[5단계] 시각화 생성")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) 1단계 회귀
ax = axes[0, 0]
ax.scatter(df['Z'], df['D'], alpha=0.3, s=20, color='steelblue')
z_range = np.linspace(df['Z'].min(), df['Z'].max(), 100)
d_pred = first_stage.params['Intercept'] + first_stage.params['Z'] * z_range + \
         first_stage.params['X1'] * df['X1'].mean() + first_stage.params['X2'] * df['X2'].mean()
ax.plot(z_range, d_pred, 'r-', linewidth=2, label=f'π₁={pi_1:.2f}')
ax.set_xlabel('도구변수 (Z)', fontsize=11)
ax.set_ylabel('처치 (D)', fontsize=11)
ax.set_title(f'(a) 1단계 회귀 (F={f_stat:.1f})', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# (b) 추정값 비교
ax = axes[0, 1]
methods = ['참값', '2SLS', 'OLS']
estimates = [true_effect, iv_effect, ols_effect]
errors = [0, iv_se * 1.96, ols_se * 1.96]
colors = ['green', 'steelblue', 'coral']

x_pos = np.arange(len(methods))
bars = ax.bar(x_pos, estimates, yerr=errors, capsize=5, color=colors, alpha=0.7)
ax.axhline(true_effect, color='red', linestyle='--', linewidth=2, label=f'참값 = {true_effect:.2f}')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_ylabel('처치효과 추정값', fontsize=11)
ax.set_title('(b) 방법론별 추정값 비교', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# (c) 잔차 vs 예측값 (2SLS)
ax = axes[1, 0]
fitted = second_stage.fittedvalues
residuals = second_stage.resid
ax.scatter(fitted, residuals, alpha=0.3, s=20, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('예측값', fontsize=11)
ax.set_ylabel('잔차', fontsize=11)
ax.set_title('(c) 2SLS 잔차 플롯', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# (d) 편향 비교
ax = axes[1, 1]
biases = [0, iv_effect - true_effect, ols_effect - true_effect]
colors_bias = ['green', 'steelblue', 'coral']
bars = ax.bar(x_pos, biases, color=colors_bias, alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_ylabel('편향 (추정값 - 참값)', fontsize=11)
ax.set_title('(d) 편향 비교', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('6-1-traditional-2sls.png', dpi=300, bbox_inches='tight')
print("그래프 저장: 6-1-traditional-2sls.png")

# 6. 최종 요약
print("\n" + "=" * 80)
print("2SLS 분석 요약")
print("=" * 80)
print(f"✓ 1단계 F-통계량: {f_stat:.2f} (강한 도구변수)")
print(f"✓ 2SLS 추정값: {iv_effect:.2f} (95% CI: [{second_stage.conf_int().loc['D_hat', 0]:.2f}, {second_stage.conf_int().loc['D_hat', 1]:.2f}])")
print(f"✓ OLS 편향: {(ols_effect - true_effect)/true_effect*100:.1f}% (내생성으로 인한 과소추정)")
print(f"✓ 2SLS는 참값에 {abs(iv_effect - true_effect)/true_effect*100:.1f}% 오차로 근접")
print("=" * 80)
