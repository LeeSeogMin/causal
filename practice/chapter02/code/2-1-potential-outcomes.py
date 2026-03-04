"""
제2장 2.1절: Potential Outcomes Framework
Rubin의 잠재적 결과 프레임워크를 통한 선택편향과 인과효과 이해
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("제2장 2.1절: Potential Outcomes Framework")
print("=" * 70)
print()

# 1. 데이터 로드
print("1. Potential Outcomes 데이터 로드")
print("-" * 70)

# CSV 파일에서 데이터 읽기
# CSV 파일에서 데이터 읽기
import os
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/2-1-potential-outcomes.csv')
df = pd.read_csv(data_path)

# 변수 추출
X = df[['X1', 'X2', 'X3', 'X4', 'X5']].values
treatment = df['treatment'].values
Y_observed = df['Y_observed'].values
Y0_true = df['Y0_true'].values
Y1_true = df['Y1_true'].values
true_propensity = df['true_propensity'].values

n = len(df)
print(f"  데이터 파일: {data_path}")
print(f"  총 샘플 수: {n}개")
print(f"  처치군: {treatment.sum()}개")
print(f"  대조군: {n - treatment.sum()}개")
print()

# 2. 인과효과 추정
print("2. 인과효과 추정")
print("-" * 70)

# 참값 ATE와 ATT
true_ATE = np.mean(Y1_true - Y0_true)
true_ATT = np.mean((Y1_true - Y0_true)[treatment == 1])

# Naive 추정 (선택편향 포함)
naive_estimate = Y_observed[treatment==1].mean() - Y_observed[treatment==0].mean()
selection_bias = naive_estimate - true_ATT
bias_ratio = (selection_bias / true_ATT) * 100

print(f"  참값 ATE: {true_ATE:.2f}")
print(f"  참값 ATT: {true_ATT:.2f}")
print(f"  Naive 추정값: {naive_estimate:.2f}")
print(f"  선택편향: {selection_bias:.2f}")
print(f"  편향 비율: {bias_ratio:.1f}%")
print()

# 3. 공변량 균형 진단
print("3. 공변량 균형 진단")
print("-" * 70)

def compute_smd(X, treatment):
    """표준화 평균 차이 계산"""
    mean_t = X[treatment==1].mean(axis=0)
    mean_c = X[treatment==0].mean(axis=0)
    std_pooled = np.sqrt((X[treatment==1].var(axis=0) + X[treatment==0].var(axis=0)) / 2)
    return (mean_t - mean_c) / std_pooled

mean_treated = X[treatment==1].mean(axis=0)
mean_control = X[treatment==0].mean(axis=0)
smd = compute_smd(X, treatment)

print("\n공변량별 균형:")
for i in range(5):
    print(f"  X{i+1}: 처치군={mean_treated[i]:6.2f}, 대조군={mean_control[i]:6.2f}, SMD={smd[i]:.2f}")
print()

# 4. Overlap 진단
print("4. Overlap (공통지지) 진단")
print("-" * 70)

ps_range_treated = [true_propensity[treatment==1].min(), true_propensity[treatment==1].max()]
ps_range_control = [true_propensity[treatment==0].min(), true_propensity[treatment==0].max()]

# 공통지지 영역 밖 개체 수
common_support_min = max(ps_range_control[0], ps_range_treated[0])
common_support_max = min(ps_range_control[1], ps_range_treated[1])

outside_support = ((true_propensity < common_support_min) |
                   (true_propensity > common_support_max)).sum()
outside_ratio = (outside_support / n) * 100

print(f"  성향점수 범위 (처치군): [{ps_range_treated[0]:.2f}, {ps_range_treated[1]:.2f}]")
print(f"  성향점수 범위 (대조군): [{ps_range_control[0]:.2f}, {ps_range_control[1]:.2f}]")
print(f"  공통지지 영역 밖 개체: {outside_support}개 ({outside_ratio:.1f}%)")
print()

# 5. 결과 요약 테이블
print("=" * 70)
print("분석 결과 요약")
print("=" * 70)
print()

print("표 2-1: Potential Outcomes 시뮬레이션 결과")
print("-" * 70)
results_df = pd.DataFrame({
    '추정량': ['참값 ATE', '참값 ATT', 'Naive 추정값', '선택편향', '편향 비율'],
    '값': [f'{true_ATE:.2f}', f'{true_ATT:.2f}', f'{naive_estimate:.2f}',
           f'{selection_bias:.2f}', f'{bias_ratio:.1f}%'],
    '설명': ['전체 모집단의 평균 처치효과', '처치군의 평균 처치효과',
            '단순 평균 차이 (편향됨)', 'Naive 추정 - 참값 ATT',
            '(선택편향 / 참값 ATT) × 100']
})
print(results_df.to_string(index=False))
print()

print("표 2-2: 공변량 균형 진단")
print("-" * 70)
balance_df = pd.DataFrame({
    '공변량': [f'X{i+1}' for i in range(5)],
    '처치군 평균': [f'{mean_treated[i]:.2f}' for i in range(5)],
    '대조군 평균': [f'{mean_control[i]:.2f}' for i in range(5)],
    'SMD': [f'{smd[i]:.2f}' for i in range(5)]
})
print(balance_df.to_string(index=False))
print()

print("표 2-3: Overlap 진단")
print("-" * 70)
print(f"  - 성향점수 범위 (처치군): [{ps_range_treated[0]:.2f}, {ps_range_treated[1]:.2f}]")
print(f"  - 성향점수 범위 (대조군): [{ps_range_control[0]:.2f}, {ps_range_control[1]:.2f}]")
print(f"  - 공통지지 영역 밖 개체: {outside_support}개 ({outside_ratio:.1f}%)")
print()

# 6. 시각화
print("5. 결과 시각화 생성 중...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (1) Potential Outcomes 분포
ax1 = axes[0, 0]
ax1.hist(Y0_true[treatment==1], bins=30, alpha=0.5, label='Y0 (Treated)', color='blue')
ax1.hist(Y1_true[treatment==1], bins=30, alpha=0.5, label='Y1 (Treated)', color='red')
ax1.axvline(Y0_true[treatment==1].mean(), color='blue', linestyle='--', linewidth=2)
ax1.axvline(Y1_true[treatment==1].mean(), color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Outcome')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# (2) 성향점수 분포
ax2 = axes[0, 1]
ax2.hist(true_propensity[treatment==1], bins=30, alpha=0.5, label='Treated', color='red')
ax2.hist(true_propensity[treatment==0], bins=30, alpha=0.5, label='Control', color='blue')
ax2.axvline(common_support_min, color='green', linestyle='--', label='Common Support')
ax2.axvline(common_support_max, color='green', linestyle='--')
ax2.set_xlabel('Propensity Score')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# (3) 공변량 균형 (SMD)
ax3 = axes[1, 0]
covariates = [f'X{i+1}' for i in range(5)]
colors = ['red' if abs(s) > 0.1 else 'green' for s in smd]
ax3.barh(covariates, np.abs(smd), color=colors, alpha=0.7)
ax3.axvline(0.1, color='red', linestyle='--', linewidth=2, label='SMD = 0.1 threshold')
ax3.set_xlabel('Absolute SMD')
ax3.set_ylabel('Covariate')
ax3.legend()
ax3.grid(True, alpha=0.3)

# (4) 선택편향 분해
ax4 = axes[1, 1]
categories = ['True ATT', 'Selection\nBias', 'Naive\nEstimate']
values = [true_ATT, selection_bias, naive_estimate]
colors_bar = ['green', 'red', 'orange']
bars = ax4.bar(categories, values, color=colors_bar, alpha=0.7)
ax4.axhline(true_ATT, color='green', linestyle='--', linewidth=2, label='True ATT')
ax4.set_ylabel('Effect Size')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 값 표시
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('potential_outcomes_analysis.png', dpi=150, bbox_inches='tight')
print("  시각화 저장 완료: potential_outcomes_analysis.png")
print()

print("=" * 70)
print("분석 완료!")
print("=" * 70)
print()
print("주요 시사점:")
print(f"  1. 선택편향: Naive 추정값({naive_estimate:.2f})은 참값 ATT({true_ATT:.2f})보다")
print(f"     {bias_ratio:.1f}% 과대추정되어 있음")
print(f"  2. 공변량 불균형: X1, X2, X3의 SMD > 0.5로 심각한 불균형")
print(f"  3. Overlap: {outside_support}개({outside_ratio:.1f}%) 개체가 공통지지 밖")
print()
print("※ 본 코드는 교육 목적의 시뮬레이션입니다.")
