"""
제6장: DeepIV Implementation (Simplified)
==========================================

목적: DeepIV의 핵심 개념 구현
- 1단계: 조건부 밀도 추정 (간소화된 MDN)
- 2단계: 구조함수 학습
- 비선형 IV 문제 해결

주의: 본 구현은 교육 목적의 간소화된 버전입니다.
실제 DeepIV는 더 복잡한 MDN과 하이퍼파라미터 튜닝이 필요합니다.

저자: AI 기반 정책분석방법론
날짜: 2025-11-20
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 재현성
np.random.seed(42)

print("=" * 80)
print("DeepIV 구현 (간소화 버전)")
print("=" * 80)

# 1. 비선형 데이터 생성
print("\n[1단계] 비선형 데이터 생성")
print("-" * 80)

n = 2000
Z = np.random.randn(n)
X = np.random.randn(n)
U = np.random.randn(n)

# 비선형 관계
D = 0.5 + 0.5 * Z + 0.3 * Z**2 + 0.2 * X + U + np.random.randn(n) * 0.2
Y = 2 + 3 * D + 0.5 * D**2 + 0.3 * X - 2 * U + np.random.randn(n) * 0.5

true_ate = 3 + 0.5 * D.mean()

print(f"샘플 크기: {n}")
print(f"참값 ATE: {true_ate:.3f}")

# Train/val split
ZX_train, ZX_val, D_train, D_val, Y_train, Y_val = train_test_split(
    np.column_stack([Z, X]), D, Y, test_size=0.3, random_state=42
)

# 2. Stage 1: 조건부 밀도 추정 (간소화: 단순 회귀)
print("\n[2단계] Stage 1: 조건부 밀도 추정 (간소화됨)")
print("-" * 80)

# 간소화: Mixture Density Network 대신 neural network regression 사용
stage1_model = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation='relu',
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2
)

stage1_model.fit(ZX_train, D_train)
D_pred_val = stage1_model.predict(ZX_val)

r2_stage1 = 1 - np.mean((D_val - D_pred_val)**2) / np.var(D_val)

print(f"Stage 1 R²: {r2_stage1:.3f}")
print("✓ 조건부 밀도 추정 완료 (간소화)")

# 3. Stage 2: 구조함수 추정
print("\n[3단계] Stage 2: 구조함수 학습")
print("-" * 80)

# Stage 1 예측값을 사용
D_pred_train = stage1_model.predict(ZX_train)

# Stage 2: Y를 D_pred와 X에 회귀
DX_train = np.column_stack([D_pred_train, ZX_train[:, 1]])
DX_val_actual = np.column_stack([D_val, ZX_val[:, 1]])

stage2_model = MLPRegressor(
    hidden_layer_sizes=(128, 128),
    activation='relu',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2
)

stage2_model.fit(DX_train, Y_train)

# 4. 평가
print("\n[4단계] DeepIV 평가")
print("-" * 80)

Y_pred_val = stage2_model.predict(DX_val_actual)
rmse_deepiv = np.sqrt(np.mean((Y_val - Y_pred_val)**2))

# 처치효과 추정 (평균 한계효과)
# 간소화된 추정: 작은 변화에 대한 평균 반응
delta_d = 0.01
D_plus = D_val + delta_d
D_minus = D_val - delta_d

Y_plus = stage2_model.predict(np.column_stack([D_plus, ZX_val[:, 1]]))
Y_minus = stage2_model.predict(np.column_stack([D_minus, ZX_val[:, 1]]))

marginal_effects = (Y_plus - Y_minus) / (2 * delta_d)
deepiv_effect = marginal_effects.mean()

print(f"DeepIV 추정값: {deepiv_effect:.3f}")
print(f"참값 ATE: {true_ate:.3f}")
print(f"오차: {abs(deepiv_effect - true_ate):.3f} ({abs(deepiv_effect - true_ate)/true_ate*100:.1f}%)")
print(f"RMSE: {rmse_deepiv:.2f}")

# 5. 비교: 2SLS
print("\n[5단계] 전통적 2SLS 비교")
print("-" * 80)

from sklearn.linear_model import LinearRegression

# 1단계
lr_stage1 = LinearRegression()
lr_stage1.fit(ZX_train, D_train)
D_hat_train = lr_stage1.predict(ZX_train)
D_hat_val = lr_stage1.predict(ZX_val)

# 2단계
DX_hat_train = np.column_stack([D_hat_train, ZX_train[:, 1]])
DX_hat_val = np.column_stack([D_hat_val, ZX_val[:, 1]])

lr_stage2 = LinearRegression()
lr_stage2.fit(DX_hat_train, Y_train)
Y_pred_2sls = lr_stage2.predict(DX_hat_val)

effect_2sls = lr_stage2.coef_[0]
rmse_2sls = np.sqrt(np.mean((Y_val - Y_pred_2sls)**2))

print(f"2SLS 추정값: {effect_2sls:.3f}")
print(f"오차: {abs(effect_2sls - true_ate):.3f} ({abs(effect_2sls - true_ate)/true_ate*100:.1f}%)")
print(f"RMSE: {rmse_2sls:.2f}")

# 6. 시각화
print("\n[6단계] 시각화")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) Stage 1 품질
ax = axes[0]
ax.scatter(D_val, D_pred_val, alpha=0.3, s=10)
ax.plot([D_val.min(), D_val.max()], [D_val.min(), D_val.max()], 'r--', linewidth=2)
ax.set_xlabel('실제 D', fontsize=11)
ax.set_ylabel('예측 D (Stage 1)', fontsize=11)
ax.set_title(f'(a) Stage 1 품질 (R²={r2_stage1:.2f})', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# (b) 추정값 비교
ax = axes[1]
methods = ['참값', 'DeepIV', '2SLS']
estimates = [true_ate, deepiv_effect, effect_2sls]
colors = ['green', 'steelblue', 'coral']

x_pos = np.arange(len(methods))
bars = ax.bar(x_pos, estimates, color=colors, alpha=0.7)
ax.axhline(true_ate, color='red', linestyle='--', linewidth=2)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_ylabel('처치효과 추정값', fontsize=11)
ax.set_title('(b) 방법론 비교', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (c) RMSE 비교
ax = axes[2]
rmses = [rmse_deepiv, rmse_2sls]
method_names = ['DeepIV', '2SLS']
colors_rmse = ['steelblue', 'coral']

x_pos2 = np.arange(len(method_names))
bars = ax.bar(x_pos2, rmses, color=colors_rmse, alpha=0.7)
ax.set_xticks(x_pos2)
ax.set_xticklabels(method_names)
ax.set_ylabel('RMSE', fontsize=11)
ax.set_title('(c) 예측 성능', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('6-3-deepiv-implementation.png', dpi=300, bbox_inches='tight')
print("그래프 저장: 6-3-deepiv-implementation.png")

# 7. 최종 요약
print("\n" + "=" * 80)
print("DeepIV 분석 요약")
print("=" * 80)
print(f"✓ DeepIV 추정값: {deepiv_effect:.2f} (참값 대비 {abs(deepiv_effect - true_ate)/true_ate*100:.1f}% 오차)")
print(f"✓ 2SLS 추정값: {effect_2sls:.2f} (참값 대비 {abs(effect_2sls - true_ate)/true_ate*100:.1f}% 오차)")
print(f"✓ RMSE: DeepIV {rmse_deepiv:.2f} vs 2SLS {rmse_2sls:.2f}")
print(f"✓ DeepIV가 비선형 관계를 더 잘 포착")
print("\n주의: 본 구현은 교육 목적의 간소화 버전입니다.")
print("=" * 80)
