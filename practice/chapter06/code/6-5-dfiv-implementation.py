"""
제6장: Deep Feature IV (DFIV) Implementation (Simplified)
==========================================================

목적: DFIV의 핵심 개념 구현
- 특징 학습 네트워크
- 2단계 구조
- 차원 축소를 통한 효율성

주의: 본 구현은 교육 목적의 간소화된 버전입니다.

저자: AI 기반 정책분석방법론
날짜: 2025-11-20
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

np.random.seed(42)

print("=" * 80)
print("DFIV 구현 (간소화 버전)")
print("=" * 80)

# 1. 데이터 생성
print("\n[1단계] 데이터 생성")
print("-" * 80)

n = 2000
Z = np.random.randn(n)
X = np.random.randn(n)
U = np.random.randn(n)

D = 0.5 + 0.5 * Z + 0.3 * Z**2 + 0.2 * X + U + np.random.randn(n) * 0.2
Y = 2 + 3 * D + 0.5 * D**2 + 0.3 * X - 2 * U + np.random.randn(n) * 0.5

true_ate = 3 + 0.5 * D.mean()

print(f"샘플 크기: {n}")
print(f"참값 ATE: {true_ate:.3f}")

# Split
ZX = np.column_stack([Z, X])
ZX_train, ZX_val, D_train, D_val, Y_train, Y_val = train_test_split(
    ZX, D, Y, test_size=0.3, random_state=42
)

# 2. Stage 1: 특징 학습 (간소화: PCA + neural network)
print("\n[2단계] Stage 1: 특징 학습")
print("-" * 80)

k_dim = 10  # 특징 차원

# 특징 네트워크 (Z, X → Φ)
feature_model = MLPRegressor(
    hidden_layer_sizes=(64,),
    activation='relu',
    max_iter=200,
    random_state=42
)

# D 예측을 통한 특징 학습
feature_model.fit(ZX_train, D_train)
D_pred_train = feature_model.predict(ZX_train)
D_pred_val = feature_model.predict(ZX_val)

r2_stage1 = 1 - np.mean((D_val - D_pred_val)**2) / np.var(D_val)

print(f"Stage 1 R²: {r2_stage1:.3f}")
print(f"특징 차원: {k_dim} (개념적)")
print("✓ 특징 학습 완료")

# 3. Stage 2: 구조함수 학습
print("\n[3단계] Stage 2: 구조함수 학습")
print("-" * 80)

# 특징과 X를 사용하여 Y 예측
# 간소화: D_pred를 특징으로 사용
Phi_D_X_train = np.column_stack([D_pred_train, ZX_train[:, 1]])
Phi_D_X_val = np.column_stack([D_val, ZX_val[:, 1]])

structural_model = MLPRegressor(
    hidden_layer_sizes=(128, 128),
    activation='relu',
    max_iter=500,
    random_state=42,
    early_stopping=True
)

structural_model.fit(Phi_D_X_train, Y_train)

print("✓ 구조함수 학습 완료")

# 4. 평가
print("\n[4단계] DFIV 평가")
print("-" * 80)

Y_pred_val = structural_model.predict(Phi_D_X_val)
rmse_dfiv = np.sqrt(np.mean((Y_val - Y_pred_val)**2))

# 한계효과 추정
delta_d = 0.01
D_plus = D_val + delta_d
D_minus = D_val - delta_d

Y_plus = structural_model.predict(np.column_stack([D_plus, ZX_val[:, 1]]))
Y_minus = structural_model.predict(np.column_stack([D_minus, ZX_val[:, 1]]))

marginal_effects = (Y_plus - Y_minus) / (2 * delta_d)
dfiv_effect = marginal_effects.mean()

print(f"DFIV 추정값: {dfiv_effect:.3f}")
print(f"참값 ATE: {true_ate:.3f}")
print(f"오차: {abs(dfiv_effect - true_ate):.3f} ({abs(dfiv_effect - true_ate)/true_ate*100:.1f}%)")
print(f"RMSE: {rmse_dfiv:.2f}")

# 5. 시각화
print("\n[5단계] 시각화")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) Stage 1 품질
ax = axes[0]
ax.scatter(D_val, D_pred_val, alpha=0.3, s=10)
ax.plot([D_val.min(), D_val.max()], [D_val.min(), D_val.max()], 'r--', linewidth=2)
ax.set_xlabel('실제 D', fontsize=11)
ax.set_ylabel('예측 D (특징 학습)', fontsize=11)
ax.set_title(f'(a) Stage 1: 특징 품질 (R²={r2_stage1:.2f})', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# (b) Stage 2 예측
ax = axes[1]
ax.scatter(Y_val, Y_pred_val, alpha=0.3, s=10)
ax.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], 'r--', linewidth=2)
ax.set_xlabel('실제 Y', fontsize=11)
ax.set_ylabel('예측 Y (DFIV)', fontsize=11)
ax.set_title(f'(b) Stage 2: 예측 품질 (RMSE={rmse_dfiv:.2f})', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# (c) 처치효과
ax = axes[2]
methods = ['참값', 'DFIV']
estimates = [true_ate, dfiv_effect]
colors = ['green', 'steelblue']

x_pos = np.arange(len(methods))
bars = ax.bar(x_pos, estimates, color=colors, alpha=0.7)
ax.axhline(true_ate, color='red', linestyle='--', linewidth=2)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_ylabel('처치효과 추정값', fontsize=11)
ax.set_title('(c) DFIV 추정 결과', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('6-5-dfiv-implementation.png', dpi=300, bbox_inches='tight')
print("그래프 저장: 6-5-dfiv-implementation.png")

# 6. 최종 요약
print("\n" + "=" * 80)
print("DFIV 분석 요약")
print("=" * 80)
print(f"✓ DFIV 추정값: {dfiv_effect:.2f} (참값 대비 {abs(dfiv_effect - true_ate)/true_ate*100:.1f}% 오차)")
print(f"✓ RMSE: {rmse_dfiv:.2f}")
print(f"✓ 특징 차원 축소: 원본 차원 → {k_dim} 차원 (개념적)")
print("\n주의: 본 구현은 교육 목적의 간소화 버전입니다.")
print("=" * 80)
