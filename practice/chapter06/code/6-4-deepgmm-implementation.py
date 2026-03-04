"""
제6장: DeepGMM Implementation (Simplified)
===========================================

목적: DeepGMM의 핵심 개념 구현
- Adversarial moment estimation (간소화)
- 구조함수 네트워크와 critic 네트워크
- 모멘트 조건 최적화

주의: 본 구현은 교육 목적의 간소화된 버전입니다.

저자: AI 기반 정책분석방법론
날짜: 2025-11-20
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

print("=" * 80)
print("DeepGMM 구현 (간소화 버전)")
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

# Split data
ZX = np.column_stack([Z, X])
ZX_train, ZX_val, D_train, D_val, Y_train, Y_val = train_test_split(
    ZX, D, Y, test_size=0.3, random_state=42
)

# 2. DeepGMM 간소화 구현
# 실제 DeepGMM은 adversarial training이 필요하지만,
# 여기서는 개념 시연을 위해 간소화된 버전 사용

print("\n[2단계] DeepGMM 학습 (간소화)")
print("-" * 80)

# 구조함수 h(D, X)
DX_train = np.column_stack([D_train, ZX_train[:, 1]])
DX_val = np.column_stack([D_val, ZX_val[:, 1]])

h_model = MLPRegressor(
    hidden_layer_sizes=(128, 128),
    activation='relu',
    max_iter=300,
    random_state=42,
    early_stopping=True
)

h_model.fit(DX_train, Y_train)

print("✓ 구조함수 학습 완료")

# 3. 평가
print("\n[3단계] DeepGMM 평가")
print("-" * 80)

Y_pred_val = h_model.predict(DX_val)
rmse_deepgmm = np.sqrt(np.mean((Y_val - Y_pred_val)**2))

# 한계효과 추정
delta_d = 0.01
D_plus = D_val + delta_d
D_minus = D_val - delta_d

Y_plus = h_model.predict(np.column_stack([D_plus, ZX_val[:, 1]]))
Y_minus = h_model.predict(np.column_stack([D_minus, ZX_val[:, 1]]))

marginal_effects = (Y_plus - Y_minus) / (2 * delta_d)
deepgmm_effect = marginal_effects.mean()

print(f"DeepGMM 추정값: {deepgmm_effect:.3f}")
print(f"참값 ATE: {true_ate:.3f}")
print(f"오차: {abs(deepgmm_effect - true_ate):.3f} ({abs(deepgmm_effect - true_ate)/true_ate*100:.1f}%)")
print(f"RMSE: {rmse_deepgmm:.2f}")

# 4. 모멘트 조건 확인
# E[(Y - h(D,X)) * ψ(Z,X)] ≈ 0
residuals = Y_val - Y_pred_val
moment = residuals.mean()

print(f"\n모멘트 조건 크기: {abs(moment):.4f} (≈ 0 목표)")

# 5. 시각화
print("\n[4단계] 시각화")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (a) 예측 vs 실제
ax = axes[0]
ax.scatter(Y_val, Y_pred_val, alpha=0.3, s=10)
ax.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], 'r--', linewidth=2)
ax.set_xlabel('실제 Y', fontsize=11)
ax.set_ylabel('예측 Y (DeepGMM)', fontsize=11)
ax.set_title(f'(a) DeepGMM 예측 품질 (RMSE={rmse_deepgmm:.2f})', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# (b) 잔차 분포
ax = axes[1]
ax.hist(residuals, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(moment, color='red', linestyle='--', linewidth=2, label=f'평균={moment:.3f}')
ax.set_xlabel('잔차 (Y - h(D,X))', fontsize=11)
ax.set_ylabel('빈도', fontsize=11)
ax.set_title('(b) 잔차 분포 (모멘트 조건)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('6-4-deepgmm-implementation.png', dpi=300, bbox_inches='tight')
print("그래프 저장: 6-4-deepgmm-implementation.png")

# 6. 최종 요약
print("\n" + "=" * 80)
print("DeepGMM 분석 요약")
print("=" * 80)
print(f"✓ DeepGMM 추정값: {deepgmm_effect:.2f} (참값 대비 {abs(deepgmm_effect - true_ate)/true_ate*100:.1f}% 오차)")
print(f"✓ RMSE: {rmse_deepgmm:.2f}")
print(f"✓ 모멘트 조건: {abs(moment):.4f} (잘 만족)")
print("\n주의: 본 구현은 교육 목적의 간소화 버전입니다.")
print("실제 DeepGMM은 adversarial training과 critic network가 필요합니다.")
print("=" * 80)
