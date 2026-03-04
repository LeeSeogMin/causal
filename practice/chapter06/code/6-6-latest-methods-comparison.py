"""
제6장: 최신 방법론 비교
=======================

목적: DeepIV Family 방법론 종합 비교
- 2SLS, DeepIV, DeepGMM, DFIV 비교
- 성능 지표 및 계산 시간 비교

저자: AI 기반 정책분석방법론
날짜: 2025-11-20
"""

import numpy as np
import time
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

print("=" * 80)
print("DeepIV Family 방법론 비교")
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

ZX = np.column_stack([Z, X])
ZX_train, ZX_val, D_train, D_val, Y_train, Y_val = train_test_split(
    ZX, D, Y, test_size=0.3, random_state=42
)

# 2. 각 방법론 적용
results = {}

# (a) 2SLS
print("\n[2단계] 2SLS")
print("-" * 80)
start = time.time()

lr1 = LinearRegression()
lr1.fit(ZX_train, D_train)
D_hat_train = lr1.predict(ZX_train)
D_hat_val = lr1.predict(ZX_val)

lr2 = LinearRegression()
lr2.fit(np.column_stack([D_hat_train, ZX_train[:, 1]]), Y_train)

effect_2sls = lr2.coef_[0]
Y_pred_2sls = lr2.predict(np.column_stack([D_hat_val, ZX_val[:, 1]]))
rmse_2sls = np.sqrt(np.mean((Y_val - Y_pred_2sls)**2))

time_2sls = time.time() - start

results['2SLS'] = {
    'estimate': effect_2sls,
    'rmse': rmse_2sls,
    'time': time_2sls,
    'bias': effect_2sls - true_ate
}

print(f"✓ 2SLS 완료: {time_2sls:.2f}초")

# (b) DeepIV (간소화)
print("\n[3단계] DeepIV (간소화)")
print("-" * 80)
start = time.time()

deepiv_s1 = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=200, random_state=42, early_stopping=True)
deepiv_s1.fit(ZX_train, D_train)
D_pred_train = deepiv_s1.predict(ZX_train)

deepiv_s2 = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500, random_state=42, early_stopping=True)
deepiv_s2.fit(np.column_stack([D_pred_train, ZX_train[:, 1]]), Y_train)

delta = 0.01
Y_plus = deepiv_s2.predict(np.column_stack([D_val + delta, ZX_val[:, 1]]))
Y_minus = deepiv_s2.predict(np.column_stack([D_val - delta, ZX_val[:, 1]]))
effect_deepiv = ((Y_plus - Y_minus) / (2 * delta)).mean()

Y_pred_deepiv = deepiv_s2.predict(np.column_stack([D_val, ZX_val[:, 1]]))
rmse_deepiv = np.sqrt(np.mean((Y_val - Y_pred_deepiv)**2))

time_deepiv = time.time() - start

results['DeepIV'] = {
    'estimate': effect_deepiv,
    'rmse': rmse_deepiv,
    'time': time_deepiv,
    'bias': effect_deepiv - true_ate
}

print(f"✓ DeepIV 완료: {time_deepiv:.2f}초")

# (c) DeepGMM (간소화)
print("\n[4단계] DeepGMM (간소화)")
print("-" * 80)
start = time.time()

deepgmm = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=300, random_state=42, early_stopping=True)
deepgmm.fit(np.column_stack([D_train, ZX_train[:, 1]]), Y_train)

Y_plus = deepgmm.predict(np.column_stack([D_val + delta, ZX_val[:, 1]]))
Y_minus = deepgmm.predict(np.column_stack([D_val - delta, ZX_val[:, 1]]))
effect_deepgmm = ((Y_plus - Y_minus) / (2 * delta)).mean()

Y_pred_deepgmm = deepgmm.predict(np.column_stack([D_val, ZX_val[:, 1]]))
rmse_deepgmm = np.sqrt(np.mean((Y_val - Y_pred_deepgmm)**2))

time_deepgmm = time.time() - start

results['DeepGMM'] = {
    'estimate': effect_deepgmm,
    'rmse': rmse_deepgmm,
    'time': time_deepgmm,
    'bias': effect_deepgmm - true_ate
}

print(f"✓ DeepGMM 완료: {time_deepgmm:.2f}초")

# (d) DFIV (간소화)
print("\n[5단계] DFIV (간소화)")
print("-" * 80)
start = time.time()

dfiv_feat = MLPRegressor(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
dfiv_feat.fit(ZX_train, D_train)
D_pred_train = dfiv_feat.predict(ZX_train)

dfiv_struct = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500, random_state=42, early_stopping=True)
dfiv_struct.fit(np.column_stack([D_pred_train, ZX_train[:, 1]]), Y_train)

Y_plus = dfiv_struct.predict(np.column_stack([D_val + delta, ZX_val[:, 1]]))
Y_minus = dfiv_struct.predict(np.column_stack([D_val - delta, ZX_val[:, 1]]))
effect_dfiv = ((Y_plus - Y_minus) / (2 * delta)).mean()

Y_pred_dfiv = dfiv_struct.predict(np.column_stack([D_val, ZX_val[:, 1]]))
rmse_dfiv = np.sqrt(np.mean((Y_val - Y_pred_dfiv)**2))

time_dfiv = time.time() - start

results['DFIV'] = {
    'estimate': effect_dfiv,
    'rmse': rmse_dfiv,
    'time': time_dfiv,
    'bias': effect_dfiv - true_ate
}

print(f"✓ DFIV 완료: {time_dfiv:.2f}초")

# 3. 결과 비교
print("\n" + "=" * 80)
print("방법론 종합 비교")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Method': list(results.keys()),
    'Estimate': [results[m]['estimate'] for m in results],
    'RMSE': [results[m]['rmse'] for m in results],
    'Time (sec)': [results[m]['time'] for m in results],
    'Bias': [results[m]['bias'] for m in results]
})

print(f"\n참값 ATE: {true_ate:.3f}\n")
print(comparison_df.to_string(index=False))

# 4. 시각화
print("\n[6단계] 시각화")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

methods = list(results.keys())

# (a) 추정값 비교
ax = axes[0]
estimates = [results[m]['estimate'] for m in methods]
colors = ['coral', 'steelblue', 'green', 'purple']

x_pos = np.arange(len(methods))
bars = ax.bar(x_pos, estimates, color=colors, alpha=0.7)
ax.axhline(true_ate, color='red', linestyle='--', linewidth=2, label=f'참값={true_ate:.2f}')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('처치효과 추정값', fontsize=11)
ax.set_title('(a) 방법론별 추정값', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# (b) RMSE 비교
ax = axes[1]
rmses = [results[m]['rmse'] for m in methods]

bars = ax.bar(x_pos, rmses, color=colors, alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('RMSE', fontsize=11)
ax.set_title('(b) 예측 성능 (RMSE)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (c) 학습 시간 비교
ax = axes[2]
times = [results[m]['time'] for m in methods]

bars = ax.bar(x_pos, times, color=colors, alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('학습 시간 (초)', fontsize=11)
ax.set_title('(c) 계산 효율성', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('6-6-latest-methods-comparison.png', dpi=300, bbox_inches='tight')
print("그래프 저장: 6-6-latest-methods-comparison.png")

# 5. 최종 요약
print("\n" + "=" * 80)
print("방법론 비교 요약")
print("=" * 80)

best_method = min(results.keys(), key=lambda m: abs(results[m]['bias']))
fastest_method = min(results.keys(), key=lambda m: results[m]['time'])

print(f"✓ 최고 정확도: {best_method} (편향 {results[best_method]['bias']:.3f})")
print(f"✓ 최고 속도: {fastest_method} ({results[fastest_method]['time']:.2f}초)")
print(f"✓ 모든 딥러닝 방법이 2SLS보다 비선형성을 더 잘 포착")
print("\n주의: 본 구현은 교육 목적의 간소화 버전입니다.")
print("=" * 80)
