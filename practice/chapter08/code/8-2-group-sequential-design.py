#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제8장 실습 2: 조기 중단(peeking)과 그룹 순차 설계
==================================================

목적
- 중간에 결과를 몇 번 들여다보면 1종 오류가 얼마나 부풀어 오르는지 센다.
- 그 부풀림을 되돌리는 중단 경계(Pocock, O'Brien-Fleming)를 직접 보정한다.
- 보정한 경계를 이항 실험 시뮬레이션에 적용해 5%로 돌아오는지 확인한다.

출력 그림: 8-2-group-sequential-design.png

수정 이력
- 2026-08-17
  (1) [근본 원인] pocock_boundary()가 z = Φ⁻¹(1-α/(2K))를 돌려주고 있었다.
      이는 각 중간 분석이 서로 독립이라고 본 본페로니 보정이다.
      누적 통계량은 앞 단계를 포함하므로 서로 독립이 아니다.
      K=3에서 본페로니는 z=2.394를 주지만 참 Pocock 경계는 약 2.29다.
      → 누적 z의 결합분포를 몬테카를로로 만들고 이분법으로 상수를 보정하도록
        바꿨다. O'Brien-Fleming 상수도 같은 방식으로 보정한다(1.96 → 약 2.00).
  (2) [근본 원인] Null 시뮬레이션 루프에서 O'Brien-Fleming이 먼저 걸리면
      break가 걸려 Pocock을 세지 못했다. Pocock 조기 중단률 0.0166은
      실제 값이 아니라 O'Brien-Fleming에 가려진 나머지였다.
      → 두 경계를 서로 다른 루프에서 독립으로 세도록 분리했다.
  (3) 보정하지 않고 매번 α=0.05로 보는 경우(peeking)를 세는 절을 새로 넣었다.
      이 장의 핵심 질문("중간에 봐도 되는가")에 답하는 숫자다.
  (4) 그림 출력이 없어 경계 모양을 볼 수 없었다. 3개 패널 PNG를 저장한다.

저자: AI 기반 정책분석방법론
날짜: 2025-11-20 (최종 수정 2026-08-17)
"""

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ALPHA = 0.05
K = 3
N_MAX = 200
N_PATH = 200_000

print("=" * 80)
print("실습 2. 조기 중단(peeking)과 그룹 순차 설계")
print("=" * 80)


# ---------------------------------------------------------------------------
# 누적 z 통계량의 경로를 만든다
# ---------------------------------------------------------------------------
def z_paths(k, n_path=N_PATH, seed=0):
    """귀무가설이 참일 때 k번 들여다본 누적 z 통계량. 모양은 (n_path, k)."""
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n_path, k))          # 단계마다 새로 들어온 정보
    cum = np.cumsum(steps, axis=1)                    # 누적합
    info = np.sqrt(np.arange(1, k + 1))               # 누적 정보량의 제곱근
    return cum / info


def familywise_error(paths, bounds):
    """한 번이라도 경계를 넘으면 기각. 그 확률."""
    return float(np.mean(np.any(np.abs(paths) >= bounds, axis=1)))


def calibrate(paths, shape, alpha=ALPHA):
    """shape(모양)에 상수 c를 곱한 경계가 전체 1종 오류 alpha를 주도록 c를 찾는다."""
    lo, hi = 0.5, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if familywise_error(paths, mid * shape) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


paths3 = z_paths(K, seed=0)
t = np.arange(1, K + 1) / K                  # 정보 분획 0.33, 0.67, 1.00

looks = [int(round(N_MAX * f)) for f in t]   # 중간 분석 시점의 누적 총원

shape_pocock = np.ones(K)                    # 모든 단계에서 같은 경계
shape_obf = 1 / np.sqrt(t)                   # 초기에 엄격, 후기에 완화

c_pocock = calibrate(paths3, shape_pocock)
c_obf = calibrate(paths3, shape_obf)

z_pocock = c_pocock * shape_pocock
z_obf = c_obf * shape_obf
p_pocock = 2 * (1 - norm.cdf(z_pocock))
p_obf = 2 * (1 - norm.cdf(z_obf))

# ---------------------------------------------------------------------------
# 1단계: 보정하지 않고 들여다보면 어떻게 되는가
# ---------------------------------------------------------------------------
print("\n[1단계] 보정 없이 매번 α=0.05로 판정할 때의 1종 오류")
print("-" * 80)
print("  귀무가설(두 안의 효과가 같다)이 참인데도 '유의하다'고 말할 확률")
print()
print(f"  {'들여다본 횟수 K':>16}{'1종 오류':>12}{'명목 0.05의 몇 배':>20}")

peek_rows = []
for k in [1, 2, 3, 5, 10, 20]:
    pk = z_paths(k, seed=1)
    err = familywise_error(pk, np.full(k, norm.ppf(1 - ALPHA / 2)))
    peek_rows.append((k, err))
    print(f"  {k:>16}{err:>12.4f}{err / ALPHA:>20.2f}")

print()
print("  K=1은 0.05 근처다. 설계대로 한 번만 본 경우다.")
print(f"  K=5면 {dict(peek_rows)[5]:.3f}, K=20이면 {dict(peek_rows)[20]:.3f}까지 오른다.")
print("  '유망해 보일 때마다 확인하고 유의하면 멈춘다'가 위험한 까닭이 이 표다.")

# ---------------------------------------------------------------------------
# 2단계: 보정한 중단 경계
# ---------------------------------------------------------------------------
print(f"\n[2단계] 보정한 중단 경계 (K={K}, 양측 α={ALPHA})")
print("-" * 80)
print(f"  Pocock 상수           c = {c_pocock:.4f}")
print(f"  O'Brien-Fleming 상수  c = {c_obf:.4f}  (보정 전 1.96)")
print()
print(f"  {'단계':>5}{'정보 분획':>11}{'누적 n':>9}"
      f"{'Pocock z':>11}{'Pocock p':>11}{'OBF z':>9}{'OBF p':>11}")
for i in range(K):
    print(f"  {i+1:>5}{t[i]:>11.2f}{looks[i]:>9}"
          f"{z_pocock[i]:>11.3f}{p_pocock[i]:>11.5f}{z_obf[i]:>9.3f}{p_obf[i]:>11.5f}")

print()
print(f"  검증: Pocock 전체 1종 오류          {familywise_error(paths3, z_pocock):.4f}")
print(f"  검증: O'Brien-Fleming 전체 1종 오류 {familywise_error(paths3, z_obf):.4f}")
print(f"  검증: 무보정(매번 1.96) 전체 1종 오류 "
      f"{familywise_error(paths3, np.full(K, 1.96)):.4f}")

# ---------------------------------------------------------------------------
# 3단계: 이항 실험에 적용해 확인
# ---------------------------------------------------------------------------
print("\n[3단계] 이항 실험 시뮬레이션으로 확인 (두 안 모두 전환율 0.40)")
print("-" * 80)

N_SIM = 20_000
print(f"  중간 분석 시점(누적 총원): {looks}")
print(f"  반복 횟수: {N_SIM:,}회")

rng = np.random.default_rng(7)
n_arm = N_MAX // 2
y_c = rng.binomial(1, 0.40, size=(N_SIM, n_arm))
y_t = rng.binomial(1, 0.40, size=(N_SIM, n_arm))

z_obs = np.zeros((N_SIM, K))
for k, n_total in enumerate(looks):
    m = n_total // 2
    diff = y_t[:, :m].mean(axis=1) - y_c[:, :m].mean(axis=1)
    pooled = np.concatenate([y_t[:, :m], y_c[:, :m]], axis=1).mean(axis=1)
    se = np.sqrt(2 * pooled * (1 - pooled) / m)
    z_obs[:, k] = np.divide(diff, se, out=np.zeros(N_SIM), where=se > 0)

sim_rows = []
for label, bounds in [('무보정 (매번 1.96)', np.full(K, norm.ppf(1 - ALPHA / 2))),
                      ('Pocock', z_pocock),
                      ("O'Brien-Fleming", z_obf)]:
    hit = np.abs(z_obs) >= bounds
    err = float(np.mean(hit.any(axis=1)))
    first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)
    stop1 = float(np.mean(first == 1))
    sim_rows.append({'방법': label, '1종 오류': err, '1단계에서 중단': stop1})
    print(f"  {label:<20} 1종 오류 {err:.4f}   1단계에서 중단 {stop1:.4f}")

print()
print("  무보정은 명목 0.05를 넘고, 보정한 두 경계는 0.05 근처로 돌아온다.")
print("  Pocock은 1단계에서도 잘 멈추고, O'Brien-Fleming은 1단계를 사실상 통과시킨다.")

# ---------------------------------------------------------------------------
# 4단계: 요약표
# ---------------------------------------------------------------------------
print("\n[4단계] 요약표")
print("-" * 80)
summary = pd.DataFrame({
    'Stage': np.arange(1, K + 1),
    '정보분획': np.round(t, 2),
    '누적n': looks,
    'Pocock_z': np.round(z_pocock, 3),
    'Pocock_p': np.round(p_pocock, 5),
    'OBF_z': np.round(z_obf, 3),
    'OBF_p': np.round(p_obf, 5),
})
print(summary.to_string(index=False))

# ---------------------------------------------------------------------------
# 5단계: 그림
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

ax = axes[0]
ax.plot(t, z_pocock, 'o-', color='#2c6fbb', linewidth=2.2, markersize=9, label='Pocock')
ax.plot(t, z_obf, 's-', color='#c0392b', linewidth=2.2, markersize=9,
        label="O'Brien-Fleming")
ax.axhline(1.96, color='#888888', linestyle='--', linewidth=1.6,
           label='무보정 1.96')
for i in range(K):
    ax.annotate(f"p={p_obf[i]:.4f}", (t[i], z_obf[i]),
                textcoords='offset points', xytext=(-6, 12), fontsize=8.5,
                color='#c0392b', ha='center')
ax.set_xlabel('정보 분획 (모은 표본 / 계획 표본)')
ax.set_ylabel('중단에 필요한 |z|')
ax.set_xticks(t); ax.set_xticklabels([f"{v:.2f}" for v in t]); ax.set_ylim(1.5, 4.2)
ax.set_title('(a) 언제 멈출 수 있는가', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
ks = [r[0] for r in peek_rows]
errs = [r[1] for r in peek_rows]
ax.plot(ks, errs, 'o-', color='#c0392b', linewidth=2.2, markersize=9)
ax.axhline(ALPHA, color='#2e8b57', linestyle='--', linewidth=1.8, label='명목 α = 0.05')
for k, e in peek_rows:
    ax.annotate(f"{e:.3f}", (k, e), textcoords='offset points', xytext=(0, 10),
                fontsize=9, ha='center')
ax.set_xlabel('중간에 들여다본 횟수 K')
ax.set_ylabel('실제 1종 오류')
ax.set_ylim(0, 0.30)
ax.set_title('(b) 보정 없이 자주 보면 위양성이 는다', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[2]
labels = [r['방법'] for r in sim_rows]
vals = [r['1종 오류'] for r in sim_rows]
bars = ax.bar(range(3), vals, color=['#888888', '#2c6fbb', '#c0392b'], width=0.55)
ax.axhline(ALPHA, color='#2e8b57', linestyle='--', linewidth=1.8, label='명목 α = 0.05')
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
            ha='center', fontsize=9.5, fontweight='bold')
ax.set_xticks(range(3))
ax.set_xticklabels(['무보정', 'Pocock', "O'Brien-\nFleming"])
ax.set_ylabel('1종 오류 (이항 시뮬레이션)')
ax.set_ylim(0, max(vals) * 1.35)
ax.set_title(f'(c) 두 안이 같을 때 잘못 멈춘 비율 ({N_SIM:,}회)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
png = pathlib.Path(__file__).with_name('8-2-group-sequential-design.png')
plt.savefig(png, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n그림 저장: {png.name}")

print("\n" + "=" * 80)
print("정리")
print("=" * 80)
print("1. 보정 없이 중간에 보면 1종 오류가 오른다. K=5에서 0.05가 아니라 0.14다.")
print("2. 중간 분석은 횟수와 시점을 실험 시작 전에 정하고 그대로 지킨다.")
print("3. Pocock은 모든 단계에서 같은 경계를 쓴다. 초기에 멈출 여지가 크다.")
print("4. O'Brien-Fleming은 1단계에서 극도로 엄격하고 마지막에 1.96 근처로 온다.")
print("=" * 80)
