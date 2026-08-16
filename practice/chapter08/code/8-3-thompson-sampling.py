#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제8장 실습 3: 톰슨 샘플링으로 배분 비율을 바꿔 가며 실험하기
=============================================================

목적
- 시행이 쌓일수록 배분 비율이 어떻게 움직이는지 본다.
- 고정 배분과 톰슨 샘플링의 손실(regret)을 같은 기준으로 잰다.
- 적게 배분된 팔의 추정값이 왜 나빠지는지 확인한다.

데이터: practice/chapter08/data/8-3-thompson-sampling.csv
        (톰슨 샘플링을 T=1,000회 돌린 기록. 매 시행의 선택 arm과 보상)
출력 그림: 8-3-thompson-sampling.png

수정 이력
- 2026-08-17
  (1) [근본 원인] Regret을 '최적 팔의 기대보상 x T - 실제로 받은 보상'으로
      계산해 -22.0이라는 음수가 나왔다. 앞항은 기댓값, 뒷항은 실현값이라
      단위가 섞여 있었고, 운이 좋으면 음수가 나온다. 손실이 음수일 수는 없다.
      → Σ n_a x (p* - p_a) 형태의 기대 regret으로 바꿨다. 항상 0 이상이고
        '어느 팔에 몇 명을 보냈는가'만으로 정해져 배분 전략을 직접 잰다.
        실현 보상은 따로 별도 열에 적어 둔다.
  (2) [근본 원인] Uniform은 기댓값(416.7), 톰슨 샘플링은 실현값(572)으로
      계산해 서로 비교할 수 없는 두 수를 한 표에 올려놨다.
      → 세 방법 모두 기대 regret 기준으로 통일했다.
  (3) [근본 원인] Epsilon-Greedy가 np.argmax(보상합 / (횟수 + 1e-5))로
      평균을 구해, 한 번도 안 뽑은 팔과 뽑았지만 실패한 팔이 똑같이 0이 됐다.
      동점이면 argmax가 항상 인덱스 0을 골라 초기에 나쁜 팔에 고정됐다.
      → 각 팔을 한 번씩 먼저 뽑아 초기화하고 평균을 제대로 계산하도록 고쳤다.
  (4) 그림 출력이 없어 배분 비율의 시간 변화를 볼 수 없었다.
      → 3개 패널 PNG를 저장한다.

저자: AI 기반 정책분석방법론
날짜: 2025-11-20 (최종 수정 2026-08-17)
"""

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("실습 3. 톰슨 샘플링으로 배분 비율을 바꿔 가며 실험하기")
print("=" * 80)

# ---------------------------------------------------------------------------
# 0단계: 데이터 로드
# ---------------------------------------------------------------------------
base_dir = pathlib.Path(__file__).resolve().parents[1]
data_path = base_dir / 'data' / '8-3-thompson-sampling.csv'
df = pd.read_csv(data_path)

print("\n[0단계] 데이터 로드")
print("-" * 80)
print(f"  파일: {data_path.name}")
print(f"  총 시행 횟수: {len(df)}")
print(f"  팔(arm) 수: {df['arm_selected'].nunique()}")

true_probs = df.groupby('arm_selected')['true_prob'].first().sort_index().values
T = len(df)
K = len(true_probs)
best = int(np.argmax(true_probs))
p_star = true_probs[best]

print("\n[1단계] 설정")
print("-" * 80)
for i, p in enumerate(true_probs):
    mark = '  ← 최적' if i == best else ''
    print(f"  Arm {i} 참 전환율: {p:.2f}{mark}")
print(f"  총 시행 T = {T}")


def expected_regret(counts):
    """기대 손실 = 각 팔에 보낸 인원 x (최적 팔과의 전환율 차이)."""
    return float(np.sum(np.asarray(counts) * (p_star - true_probs)))


# ---------------------------------------------------------------------------
# 2단계: 톰슨 샘플링 결과
# ---------------------------------------------------------------------------
print("\n[2단계] 톰슨 샘플링 기록 요약")
print("-" * 80)

counts_ts = df['arm_selected'].value_counts().sort_index().reindex(
    range(K), fill_value=0).values
success_ts = df.groupby('arm_selected')['reward'].sum().reindex(
    range(K), fill_value=0).values
reward_ts = int(df['reward'].sum())

print(f"  {'Arm':>5}{'참 전환율':>11}{'배분 인원':>11}{'배분 비율':>11}"
      f"{'성공':>7}{'사후평균':>10}{'참값과 차이':>12}")
post_mean = (1 + success_ts) / (2 + counts_ts)
for i in range(K):
    print(f"  {i:>5}{true_probs[i]:>11.2f}{counts_ts[i]:>11}"
          f"{counts_ts[i] / T:>11.1%}{success_ts[i]:>7}"
          f"{post_mean[i]:>10.3f}{post_mean[i] - true_probs[i]:>+12.3f}")

print(f"\n  실현 보상: {reward_ts}/{T} = {reward_ts / T:.3f}")
print(f"  기대 손실(regret): {expected_regret(counts_ts):.2f}")
print(f"    = {counts_ts[0]}x{p_star - true_probs[0]:.2f} + "
      f"{counts_ts[1]}x{p_star - true_probs[1]:.2f} + "
      f"{counts_ts[2]}x{p_star - true_probs[2]:.2f}")

# ---------------------------------------------------------------------------
# 3단계: 배분 비율의 시간 변화
# ---------------------------------------------------------------------------
print("\n[3단계] 배분 비율이 시간에 따라 어떻게 바뀌었나")
print("-" * 80)
print(f"  {'시점 t':>8}" + "".join(f"{'Arm ' + str(i):>10}" for i in range(K)))
checkpoints = [50, 100, 200, 500, 1000]
for cp in checkpoints:
    head = df.iloc[:cp]
    row = [(head['arm_selected'] == i).sum() / cp for i in range(K)]
    print(f"  {cp:>8}" + "".join(f"{v:>10.1%}" for v in row))

print("\n  최적 팔(Arm 1)의 비율이 오르고 나머지가 줄어든다.")
print("  고정 배분이면 이 세 값이 계속 33.3%로 머문다.")

# ---------------------------------------------------------------------------
# 4단계: 세 배분 방식 비교 (같은 기준: 기대 손실)
# ---------------------------------------------------------------------------
print("\n[4단계] 세 배분 방식 비교")
print("-" * 80)

# 고정 배분 1:1:1
counts_uniform = np.full(K, T / K)

# Epsilon-Greedy (각 팔을 한 번씩 먼저 뽑아 초기화)
rng = np.random.default_rng(42)
EPS = 0.10
sums = np.zeros(K)
cnts = np.zeros(K)
for a in range(K):
    sums[a] += rng.binomial(1, true_probs[a])
    cnts[a] += 1
for _ in range(T - K):
    if rng.random() < EPS:
        a = rng.integers(K)
    else:
        a = int(np.argmax(sums / cnts))
    sums[a] += rng.binomial(1, true_probs[a])
    cnts[a] += 1
counts_eg = cnts
reward_eg = int(sums.sum())

comp = pd.DataFrame({
    '배분 방식': ['고정 1:1:1', f'Epsilon-Greedy (ε={EPS})', '톰슨 샘플링', '최적(항상 Arm 1)'],
    '최적팔 배분비율': [f"{counts_uniform[best] / T:.1%}",
                  f"{counts_eg[best] / T:.1%}",
                  f"{counts_ts[best] / T:.1%}", "100.0%"],
    '기대 손실': [expected_regret(counts_uniform), expected_regret(counts_eg),
              expected_regret(counts_ts), 0.0],
    '실현 보상': ['-', reward_eg, reward_ts, '-'],
})
print(comp.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

r_uni = expected_regret(counts_uniform)
r_ts = expected_regret(counts_ts)
print(f"\n  고정 배분 대비 손실 감소: {r_uni:.2f} → {r_ts:.2f} "
      f"({(1 - r_ts / r_uni):.1%} 감소)")
print(f"  뜻: 열등한 안을 겪는 사용자가 약 {r_uni - r_ts:.0f}명분 줄었다.")

# ---------------------------------------------------------------------------
# 5단계: 적게 배분된 팔의 추정은 나쁘다
# ---------------------------------------------------------------------------
print("\n[5단계] 적게 배분된 팔의 추정 정확도")
print("-" * 80)
print(f"  {'Arm':>5}{'배분 인원':>11}{'사후평균':>10}{'참값':>8}{'절대오차':>10}"
      f"{'95% 신용구간':>22}")
for i in range(K):
    a, b = 1 + success_ts[i], 1 + counts_ts[i] - success_ts[i]
    lo, hi = beta_dist.ppf([0.025, 0.975], a, b)
    print(f"  {i:>5}{counts_ts[i]:>11}{post_mean[i]:>10.3f}{true_probs[i]:>8.2f}"
          f"{abs(post_mean[i] - true_probs[i]):>10.3f}"
          f"{f'[{lo:.3f}, {hi:.3f}]':>22}")

print("\n  배분이 적은 팔일수록 오차가 크고 구간이 넓다.")
print("  적응형 배분은 최적 팔을 잘 찾는 대신 열등한 팔의 효과 추정을 포기한다.")

# ---------------------------------------------------------------------------
# 6단계: 그림
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
colors = ['#2c6fbb', '#c0392b', '#2e8b57']

ax = axes[0]
tgrid = np.arange(1, T + 1)
for i in range(K):
    cum = np.cumsum((df['arm_selected'] == i).values) / tgrid
    ax.plot(tgrid, cum, color=colors[i], linewidth=2,
            label=f'Arm {i} (참 {true_probs[i]:.2f})')
ax.axhline(1 / K, color='#888888', linestyle='--', linewidth=1.6,
           label='고정 배분 33.3%')
ax.set_xlabel('시행 t'); ax.set_ylabel('누적 배분 비율')
ax.set_ylim(0, 1.05)
ax.set_title('(a) 배분 비율이 시간에 따라 갈라진다', fontweight='bold')
ax.legend(fontsize=9, loc='center right'); ax.grid(alpha=0.3)

ax = axes[1]
gap = p_star - true_probs
reg_ts = np.cumsum(gap[df['arm_selected'].values])
ax.plot(tgrid, reg_ts, color='#2c6fbb', linewidth=2.2, label='톰슨 샘플링')
ax.plot(tgrid, tgrid * gap.mean(), color='#888888', linestyle='--',
        linewidth=2, label='고정 1:1:1')
ax.annotate(f"{reg_ts[-1]:.1f}", (T, reg_ts[-1]), textcoords='offset points',
            xytext=(-38, 10), fontsize=10, color='#2c6fbb', fontweight='bold')
ax.annotate(f"{T * gap.mean():.1f}", (T, T * gap.mean()),
            textcoords='offset points', xytext=(-40, -16), fontsize=10,
            color='#555555', fontweight='bold')
ax.set_xlabel('시행 t'); ax.set_ylabel('누적 기대 손실')
ax.set_title('(b) 누적 손실: 직선 vs 눕는 곡선', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[2]
xs = np.linspace(0, 1, 500)
for i in range(K):
    a, b = 1 + success_ts[i], 1 + counts_ts[i] - success_ts[i]
    ax.plot(xs, beta_dist.pdf(xs, a, b), color=colors[i], linewidth=2,
            label=f'Arm {i} (n={counts_ts[i]})')
    ax.axvline(true_probs[i], color=colors[i], linestyle=':', linewidth=1.6)
ax.set_xlabel('전환율'); ax.set_ylabel('사후분포 밀도')
ax.set_xlim(0, 1)
ax.set_title('(c) 점선이 참값. 배분이 적으면 빗나간다', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
png = pathlib.Path(__file__).with_name('8-3-thompson-sampling.png')
plt.savefig(png, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n그림 저장: {png.name}")

print("\n" + "=" * 80)
print("정리")
print("=" * 80)
print(f"1. 톰슨 샘플링은 최적 팔에 {counts_ts[best] / T:.1%}를 보냈다. 고정 배분은 33.3%다.")
print(f"2. 기대 손실이 {r_uni:.1f}에서 {r_ts:.1f}로 줄었다.")
print("3. 대신 열등한 팔의 전환율 추정이 크게 빗나간다. 효과 크기를 재려면 쓰지 않는다.")
print("=" * 80)
