#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제8장 실습 5: 콘텐츠 5개를 놓고 배분 비율을 바꿔 가는 서비스 실험
==================================================================

목적
- 선택지가 2개가 아니라 5개일 때 적응형 배분이 어떻게 움직이는지 본다.
- 고정 배분과 비교해 성공자 수가 얼마나 늘었는지 같은 기준으로 잰다.
- 배분이 적었던 콘텐츠의 추정값이 왜 못 미더운지 확인한다.

사례: 가상의 불면증 케어 앱. 콘텐츠 5종 중 수면 개선 반응이 좋은 것을 찾는다.
데이터: practice/chapter08/data/8-5-korean-dtx.csv (톰슨 샘플링 800명 기록)
출력 그림: 8-5-korean-dtx-case.png

수정 이력
- 2026-08-17
  (1) [근본 원인] arm_names = df['strategy_name'].unique()가 '처음 나온 순서'로
      이름을 돌려준다. CSV의 첫 행이 strategy=1(호흡 훈련)이라
      arm_names[0]='호흡 훈련'이 됐는데, 코드는 enumerate로 0번 이름을
      strategy=0(인지 재구성)의 숫자와 짝지었다.
      결과표의 이름이 전부 한 칸씩 밀려서, 최고 전략을 '운동 독려'(참 0.35, 실제로는
      최저)로, 최저 전략을 '수면 개선'(참 0.48)으로 뒤집어 보고했다.
      → strategy 열을 키로 이름을 뽑아 짝을 맞췄다. 바로잡으면 최고 전략은
        마음챙김(참 0.52, 354명), 최저는 운동 독려(참 0.35, 89명)다.
  (2) [근본 원인] 성과 비교에서 적응형은 실현 성공자(382명), 고정 배분은
      기댓값(348.8명)을 써서 서로 다른 종류의 수를 뺐다.
      운이 좋으면 개선폭이 부풀고 나쁘면 줄어든다.
      → 배분 인원만으로 정해지는 기대 성공자를 함께 계산해 두 기준을 나란히 적는다.
  (3) 그림 출력이 없어 배분 비율의 시간 변화를 볼 수 없었다.
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
print("실습 5. 콘텐츠 5개를 놓고 배분 비율을 바꿔 가는 서비스 실험")
print("=" * 80)

base_dir = pathlib.Path(__file__).resolve().parents[1]
data_path = base_dir / 'data' / '8-5-korean-dtx.csv'
df = pd.read_csv(data_path)

print("\n[0단계] 데이터 로드")
print("-" * 80)
print(f"  파일: {data_path.name}")
print(f"  총 참가자 수: {len(df)}명")
print(f"  콘텐츠 수: {df['strategy'].nunique()}개")

# strategy 열을 키로 이름과 참 성공률을 뽑는다. unique() 순서에 기대지 않는다.
info = (df.groupby('strategy')
          .agg(name=('strategy_name', 'first'), true_rate=('true_rate', 'first'),
               n=('success', 'size'), s=('success', 'sum'))
          .sort_index())
names = info['name'].tolist()
true_rates = info['true_rate'].to_numpy()
counts = info['n'].to_numpy()
successes = info['s'].to_numpy()

K = len(names)
N = len(df)
best = int(np.argmax(true_rates))
worst = int(np.argmin(true_rates))

print("\n[1단계] 콘텐츠별 참 성공률 (수면 개선 반응률)")
print("-" * 80)
for i in range(K):
    mark = '  ← 가장 높음' if i == best else ('  ← 가장 낮음' if i == worst else '')
    print(f"  strategy {i}  {names[i]:<8} {true_rates[i]:.2f}{mark}")

# ---------------------------------------------------------------------------
# 2단계: 배분 비율의 시간 변화
# ---------------------------------------------------------------------------
print("\n[2단계] 중간 시점별 배분 비율")
print("-" * 80)
checkpoints = [200, 400, 600, 800]
print(f"  {'콘텐츠':<10}{'참 성공률':>10}" + "".join(f"{'n=' + str(c):>10}" for c in checkpoints))
for i in range(K):
    row = []
    for c in checkpoints:
        head = df[df['participant_id'] <= c]
        row.append((head['strategy'] == i).sum() / c)
    print(f"  {names[i]:<10}{true_rates[i]:>10.2f}" + "".join(f"{v:>10.1%}" for v in row))
print(f"  {'고정 배분이면':<10}{'':>10}" + "".join(f"{1/K:>10.1%}" for _ in checkpoints))

print("\n  참 성공률이 높은 콘텐츠의 비율이 오르고 낮은 쪽이 줄어든다.")
print("  다만 순서가 완전히 맞지는 않는다. 5단계에서 그 이유를 본다.")

# ---------------------------------------------------------------------------
# 3단계: 최종 결과표
# ---------------------------------------------------------------------------
print("\n[3단계] 최종 결과표 (n=800)")
print("-" * 80)
post_mean = (1 + successes) / (2 + counts)
ci = np.array([beta_dist.ppf([0.025, 0.975], 1 + successes[i],
                             1 + counts[i] - successes[i]) for i in range(K)])
table = pd.DataFrame({
    '콘텐츠': names,
    '배분 인원': counts,
    '배분 비율': [f"{c/N:.1%}" for c in counts],
    '성공자': successes,
    '사후평균': np.round(post_mean, 3),
    '참 성공률': true_rates,
    '95% 신용구간': [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in ci],
})
print(table.to_string(index=False))

# ---------------------------------------------------------------------------
# 4단계: 성과 비교
# ---------------------------------------------------------------------------
print("\n[4단계] 고정 배분과 비교")
print("-" * 80)

realized = int(successes.sum())
expected_adaptive = float(np.sum(counts * true_rates))     # 실제 배분의 기대 성공자
expected_fixed = float(np.mean(true_rates) * N)            # 1:1:1:1:1의 기대 성공자
expected_optimal = float(true_rates[best] * N)             # 전부 최고 콘텐츠

print(f"  적응형 배분, 실제 성공자      : {realized}명")
print(f"  적응형 배분, 기대 성공자      : {expected_adaptive:.1f}명")
print(f"  고정 배분(1:1:1:1:1) 기대 성공자: {expected_fixed:.1f}명")
print(f"  전부 최고 콘텐츠일 때 기대     : {expected_optimal:.1f}명")
print()
print(f"  기대값끼리 비교(운의 영향 없음): {expected_fixed:.1f} → {expected_adaptive:.1f}명 "
      f"(+{expected_adaptive - expected_fixed:.1f}명, "
      f"+{(expected_adaptive/expected_fixed - 1)*100:.1f}%)")
print(f"  실현값과 고정 배분 기대값 비교 : {expected_fixed:.1f} → {realized}명 "
      f"(+{realized - expected_fixed:.1f}명, "
      f"+{(realized/expected_fixed - 1)*100:.1f}%)")
print()
print(f"  두 수치가 다른 이유: 앞은 배분만 보고, 뒤는 이번 실행의 운까지 섞여 있다.")
print(f"  보고서에는 기댓값끼리 비교한 앞 숫자를 쓴다.")

gap_closed = (expected_adaptive - expected_fixed) / (expected_optimal - expected_fixed)
print(f"\n  최적까지의 격차 중 {gap_closed:.1%}를 메웠다 "
      f"(고정 {expected_fixed:.1f} → 적응형 {expected_adaptive:.1f} → 최적 {expected_optimal:.1f})")

print(f"\n  사용자 경험 쪽 계산")
fixed_per_arm = N / K
print(f"  가장 나쁜 콘텐츠({names[worst]}, 참 {true_rates[worst]:.2f}) 노출:")
print(f"    고정 배분이면 {fixed_per_arm:.0f}명 → 적응형은 {counts[worst]}명")
print(f"    {fixed_per_arm - counts[worst]:.0f}명이 더 나은 콘텐츠를 받았다.")

# ---------------------------------------------------------------------------
# 5단계: 배분 순서가 참값 순서와 어긋나는 지점
# ---------------------------------------------------------------------------
print("\n[5단계] 배분 순서와 참값 순서 비교")
print("-" * 80)
order_true = np.argsort(-true_rates)
order_alloc = np.argsort(-counts)
print(f"  참 성공률 순위 : " + " > ".join(f"{names[i]}({true_rates[i]:.2f})" for i in order_true))
print(f"  배분 인원 순위 : " + " > ".join(f"{names[i]}({counts[i]}명)" for i in order_alloc))
mismatch = [names[i] for i in range(K)
            if list(order_true).index(i) != list(order_alloc).index(i)]
print(f"  순위가 어긋난 콘텐츠: {', '.join(mismatch) if mismatch else '없음'}")
print()
print("  참 성공률 차이가 작은 콘텐츠끼리는 800명으로도 순위가 갈리지 않는다.")
print("  적응형 배분의 목적은 순위 매기기가 아니라 손실 줄이기다.")

# ---------------------------------------------------------------------------
# 6단계: 그림
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
palette = ['#2c6fbb', '#c0392b', '#2e8b57', '#d6a300', '#8e44ad']

ax = axes[0]
grid = np.arange(1, N + 1)
for i in range(K):
    cum = np.cumsum((df['strategy'] == i).values) / grid
    ax.plot(grid, cum, color=palette[i], linewidth=2,
            label=f'{names[i]} (참 {true_rates[i]:.2f})')
ax.axhline(1 / K, color='#888888', linestyle='--', linewidth=1.6,
           label=f'고정 배분 {1/K:.0%}')
ax.set_xlabel('참가자 수'); ax.set_ylabel('누적 배분 비율')
ax.set_ylim(0, 0.62)
ax.set_title('(a) 배분 비율이 시간에 따라 벌어진다', fontweight='bold')
ax.legend(fontsize=8.5, loc='upper right'); ax.grid(alpha=0.3)

ax = axes[1]
ys = np.arange(K)
ax.barh(ys, post_mean, color=palette, alpha=0.75, height=0.55, label='사후평균')
ax.errorbar(post_mean, ys, xerr=[post_mean - ci[:, 0], ci[:, 1] - post_mean],
            fmt='none', ecolor='#333333', capsize=5, linewidth=1.6)
ax.plot(true_rates, ys, '^', color='#c0392b', markersize=12, label='참 성공률')
for i in range(K):
    ax.text(0.02, i, f"n={counts[i]}", va='center', fontsize=9, color='white',
            fontweight='bold')
ax.set_yticks(ys); ax.set_yticklabels(names)
ax.set_xlabel('성공률'); ax.set_xlim(0, 0.75)
ax.set_title('(b) 배분이 적으면 구간이 넓다', fontweight='bold')
ax.legend(fontsize=9, loc='lower right'); ax.grid(axis='x', alpha=0.3)

ax = axes[2]
labels = ['고정 배분\n(기대)', '적응형 배분\n(기대)', '적응형 배분\n(실현)', '전부 최고\n(기대)']
vals = [expected_fixed, expected_adaptive, realized, expected_optimal]
bars = ax.bar(range(4), vals, color=['#888888', '#2c6fbb', '#5b9bd5', '#2e8b57'],
              width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.1f}",
            ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('성공자 수 (전체 800명 중)')
ax.set_ylim(300, expected_optimal * 1.09)
ax.set_title('(c) 고정 배분보다 얼마나 늘었나', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
png = pathlib.Path(__file__).with_name('8-5-korean-dtx-case.png')
plt.savefig(png, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n그림 저장: {png.name}")

print("\n" + "=" * 80)
print("정리")
print("=" * 80)
print(f"1. 최고 콘텐츠({names[best]})에 {counts[best]/N:.1%}를 보냈다. 고정 배분이면 {1/K:.0%}다.")
print(f"2. 기대 성공자가 {expected_fixed:.1f}명에서 {expected_adaptive:.1f}명으로 늘었다.")
print(f"3. 최저 콘텐츠 노출이 {fixed_per_arm:.0f}명에서 {counts[worst]}명으로 줄었다.")
print("4. 배분이 적은 콘텐츠는 신용구간이 넓다. 콘텐츠 간 효과 비교에는 쓰지 않는다.")
print("=" * 80)
