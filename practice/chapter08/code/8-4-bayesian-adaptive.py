#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제8장 실습 4: 베이지안 적응형 설계로 중단 시점 정하기
=====================================================

목적
- 중간 분석마다 P(B안이 A안보다 낫다)를 계산한다.
- 그 확률로 조기 중단(효능 / 무용성)을 판정한다.
- 세 시나리오에서 최종 표본 크기가 어떻게 달라지는지 본다.

데이터: practice/chapter08/data/8-4-bayesian-adaptive.csv
출력 그림: 8-4-bayesian-adaptive.png

수정 이력
- 2026-08-17
  (1) [근본 원인] CSV의 participant_id가 시나리오별이 아니라 파일 전체에서
      1~600으로 이어져 있다(large_effect 1~200, null_effect 201~400,
      medium_effect 401~600). 그런데 코드는 시나리오마다
      participant_id <= 50/100/150/200으로 잘라 냈다.
      null_effect와 medium_effect는 매번 0행이 잡혔고, Beta(1,1) 대 Beta(1,1)
      비교가 되어 사후확률이 언제나 0.5 근처로 찍혔다.
      직전 실행에서 medium_effect의 사후확률이 n=50,100,150,200에서
      0.4913 / 0.5010 / 0.4878 / 0.5068이었던 것은 데이터를 하나도
      안 보고 사전분포만 비교한 결과였다.
      → 시나리오 안에서 1부터 다시 세는 순번(seq)을 만들어 자르도록 고쳤다.
  (2) [근본 원인] CSV가 대조군 100명 뒤에 처치군 100명이 오는 순서로
      정렬돼 있었다. 순차 실험은 두 군이 번갈아 도착해야 한다.
      정렬된 채로 자르면 n=100 시점에 처치군이 0명이라 비교 자체가 안 된다.
      large_effect가 n=50, n=100에서 0.5925, 0.6092로 나온 원인이 이것이다.
      → 군별로 섞은 뒤 1:1로 번갈아 세우는 도착 순서를 seed 고정으로 만든다.
        관측값(outcome)은 그대로 쓰고 도착 순서만 바로잡는다.
  (3) 사후확률 몬테카를로에 시드가 없어 실행마다 값이 흔들렸다.
      → 시드를 고정했다. 도착 순서 시드도 hash()가 아니라 상수로 적는다
        (파이썬 문자열 hash는 실행마다 값이 달라진다).
  (4) 그림 출력이 없었다. 2개 패널 PNG를 저장한다.

저자: AI 기반 정책분석방법론
날짜: 2025-11-20 (최종 수정 2026-08-17)
"""

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

THETA_EFFICACY = 0.99      # 이 이상이면 B안 채택하고 중단
THETA_FUTILITY = 0.05      # 이 이하이면 B안 접고 중단
INTERIM_POINTS = [50, 100, 150, 200]
MC = 200_000

print("=" * 80)
print("실습 4. 베이지안 적응형 설계로 중단 시점 정하기")
print("=" * 80)

base_dir = pathlib.Path(__file__).resolve().parents[1]
data_path = base_dir / 'data' / '8-4-bayesian-adaptive.csv'
df = pd.read_csv(data_path)

print("\n[0단계] 데이터 로드")
print("-" * 80)
print(f"  파일: {data_path.name}")
print(f"  총 행 수: {len(df)}")
print(f"  시나리오: {list(df['scenario'].unique())}")
print(f"  participant_id 범위: {df['participant_id'].min()}~{df['participant_id'].max()}"
      "  (시나리오별이 아니라 파일 전체 번호다)")


def arrival_order(d, seed=0):
    """대조군과 처치군을 1:1로 번갈아 세운 도착 순서를 만든다."""
    rng = np.random.default_rng(seed)
    ctrl = d[d['treatment'] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
    trt = d[d['treatment'] == 1].sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    m = min(len(ctrl), len(trt))
    rows = []
    for i in range(m):
        pair = [ctrl.iloc[i], trt.iloc[i]]
        if rng.random() < 0.5:                 # 쌍 안에서 순서도 무작위
            pair = pair[::-1]
        rows.extend(pair)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out['seq'] = np.arange(1, len(out) + 1)    # 시나리오 안에서 1부터 다시 센다
    return out


rng_mc = np.random.default_rng(2026)

print(f"\n[1단계] 중단 규칙")
print("-" * 80)
print(f"  P(p_B > p_A | 데이터) >= {THETA_EFFICACY}  →  B안 채택하고 중단")
print(f"  P(p_B > p_A | 데이터) <= {THETA_FUTILITY}  →  B안 접고 중단")
print(f"  그 사이면 계속 모집. 중간 분석 시점: {INTERIM_POINTS}")
print(f"  사전분포는 두 군 모두 Beta(1,1) = 균등분포")

scenario_order = ['large_effect', 'medium_effect', 'null_effect']
scenario_ko = {'large_effect': '대효과', 'medium_effect': '중효과', 'null_effect': '무효과'}
# 도착 순서용 시드. hash()는 실행마다 달라지므로 값을 직접 적어 둔다.
scenario_seed = {'large_effect': 11, 'medium_effect': 22, 'null_effect': 33}

summary = []
traces = {}

for sid in scenario_order:
    d = arrival_order(df[df['scenario'] == sid], seed=scenario_seed[sid])
    p_c_true = d['p_control_true'].iloc[0]
    p_t_true = d['p_treatment_true'].iloc[0]

    print(f"\n[2단계-{scenario_ko[sid]}] {sid}")
    print("-" * 80)
    print(f"  참 전환율: A안 {p_c_true:.2f}, B안 {p_t_true:.2f} "
          f"(참 효과 {p_t_true - p_c_true:+.2f})")
    print(f"  {'누적 n':>8}{'A안 성공/인원':>15}{'B안 성공/인원':>15}"
          f"{'P(p_B>p_A)':>13}{'결정':>16}")

    trace = []
    stopped, reason, final_n, final_pp = False, None, None, None

    for n in INTERIM_POINTS:
        d_n = d[d['seq'] <= n]
        ctrl, trt = d_n[d_n['treatment'] == 0], d_n[d_n['treatment'] == 1]
        s_c, n_c = int(ctrl['outcome'].sum()), len(ctrl)
        s_t, n_t = int(trt['outcome'].sum()), len(trt)

        # Beta(1,1) 사전분포 + 이항 관측 → Beta(1+성공, 1+실패)
        pa = rng_mc.beta(1 + s_c, 1 + n_c - s_c, MC)
        pb = rng_mc.beta(1 + s_t, 1 + n_t - s_t, MC)
        pp = float(np.mean(pb > pa))

        if pp >= THETA_EFFICACY:
            decision, stopped, reason = 'B안 채택·중단', True, 'efficacy'
        elif pp <= THETA_FUTILITY:
            decision, stopped, reason = 'B안 폐기·중단', True, 'futility'
        else:
            decision = '계속'

        trace.append({'n': n, 'pp': pp, 'decision': decision})
        print(f"  {n:>8}{f'{s_c}/{n_c}':>15}{f'{s_t}/{n_t}':>15}"
              f"{pp:>13.4f}{decision:>16}")

        final_n, final_pp = n, pp
        if stopped:
            break

    traces[sid] = trace
    summary.append({
        '시나리오': scenario_ko[sid],
        '참 A': p_c_true, '참 B': p_t_true,
        '최종 n': final_n,
        '조기 중단': '예' if stopped else '아니오',
        '이유': reason or '-',
        '최종 사후확률': final_pp,
        '절약한 인원': 200 - final_n,
    })

print("\n[3단계] 세 시나리오 요약")
print("-" * 80)
sm = pd.DataFrame(summary)
print(sm.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

saved = int(sm['절약한 인원'].sum())
print(f"\n  세 실험을 다 돌렸다면 600명이 필요했다. 실제로 쓴 인원 {600 - saved}명.")
print(f"  {saved}명을 아꼈다.")

print("\n[4단계] 판정이 뒤집히는 지점")
print("-" * 80)
for sid in scenario_order:
    tr = traces[sid]
    print(f"  {scenario_ko[sid]:>4}: " +
          " → ".join(f"n={x['n']} {x['pp']:.3f}" for x in tr))
print(f"\n  효능 기준을 {THETA_EFFICACY}에서 0.95로 낮추면 언제 멈추는가:")
for sid in scenario_order:
    hit = [x['n'] for x in traces[sid] if x['pp'] >= 0.95]
    print(f"    {scenario_ko[sid]:>4}: " +
          (f"n={hit[0]}에서 중단" if hit else "중단 없음 (200명 완주)"))

# ---------------------------------------------------------------------------
# 5단계: 그림
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
colors = {'large_effect': '#2e8b57', 'medium_effect': '#d6a300', 'null_effect': '#c0392b'}

ax = axes[0]
for sid in scenario_order:
    tr = traces[sid]
    ax.plot([x['n'] for x in tr], [x['pp'] for x in tr], 'o-',
            color=colors[sid], linewidth=2.2, markersize=9,
            label=f"{scenario_ko[sid]} (참 효과 "
                  f"{sm[sm['시나리오'] == scenario_ko[sid]]['참 B'].iloc[0] - sm[sm['시나리오'] == scenario_ko[sid]]['참 A'].iloc[0]:+.2f})")
    for x in tr:
        ax.annotate(f"{x['pp']:.3f}", (x['n'], x['pp']),
                    textcoords='offset points', xytext=(0, 9), fontsize=8,
                    ha='center', color=colors[sid])
ax.axhline(THETA_EFFICACY, color='#2e8b57', linestyle='--', linewidth=1.8)
ax.axhline(THETA_FUTILITY, color='#c0392b', linestyle='--', linewidth=1.8)
ax.text(202, THETA_EFFICACY, ' 효능 0.99', va='center', fontsize=9, color='#2e8b57')
ax.text(202, THETA_FUTILITY, ' 무용성 0.05', va='center', fontsize=9, color='#c0392b')
ax.set_xlabel('누적 표본 크기'); ax.set_ylabel('P(p_B > p_A | 데이터)')
ax.set_xlim(30, 245); ax.set_ylim(-0.05, 1.12)
ax.set_xticks(INTERIM_POINTS)
ax.set_title('(a) 중간 분석마다 계산한 사후확률', fontweight='bold')
ax.legend(fontsize=9, loc='center left'); ax.grid(alpha=0.3)

ax = axes[1]
labs = [r['시나리오'] for r in summary]
used = [r['최종 n'] for r in summary]
xs = np.arange(len(labs))
ax.bar(xs - 0.2, [200] * len(labs), width=0.4, color='#888888', label='고정 설계 200명')
ax.bar(xs + 0.2, used, width=0.4,
       color=[colors[s] for s in scenario_order], label='적응형 설계 실제 사용')
for i, v in enumerate(used):
    ax.text(i + 0.2, v + 4, f"{v}명", ha='center', fontsize=10, fontweight='bold')
    if v < 200:
        ax.text(i, 210, f"{200 - v}명 절약", ha='center', fontsize=9, color='#2e8b57')
ax.set_xticks(xs); ax.set_xticklabels(labs)
ax.set_ylabel('사용한 표본 크기'); ax.set_ylim(0, 240)
ax.set_title('(b) 효과가 크면 일찍 멈춘다', fontweight='bold')
ax.legend(fontsize=9, loc='lower right'); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
png = pathlib.Path(__file__).with_name('8-4-bayesian-adaptive.png')
plt.savefig(png, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n그림 저장: {png.name}")

print("\n" + "=" * 80)
print("정리")
print("=" * 80)
print("1. 사후확률은 'B안이 나을 확률'로 바로 읽는다. p값과 달리 해석에 조건이 붙지 않는다.")
print("2. 효과가 크면 사후확률이 빨리 0.99를 넘어 표본을 아낀다.")
print("3. 첫 중간 분석은 각 군 25명뿐이라 흔들린다. 무효과 시나리오가")
print("   n=50에서 0.042로 떨어져 B안을 접었는데, 참 효과는 0이 아니라 '차이 없음'이었다.")
print("   중간 분석을 너무 이른 시점에 두면 이런 판정이 나온다.")
print("4. 중단 기준(0.99 / 0.05)과 중간 분석 시점은 실험 전에 정해 두고 바꾸지 않는다.")
print("=" * 80)
