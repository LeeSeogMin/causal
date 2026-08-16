"""
7장 강의노트용 개념도 생성
==========================

실행: python diagrams/make_chapter07.py
출력: diagrams/7-1.png ~ 7-5.png

- 7-1.png: 평균이 같아도 분포가 다르다
- 7-2.png: 분위수 읽는 법 (소득 줄세우기)
- 7-3.png: 분위별로 효과가 달라지는 정책 (최저임금)
- 7-4.png: 체크 함수 - OLS와 분위 회귀가 벌하는 방식
- 7-5.png: 질문에 따라 방법을 고르는 순서
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))

BLUE = '#d9e8ff'; BLUE_E = '#2c6fbb'
GREEN = '#d9f2d9'; GREEN_E = '#2e8b57'
RED = '#ffd9d9'; RED_E = '#c0392b'
YELLOW = '#fff2cc'; YELLOW_E = '#d6a300'
GRAY = '#eeeeee'; GRAY_E = '#888888'


def box(ax, x, y, w, h, text, fc, ec, fs=11, bold=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor=fc, edgecolor=ec, linewidth=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal')


def arrow(ax, p1, p2, color='#333333', style='-', label=None,
          lx=0, ly=0, lfs=10, lcolor=None, width=2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=20,
                                 color=color, linewidth=width, linestyle=style,
                                 shrinkA=6, shrinkB=6))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly
        ax.text(mx, my, label, ha='center', va='center', fontsize=lfs,
                color=lcolor or color, fontweight='bold')


# ---------------------------------------------------------------------------
# 7-1. 평균이 같아도 분포가 다르다
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
n = 40000
# 마을 A: 모두 비슷한 소득
a = rng.normal(300, 35, n)
# 마을 B: 평균은 같지만 저소득과 고소득으로 갈라짐
b = np.concatenate([rng.normal(215, 30, n // 2), rng.normal(385, 30, n // 2)])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
bins = np.linspace(120, 500, 70)

for ax, data, name, note in [
        (axes[0], a, '마을 A', '한 봉우리, 폭이 좁다'),
        (axes[1], b, '마을 B', '두 봉우리로 갈라져 있다')]:
    ax.hist(data, bins=bins, density=True, color='#cccccc', edgecolor='#555555',
            linewidth=0.4)
    ax.axvline(data.mean(), color=RED_E, linewidth=2.5, linestyle='--')
    ax.text(data.mean() + 6, ax.get_ylim()[1] * 0.92,
            f'평균 {data.mean():.0f}만원', color=RED_E, fontsize=10.5,
            fontweight='bold', va='top')
    p10, p90 = np.percentile(data, [10, 90])
    ax.axvline(p10, color=BLUE_E, linewidth=2)
    ax.axvline(p90, color=BLUE_E, linewidth=2)
    ax.text(p10, -ax.get_ylim()[1] * 0.11, f'하위 10%\n{p10:.0f}',
            color=BLUE_E, fontsize=9.5, ha='center', va='top')
    ax.text(p90, -ax.get_ylim()[1] * 0.11, f'상위 10%\n{p90:.0f}',
            color=BLUE_E, fontsize=9.5, ha='center', va='top')
    ax.set_title(f'{name} — {note}', fontsize=12, fontweight='bold')
    ax.set_xlabel('월 소득 (만원)', fontsize=10.5)
    ax.set_yticks([])
    ax.set_xlim(120, 500)

axes[0].set_ylabel('사람 수', fontsize=10.5)
fig.suptitle('그림 7-1. 두 마을의 평균 소득은 같다. 사는 모습은 같지 않다.',
             fontsize=13, fontweight='bold', y=1.02)
fig.text(0.5, -0.10,
         '평균 하나만 보고하면 두 마을이 구별되지 않는다. '
         '하위 10%와 상위 10%를 같이 보면 갈라진다.',
         ha='center', fontsize=10.5, color=RED_E)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '7-1.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 7-2. 분위수 읽는 법
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.set_xlim(0, 10.6)
ax.set_ylim(-1.5, 4.4)
ax.axis('off')

# 사람 100명을 줄세운 막대
xs = np.linspace(0.6, 9.8, 100)
vals = 150 + 260 * (np.linspace(0, 1, 100) ** 1.9)
ax.bar(xs, vals / 130, width=0.075, color='#dddddd', edgecolor='#999999',
       linewidth=0.3, bottom=0)

for pos, tau, name, col in [(0.10, 0.10, '하위 10%\n(τ=0.10)', BLUE_E),
                            (0.50, 0.50, '중위\n(τ=0.50)', GREEN_E),
                            (0.90, 0.90, '상위 10%\n(τ=0.90)', RED_E)]:
    xi = 0.6 + 9.2 * pos
    yi = (150 + 260 * (pos ** 1.9)) / 130
    ax.plot([xi, xi], [0, yi], color=col, linewidth=3)
    ax.plot(xi, yi, 'o', color=col, markersize=9)
    ax.text(xi, yi + 0.30, f'{150 + 260 * (pos ** 1.9):.0f}만원',
            ha='center', fontsize=10.5, color=col, fontweight='bold')
    ax.text(xi, -0.72, name, ha='center', va='center', fontsize=10, color=col)

ax.annotate('', xy=(9.9, -0.05), xytext=(0.5, -0.05),
            arrowprops=dict(arrowstyle='-|>', color='#333333', linewidth=1.8))
ax.text(0.5, -1.28, '소득이 낮은 사람', fontsize=10, ha='left')
ax.text(9.9, -1.28, '소득이 높은 사람', fontsize=10, ha='right')
ax.text(5.2, 4.05, '100명을 소득 순으로 줄 세운다. τ분위수는 앞에서 τ×100번째 사람의 소득이다.',
        ha='center', fontsize=11)
ax.set_title('그림 7-2. 분위수는 줄 세운 뒤 몇 번째 사람인지를 가리킨다',
             fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '7-2.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 7-3. 분위별로 효과가 달라지는 정책
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 4.6))

tau = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
eff = np.array([750, 640, 590, 400, 200, 130])
mean_eff = 400

ax.plot(tau, eff, 'o-', color='black', linewidth=2.5, markersize=9,
        label='분위별 효과')
ax.axhline(mean_eff, color=RED_E, linestyle='--', linewidth=2.5,
           label=f'평균 효과 {mean_eff}원')
ax.fill_between(tau, mean_eff, eff, where=(eff > mean_eff), color=BLUE,
                alpha=0.7)
ax.fill_between(tau, mean_eff, eff, where=(eff < mean_eff), color=RED,
                alpha=0.7)

ax.annotate('평균보다 훨씬 큰 효과', xy=(0.09, 660), xytext=(0.16, 780),
            fontsize=10.5, color=BLUE_E, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BLUE_E, linewidth=1.6))
ax.annotate('평균보다 훨씬 작은 효과', xy=(0.87, 145), xytext=(0.50, 90),
            fontsize=10.5, color=RED_E, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED_E, linewidth=1.6))

ax.set_xlabel('임금 분위 τ (0에 가까울수록 저임금)', fontsize=11)
ax.set_ylabel('시간당 임금 상승분 (원)', fontsize=11)
ax.set_ylim(40, 880)
ax.set_xlim(0, 1)
ax.legend(fontsize=10.5, loc='upper right')
ax.grid(alpha=0.3)
ax.set_title('그림 7-3. 최저임금 인상은 저임금 쪽에서 크게, 고임금 쪽에서 작게 작동한다',
             fontsize=12.5, fontweight='bold', pad=10)
fig.text(0.5, -0.04,
         '평균 400원 하나만 보고하면 750원과 130원이 같은 숫자로 뭉개진다. (모양을 보이려고 그린 예시 값)',
         ha='center', fontsize=10.5, color='#333333')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '7-3.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 7-4. 체크 함수
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
u = np.linspace(-3, 3, 400)

panels = [
    (axes[0], u ** 2, 'OLS: 오차의 제곱', '위아래를 똑같이 벌한다\n→ 평균으로 끌린다', GRAY_E),
    (axes[1], np.where(u > 0, 0.5 * u, -0.5 * u), '분위 회귀 τ=0.50',
     '위아래를 똑같이 벌하되 제곱하지 않는다\n→ 중위수로 끌린다', GREEN_E),
    (axes[2], np.where(u > 0, 0.9 * u, -0.1 * u), '분위 회귀 τ=0.90',
     '넘친 쪽(u>0)을 9배 더 벌한다\n→ 상위 10% 선으로 끌린다', RED_E),
]

for ax, y, title, note, col in panels:
    ax.plot(u, y, color=col, linewidth=2.8)
    ax.axvline(0, color='#888888', linewidth=1)
    ax.axhline(0, color='#888888', linewidth=1)
    ax.set_title(title, fontsize=11.5, fontweight='bold')
    ax.set_xlabel('오차 u = 실제값 − 예측값', fontsize=10)
    ax.text(0, -0.62, note, transform=ax.transAxes, fontsize=10,
            ha='left', va='top', color=col)
    ax.set_xlim(-3, 3)
    ax.set_yticks([])
    ax.grid(alpha=0.25)

axes[0].set_ylabel('벌점(손실)', fontsize=10.5)
fig.suptitle('그림 7-4. 어느 쪽 오차를 더 아프게 벌하느냐로 목표 지점이 정해진다',
             fontsize=13, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '7-4.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 7-5. 질문에 따라 방법을 고르는 순서
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.6, 6.6))
ax.set_xlim(0, 11.6)
ax.set_ylim(0, 7.4)
ax.axis('off')

box(ax, 5.8, 6.75, 7.4, 0.8, '무엇을 묻고 있는가', YELLOW, YELLOW_E, fs=12)

box(ax, 2.0, 5.0, 3.6, 1.05,
    '같은 조건인 사람 안에서\n분위별 차이', BLUE, BLUE_E, fs=10.5)
box(ax, 6.6, 5.0, 3.6, 1.05,
    '모집단 전체 분위수가\n정책에 반응하는 크기', GREEN, GREEN_E, fs=10.5)
box(ax, 10.3, 5.0, 2.4, 1.05,
    '처치군의\n분위별 변화', RED, RED_E, fs=10.5)

arrow(ax, (4.2, 6.32), (2.4, 5.58))
arrow(ax, (6.0, 6.32), (6.4, 5.58))
arrow(ax, (8.6, 6.32), (10.1, 5.58))

box(ax, 2.0, 3.0, 3.6, 1.05, '분위 회귀 quantreg\n실습 1 (7-1)', GRAY, GRAY_E, fs=10.5)
box(ax, 6.6, 3.0, 3.6, 1.05, 'RIF 회귀\n실습 2 (7-2)', GRAY, GRAY_E, fs=10.5)
box(ax, 10.3, 3.0, 2.4, 1.05, 'DID·CIC\n실습 3 (7-3)', GRAY, GRAY_E, fs=10.5)

arrow(ax, (2.0, 4.44), (2.0, 3.58))
arrow(ax, (6.6, 4.44), (6.6, 3.58))
arrow(ax, (10.3, 4.44), (10.3, 3.58))

box(ax, 3.3, 1.35, 5.4, 0.95,
    '관계가 곡선이거나 변수가 많으면\nQRF·LightGBM 실습 4 (7-4)', BLUE, BLUE_E, fs=10.5)
box(ax, 9.0, 1.35, 4.6, 0.95,
    '보고서에는 분위별 표와\n평균 효과를 함께 싣는다', GREEN, GREEN_E, fs=10.5)

arrow(ax, (2.0, 2.44), (3.0, 1.85))
arrow(ax, (6.6, 2.44), (5.6, 1.85))
arrow(ax, (10.3, 2.44), (9.4, 1.85))

ax.text(5.8, 0.30, '평균 하나만 실은 보고서는 이 표의 어느 칸도 답하지 못한다',
        ha='center', fontsize=11, color=RED_E, fontweight='bold')
ax.set_title('그림 7-5. 질문이 정해지면 방법이 정해진다',
             fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '7-5.png'), dpi=200, bbox_inches='tight')
plt.close()

print("생성 완료:")
for f in ['7-1.png', '7-2.png', '7-3.png', '7-4.png', '7-5.png']:
    p = os.path.join(OUT, f)
    print(f"  {f}  ({os.path.getsize(p) / 1024:.0f} KB)")
