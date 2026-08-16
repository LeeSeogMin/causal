"""
5장 강의노트용 개념도 생성
==========================

실행: python diagrams/make_chapter05.py
출력: diagrams/5-1.png ~ 5-5.png

- 5-1.png: Sharp RDD의 구조 (기준점 좌우 산점도와 수직 점프)
- 5-2.png: Sharp와 Fuzzy의 차이 (기준점에서 처치 확률이 얼마나 뛰는가)
- 5-3.png: 대역폭을 좁힐 때와 넓힐 때 무엇이 달라지는가
- 5-4.png: 배정 변수 밀도로 조작을 찾는 방법 (McCrary 검정의 발상)
- 5-5.png: RDD 분석 순서

주의: 5-1, 5-3, 5-4의 점과 곡선은 개념 설명용으로 이 파일 안에서 만든 값이다.
      실습 결과 숫자가 아니므로 추정값을 표시하지 않는다.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_installed = {f.name for f in fm.fontManager.ttflist}
for _name in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR']:
    if _name in _installed:
        plt.rcParams['font.family'] = _name
        break
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
# 5-1. Sharp RDD의 구조
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
c = 50.0
x = rng.uniform(30, 70, 320)
jump = 4.0
y = 20 + 0.30 * (x - c) + jump * (x >= c) + rng.normal(0, 1.6, len(x))

fig, ax = plt.subplots(figsize=(9.6, 5.6))
left, right = x < c, x >= c
ax.scatter(x[left], y[left], s=16, color=BLUE_E, alpha=0.45, label='대조군 (기준점 미만)')
ax.scatter(x[right], y[right], s=16, color=RED_E, alpha=0.45, label='처치군 (기준점 이상)')

gl = np.polyfit(x[left], y[left], 1)
gr = np.polyfit(x[right], y[right], 1)
xl = np.linspace(30, c, 50); xr = np.linspace(c, 70, 50)
ax.plot(xl, np.polyval(gl, xl), color=BLUE_E, linewidth=3)
ax.plot(xr, np.polyval(gr, xr), color=RED_E, linewidth=3)

yl, yr = np.polyval(gl, c), np.polyval(gr, c)
ax.axvline(c, color='#333333', linestyle='--', linewidth=2)
ax.plot([c, c], [yl, yr], color=YELLOW_E, linewidth=5, solid_capstyle='butt', zorder=5)
ax.annotate('', xy=(c, yr), xytext=(c, yl),
            arrowprops=dict(arrowstyle='<->', color=YELLOW_E, linewidth=2.5))
ax.text(c + 0.9, (yl + yr) / 2, 'τ = 기준점에서의\n수직 점프',
        fontsize=12, color='#8a6d00', fontweight='bold', va='center')

ax.axvspan(c - 5, c + 5, color=GREEN, alpha=0.45, zorder=0)
ax.text(c, 12.2, '대역폭 안 (여기 자료만 쓴다)', ha='center', fontsize=10.5, color=GREEN_E,
        fontweight='bold')
ax.text(c - 0.8, 32.0, '기준점 c', ha='right', fontsize=11.5, fontweight='bold')

ax.set_xlim(30, 70); ax.set_ylim(11, 34)
ax.set_xlabel('배정 변수 X (예: 시험 점수)', fontsize=12)
ax.set_ylabel('결과 변수 Y', fontsize=12)
ax.set_title('그림 5-1. Sharp RDD의 구조', fontsize=13.5, fontweight='bold', pad=10)
ax.legend(fontsize=10.5, loc='upper left')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '5-1.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 5-2. Sharp와 Fuzzy의 차이
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
grid = np.linspace(30, 70, 400)

ax = axes[0]
p_sharp = (grid >= c).astype(float)
ax.plot(grid[grid < c], p_sharp[grid < c], color=BLUE_E, linewidth=3.5)
ax.plot(grid[grid >= c], p_sharp[grid >= c], color=RED_E, linewidth=3.5)
ax.axvline(c, color='#333333', linestyle='--', linewidth=1.8)
ax.annotate('', xy=(c, 1.0), xytext=(c, 0.0),
            arrowprops=dict(arrowstyle='<->', color=YELLOW_E, linewidth=2.5))
ax.text(c + 1.2, 0.5, '점프 폭 = 1', fontsize=11.5, color='#8a6d00', fontweight='bold',
        va='center')
ax.text(37, 0.08, '아무도\n처치 안 받음', ha='center', fontsize=10.5, color=BLUE_E)
ax.text(63, 0.88, '전원\n처치 받음', ha='center', fontsize=10.5, color=RED_E)
ax.set_ylim(-0.12, 1.16); ax.set_xlim(30, 70)
ax.set_xlabel('배정 변수 X', fontsize=11.5)
ax.set_ylabel('처치받을 확률  P(D=1 | X)', fontsize=11.5)
ax.set_title('(a) Sharp: 기준점이 처치를 전부 결정', fontsize=12, fontweight='bold')
ax.grid(alpha=0.2)

ax = axes[1]
p_fuzzy = 0.14 + 0.02 * (grid - c) / 20 + 0.48 * (grid >= c)
ax.plot(grid[grid < c], p_fuzzy[grid < c], color=BLUE_E, linewidth=3.5)
ax.plot(grid[grid >= c], p_fuzzy[grid >= c], color=RED_E, linewidth=3.5)
ax.axvline(c, color='#333333', linestyle='--', linewidth=1.8)
ax.annotate('', xy=(c, 0.62), xytext=(c, 0.14),
            arrowprops=dict(arrowstyle='<->', color=YELLOW_E, linewidth=2.5))
ax.text(c + 1.2, 0.38, '점프 폭 < 1\n(1단계 계수)', fontsize=11.5, color='#8a6d00',
        fontweight='bold', va='center')
ax.text(37, 0.28, '기준 미달인데\n일부는 받음', ha='center', fontsize=10.5, color=BLUE_E)
ax.text(63, 0.50, '기준 넘겼는데\n일부는 안 받음', ha='center', fontsize=10.5, color=RED_E)
ax.set_ylim(-0.12, 1.16); ax.set_xlim(30, 70)
ax.set_xlabel('배정 변수 X', fontsize=11.5)
ax.set_ylabel('처치받을 확률  P(D=1 | X)', fontsize=11.5)
ax.set_title('(b) Fuzzy: 기준점은 확률만 올림', fontsize=12, fontweight='bold')
ax.grid(alpha=0.2)

fig.suptitle('그림 5-2. Sharp와 Fuzzy의 차이는 세로 점프 폭 하나다',
             fontsize=13.5, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '5-2.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 5-3. 대역폭을 좁힐 때와 넓힐 때
# ---------------------------------------------------------------------------
rng = np.random.default_rng(11)
x3 = rng.uniform(20, 80, 600)
true_curve = lambda t: 20 + 0.30 * (t - c) - 0.010 * (t - c) ** 2
y3 = true_curve(x3) + 4.0 * (x3 >= c) + rng.normal(0, 1.5, len(x3))

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
for ax, h, title in zip(axes, [4.0, 25.0],
                        ['(a) 대역폭이 좁을 때', '(b) 대역폭이 넓을 때']):
    inside = np.abs(x3 - c) <= h
    ax.scatter(x3[~inside], y3[~inside], s=10, color='#cccccc', alpha=0.6)
    ax.scatter(x3[inside & (x3 < c)], y3[inside & (x3 < c)], s=14, color=BLUE_E, alpha=0.55)
    ax.scatter(x3[inside & (x3 >= c)], y3[inside & (x3 >= c)], s=14, color=RED_E, alpha=0.55)

    ml = inside & (x3 < c); mr = inside & (x3 >= c)
    fl = np.polyfit(x3[ml], y3[ml], 1); fr = np.polyfit(x3[mr], y3[mr], 1)
    tl = np.linspace(c - h, c, 30); tr = np.linspace(c, c + h, 30)
    ax.plot(tl, np.polyval(fl, tl), color=BLUE_E, linewidth=3)
    ax.plot(tr, np.polyval(fr, tr), color=RED_E, linewidth=3)

    tt = np.linspace(20, 80, 200)
    ax.plot(tt[tt < c], true_curve(tt[tt < c]), color='#444444', linestyle=':', linewidth=2)
    ax.plot(tt[tt >= c], true_curve(tt[tt >= c]) + 4.0, color='#444444', linestyle=':',
            linewidth=2, label='실제 관계 (곡선)')

    ax.axvline(c, color='#333333', linestyle='--', linewidth=1.8)
    ax.axvspan(c - h, c + h, color=GREEN, alpha=0.30, zorder=0)
    ax.set_xlim(20, 80); ax.set_ylim(0, 37)
    ax.set_xlabel('배정 변수 X', fontsize=11.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9.5, loc='lower center')

axes[0].set_ylabel('결과 변수 Y', fontsize=11.5)
axes[0].text(50, 33.5, '쓰는 자료가 적다 → 신뢰구간이 넓다', ha='center', fontsize=11,
             color=RED_E, fontweight='bold')
axes[1].text(50, 33.5, '직선이 곡선을 못 따라간다 → 편향', ha='center', fontsize=11,
             color=RED_E, fontweight='bold')
fig.suptitle('그림 5-3. 대역폭은 자료 양과 직선 근사 정확도를 맞바꾼다',
             fontsize=13.5, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '5-3.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 5-4. 배정 변수 밀도로 조작을 찾는다
# ---------------------------------------------------------------------------
rng = np.random.default_rng(23)
base = rng.normal(50, 9, 4000)
base = base[(base > 25) & (base < 75)]

moved = base.copy()
pick = (moved > c - 3) & (moved < c)
idx = np.where(pick)[0]
take = rng.choice(idx, size=int(len(idx) * 0.60), replace=False)
moved[take] = rng.uniform(c, c + 3.0, len(take))

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
bins = np.arange(26, 75.1, 1.5)   # 기준점 50이 정확히 구간 경계가 되도록 맞춘다
for ax, dat, title, msg, col in zip(
        axes, [base, moved],
        ['(a) 조작이 없을 때', '(b) 조작이 있을 때'],
        ['기준점을 지나며 밀도가 매끄럽게 이어진다',
         '기준점 바로 위에 사람이 몰려 있다'],
        [GREEN_E, RED_E]):
    ax.hist(dat[dat < c], bins=bins, color=BLUE_E, alpha=0.65)
    ax.hist(dat[dat >= c], bins=bins, color=RED_E, alpha=0.65)
    ax.axvline(c, color='#333333', linestyle='--', linewidth=2)
    ax.set_xlim(25, 75); ax.set_ylim(0, 500)
    ax.set_xlabel('배정 변수 X', fontsize=11.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.text(50, 460, msg, ha='center', fontsize=11, color=col, fontweight='bold')
    ax.grid(alpha=0.2)

axes[0].set_ylabel('사람 수', fontsize=11.5)
axes[1].annotate('여기가 솟는다', xy=(51.3, 390), xytext=(61, 390),
                 fontsize=11, color=RED_E, fontweight='bold', va='center',
                 arrowprops=dict(arrowstyle='->', color=RED_E, linewidth=2))
fig.suptitle('그림 5-4. 결과가 아니라 사람 수를 세어 조작을 찾는다',
             fontsize=13.5, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '5-4.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 5-5. RDD 분석 순서
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.6, 7.0))
ax.set_xlim(0, 11.4); ax.set_ylim(0, 8.0); ax.axis('off')

box(ax, 5.7, 7.4, 5.6, 0.80, '① 배정 변수 밀도를 먼저 본다', YELLOW, YELLOW_E, fs=11.5)
box(ax, 1.9, 5.9, 3.2, 0.90, '기준점 근처에\n사람이 몰렸다\n→ 여기서 멈춘다', RED, RED_E, fs=10)
box(ax, 7.6, 5.9, 4.4, 0.90, '② 공변량이 기준점에서\n튀지 않는지 본다', GRAY, GRAY_E, fs=10.5)
box(ax, 7.6, 4.3, 4.4, 0.80, '③ 대역폭을 자료에서 고른다', BLUE, BLUE_E, fs=10.5)
box(ax, 7.6, 2.8, 4.4, 0.80, '④ 기준점에서 처치가\n전원 바뀌는가', GRAY, GRAY_E, fs=10.5)
box(ax, 9.6, 1.3, 3.0, 0.80, 'Sharp RDD\n실습 1', GREEN, GREEN_E, fs=10.5)
box(ax, 5.5, 1.3, 3.0, 0.80, 'Fuzzy RDD\n실습 4', GREEN, GREEN_E, fs=10.5)
box(ax, 1.9, 3.4, 3.2, 1.30, '⑤ 가짜 기준점과\n도넛홀로 되짚는다\n실습 3', YELLOW, YELLOW_E,
    fs=10.5)

arrow(ax, (4.2, 7.00), (2.3, 6.42))
arrow(ax, (7.2, 7.00), (7.6, 6.42))
ax.text(2.55, 6.68, '몰림 있음', fontsize=10, fontweight='bold', color=RED_E, ha='center')
ax.text(8.35, 6.72, '몰림 없음', fontsize=10, fontweight='bold', color=GREEN_E, ha='center')

arrow(ax, (7.6, 5.42), (7.6, 4.75))
arrow(ax, (7.6, 3.88), (7.6, 3.24))
arrow(ax, (8.9, 2.38), (9.5, 1.75))
arrow(ax, (6.3, 2.38), (5.7, 1.75))
ax.text(9.55, 2.05, '예', fontsize=10, fontweight='bold')
ax.text(5.55, 2.05, '아니오', fontsize=10, fontweight='bold')

arrow(ax, (4.0, 1.3), (3.5, 2.80), style=':')
ax.text(2.0, 1.85, '추정한 뒤 반드시\n되짚어 본다', fontsize=10, ha='center', color='#333333')

ax.text(5.7, 0.30, '②와 ⑤를 건너뛴 추정값은 보고서에 싣지 않는다',
        ha='center', fontsize=11.5, color=RED_E, fontweight='bold')
ax.set_title('그림 5-5. RDD 분석 순서', fontsize=13.5, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '5-5.png'), dpi=200, bbox_inches='tight')
plt.close()

print("생성 완료:")
for f in ['5-1.png', '5-2.png', '5-3.png', '5-4.png', '5-5.png']:
    p = os.path.join(OUT, f)
    print(f"  {f}  ({os.path.getsize(p)/1024:.0f} KB)")
