"""
4장 강의노트용 개념도 생성
==========================

실행: python diagrams/make_chapter04.py
출력: diagrams/4-1.png ~ 4-4.png

- 4-1.png: 평행추세 가정과 반사실 (DID가 무엇을 재는가)
- 4-2.png: 두 번 빼는 구조 (2x2 차분)
- 4-3.png: 합성 대조군 구성 방식
- 4-4.png: 분석 순서와 판정 지점
"""

import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
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


def canvas(w=9, h=4.6, xlim=(0, 10), ylim=(0, 5)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis('off')
    return fig, ax


# ---------------------------------------------------------------------------
# 4-1. 평행추세 가정과 반사실
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 5.4))

t_pre = np.array([0, 1, 2, 3, 4])
t_post = np.array([4, 5, 6, 7, 8])
T0 = 4.0

ctrl_pre = 10.0 + 0.9 * t_pre
ctrl_post = 10.0 + 0.9 * t_post
trt_pre = 13.0 + 0.9 * t_pre
cf_post = 13.0 + 0.9 * t_post           # 처치가 없었다면 갔을 길
trt_post = cf_post + np.array([0, 1.6, 2.3, 2.6, 2.8])

ax.plot(t_pre, trt_pre, '-o', color=RED_E, linewidth=2.6, markersize=7,
        label='처치군 (관측)')
ax.plot(t_post, trt_post, '-o', color=RED_E, linewidth=2.6, markersize=7)
ax.plot(t_pre, ctrl_pre, '-s', color=BLUE_E, linewidth=2.6, markersize=7,
        label='대조군 (관측)')
ax.plot(t_post, ctrl_post, '-s', color=BLUE_E, linewidth=2.6, markersize=7)
ax.plot(t_post, cf_post, '--', color='#7a7a7a', linewidth=2.4,
        label='처치군의 반사실 (관측 불가)')

ax.axvline(T0, color='#555555', linestyle=':', linewidth=2)
ax.text(T0, 9.2, '정책 시행', ha='center', fontsize=11, color='#555555',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor='none'))

# 처치효과 표시
ax.annotate('', xy=(8, trt_post[-1]), xytext=(8, cf_post[-1]),
            arrowprops=dict(arrowstyle='<->', color=GREEN_E, linewidth=2.4))
ax.text(8.15, (trt_post[-1] + cf_post[-1]) / 2, 'DID가 재는 값\n(처치효과)',
        ha='left', va='center', fontsize=11, color=GREEN_E, fontweight='bold')

# 평행추세 표시
ax.annotate('', xy=(2.0, trt_pre[2]), xytext=(2.0, ctrl_pre[2]),
            arrowprops=dict(arrowstyle='<->', color='#999999', linewidth=1.8))
ax.text(1.85, (trt_pre[2] + ctrl_pre[2]) / 2, '간격 3.0',
        ha='right', va='center', fontsize=10.5, color='#666666')
ax.annotate('', xy=(6.0, cf_post[2]), xytext=(6.0, ctrl_post[2]),
            arrowprops=dict(arrowstyle='<->', color='#999999', linewidth=1.8))
ax.text(5.85, (cf_post[2] + ctrl_post[2]) / 2, '간격 3.0 유지',
        ha='right', va='center', fontsize=10.5, color='#666666')

ax.set_xlabel('시간', fontsize=11.5)
ax.set_ylabel('결과 Y', fontsize=11.5)
ax.set_xlim(-0.4, 11.2)
ax.set_ylim(8.5, 26)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='upper left', fontsize=10.5, framealpha=0.95)
ax.grid(alpha=0.25)
ax.set_title('그림 4-1. 평행추세 가정과 반사실', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '4-1.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 4-2. 두 번 빼는 구조
# ---------------------------------------------------------------------------
fig, ax = canvas(w=9.6, h=5.2, xlim=(0, 10), ylim=(0, 5.6))

box(ax, 2.3, 4.3, 2.8, 0.95, '처치군 시행 전\n-0.936', BLUE, BLUE_E, fs=11)
box(ax, 6.4, 4.3, 2.8, 0.95, '처치군 시행 후\n2.979', BLUE, BLUE_E, fs=11)
box(ax, 2.3, 2.4, 2.8, 0.95, '대조군 시행 전\n-0.651', GREEN, GREEN_E, fs=11)
box(ax, 6.4, 2.4, 2.8, 0.95, '대조군 시행 후\n0.811', GREEN, GREEN_E, fs=11)

arrow(ax, (3.8, 4.3), (4.9, 4.3), color=BLUE_E)
ax.text(4.35, 4.72, '① +3.915', ha='center', fontsize=11, color=BLUE_E,
        fontweight='bold')
arrow(ax, (3.8, 2.4), (4.9, 2.4), color=GREEN_E)
ax.text(4.35, 2.82, '② +1.462', ha='center', fontsize=11, color=GREEN_E,
        fontweight='bold')

box(ax, 8.9, 3.35, 1.8, 1.3, '3.915\n-1.462\n= 2.452', YELLOW, YELLOW_E, fs=11.5)
arrow(ax, (7.85, 4.3), (8.55, 3.85), color=YELLOW_E)
arrow(ax, (7.85, 2.4), (8.55, 2.85), color=YELLOW_E)

ax.text(5.0, 1.15,
        '① 각 집단 안에서 시간에 따른 변화를 먼저 뺀다 (변하지 않는 집단 특성이 사라진다)\n'
        '② 두 변화의 차이를 다시 뺀다 (두 집단에 공통으로 온 변화가 사라진다)',
        ha='center', va='center', fontsize=10.8, color='#333333')
ax.text(5.0, 0.35, '남는 값 2.452가 처치효과 추정치다 (실습 1 데이터의 실측 평균)',
        ha='center', fontsize=11.5, color=YELLOW_E, fontweight='bold')

ax.set_title('그림 4-2. 두 번 빼서 처치효과를 얻는 구조',
             fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '4-2.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 4-3. 합성 대조군 구성 (실습 2 실측 가중치)
# ---------------------------------------------------------------------------
fig, ax = canvas(w=10.2, h=5.4, xlim=(0, 11), ylim=(0, 5.8))

donors = [('광주', 0.35), ('울산', 0.16), ('창원', 0.12),
          ('전주', 0.12), ('성남', 0.09), ('나머지 15곳', 0.16)]
ys = [5.05, 4.25, 3.45, 2.65, 1.85, 1.05]
for (name, w), y in zip(donors, ys):
    box(ax, 1.7, y, 2.6, 0.62, f'{name}   가중치 {w:.2f}', GRAY, GRAY_E,
        fs=10.5, bold=False)
    arrow(ax, (3.05, y), (4.55, 3.05), color=GRAY_E, width=1.4)

box(ax, 5.7, 3.05, 1.9, 1.1, '가중 합\n(합 = 1)', YELLOW, YELLOW_E, fs=11)
arrow(ax, (6.7, 3.05), (7.6, 3.05), color=YELLOW_E)
box(ax, 9.0, 3.05, 2.6, 1.1, '합성 대조군\n= 가짜 서울', GREEN, GREEN_E, fs=11)

box(ax, 9.0, 5.05, 2.6, 0.85, '실제 서울', RED, RED_E, fs=11)
ax.annotate('', xy=(9.0, 3.65), xytext=(9.0, 4.6),
            arrowprops=dict(arrowstyle='<->', color=RED_E, linewidth=2.2))
ax.text(9.35, 4.15, '차이 = 처치효과', ha='left', va='center', fontsize=10.5,
        color=RED_E, fontweight='bold')

ax.text(5.5, 0.35,
        '가중치는 정책 시행 전 서울을 가장 잘 재현하도록 데이터가 정한다.\n'
        '음수 가중치를 허용하지 않고 합을 1로 묶는다.',
        ha='center', va='center', fontsize=10.8, color='#333333')

ax.set_title('그림 4-3. 대조군을 조합해서 만드는 방식 (실습 2 실측 가중치)',
             fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '4-3.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 4-4. 분석 순서와 판정 지점
# ---------------------------------------------------------------------------
fig, ax = canvas(w=10.6, h=6.8, xlim=(0, 11.6), ylim=(0, 7.4))

box(ax, 5.8, 6.7, 5.6, 0.85, '처치 단위가 여럿인가, 하나인가', YELLOW, YELLOW_E,
    fs=11.5)

box(ax, 2.5, 4.9, 4.0, 1.0, '여럿 → 사전 추세 F검정\n실습 1', BLUE, BLUE_E, fs=10.5)
box(ax, 9.0, 4.9, 4.0, 1.0, '하나 → 사전 적합도(RMSPE)\n실습 2', BLUE, BLUE_E, fs=10.5)

arrow(ax, (4.3, 6.27), (2.9, 5.45))
arrow(ax, (7.3, 6.27), (8.6, 5.45))

box(ax, 1.5, 2.9, 2.4, 1.0, 'TWFE DID\n실습 1', GREEN, GREEN_E, fs=10.5)
box(ax, 4.4, 2.9, 2.6, 1.0, 'DML로 공변량 조정\n실습 5', GREEN, GREEN_E, fs=10.5)
box(ax, 7.5, 2.9, 2.4, 1.0, 'SCM\n실습 2', GREEN, GREEN_E, fs=10.5)
box(ax, 10.3, 2.9, 2.2, 1.0, 'SDID\n실습 4', GREEN, GREEN_E, fs=10.5)

arrow(ax, (1.9, 4.4), (1.5, 3.45))
ax.text(1.05, 3.95, '공변량\n적다', fontsize=9, ha='center')
arrow(ax, (3.3, 4.4), (4.2, 3.45))
ax.text(4.15, 3.95, '공변량\n많다', fontsize=9, ha='center')
arrow(ax, (8.4, 4.4), (7.7, 3.45))
ax.text(7.55, 3.95, '적합\n좋다', fontsize=9, ha='center')
arrow(ax, (9.7, 4.4), (10.3, 3.45))
ax.text(10.7, 3.95, '평행추세도\n쓰고 싶다', fontsize=9, ha='center')

box(ax, 5.8, 1.15, 7.4, 0.95, '어느 경로든 마지막은 검증이다 (실습 3)',
    RED, RED_E, fs=11.5)
arrow(ax, (1.5, 2.4), (3.4, 1.65))
arrow(ax, (4.4, 2.4), (5.0, 1.65))
arrow(ax, (7.5, 2.4), (6.8, 1.65))
arrow(ax, (10.3, 2.4), (8.4, 1.65))

ax.text(5.8, 0.25, '플라시보 검정 · leave-one-out · 시간 플라시보',
        ha='center', fontsize=10.5, color='#333333')

ax.set_title('그림 4-4. 분석 순서와 판정 지점', fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '4-4.png'), dpi=200, bbox_inches='tight')
plt.close()

print("생성 완료:")
for f in ['4-1.png', '4-2.png', '4-3.png', '4-4.png']:
    p = os.path.join(OUT, f)
    print(f"  {f}  ({os.path.getsize(p)/1024:.0f} KB)")
