"""
6장 강의노트용 개념도 생성
==========================

실행: python diagrams/make_chapter06.py
출력: diagrams/6-1.png ~ 6-5.png

- 6-1.png: 내생성이 생기는 세 경로 (생략변수 / 측정오차 / 역인과)
- 6-2.png: 도구변수의 3조건
- 6-3.png: 2SLS 두 단계 흐름
- 6-4.png: 축약형 ÷ 1단계 나눗셈
- 6-5.png: 방법 선택 흐름
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams['font.family'] = ['Malgun Gothic', 'Apple SD Gothic Neo', 'AppleGothic']
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
# 6-1. 내생성이 생기는 세 경로
# ---------------------------------------------------------------------------
def curved(ax, p1, p2, rad, color, label=None, lx=0.0, ly=0.0, lfs=9):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=16,
                                 color=color, linewidth=2, shrinkA=6, shrinkB=6,
                                 connectionstyle='arc3,rad=%s' % rad))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=lfs, color=color, fontweight='bold')


fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
for a in axes:
    a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis('off')

# (1) 생략변수
a = axes[0]
box(a, 5.0, 8.5, 6.0, 1.6, 'U: 관측 안 되는 원인\n(건강 관심도)', RED, RED_E, fs=9.5)
box(a, 2.0, 4.0, 3.4, 1.6, 'D: 처치\n(헬스장 등록)', BLUE, BLUE_E, fs=9.5)
box(a, 8.0, 4.0, 3.4, 1.6, 'Y: 결과\n(체중 변화)', GREEN, GREEN_E, fs=9.5)
arrow(a, (4.0, 7.7), (2.4, 4.9), color=RED_E)
arrow(a, (6.0, 7.7), (7.6, 4.9), color=RED_E)
arrow(a, (3.8, 4.0), (6.2, 4.0))
a.text(5.0, 5.4, '알고 싶은 효과', ha='center', va='center', fontsize=9)
a.set_title('① 생략변수', fontsize=12, fontweight='bold', pad=6)
a.text(5.0, 1.6, 'U가 D와 Y를 함께 움직인다', ha='center', va='center',
       fontsize=9.5, color='#c0392b')

# (2) 측정오차
a = axes[1]
box(a, 2.6, 8.5, 4.4, 1.6, 'D*: 실제 처치\n(실제 운동시간)', GRAY, GRAY_E, fs=9.5)
box(a, 7.8, 6.3, 3.8, 1.6, 'Y: 결과\n(체중 변화)', GREEN, GREEN_E, fs=9.5)
box(a, 2.6, 4.0, 4.4, 1.6, 'D: 잰 값\n(설문 응답)', BLUE, BLUE_E, fs=9.5)
arrow(a, (4.9, 8.1), (6.0, 6.9))
a.text(6.6, 8.3, '알고 싶은 효과', ha='center', va='center', fontsize=9)
arrow(a, (2.6, 7.7), (2.6, 4.8), color=RED_E)
a.text(3.1, 6.3, '잴 때 오차가 섞인다', ha='left', va='center',
       fontsize=9, color=RED_E, fontweight='bold')
a.set_title('② 측정오차', fontsize=12, fontweight='bold', pad=6)
a.text(5.0, 1.6, '회귀에는 실제 값이 아니라 아래 칸을 넣는다', ha='center', va='center',
       fontsize=9.5, color='#c0392b')

# (3) 역인과
a = axes[2]
box(a, 1.9, 6.0, 3.2, 1.6, 'D: 처치\n(가격)', BLUE, BLUE_E, fs=9.5)
box(a, 8.1, 6.0, 3.2, 1.6, 'Y: 결과\n(손님 수)', GREEN, GREEN_E, fs=9.5)
curved(a, (3.5, 6.8), (6.5, 6.8), 0.35, '#333333')
a.text(5.0, 8.6, '알고 싶은 효과', ha='center', va='center', fontsize=9)
curved(a, (6.5, 5.2), (3.5, 5.2), 0.35, RED_E)
a.text(5.0, 3.4, 'Y가 거꾸로 D를 정한다', ha='center', va='center',
       fontsize=9, color=RED_E, fontweight='bold')
a.set_title('③ 역인과', fontsize=12, fontweight='bold', pad=6)
a.text(5.0, 1.6, '화살표가 양쪽으로 간다', ha='center', va='center',
       fontsize=9.5, color='#c0392b')

fig.suptitle('그림 6-1. 내생성이 생기는 세 경로', fontsize=13.5, fontweight='bold', y=1.03)
fig.text(0.5, -0.04,
         '세 경로 모두 D를 회귀식의 오차와 상관되게 만든다 — 이 상태가 내생성이다.',
         ha='center', va='center', fontsize=11, fontweight='bold', color='#c0392b')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '6-1.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 6-2. 도구변수의 3조건
# ---------------------------------------------------------------------------
fig, ax = canvas(h=5.4, ylim=(-0.9, 5.4))
box(ax, 1.6, 1.8, 2.4, 1.0, 'Z: 도구변수\n(헬스장 거리)', YELLOW, YELLOW_E, fs=10.5)
box(ax, 5.0, 1.8, 2.4, 1.0, 'D: 처치\n(등록 여부)', BLUE, BLUE_E, fs=10.5)
box(ax, 8.4, 1.8, 2.4, 1.0, 'Y: 결과\n(체중 변화)', GREEN, GREEN_E, fs=10.5)
box(ax, 6.7, 4.4, 2.8, 0.9, 'U: 건강 관심도', RED, RED_E, fs=10.5)

arrow(ax, (2.8, 1.8), (3.8, 1.8), label='① 관련성', ly=0.30, lfs=10, lcolor=YELLOW_E)
arrow(ax, (6.2, 1.8), (7.2, 1.8), label='효과', ly=0.30, lfs=10)
arrow(ax, (6.1, 4.05), (5.3, 2.35), color=RED_E)
arrow(ax, (7.3, 4.05), (8.2, 2.35), color=RED_E)

ax.add_patch(FancyArrowPatch((1.6, 1.25), (8.4, 1.25), arrowstyle='-|>',
                             mutation_scale=20, color=RED_E, linewidth=2,
                             linestyle=':', connectionstyle='arc3,rad=0.22',
                             shrinkA=6, shrinkB=6))
ax.text(5.0, -0.62, '② 배제제약: 이 경로가 없어야 한다', ha='center',
        fontsize=10.5, color=RED_E, fontweight='bold')
ax.text(2.7, 3.7, '③ 외생성:\nZ와 U가 무관해야 한다', ha='center', va='center',
        fontsize=10, color=RED_E)

ax.text(5.0, 5.15, '세 조건 중 데이터로 확인되는 것은 ①뿐이다',
        ha='center', fontsize=10.5, color='#333333')
ax.set_title('그림 6-2. 유효한 도구변수의 3조건', fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '6-2.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 6-3. 2SLS 두 단계
# ---------------------------------------------------------------------------
fig, ax = canvas(w=10, h=4.4, ylim=(0, 5))
box(ax, 1.5, 3.3, 2.6, 1.2, '원래 D\n오염된 변동 포함', GRAY, GRAY_E, fs=10.5)
box(ax, 4.3, 3.3, 2.4, 1.2, '1단계 회귀\nD ~ Z + X', YELLOW, YELLOW_E, fs=10.5)
box(ax, 7.0, 3.3, 2.4, 1.2, 'D_hat\nZ가 만든 변동만', GREEN, GREEN_E, fs=10.5)
box(ax, 7.0, 1.1, 2.4, 1.0, '2단계 회귀\nY ~ D_hat + X', BLUE, BLUE_E, fs=10.5)
box(ax, 3.6, 1.1, 3.0, 1.0, 'β₁ = 인과효과', YELLOW, YELLOW_E, fs=11.5)

arrow(ax, (2.8, 3.3), (3.1, 3.3))
arrow(ax, (5.5, 3.3), (5.8, 3.3))
arrow(ax, (7.0, 2.7), (7.0, 1.6))
arrow(ax, (5.8, 1.1), (5.1, 1.1))

ax.text(1.5, 2.3, 'Corr(D, U)\n= 0.527', ha='center', va='center',
        fontsize=10.5, color=RED_E, fontweight='bold')
ax.text(9.0, 2.3, 'Corr(D_hat, U)\n= 0.001', ha='center', va='center',
        fontsize=10.5, color=GREEN_E, fontweight='bold')
ax.text(5.0, 0.15, '1단계를 거치면 숨은 원인과의 연결이 끊어진다 (실습 6-1 실측값)',
        ha='center', fontsize=10.5, color='#333333')
ax.set_title('그림 6-3. 2SLS의 두 단계', fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '6-3.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 6-4. 축약형 ÷ 1단계 — 2SLS와 같은 답
# ---------------------------------------------------------------------------
fig, ax = canvas(w=10.5, h=4.4, xlim=(0, 12), ylim=(0, 5))
box(ax, 1.2, 2.4, 1.4, 1.2, 'Z', GREEN, GREEN_E, fs=13)
box(ax, 4.2, 3.7, 2.6, 1.1, '1단계 회귀\nD ~ Z + X', YELLOW, YELLOW_E, fs=10.5)
box(ax, 4.2, 1.1, 2.6, 1.1, '축약형 회귀\nY ~ Z + X', BLUE, BLUE_E, fs=10.5)
box(ax, 7.3, 3.7, 2.3, 1.1, '0.6096\n분모 (Z → D)', YELLOW, YELLOW_E, fs=10.5)
box(ax, 7.3, 1.1, 2.3, 1.1, '2.0438\n분자 (Z → Y)', BLUE, BLUE_E, fs=10.5)
box(ax, 10.5, 2.4, 2.6, 1.4, '2.0438 ÷ 0.6096\n= 3.3528', GREEN, GREEN_E, fs=11)

arrow(ax, (1.9, 2.8), (2.9, 3.6))
arrow(ax, (1.9, 2.0), (2.9, 1.2))
arrow(ax, (5.5, 3.7), (6.1, 3.7))
arrow(ax, (5.5, 1.1), (6.1, 1.1))
arrow(ax, (8.5, 3.5), (9.3, 2.8))
arrow(ax, (8.5, 1.3), (9.3, 2.0))

ax.text(6.0, 0.15, '2SLS가 내놓은 3.353과 같은 값이다 — 같은 추정량을 다른 순서로 계산했다',
        ha='center', fontsize=10.5, color='#333333')
ax.set_title('그림 6-4. 축약형 ÷ 1단계 — 2SLS와 같은 답', fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '6-4.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 6-4. 방법 선택 흐름
# ---------------------------------------------------------------------------
fig, ax = canvas(w=10.5, h=6.6, xlim=(0, 11.4), ylim=(0, 7.2))
box(ax, 5.7, 6.5, 4.8, 0.85, '1단계 F통계량을 먼저 본다', YELLOW, YELLOW_E, fs=11.5)
box(ax, 1.7, 4.8, 3.0, 0.95, '도구변수를\n다시 찾는다', RED, RED_E, fs=10.5)
box(ax, 7.9, 4.8, 3.8, 0.95, 'D와 Y 관계가\n곡선인가', GRAY, GRAY_E, fs=10.5)
box(ax, 9.9, 2.8, 2.6, 0.95, '2SLS\n실습 6-1', BLUE, BLUE_E, fs=10.5)
box(ax, 5.9, 2.8, 3.0, 0.95, '2SLS + D² 항\n실습 6-2', BLUE, BLUE_E, fs=10.5)
box(ax, 1.7, 2.8, 2.6, 0.95, 'DeepIV\n실습 6-3', BLUE, BLUE_E, fs=10.5)
box(ax, 5.7, 0.8, 6.4, 0.9, '결과는 항상 1단계 F와 함께 보고한다',
    GREEN, GREEN_E, fs=11.5)

arrow(ax, (4.4, 6.07), (2.2, 5.30))
arrow(ax, (7.0, 6.07), (7.4, 5.30))
ax.text(2.7, 5.85, 'F < 10', fontsize=10, fontweight='bold', color=RED_E)
ax.text(7.15, 5.85, 'F ≥ 10', fontsize=10, fontweight='bold', color=GREEN_E)

arrow(ax, (9.2, 4.32), (9.8, 3.30))
arrow(ax, (6.9, 4.32), (6.2, 3.30))
ax.text(9.55, 3.85, '아니오', fontsize=9.5)
ax.text(6.05, 3.85, '예', fontsize=9.5)

arrow(ax, (4.4, 2.8), (3.1, 2.8))
ax.text(3.75, 3.15, '곡선 모양을\n짐작 못 하면', fontsize=9, ha='center')

arrow(ax, (9.6, 2.32), (7.4, 1.30))
arrow(ax, (5.9, 2.32), (5.7, 1.28))
arrow(ax, (1.9, 2.32), (3.6, 1.30))

ax.set_title('그림 6-5. 어떤 방법을 쓸지 정하는 순서',
             fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '6-5.png'), dpi=200, bbox_inches='tight')
plt.close()

print("생성 완료:")
for f in ['6-1.png', '6-2.png', '6-3.png', '6-4.png', '6-5.png']:
    p = os.path.join(OUT, f)
    print(f"  {f}  ({os.path.getsize(p)/1024:.0f} KB)")
