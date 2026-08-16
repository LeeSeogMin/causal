"""
2장 강의노트용 개념도 생성
==========================

실행: python diagrams/make_chapter02.py
출력: diagrams/2-1.png ~ 2-5.png

- 2-1.png: 잠재적 결과와 인과추론의 근본 문제
- 2-2.png: 단순 평균 차이가 ATT와 선택편향으로 갈라지는 구조
- 2-3.png: 성향점수 매칭의 네 단계
- 2-4.png: 이중 머신러닝이 두 번 빼는 절차
- 2-5.png: ATE 하나와 CATE 분포가 답하는 질문의 차이
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))

BLUE = '#d9e8ff'; BLUE_E = '#2c6fbb'
GREEN = '#d9f2d9'; GREEN_E = '#2e8b57'
RED = '#ffd9d9'; RED_E = '#c0392b'
YELLOW = '#fff2cc'; YELLOW_E = '#d6a300'
GRAY = '#eeeeee'; GRAY_E = '#888888'
PURPLE = '#ece0f5'; PURPLE_E = '#7b52ab'


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
# 2-1. 잠재적 결과와 근본 문제
# ---------------------------------------------------------------------------
fig, ax = canvas(w=9.6, h=5.2, xlim=(0, 11), ylim=(0, 5.6))

ax.text(1.4, 4.85, '사람', ha='center', fontsize=11.5, fontweight='bold')
ax.text(4.2, 4.85, 'D = 1이면', ha='center', fontsize=11.5, fontweight='bold')
ax.text(7.4, 4.85, 'D = 0이면', ha='center', fontsize=11.5, fontweight='bold')
ax.text(9.9, 4.85, '효과', ha='center', fontsize=11.5, fontweight='bold')

box(ax, 1.4, 3.7, 1.7, 0.85, '철수\nD = 1', BLUE, BLUE_E, fs=10.5)
box(ax, 4.2, 3.7, 2.6, 0.85, 'Y(처치) = 6.2\n관측된다', GREEN, GREEN_E, fs=10.5)
box(ax, 7.4, 3.7, 2.6, 0.85, 'Y(무처치) = ?\n영원히 못 본다', RED, RED_E, fs=10.5)
ax.text(9.9, 3.7, '계산\n불가', ha='center', va='center', fontsize=10.5,
        color=RED_E, fontweight='bold')

box(ax, 1.4, 2.3, 1.7, 0.85, '영희\nD = 0', GRAY, GRAY_E, fs=10.5)
box(ax, 4.2, 2.3, 2.6, 0.85, 'Y(처치) = ?\n영원히 못 본다', RED, RED_E, fs=10.5)
box(ax, 7.4, 2.3, 2.6, 0.85, 'Y(무처치) = 2.4\n관측된다', GREEN, GREEN_E, fs=10.5)
ax.text(9.9, 2.3, '계산\n불가', ha='center', va='center', fontsize=10.5,
        color=RED_E, fontweight='bold')

ax.text(5.5, 1.05,
        '한 사람에게서 [처치 결과 - 무처치 결과]를 계산할 방법은 없다.\n'
        '그래서 개인이 아니라 집단의 평균을 목표로 바꾼다.',
        ha='center', va='center', fontsize=11, color='#333333')
ax.set_title('그림 2-1. 인과추론의 근본 문제: 한 사람의 두 결과를 동시에 볼 수 없다',
             fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '2-1.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 2-2. 단순 평균 차이의 분해
# ---------------------------------------------------------------------------
fig, ax = canvas(w=10.4, h=4.8, xlim=(0, 12.0), ylim=(0, 5.2))

box(ax, 2.0, 3.9, 3.4, 1.1, '단순 평균 차이\n3.71', YELLOW, YELLOW_E, fs=12)
ax.text(4.15, 3.9, '=', ha='center', va='center', fontsize=18, fontweight='bold')
box(ax, 6.4, 3.9, 3.2, 1.1, '알고 싶은 것\nATT = 3.00', GREEN, GREEN_E, fs=12)
ax.text(8.55, 3.9, '+', ha='center', va='center', fontsize=18, fontweight='bold')
box(ax, 10.4, 3.9, 2.6, 1.1, '선택편향\n0.71', RED, RED_E, fs=12)

box(ax, 6.6, 1.9, 4.6, 0.95,
    '처치군이 처치를 안 받았을 때의 결과가\n대조군보다 원래 높다', RED, RED_E, fs=10.5)
arrow(ax, (10.4, 3.30), (8.6, 2.38), color=RED_E)

ax.text(2.0, 2.35, '데이터에서\n바로 계산된다', ha='center', va='center',
        fontsize=10.5, color='#333333')
ax.text(2.0, 1.25, '실습 1에서 X₁의\nSMD가 0.55다', ha='center', va='center',
        fontsize=10.5, color=RED_E, fontweight='bold')

ax.text(6.0, 0.45,
        '선택편향 0.71은 참값 3.00의 23.6%다. 매칭·DML이 줄이려는 대상이 이 항이다.',
        ha='center', va='center', fontsize=11, color='#333333')
ax.set_title('그림 2-2. 단순 평균 차이는 두 조각으로 갈라진다 (실습 1 실측값)',
             fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '2-2.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 2-3. 성향점수 매칭 네 단계
# ---------------------------------------------------------------------------
fig, ax = canvas(w=10.4, h=4.4, xlim=(0, 11.6), ylim=(0, 5))

box(ax, 1.5, 3.5, 2.5, 1.15, '① 성향점수 추정\ne(X) = P(D=1|X)', YELLOW, YELLOW_E, fs=10.5)
box(ax, 4.4, 3.5, 2.5, 1.15, '② 점수가 가까운\n처치·대조 짝짓기', BLUE, BLUE_E, fs=10.5)
box(ax, 7.3, 3.5, 2.5, 1.15, '③ 균형 확인\nSMD < 0.1인가', PURPLE, PURPLE_E, fs=10.5)
box(ax, 10.2, 3.5, 2.3, 1.15, '④ 짝의 차이를\n평균 → ATT', GREEN, GREEN_E, fs=10.5)

arrow(ax, (2.8, 3.5), (3.1, 3.5))
arrow(ax, (5.7, 3.5), (6.0, 3.5))
arrow(ax, (8.6, 3.5), (9.0, 3.5))

ax.text(1.5, 2.35, '로지스틱 회귀', ha='center', fontsize=10, color='#555555')
ax.text(4.4, 2.35, '1:1 최근접 이웃\ncaliper 0.25', ha='center', va='top',
        fontsize=10, color='#555555')
ax.text(7.0, 2.35, '실습 2에서\nX₁ 0.55 → 0.01', ha='center', va='top',
        fontsize=10, color=GREEN_E, fontweight='bold')
ax.text(10.2, 2.35, '실습 2에서\n3.08 (참값 3.00)', ha='center', va='top',
        fontsize=10, color=GREEN_E, fontweight='bold')

box(ax, 5.8, 0.75, 9.4, 1.0,
    '③에서 SMD가 0.1을 넘는 공변량이 남으면 ④로 넘어가지 않는다. 성향점수 모형을 고쳐 ①로 돌아간다.',
    RED, RED_E, fs=10.5)
ax.add_patch(FancyArrowPatch((8.5, 2.92), (8.5, 1.28), arrowstyle='-|>',
                             mutation_scale=20, color=RED_E, linewidth=2,
                             linestyle=':', shrinkA=6, shrinkB=6))
ax.set_title('그림 2-3. 성향점수 매칭의 네 단계', fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '2-3.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 2-4. DML이 두 번 빼는 절차
# ---------------------------------------------------------------------------
fig, ax = canvas(w=11.0, h=5.2, xlim=(0, 12.4), ylim=(0, 5.8))

box(ax, 1.6, 4.5, 2.6, 0.95, '결과 Y\n(소득)', GRAY, GRAY_E, fs=11)
box(ax, 5.2, 4.5, 3.2, 0.95, 'ML로 예측한 값\nL(X) = E[Y|X]', YELLOW, YELLOW_E, fs=10.5)
box(ax, 9.0, 4.5, 3.0, 0.95, 'Y 잔차\nY_res = Y - L(X)', GREEN, GREEN_E, fs=10.5)
arrow(ax, (2.9, 4.5), (3.6, 4.5), label='빼기', ly=0.33, lfs=10)
arrow(ax, (6.8, 4.5), (7.5, 4.5))

box(ax, 1.6, 2.7, 2.6, 0.95, '처치 D\n(훈련 참여)', GRAY, GRAY_E, fs=11)
box(ax, 5.2, 2.7, 3.2, 0.95, 'ML로 예측한 값\nM(X) = E[D|X]', YELLOW, YELLOW_E, fs=10.5)
box(ax, 9.0, 2.7, 3.0, 0.95, 'D 잔차\nD_res = D - M(X)', GREEN, GREEN_E, fs=10.5)
arrow(ax, (2.9, 2.7), (3.6, 2.7), label='빼기', ly=0.33, lfs=10)
arrow(ax, (6.8, 2.7), (7.5, 2.7))

box(ax, 5.6, 0.85, 5.2, 1.0, 'Y_res를 D_res에 회귀 → 2.88', BLUE, BLUE_E, fs=12)
arrow(ax, (9.0, 2.22), (8.3, 1.42), color=GREEN_E)
ax.add_patch(FancyArrowPatch((10.5, 4.5), (8.15, 1.30), arrowstyle='-|>',
                             mutation_scale=20, color=GREEN_E, linewidth=2,
                             connectionstyle='arc3,rad=-0.45',
                             shrinkA=6, shrinkB=6))

ax.text(1.6, 0.85, '실습 3\n참값 2.99', ha='center', va='center',
        fontsize=10.5, color=GREEN_E, fontweight='bold')
ax.text(5.6, 5.45, '두 번 빼는 이유: X가 Y와 D 양쪽을 동시에 움직이기 때문이다',
        ha='center', fontsize=11, color='#333333')
ax.set_title('그림 2-4. 이중 머신러닝이 두 번 빼는 절차',
             fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '2-4.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 2-5. ATE 하나와 CATE 분포
# ---------------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.3))

axL.axis('off')
axL.set_xlim(0, 10); axL.set_ylim(0, 10)
axL.add_patch(FancyBboxPatch((1.2, 4.6), 7.6, 2.4,
                             boxstyle="round,pad=0.1,rounding_size=0.2",
                             facecolor=YELLOW, edgecolor=YELLOW_E, linewidth=2))
axL.text(5.0, 5.8, 'ATE = 24.89점', ha='center', va='center',
         fontsize=17, fontweight='bold')
axL.text(5.0, 3.4, '답하는 질문\n"이 정책을 쓸까 말까"', ha='center', va='center',
         fontsize=12)
axL.text(5.0, 1.4, '답 못 하는 질문\n"예산이 절반이면 누구부터"', ha='center', va='center',
         fontsize=12, color=RED_E, fontweight='bold')
axL.text(5.0, 8.6, '평균 하나만 볼 때', ha='center', fontsize=13, fontweight='bold')

rng = np.random.default_rng(2)
q = np.array([28.68, 27.11, 25.63, 23.59, 19.41])
axR.bar(np.arange(5), q, color=[GREEN_E, '#57a773', '#8fbf9f', '#c4a35a', RED_E],
        edgecolor='black', alpha=0.85)
axR.axhline(24.89, color='black', linestyle='--', linewidth=2)
axR.text(-0.45, 31.0, 'ATE 24.89', ha='left', fontsize=10.5, fontweight='bold')
for i, v in enumerate(q):
    axR.text(i, v - 1.9, f'{v:.1f}', ha='center', fontsize=10.5,
             fontweight='bold', color='white')
axR.set_xticks(np.arange(5))
axR.set_xticklabels(['Q1\n최저소득', 'Q2', 'Q3', 'Q4', 'Q5\n최고소득'], fontsize=10)
axR.set_ylabel('참값 CATE (만족도 점)', fontsize=11)
axR.set_ylim(0, 33)
axR.set_title('소득 분위별로 나눠 볼 때 (실습 4 참값)', fontsize=13, fontweight='bold')
axR.grid(True, alpha=0.3, axis='y')

fig.suptitle('그림 2-5. 평균 하나와 집단별 효과가 답하는 질문의 차이',
             fontsize=13.5, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '2-5.png'), dpi=200, bbox_inches='tight')
plt.close()

print("생성 완료:")
for f in ['2-1.png', '2-2.png', '2-3.png', '2-4.png', '2-5.png']:
    p = os.path.join(OUT, f)
    print(f"  {f}  ({os.path.getsize(p)/1024:.0f} KB)")
