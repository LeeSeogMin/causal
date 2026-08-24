"""
1장 강의노트용 개념도 생성
==========================

실행: python diagrams/make_chapter01.py
출력: diagrams/1-1.png ~ 1-3.png

- 1-1.png: 예측과 인과는 다른 질문에 답한다
- 1-2.png: 인과적 데이터 사이언스의 5단계 순환
- 1-3.png: 이 교재에서 배울 방법의 지도
"""

import os
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


def arrow(ax, p1, p2, color='#333333', style='-', width=2, rad=None):
    kw = dict(arrowstyle='-|>', mutation_scale=20, color=color,
              linewidth=width, linestyle=style, shrinkA=6, shrinkB=6)
    if rad is not None:
        kw['connectionstyle'] = f'arc3,rad={rad}'
    ax.add_patch(FancyArrowPatch(p1, p2, **kw))


def canvas(w=9, h=4.6, xlim=(0, 10), ylim=(0, 5)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis('off')
    return fig, ax


# ---------------------------------------------------------------------------
# 1-1. 예측과 인과는 다른 질문에 답한다
# ---------------------------------------------------------------------------
fig, ax = canvas(w=10, h=5.0, xlim=(0, 11), ylim=(0, 5.6))

box(ax, 2.8, 4.6, 4.4, 0.8, '예측', BLUE, BLUE_E, fs=13)
box(ax, 8.0, 4.6, 4.4, 0.8, '인과', GREEN, GREEN_E, fs=13)

rows = [
    (3.5, '이 고객이 얼마를 살 것인가', '쿠폰을 보내면 얼마가 늘어나는가'),
    (2.6, '데이터를 있는 그대로 본다', '값을 바꿔 넣었을 때를 계산한다'),
    (1.7, '맞히는 정확도로 평가한다', '참 효과와의 차이로 평가한다'),
    (0.8, '실습 1-2: R² 0.552', '실습 1-3: 실험값 2,978원'),
]
for y, left, right in rows:
    box(ax, 2.8, y, 4.4, 0.62, left, '#f7faff', BLUE_E, fs=10, bold=False)
    box(ax, 8.0, y, 4.4, 0.62, right, '#f7fff9', GREEN_E, fs=10, bold=False)

ax.text(5.5, 0.05, '실습 1-2에서 쿠폰 효과를 단순 비교하면 12,710원, 참값은 3,000원이다',
        ha='center', fontsize=10.5, color=RED_E)
ax.set_title('그림 1-1. 예측과 인과는 다른 질문에 답한다',
             fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '1-1.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 1-2. 5단계 순환
# ---------------------------------------------------------------------------
fig, ax = canvas(w=10, h=5.8, xlim=(0, 11), ylim=(0, 6.4))

stages = [
    (2.0, 5.0, '1. 문제 정의', '데이터에서 문제를 찾는다'),
    (5.5, 5.0, '2. 대안 설계', '가능한 개입을 만든다'),
    (9.0, 5.0, '3. 의사결정', '무엇을 할지 고른다'),
    (9.0, 2.2, '4. 적응형 실행', '반응을 보며 조정한다'),
    (5.5, 2.2, '5. 평가와 학습', '효과를 측정한다'),
]
colors = [YELLOW, YELLOW, BLUE, BLUE, GREEN]
edges = [YELLOW_E, YELLOW_E, BLUE_E, BLUE_E, GREEN_E]

for (x, y, title, sub), fc, ec in zip(stages, colors, edges):
    box(ax, x, y, 3.0, 1.1, f'{title}\n{sub}', fc, ec, fs=10)

arrow(ax, (3.5, 5.0), (4.0, 5.0))
arrow(ax, (7.0, 5.0), (7.5, 5.0))
arrow(ax, (9.0, 4.45), (9.0, 2.75))
arrow(ax, (7.5, 2.2), (7.0, 2.2))
arrow(ax, (4.4, 2.6), (2.4, 4.45), color=GREEN_E, rad=0.25)

ax.text(3.0, 3.3, '평가 결과가\n다음 순환의\n문제 정의로 간다',
        ha='center', va='center', fontsize=10, color=GREEN_E, fontweight='bold')
ax.text(5.5, 0.5, '전통적 모델은 1에서 5까지 한 번 지나가고 끝난다.\n'
                  '여기서는 5의 결과가 1로 돌아가 다시 돈다.',
        ha='center', fontsize=10.5)
ax.set_title('그림 1-2. 인과적 데이터 사이언스의 5단계 순환',
             fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '1-2.png'), dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 1-3. 이 교재의 지도
# ---------------------------------------------------------------------------
fig, ax = canvas(w=10.5, h=6.2, xlim=(0, 11.6), ylim=(0, 6.8))

box(ax, 5.8, 6.2, 6.4, 0.8, '무작위 배정이 가능한가', YELLOW, YELLOW_E, fs=12)

box(ax, 2.4, 4.6, 4.0, 0.95, '가능하다\n8장: A/B 테스트와 적응형 설계',
    GREEN, GREEN_E, fs=10)
box(ax, 8.6, 4.6, 4.4, 0.95, '불가능하다\n관찰 데이터에서 인과효과를 찾는다',
    RED, RED_E, fs=10)

box(ax, 1.7, 2.6, 3.0, 1.5,
    '비슷한 사람끼리 짝짓기\n\n2장 DML\n3장 성향점수 매칭', BLUE, BLUE_E, fs=9.5)
box(ax, 5.2, 2.6, 3.0, 1.5,
    '시점·경계를 이용\n\n4장 이중차분·합성통제\n5장 회귀불연속', BLUE, BLUE_E, fs=9.5)
box(ax, 8.7, 2.6, 3.0, 1.5,
    '외부 충격을 이용\n\n6장 도구변수\n7장 분위 회귀', BLUE, BLUE_E, fs=9.5)

box(ax, 5.8, 0.7, 9.6, 0.9,
    '데이터 종류가 다를 때  |  9장 시계열   10장 텍스트   11장 네트워크   12장 시뮬레이션',
    GRAY, GRAY_E, fs=10)

arrow(ax, (4.2, 5.8), (2.8, 5.15))
arrow(ax, (7.4, 5.8), (8.4, 5.15))
arrow(ax, (6.6, 4.12), (2.2, 3.38))
arrow(ax, (7.7, 4.12), (5.4, 3.38))
arrow(ax, (8.9, 4.12), (8.8, 3.38))

ax.set_title('그림 1-3. 이 교재에서 배울 방법의 지도',
             fontsize=13, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '1-3.png'), dpi=200, bbox_inches='tight')
plt.close()

print("생성 완료:")
for f in ['1-1.png', '1-2.png', '1-3.png']:
    p = os.path.join(OUT, f)
    print(f"  {f}  ({os.path.getsize(p)/1024:.0f} KB)")
