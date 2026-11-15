"""
10장 쉬운 읽기(lecture/ch10.md)용 그림 생성
==========================================

실행: python diagrams/make_chapter10_easy.py
출력: diagrams/10e-1.png ~ 10e-5.png

- 10e-1.png: 네 문장의 문서-단어 행렬(4행 x 17열)과 A-C 유사도 계산
- 10e-2.png: 문장 쌍 세 개의 유사도 — 단어 세기와 문장 임베딩
- 10e-3.png: 수치화 방식별 분류 성적과 학습 문서 수에 따른 정확도
- 10e-4.png: 딥러닝과 준실험설계의 역할 분담
- 10e-5.png: 어텐션 — 한 단어의 숫자를 만들 때 같은 문장의 단어를 참고하는 정도

10e-2의 수치는 practice/chapter10/code/10-1-similarity-table.csv,
10e-3 왼쪽의 수치는 practice/chapter10/code/10-2-results-table.csv 에서 옮긴 값이다.
10e-3 오른쪽의 수치는 lecture/ch10.md 10.4절이 인용한
results/10-2-document-classification.log 6단계 값이다.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_available = {f.name for f in font_manager.fontManager.ttflist}
for _name in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans KR']:
    if _name in _available:
        plt.rcParams['font.family'] = _name
        break
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))

BLUE = '#d9e8ff'; BLUE_E = '#2c6fbb'
GREEN = '#d9f2d9'; GREEN_E = '#2e8b57'
RED = '#ffd9d9'; RED_E = '#c0392b'
YELLOW = '#fff2cc'; YELLOW_E = '#d6a300'
GRAY = '#eeeeee'; GRAY_E = '#888888'


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=200, bbox_inches='tight')
    plt.close()


def box(ax, x, y, w, h, text, fc, ec, fs=11, bold=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fc, edgecolor=ec, linewidth=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal')


def arrow(ax, p1, p2, color='#333333', width=2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=18,
                                 color=color, linewidth=width, shrinkA=4, shrinkB=4))


# ---------------------------------------------------------------------------
# 10e-1. 네 문장의 문서-단어 행렬(4행 x 17열)
#   값은 CountVectorizer(token_pattern=r"(?u)\b\w+\b")를 네 문장에 적용한 결과다.
#   열 순서는 CountVectorizer가 정한 순서(유니코드 순)를 그대로 따른다.
# ---------------------------------------------------------------------------
vocab = ['31도입니다', '걸려', '낮', '너무', '느립니다', '만족합니다', '민원', '불편합니다', '빨라져서',
         '서울', '속도가', '오늘', '오래', '절차가', '처리', '최고기온은', '행정']
dtm = np.array([
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],   # A
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],   # B
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],   # C
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],   # D
])
rows = ['A  민원 처리 속도가 너무 느립니다', 'B  행정 절차가 오래 걸려 불편합니다',
        'C  민원 처리 속도가 빨라져서 만족합니다', 'D  오늘 서울 낮 최고기온은 31도입니다']
shared = [j for j in range(17) if dtm[0, j] == 1 and dtm[2, j] == 1]   # A와 C가 함께 1인 열
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(-6.2, 18.6); ax.set_ylim(-2.6, 6.4); ax.axis('off')
for j, wd in enumerate(vocab):
    ax.text(j + 0.5, 4.15, wd, ha='left', va='bottom', rotation=45, fontsize=9.5,
            color=RED_E if j in shared else '#333333',
            fontweight='bold' if j in shared else 'normal')
for i in range(4):
    y = 3 - i
    ax.text(-0.15, y + 0.5, rows[i], ha='right', va='center', fontsize=10, fontweight='bold')
    for j in range(17):
        v = dtm[i, j]
        ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=BLUE if v else 'white',
                                   edgecolor=GRAY_E, linewidth=0.8))
        ax.text(j + 0.5, y + 0.5, str(v), ha='center', va='center', fontsize=10.5,
                fontweight='bold' if v else 'normal', color='#111111' if v else '#aaaaaa')
    ax.text(17.25, y + 0.5, f'1이 {dtm[i].sum()}칸', ha='left', va='center', fontsize=9.5)
for j in shared:   # A행과 C행에서 함께 1인 칸에 빨간 테두리
    for y in (3, 1):
        ax.add_patch(plt.Rectangle((j, y), 1, 1, fill=False, edgecolor=RED_E, linewidth=2.5))
ax.text(8.5, -0.75, 'A와 C가 함께 1인 칸(빨간 테두리) 3개 → 내적 3      A와 B, A와 D가 함께 1인 칸 0개 → 내적 0',
        ha='center', va='center', fontsize=10.5, fontweight='bold', color=RED_E)
ax.text(8.5, -1.75, 'A와 C의 유사도 = 함께 1인 칸 3개 ÷ 각 문장에서 1인 칸 5개 = 0.600',
        ha='center', va='center', fontsize=11.5, fontweight='bold')
save('10e-1.png')

# ---------------------------------------------------------------------------
# 10e-2. 문장 쌍 세 개의 유사도
# ---------------------------------------------------------------------------
pairs = ['A-B\n뜻이 가깝고\n겹치는 단어가 없다', 'A-C\n단어가 겹치고\n뜻이 반대다', 'A-D\n주제가 다르다']
count = [0.0, 0.6, 0.0]
emb = [0.745, 0.805, 0.357]
x = np.arange(3); w = 0.36
fig, ax = plt.subplots(figsize=(9, 4.4))
b1 = ax.bar(x - w / 2, count, w, color=GRAY, edgecolor=GRAY_E, linewidth=1.5, label='단어 세기')
b2 = ax.bar(x + w / 2, emb, w, color=BLUE, edgecolor=BLUE_E, linewidth=1.5, label='문장 임베딩')
for bars, vals in [(b1, count), (b2, emb)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.3f}', ha='center',
                va='bottom', fontsize=11, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=10.5)
ax.set_ylim(0, 1.0); ax.set_ylabel('코사인 유사도')
ax.set_title('두 방식 모두 뜻이 반대인 A-C를 가장 가깝다고 판정한다', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
save('10e-2.png')

# ---------------------------------------------------------------------------
# 10e-3. 분류 성적과 학습 문서 수
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
a = axes[0]
x = np.arange(2); w = 0.34
acc = [0.467, 0.900]; f1 = [0.419, 0.912]
b1 = a.bar(x - w / 2, acc, w, color=BLUE, edgecolor=BLUE_E, linewidth=1.5, label='정확도')
b2 = a.bar(x + w / 2, f1, w, color=GREEN, edgecolor=GREEN_E, linewidth=1.5, label='매크로 F1')
for bars, vals in [(b1, acc), (b2, f1)]:
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.3f}', ha='center',
               va='bottom', fontsize=10.5, fontweight='bold')
a.axhline(0.1, color=RED_E, linestyle='--', linewidth=1.5)
a.text(0.5, 0.12, '무작위로 고르면 0.1', ha='center', va='bottom', fontsize=9.5, color=RED_E)
a.set_xticks(x); a.set_xticklabels(['단어 빈도(TF-IDF)\n30건 중 16건 오답', '문장 임베딩\n30건 중 3건 오답'], fontsize=10)
a.set_ylim(0, 1.1); a.set_title('같은 분류기, 수치화 방식만 교체', fontsize=12)
a.legend(loc='upper left', fontsize=9.5); a.grid(axis='y', alpha=0.3)

a = axes[1]
n = [1, 2, 3, 5, 7]; curve = [0.533, 0.767, 0.867, 0.800, 0.900]
a.plot(n, curve, color=BLUE_E, linewidth=2.2, marker='o', markersize=7)
for xi, v in zip(n, curve):
    if xi == 1:
        a.text(xi + 0.15, v - 0.01, f'{v:.3f}', ha='left', va='top', fontsize=10, fontweight='bold')
    else:
        a.text(xi, v + 0.03, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
a.set_xticks(n); a.set_xlabel('카테고리당 학습 문서 수(건)')
a.set_ylim(0.4, 1.0); a.set_xlim(0.6, 7.4); a.set_ylabel('정확도')
a.set_title('학습 문서를 늘릴 때의 정확도(문장 임베딩)', fontsize=12)
a.grid(alpha=0.3)
save('10e-3.png')

# ---------------------------------------------------------------------------
# 10e-4. 딥러닝과 준실험설계의 역할 분담
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis('off')
labels = ['① 기사·민원\n수집', '② 감성 분류로 부정도를\n월별·지역별 집계',
          '③ 정책 도입 시점 기준\n이중차분', '④ 다섯 함정 점검 결과를\n추정치와 함께 보고']
cols = [(GRAY, GRAY_E), (BLUE, BLUE_E), (GREEN, GREEN_E), (YELLOW, YELLOW_E)]
xs = [1.5, 4.5, 7.5, 10.5]
for x, s, (fc, ec) in zip(xs, labels, cols):
    box(ax, x, 2.6, 2.5, 1.2, s, fc, ec, fs=10)
for i in range(3):
    arrow(ax, (xs[i] + 1.25, 2.6), (xs[i + 1] - 1.25, 2.6))
ax.plot([0.3, 5.7], [1.5, 1.5], color=BLUE_E, linewidth=3)
ax.text(3.0, 1.05, '딥러닝: 변수를 만든다 (상관을 학습한다)', ha='center', va='center',
        fontsize=11, fontweight='bold', color=BLUE_E)
ax.plot([6.3, 11.7], [1.5, 1.5], color=GREEN_E, linewidth=3)
ax.text(9.0, 1.05, '준실험설계(3~9장): 인과효과를 추정한다', ha='center', va='center',
        fontsize=11, fontweight='bold', color=GREEN_E)
save('10e-4.png')

# ---------------------------------------------------------------------------
# 10e-5. 어텐션 — 한 단어의 숫자를 만들 때 같은 문장의 다른 단어를 참고하는 정도
#   왼쪽의 선 굵기는 개념도다(측정값이 아니다). lecture/ch10.md 10.2절의 서술
#   "정부·발표를 많이 참고하고 조사 '는'은 거의 무시한다"만 옮겼다.
#   오른쪽은 같은 절의 가상 수치 [6, 2, 1] -> [0.976, 0.018, 0.007] 이다.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [2.1, 1]})
ax = axes[0]
ax.set_xlim(0, 12); ax.set_ylim(-1.0, 5.2); ax.axis('off')
toks = ['정부', '는', '새로운', '정책', '을', '발표했다']
xs = [1.2, 2.9, 4.8, 6.9, 8.5, 10.4]
ws = [1.4, 0.9, 1.6, 1.5, 0.9, 1.9]
fills = [(BLUE, BLUE_E), (GRAY, GRAY_E), ('white', GRAY_E), (YELLOW, YELLOW_E), ('white', GRAY_E), (BLUE, BLUE_E)]
for x, t, w, (fc, ec) in zip(xs, toks, ws, fills):
    box(ax, x, 1.6, w, 0.9, t, fc, ec, fs=13)
ax.add_patch(FancyArrowPatch((xs[0], 2.1), (xs[3] - 0.25, 2.1), connectionstyle='arc3,rad=-0.38',
                             arrowstyle='-|>', mutation_scale=22, color=BLUE_E, linewidth=5))
ax.add_patch(FancyArrowPatch((xs[5], 2.1), (xs[3] + 0.25, 2.1), connectionstyle='arc3,rad=0.42',
                             arrowstyle='-|>', mutation_scale=22, color=BLUE_E, linewidth=5))
ax.add_patch(FancyArrowPatch((xs[1], 1.1), (xs[3] - 0.45, 1.1), connectionstyle='arc3,rad=0.4',
                             arrowstyle='-|>', mutation_scale=12, color=GRAY_E, linewidth=1, linestyle='--'))
ax.text(2.6, 4.25, '많이 참고한다', ha='center', fontsize=11, fontweight='bold', color=BLUE_E)
ax.text(9.6, 3.55, '많이 참고한다', ha='center', fontsize=11, fontweight='bold', color=BLUE_E)
ax.text(4.3, 0.0, '거의 무시한다', ha='center', fontsize=10, color='#555555')
ax.text(xs[3] - 0.1, 0.72, '숫자를 만들 단어', ha='left', fontsize=10, fontweight='bold', color=YELLOW_E)
ax.text(6.0, -0.8, '선의 굵기 = 참고하는 정도(가중치).  한 단어가 받는 가중치를 모두 더하면 1이다',
        ha='center', fontsize=10.5, fontweight='bold')
ax.set_title('"정책"의 숫자를 만들 때 같은 문장의 어느 단어를 참고하는가', fontsize=12)

a = axes[1]
wts = [0.976, 0.018, 0.007]
bars = a.bar(range(3), wts, 0.55, color=BLUE, edgecolor=BLUE_E, linewidth=1.5)
for b, v in zip(bars, wts):
    a.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.3f}', ha='center', va='bottom',
           fontsize=11, fontweight='bold')
a.set_xticks(range(3)); a.set_xticklabels(['단어 1\n점수 6', '단어 2\n점수 2', '단어 3\n점수 1'], fontsize=10)
a.set_ylim(0, 1.15); a.set_ylabel('가중치(합 = 1)')
a.set_title('가상의 예: 점수를 가중치로 바꾼다', fontsize=12)
a.grid(axis='y', alpha=0.3)
save('10e-5.png')

print('saved: 10e-1.png ~ 10e-5.png')
