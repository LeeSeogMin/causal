"""
8장 쉬운 읽기(lecture/ch08-easy.md)용 그림 생성
================================================

실행: python diagrams/make_chapter08_easy.py
출력: diagrams/8e-1.png ~ 8e-3.png

- 8e-1.png: 똑같은 두 화면도 100명씩 보여 주면 결과가 달라진다
- 8e-2.png: 중간에 여러 번 들여다보면 속는 실험이 늘어난다
- 8e-3.png: 좋아 보이는 화면에 손님을 점점 더 많이 보낸다

8e-2, 8e-3의 수치는 practice/chapter08/results/*.log 에서 옮긴 값이다.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

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


# ---------------------------------------------------------------------------
# 8e-1. 똑같은 두 화면을 100명씩 보여 주는 실험을 20번 반복
# ---------------------------------------------------------------------------
rng = np.random.default_rng(11)
n_rep, n = 20, 100
a = rng.binomial(n, 0.40, n_rep)        # 두 화면 모두 실제로는 100명 중 40명이 누른다
b = rng.binomial(n, 0.40, n_rep)
diff = b - a
big = int(np.sum(np.abs(diff) >= 10))

fig, ax = plt.subplots(figsize=(10, 4.4))
cols = [RED if abs(d) >= 10 else GRAY for d in diff]
edges = [RED_E if abs(d) >= 10 else GRAY_E for d in diff]
ax.bar(np.arange(1, n_rep + 1), diff, color=cols, edgecolor=edges, linewidth=1.6)
ax.axhline(0, color='black', linewidth=1.2)
for lim in (10, -10):
    ax.axhline(lim, color=RED_E, linestyle='--', linewidth=1.5)
ax.text(20.6, 10.5, '10명 이상 차이', color=RED_E, fontsize=10, ha='right',
        va='bottom', fontweight='bold')
ax.set_xticks(np.arange(1, n_rep + 1))
ax.set_xlabel('실험 번호 (같은 실험을 20번 반복)', fontsize=11)
ax.set_ylabel('B를 누른 사람 수에서\nA를 누른 사람 수를 뺀 값 (명)', fontsize=10.5)
ax.set_ylim(min(diff.min(), -12) - 3, max(diff.max(), 12) + 4)
ax.grid(axis='y', alpha=0.3)
ax.set_title('그림 8e-1. 똑같은 두 화면도 100명씩 보여 주면 누른 사람 수가 매번 다르다',
             fontsize=12.5, fontweight='bold', pad=10)
fig.text(0.5, -0.04,
         f'두 화면 모두 실제로는 100명 중 40명이 누르도록 만든 가상 실험이다. '
         f'20번 중 {big}번은 10명 이상 차이가 났다.',
         ha='center', fontsize=10.5, color='#333333')
save('8e-1.png')
print('8e-1 차이:', diff.tolist(), '10명 이상:', big)


# ---------------------------------------------------------------------------
# 8e-2. 중간에 들여다본 횟수와 속는 실험의 비율 (8-2 로그 1단계)
# ---------------------------------------------------------------------------
K = ['1번', '2번', '3번', '5번', '10번', '20번']
err = [0.0498, 0.0825, 0.1062, 0.1409, 0.1911, 0.2484]
fig, ax = plt.subplots(figsize=(9.5, 4.4))
bars = ax.bar(K, [e * 100 for e in err], color=RED, edgecolor=RED_E, linewidth=1.8)
for bar, e in zip(bars, err):
    ax.text(bar.get_x() + bar.get_width() / 2, e * 100 + 0.6,
            f'100번 중\n{e * 100:.0f}번', ha='center', fontsize=10,
            fontweight='bold')
ax.set_xlabel('실험 도중에 결과를 들여다본 횟수', fontsize=11)
ax.set_ylabel('효과가 없는데 "있다"고 속은\n실험의 비율 (%)', fontsize=10.5)
ax.set_ylim(0, 31)
ax.grid(axis='y', alpha=0.3)
ax.set_title('그림 8e-2. 좋아 보일 때 멈추면, 많이 들여다볼수록 더 자주 속는다',
             fontsize=12.5, fontweight='bold', pad=10)
fig.text(0.5, -0.03,
         '두 화면이 실제로 똑같은 실험을 20만 번 반복해서 센 값이다(실습 8-2).',
         ha='center', fontsize=10.5, color='#333333')
save('8e-2.png')


# ---------------------------------------------------------------------------
# 8e-3. 좋아 보이는 화면에 보내는 비율 (8-3 로그 3단계)
# ---------------------------------------------------------------------------
t = ['50명째', '100명째', '200명째', '500명째', '1,000명째']
best = [76.0, 84.0, 91.0, 96.0, 97.9]
others = [100 - v for v in best]
fig, ax = plt.subplots(figsize=(9.5, 4.4))
x = np.arange(len(t))
ax.bar(x, best, color=GREEN, edgecolor=GREEN_E, linewidth=1.8,
       label='가장 좋은 화면(100명 중 55명이 누름)')
ax.bar(x, others, bottom=best, color=GRAY, edgecolor=GRAY_E, linewidth=1.8,
       label='나머지 두 화면(30명, 40명이 누름)')
for xi, v in zip(x, best):
    ax.text(xi, v / 2, f'{v:g}%', ha='center', va='center', fontsize=11.5,
            fontweight='bold', color=GREEN_E)
ax.axhline(33.3, color=BLUE_E, linestyle='--', linewidth=2,
           label='세 화면에 똑같이 나눴다면 33%')
ax.set_xticks(x)
ax.set_xticklabels(t)
ax.set_ylabel('그때까지 보낸 손님의 비율 (%)', fontsize=10.5)
ax.set_ylim(0, 128)
ax.legend(fontsize=9.5, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
ax.set_title('그림 8e-3. 좋아 보이는 화면에 손님을 점점 더 많이 보낸다',
             fontsize=12.5, fontweight='bold', pad=10)
save('8e-3.png')

print("생성 완료:")
for f in ['8e-1.png', '8e-2.png', '8e-3.png']:
    p = os.path.join(OUT, f)
    print(f"  {f}  ({os.path.getsize(p) / 1024:.0f} KB)")
