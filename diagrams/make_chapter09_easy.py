"""
9장 쉬운 읽기(lecture/ch09.md)용 그림 생성
==========================================

실행: python diagrams/make_chapter09_easy.py
출력: diagrams/9e-1.png ~ 9e-3.png

- 9e-1.png: 전력 수요 한 주와 "어제 같은 시각" 예측
- 9e-2.png: 기준선 세 개와 딥러닝 모델 세 개의 평균 오차(MAE)
- 9e-3.png: 개입 전 직선을 그대로 늘린 반사실이 효과를 뒤집는다

9e-1은 practice/chapter09/data/9-1-preprocessed-data.csv 를 읽는다.
9e-2의 수치는 practice/chapter09/results/9-3-train-evaluate.log 에서 옮긴 값이다
(lecture/ch09-ex.md 9.5.2절 발췌).
9e-3은 practice/chapter09/code/9-counterfactual-with-stats.py 의 합성 자료와
직선 반사실 계산을 그대로 재현한다.
"""

import os
import numpy as np
import pandas as pd
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
ROOT = os.path.dirname(OUT)

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
# 9e-1. 전력 수요 한 주(월~일)와 "어제 같은 시각" 예측
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(ROOT, 'practice/chapter09/data/9-1-preprocessed-data.csv'),
                 parse_dates=['datetime'], encoding='utf-8-sig')
start = pd.Timestamp('2024-04-08 00:00:00')            # 월요일
week = df[(df['datetime'] >= start) & (df['datetime'] < start + pd.Timedelta(days=7))]
hours = np.arange(len(week))

fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(hours, week['demand'] / 1000, color=BLUE_E, linewidth=2.2, label='실제 수요')
ax.plot(hours, week['demand_lag_24h'] / 1000, color=RED_E, linewidth=1.6, linestyle='--',
        label='"어제 같은 시각" 값을 그대로 쓴 예측')
days = ['월', '화', '수', '목', '금', '토', '일']
ax.set_xticks(np.arange(12, 168, 24))
ax.set_xticklabels([f'{d}요일' for d in days])
for x in range(24, 168, 24):
    ax.axvline(x, color=GRAY_E, linewidth=0.6, alpha=0.5)
ax.set_ylabel('전력 수요 (천 MW)')
ax.set_title('전력 수요 한 주: 하루마다 비슷한 모양(낮에 오르고 밤에 내림)이 되풀이된다', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
save('9e-1.png')

# ---------------------------------------------------------------------------
# 9e-2. 평균 오차(MAE) 비교
# ---------------------------------------------------------------------------
names = ['직전 값\n반복', '최근 24시간\n평균', '어제\n같은 시각',
         'LSTM+\n어텐션', 'LSTM', 'GRU']
mae = [4451.4, 3299.8, 2023.3, 1559.5, 1477.5, 1456.0]
cols = [GRAY, GRAY, YELLOW, GREEN, GREEN, GREEN]
edges = [GRAY_E, GRAY_E, YELLOW_E, GREEN_E, GREEN_E, GREEN_E]

fig, ax = plt.subplots(figsize=(10, 4.4))
bars = ax.bar(names, mae, color=cols, edgecolor=edges, linewidth=1.6)
for b_, v in zip(bars, mae):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 80, f'{v:,.0f}', ha='center', fontsize=11)
ax.axhline(2023.3, color=YELLOW_E, linestyle='--', linewidth=1.5)
ax.text(5.45, 2023.3 + 90, '넘어야 할 선 (2,023)', ha='right', color=YELLOW_E, fontsize=10)
ax.set_ylabel('평균적으로 틀린 양 (MW)')
ax.set_ylim(0, 5000)
ax.set_title('회색·노랑: 학습 없는 규칙(기준선)   초록: 딥러닝 모델', fontsize=13)
ax.grid(axis='y', alpha=0.3)
save('9e-2.png')

# ---------------------------------------------------------------------------
# 9e-3. 직선 반사실이 효과를 뒤집는다 (9-counterfactual-with-stats.py 합성 자료)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 24 * 365
t = np.arange(n)
series = (60000.0 + 4000.0 * np.sin(2 * np.pi * t / (24 * 365))
          + 2500.0 * np.sin(2 * np.pi * (t % 24) / 24) + rng.normal(0.0, 800.0, size=n))
iv = int(n * 0.6)
slope, intercept = np.polyfit(t[:iv], series[:iv], 1)
straight = intercept + slope * t
actual = series.copy()
actual[iv:] += 600.0                                    # 참 효과 +600

day = lambda x: x.reshape(-1, 24).mean(axis=1)           # 하루 평균으로 줄여 그린다
d = np.arange(365)
iv_d = iv // 24
ev_end = iv_d + 30                                      # 개입 후 30일

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(d, day(actual) / 1000, color=BLUE_E, linewidth=1.4, label='실제 관측 (캠페인 효과 +600 포함)')
ax.plot(d[iv_d:], day(series)[iv_d:] / 1000, color=GREEN_E, linewidth=1.4, alpha=0.8,
        label='캠페인이 없었다면 (정답, 실제로는 볼 수 없음)')
ax.plot(d, day(straight) / 1000, color=RED_E, linewidth=2.2, linestyle='--',
        label='개입 전 추세를 직선으로 늘린 "가상의 수요"')
ax.axvline(iv_d, color='black', linewidth=1.2)
ax.text(iv_d + 2, 64.6, '캠페인 시작', fontsize=10)
ax.axvspan(iv_d, ev_end, color=YELLOW, alpha=0.6)
ax.text(iv_d + 16, 54.6, '효과를\n계산한 30일', ha='center', fontsize=9, color=YELLOW_E)
ax.set_xlabel('날짜 (1월 1일부터 센 날)')
ax.set_ylabel('하루 평균 수요 (천 MW)')
ax.set_ylim(54, 66)
ax.set_title('빨간 점선이 실제보다 높게 그어져, 수요가 "줄었다"고 잘못 계산된다', fontsize=13)
ax.legend(loc='lower left', fontsize=9.5)
ax.grid(alpha=0.3)
save('9e-3.png')

eff = (actual[iv:iv + 24 * 30] - straight[iv:iv + 24 * 30]).mean()
print(f'9e-3 확인: 직선 반사실로 계산한 평균 효과 = {eff:,.2f} (참 효과 +600)')
