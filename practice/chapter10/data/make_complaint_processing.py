"""
10.7 실습 자료 만들기: 가상 민원 처리 기록 1,200건
==================================================

목적: 10-7-text-variable-regression.py가 읽을 가상 민원 처리 기록을 만든다.
- 민원 문장은 complaint_sentiment_sample.csv의 평가용 문장 36개에서 뽑는다
  (10-4b·10-7과 같은 분할이므로 감성 분류기가 학습에 쓰지 않은 문장이다)
- 처리 일수 = 3일 + 부정 민원이면 2일 + 건축 4일·복지 1일 + 잡음(표준편차 1일), 정수로 반올림, 최소 1일
- 출력: complaint_processing_sample.csv (민원번호, 분야, 민원문장, 접수일, 완료일)
  정답 감성 라벨과 처리 일수 규칙은 파일에 넣지 않는다. 분석하는 쪽은 현장처럼 날짜와 글만 받는다.

수정 이력
  2026-11-18  새로 작성.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
N = 1200
TRUE_EFFECT = 2          # 부정 민원이면 처리 일수 +2일

src = pd.read_csv(os.path.join(HERE, "complaint_sentiment_sample.csv"))
_, X_te, _, y_te = train_test_split(
    src["민원문장"].tolist(), src["감성"].tolist(),
    test_size=0.3, random_state=SEED, stratify=src["감성"])
sent = pd.DataFrame({"문장": X_te, "정답": y_te})

rng = np.random.default_rng(SEED)
FIELDS = ["일반행정", "건축", "복지"]
field = rng.choice(FIELDS, size=N, p=[0.5, 0.25, 0.25])
p_negative = {"일반행정": 0.2, "건축": 0.6, "복지": 0.3}
is_negative = rng.random(N) < np.array([p_negative[f] for f in field])

neg_sent = sent.loc[sent["정답"] == "부정", "문장"].to_numpy()
other_sent = sent.loc[sent["정답"] != "부정", "문장"].to_numpy()
text = np.where(is_negative, rng.choice(neg_sent, size=N), rng.choice(other_sent, size=N))

extra_days = {"일반행정": 0, "건축": 4, "복지": 1}
days = (3 + TRUE_EFFECT * is_negative + np.array([extra_days[f] for f in field])
        + rng.normal(0, 1.0, N))
raw_days = np.rint(days)
days = np.maximum(raw_days, 1).astype(int)

received = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
out = pd.DataFrame({
    "민원번호": [f"C{i + 1:04d}" for i in range(N)],
    "분야": field,
    "민원문장": text,
    "접수일": received.strftime("%Y-%m-%d"),
    "완료일": (received + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d"),
})
path = os.path.join(HERE, "complaint_processing_sample.csv")
out.to_csv(path, index=False)
print(f"저장: {path} ({len(out)}건)")
print(f"처리 일수: 최소 {days.min()}일, 최대 {days.max()}일, 1일 미만이라 1일로 올린 건수 {(raw_days < 1).sum()}")
