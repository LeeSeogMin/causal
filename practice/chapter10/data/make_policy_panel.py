"""
10장 보론 실습 자료 만들기: 지역 20곳 × 24개월 민원 기록과 정책 도입 표
======================================================================

목적: 10-appendix-did.py(이중차분)가 읽을 가상 자료를 만든다.
- 지역 20곳, 2024.01~2025.12, 지역·달마다 민원 40건 = 19,200건
- 10개 지역이 2025.01부터 "민원 전담팀"을 도입했다
- 부정 민원일 확률 = 지역별 기본값 + 모든 지역에 공통인 달별 변화 + 정책 효과
    지역별 기본값: 0.30 ± 0.08, 도입 지역은 여기에 +0.06 (부정 민원이 많은 지역에 먼저 도입)
    달별 변화: 매달 +0.002, 여름(6~8월) +0.03
    정책 효과: 도입 지역 × 2025.01 이후 −0.10 (부정 민원 비율 10%p 감소)
- 민원 문장은 complaint_sentiment_sample.csv의 평가용 문장 36개에서 뽑는다
  (감성 분류기가 학습에 쓰지 않은 문장이다)

출력
- complaint_policy_panel.csv : 민원번호, 지역, 접수일, 민원문장
- policy_regions.csv         : 지역, 전담팀_도입일 (도입하지 않은 지역은 빈칸)
정답 감성 라벨과 위 규칙은 파일에 넣지 않는다.

수정 이력
  2026-11-18  새로 작성.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
N_REGIONS, N_MONTHS, PER_CELL = 20, 24, 40
POLICY_START = 12                  # 분기 번호가 아니라 달 번호: 12 = 2025.01
POLICY_EFFECT = -0.10

src = pd.read_csv(os.path.join(HERE, "complaint_sentiment_sample.csv"))
_, X_te, _, y_te = train_test_split(
    src["민원문장"].tolist(), src["감성"].tolist(),
    test_size=0.3, random_state=SEED, stratify=src["감성"])
sent = pd.DataFrame({"문장": X_te, "정답": y_te})
neg_sent = sent.loc[sent["정답"] == "부정", "문장"].to_numpy()
other_sent = sent.loc[sent["정답"] != "부정", "문장"].to_numpy()

rng = np.random.default_rng(SEED)
regions = [f"지역{i + 1:02d}" for i in range(N_REGIONS)]
treated = set(rng.choice(regions, size=N_REGIONS // 2, replace=False))
base = {r: 0.30 + rng.uniform(-0.08, 0.08) + (0.06 if r in treated else 0.0) for r in regions}

rows = []
for r in regions:
    for t in range(N_MONTHS):
        month = t % 12 + 1
        p = base[r] + 0.002 * t + (0.03 if month in (6, 7, 8) else 0.0)
        if r in treated and t >= POLICY_START:
            p += POLICY_EFFECT
        is_neg = rng.random(PER_CELL) < p
        texts = np.where(is_neg, rng.choice(neg_sent, PER_CELL), rng.choice(other_sent, PER_CELL))
        days = rng.integers(1, 29, PER_CELL)
        for d, text in zip(days, texts):
            rows.append((r, f"{2024 + t // 12}-{month:02d}-{d:02d}", text))

panel = pd.DataFrame(rows, columns=["지역", "접수일", "민원문장"])
panel.insert(0, "민원번호", [f"P{i + 1:05d}" for i in range(len(panel))])
panel.to_csv(os.path.join(HERE, "complaint_policy_panel.csv"), index=False)

policy = pd.DataFrame({"지역": regions,
                       "전담팀_도입일": ["2025-01-01" if r in treated else "" for r in regions]})
policy.to_csv(os.path.join(HERE, "policy_regions.csv"), index=False)

print(f"민원 {len(panel)}건 저장, 도입 지역 {len(treated)}곳: {', '.join(sorted(treated))}")
