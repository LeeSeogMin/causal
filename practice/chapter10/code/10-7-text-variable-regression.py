"""
실습 10-7. 텍스트로 만든 변수를 회귀식에 넣는다
================================================

목적: 이 장의 질문 "민원이 부정적일수록 처리 기간이 길어지는가"를 처음부터 끝까지 분석한다
- 1단계: 민원 처리 기록을 읽고 처리기간 = 완료일 - 접수일 을 계산한다
- 2단계: 감성 분류기로 민원마다 부정 여부(0/1)를 붙인다 (10-4b와 같은 분류기)
- 3단계: 처리기간 ~ 부정 + 분야 를 추정한다 (분야 통제 없는 식과 비교)
- 4단계: 자료를 만들 때 정한 효과(부정 민원 +2일)와 비교한다

입력: practice/chapter10/data/complaint_processing_sample.csv (가상 민원 처리 기록 1,200건,
      make_complaint_processing.py로 만든다)
      practice/chapter10/data/complaint_sentiment_sample.csv (감성 분류기 학습용)
출력: results/10-7-text-variable-regression.log

수정 이력
  2026-11-18  새로 작성.
  2026-11-18  사전 방식과 정답 라벨로 추정한 모형을 뺐다. 실제 분석처럼 분류기 라벨만 쓴다.
  2026-11-18  처리기간을 스크립트 안에서 만들지 않고, 처리 기록 파일의 접수일·완료일로 계산한다.
"""

# == RESULTS LOG (auto): 콘솔 출력과 그림을 results/ 에 저장 ==
import sys as _sys
from pathlib import Path as _Path

_results_dir = _Path(__file__).resolve().parent.parent / "results"
_results_dir.mkdir(exist_ok=True)

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for _s in self._streams:
            _s.write(data)
            _s.flush()
    def flush(self):
        for _s in self._streams:
            _s.flush()

_sys.stdout = _Tee(_sys.stdout, open(_results_dir / (_Path(__file__).stem + ".log"), "w", encoding="utf-8"))
# == /RESULTS LOG ==


import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
SEED = 42
TRUE_EFFECT = 2.0   # 자료를 만들 때 정한 값: 부정 민원이면 처리기간 +2일 (make_complaint_processing.py)

print("=" * 78)
print("실습 10-7. 텍스트로 만든 변수를 회귀식에 넣는다")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1단계. 민원 처리 기록 읽기와 처리기간 계산
# ---------------------------------------------------------------------------
print("\n[1단계] 민원 처리 기록 읽기: 처리기간 = 완료일 - 접수일")
print("-" * 78)

data = pd.read_csv(os.path.join(DATA_DIR, "complaint_processing_sample.csv"),
                   parse_dates=["접수일", "완료일"])
data["처리기간"] = (data["완료일"] - data["접수일"]).dt.days

print(f"민원 {len(data)}건, 열: {', '.join(data.columns[:5])}")
print(data[["분야", "민원문장", "접수일", "완료일", "처리기간"]].head(3).to_string(index=False))
print(f"\n처리기간: 평균 {data['처리기간'].mean():.2f}일, 최소 {data['처리기간'].min()}일, 최대 {data['처리기간'].max()}일")

# ---------------------------------------------------------------------------
# 2단계. 감성 분류기로 부정 여부 붙이기
# ---------------------------------------------------------------------------
print("\n[2단계] 감성 분류기로 민원마다 부정 여부(0/1) 붙이기")
print("-" * 78)

src = pd.read_csv(os.path.join(DATA_DIR, "complaint_sentiment_sample.csv"))
X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
    src["민원문장"].tolist(), src["감성"].tolist(),
    test_size=0.3, random_state=SEED, stratify=src["감성"])

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(
    model.encode(X_tr_txt, show_progress_bar=False), y_tr)

# 분류기 품질: 사람이 라벨을 붙인 검증 문장 36건과 대조
kappa = cohen_kappa_score(y_te, clf.predict(model.encode(X_te_txt, show_progress_bar=False)))
print(f"분류기 학습 {len(X_tr_txt)}건, 검증 문장 {len(X_te_txt)}건과 대조한 카파(κ) {kappa:.3f}")

data["감성"] = clf.predict(model.encode(data["민원문장"].tolist(), show_progress_bar=False))
data["부정"] = (data["감성"] == "부정").astype(int)
print(f"부정으로 분류된 민원: {data['부정'].sum()}건 / {len(data)}건 = {data['부정'].mean():.1%}")

print("\n분야별 민원 수, 부정 비율, 평균 처리기간:")
for f, sub in data.groupby("분야", sort=False):
    print(f"  {f:<5} {len(sub):>4}건, 부정 {sub['부정'].mean():.1%}, 평균 처리기간 {sub['처리기간'].mean():.2f}일")

# ---------------------------------------------------------------------------
# 3단계. 회귀분석
# ---------------------------------------------------------------------------
print("\n[3단계] 회귀분석: 처리기간 ~ 부정 + 분야")
print("-" * 78)

specs = [
    ("① 분야 통제함", "처리기간 ~ 부정 + C(분야)"),
    ("② 분야 통제 안 함", "처리기간 ~ 부정"),
]
rows = []
for name, formula in specs:
    fit = smf.ols(formula, data=data).fit()
    lo, hi = fit.conf_int().loc["부정"]
    rows.append((name, fit.params["부정"], lo, hi))

print(f"{'모형':<18}{'부정 계수(일)':>12}{'95% 신뢰구간':>22}")
for name, b, lo, hi in rows:
    print(f"{name:<18}{b:>10.2f}      [{lo:.2f}, {hi:.2f}]")

# ---------------------------------------------------------------------------
# 4단계. 자료를 만들 때 정한 효과와 비교
# ---------------------------------------------------------------------------
print("\n[4단계] 자료를 만들 때 정한 효과 2일과 비교")
print("-" * 78)
for name, b, lo, hi in rows:
    inside = "포함" if lo <= TRUE_EFFECT <= hi else "벗어남"
    print(f"{name:<18} 추정 {b:.2f}일 = 2일의 {b / TRUE_EFFECT:.0%}, 신뢰구간이 2일을 {inside}")
