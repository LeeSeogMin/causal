"""
실습 10 보론. 텍스트로 만든 결과변수로 이중차분을 한다
======================================================

목적: "민원 전담팀을 도입한 지역에서 부정 민원 비율이 줄었는가"를 이중차분으로 추정한다
- 1단계: 민원 기록(지역 20곳 × 24개월)과 정책 도입 표를 읽는다
- 2단계: 감성 분류기로 민원마다 부정 여부를 붙이고, 지역·달별 부정 비율을 계산한다
- 3단계: 도입 지역과 비교 지역의 달별 부정 비율을 그린다 (평행 추세 확인)
- 4단계: 2×2 평균표와 이중차분 회귀(지역·달 고정효과)로 정책 효과를 추정한다
- 5단계: 도입 전 기간만으로 가짜 도입 시점을 넣어 본다 (위약 검사)
- 6단계: 자료를 만들 때 넣어 둔 효과(−10%p)와 비교한다

입력: practice/chapter10/data/complaint_policy_panel.csv, policy_regions.csv
      (make_policy_panel.py로 만든 가상 자료)
      practice/chapter10/data/complaint_sentiment_sample.csv (감성 분류기 학습용)
출력: results/10-appendix-did.log, diagrams/10-did-negative-share.png

수정 이력
  2026-11-18  새로 작성.
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

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sentence_transformers import SentenceTransformer

matplotlib.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
DIAGRAM = os.path.join(HERE, "..", "..", "..", "diagrams", "10-did-negative-share.png")
SEED = 42
BUILT_IN_EFFECT = -10.0   # 자료를 만들 때 넣어 둔 효과(%p): make_policy_panel.py

print("=" * 78)
print("실습 10 보론. 텍스트로 만든 결과변수로 이중차분을 한다")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1단계. 자료 읽기
# ---------------------------------------------------------------------------
print("\n[1단계] 민원 기록과 정책 도입 표 읽기")
print("-" * 78)

panel = pd.read_csv(os.path.join(DATA_DIR, "complaint_policy_panel.csv"), parse_dates=["접수일"])
policy = pd.read_csv(os.path.join(DATA_DIR, "policy_regions.csv"))
policy["도입"] = policy["전담팀_도입일"].notna().astype(int)
print(f"민원 {len(panel)}건, 지역 {panel['지역'].nunique()}곳, 기간 {panel['접수일'].min():%Y.%m} ~ {panel['접수일'].max():%Y.%m}")
print(f"전담팀 도입 지역 {policy['도입'].sum()}곳 (2025.01 도입), 비교 지역 {(1 - policy['도입']).sum()}곳")

# ---------------------------------------------------------------------------
# 2단계. 감성 분류 → 지역·달별 부정 비율
# ---------------------------------------------------------------------------
print("\n[2단계] 감성 분류기로 부정 여부를 붙이고 지역·달별 부정 비율 계산")
print("-" * 78)

src = pd.read_csv(os.path.join(DATA_DIR, "complaint_sentiment_sample.csv"))
X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
    src["민원문장"].tolist(), src["감성"].tolist(),
    test_size=0.3, random_state=SEED, stratify=src["감성"])
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(
    model.encode(X_tr_txt, show_progress_bar=False), y_tr)

# 분류기가 부정을 얼마나 잘 잡는지: 사람이 라벨을 붙인 검증 문장 36건과 대조
pred_te = clf.predict(model.encode(X_te_txt, show_progress_bar=False))
tn, fp, fn, tp = confusion_matrix(np.array(y_te) == "부정", pred_te == "부정").ravel()
sens, spec = tp / (tp + fn), tn / (tn + fp)
print(f"검증 문장 36건: 민감도(부정을 부정으로) {sens:.3f}, 특이도(부정 아님을 부정 아님으로) {spec:.3f}")
print(f"→ 비율 차이가 줄어드는 배수 (민감도 + 특이도 − 1) = {sens + spec - 1:.3f}")

unique_text = panel["민원문장"].unique()
label_of = dict(zip(unique_text, clf.predict(model.encode(list(unique_text), show_progress_bar=False))))
panel["부정"] = (panel["민원문장"].map(label_of) == "부정").astype(int)
panel["월"] = panel["접수일"].dt.to_period("M").astype(str)

cell = panel.groupby(["지역", "월"]).agg(민원수=("부정", "size"), 부정비율=("부정", "mean")).reset_index()
cell["부정비율"] *= 100
cell = cell.merge(policy[["지역", "도입"]], on="지역")
cell["이후"] = (cell["월"] >= "2025-01").astype(int)
cell["처치"] = cell["도입"] * cell["이후"]
print(f"분석 단위: 지역 × 달 = {len(cell)}칸, 칸마다 민원 {cell['민원수'].min()}~{cell['민원수'].max()}건")

# ---------------------------------------------------------------------------
# 3단계. 그림: 도입 지역과 비교 지역의 달별 부정 비율
# ---------------------------------------------------------------------------
print("\n[3단계] 도입 지역과 비교 지역의 달별 부정 비율 (평행 추세 확인)")
print("-" * 78)

trend = cell.groupby(["월", "도입"])["부정비율"].mean().unstack()
pre = trend[trend.index < "2025-01"]
gap = pre[1] - pre[0]
print(f"도입 전 12개월 동안 두 집단의 차이: 최소 {gap.min():.1f}%p ~ 최대 {gap.max():.1f}%p, 평균 {gap.mean():.1f}%p")

fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(trend))
ax.plot(x, trend[1], marker='o', linewidth=2, color='#d62728', label='도입 지역 (10곳)')
ax.plot(x, trend[0], marker='s', linewidth=2, color='#1f77b4', label='비교 지역 (10곳)')
ax.axvline(11.5, color='gray', linestyle='--')
ax.text(11.6, ax.get_ylim()[1] * 0.98, ' 2025.01 전담팀 도입', va='top', fontsize=11)
ax.set_xticks(x[::3])
ax.set_xticklabels([trend.index[i].replace('-', '.') for i in range(0, len(trend), 3)], rotation=45)
ax.set_xlabel('월', fontsize=12)
ax.set_ylabel('부정 민원 비율 (%, 분류기 판정)', fontsize=12)
ax.set_title('도입 지역과 비교 지역의 부정 민원 비율', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DIAGRAM, dpi=200, bbox_inches='tight')
print("✓ 그래프 저장: diagrams/10-did-negative-share.png")

# ---------------------------------------------------------------------------
# 4단계. 이중차분
# ---------------------------------------------------------------------------
print("\n[4단계] 이중차분: 2×2 평균표와 회귀")
print("-" * 78)

means = cell.groupby(["도입", "이후"])["부정비율"].mean().unstack()
means.index = ["비교 지역", "도입 지역"]
means.columns = ["도입 전", "도입 후"]
means["차이(후 − 전)"] = means["도입 후"] - means["도입 전"]
print(means.round(2).to_string())
did_simple = means.loc["도입 지역", "차이(후 − 전)"] - means.loc["비교 지역", "차이(후 − 전)"]
print(f"이중차분 = (도입 지역 변화) − (비교 지역 변화) = {did_simple:.2f}%p")

fit = smf.ols("부정비율 ~ 처치 + C(지역) + C(월)", data=cell).fit(
    cov_type="cluster", cov_kwds={"groups": cell["지역"]})
b = fit.params["처치"]
lo, hi = fit.conf_int().loc["처치"]
print(f"\n회귀: 부정비율 ~ 처치(도입 지역 × 도입 후) + 지역 고정효과 + 달 고정효과")
print(f"처치 계수 {b:.2f}%p, 95% 신뢰구간 [{lo:.2f}, {hi:.2f}] (지역 단위 군집 표준오차)")

# ---------------------------------------------------------------------------
# 5단계. 위약 검사: 도입 전 12개월만으로 가짜 도입 시점(2024.07)을 넣는다
# ---------------------------------------------------------------------------
print("\n[5단계] 위약 검사: 도입 전 기간에 가짜 도입 시점 2024.07을 넣는다")
print("-" * 78)

pre_cell = cell[cell["월"] < "2025-01"].copy()
pre_cell["가짜처치"] = pre_cell["도입"] * (pre_cell["월"] >= "2024-07").astype(int)
fit_p = smf.ols("부정비율 ~ 가짜처치 + C(지역) + C(월)", data=pre_cell).fit(
    cov_type="cluster", cov_kwds={"groups": pre_cell["지역"]})
lo_p, hi_p = fit_p.conf_int().loc["가짜처치"]
print(f"가짜 처치 계수 {fit_p.params['가짜처치']:.2f}%p, 95% 신뢰구간 [{lo_p:.2f}, {hi_p:.2f}]")

# ---------------------------------------------------------------------------
# 6단계. 넣어 둔 효과와 비교
# ---------------------------------------------------------------------------
print("\n[6단계] 자료에 넣어 둔 효과 −10%p와 비교")
print("-" * 78)
expected = BUILT_IN_EFFECT * (sens + spec - 1)
print(f"추정 {b:.2f}%p = 넣어 둔 효과의 {b / BUILT_IN_EFFECT:.0%}")
print(f"분류기 오차로 줄어들 것으로 계산한 값: −10 × {sens + spec - 1:.3f} = {expected:.2f}%p")
