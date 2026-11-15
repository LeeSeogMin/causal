"""
실습 10-4b. 민원 문장에서 감성 변수 만들기
==========================================

목적
  민원 문장 120건에 긍정·중립·부정을 붙여 "부정 비율"이라는 숫자를 만든다.
  두 가지 방법(단어 목록 대조, 학습한 분류기)으로 붙인 결과를
  정답 라벨과 대조해, 부정 비율에 오차가 얼마나 섞이는지 확인한다.

입력
  ../data/complaint_sentiment_sample.csv  (민원문장, 감성)
  - 실제 민원이 아니라 실습용으로 지어 쓴 가상 문장이다.
  - 감성 열은 문장을 쓸 때 함께 적은 값이다. 여러 사람이 따로 읽고
    붙인 라벨이 아니므로, 실제 연구의 "사람 라벨"보다 일관성이 높다.

출력
  10-4b-sentiment-results.csv   (평가 문장별 정답·사전 판정·분류기 판정)

수정 이력
  2026-11-15  새로 작성. 10-4-policy-analysis-suite.py의 감성 부분은
    문장 10개에 단어 11개를 대조할 뿐이어서 정답과 대조하는 단계가 없었다.
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

# 그림 저장을 results/ 로 돌린다 (강의자료용 diagrams/ 경로는 그대로 둔다)
try:
    from matplotlib.figure import Figure as _Fig
    if not getattr(_Fig, "_results_patched", False):
        _orig_savefig = _Fig.savefig
        def _savefig_to_results(self, fname, *a, **k):
            try:
                _p = _Path(fname)
                if "diagrams" not in [s.lower() for s in _p.parts]:
                    fname = _results_dir / _p.name
                    print(f"[그림 저장 → {fname}]")
            except TypeError:
                pass
            return _orig_savefig(self, fname, *a, **k)
        _Fig.savefig = _savefig_to_results
        _Fig._results_patched = True
except ImportError:
    pass
# == /RESULTS LOG ==


import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             cohen_kappa_score)
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "complaint_sentiment_sample.csv")
SEED = 42
ORDER = ["긍정", "중립", "부정"]

print("=" * 78)
print("실습 10-4b. 민원 문장에서 감성 변수 만들기")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1단계. 데이터
# ---------------------------------------------------------------------------
print("\n[1단계] 데이터 확인")
print("-" * 78)

df = pd.read_csv(DATA)
docs = df["민원문장"].tolist()
labels = df["감성"].tolist()

print(f"문장 수: {len(df)}")
for s in ORDER:
    print(f"  {s}: {(df['감성'] == s).sum()}건")
for i in range(3):
    print(f"예 {i + 1}: [{labels[i]}] {docs[i]}")

X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
    docs, labels, test_size=0.3, random_state=SEED, stratify=labels)
print(f"\n학습 {len(X_tr_txt)}건 / 평가 {len(X_te_txt)}건")

# ---------------------------------------------------------------------------
# 2단계. 사전 방식 — 단어 목록 대조
# ---------------------------------------------------------------------------
print("\n[2단계] 사전 방식: 긍정·부정 단어를 세어 많은 쪽으로 판정")
print("-" * 78)

positive_words = ["만족", "감사", "고맙", "친절", "좋", "편리", "편해",
                  "깨끗", "안전", "도움"]
negative_words = ["불편", "불친절", "느립", "늦", "실망", "걱정", "고장",
                  "악취", "않습니다", "없습니다"]


def lexicon_label(text):
    pos = sum(w in text for w in positive_words)
    neg = sum(w in text for w in negative_words)
    if pos > neg:
        return "긍정"
    if neg > pos:
        return "부정"
    return "중립"   # 같거나 둘 다 0


pred_lex = [lexicon_label(t) for t in X_te_txt]
print(f"긍정 단어 {len(positive_words)}개, 부정 단어 {len(negative_words)}개")
print("단어 수가 같거나 둘 다 0이면 중립으로 둔다")

# ---------------------------------------------------------------------------
# 3단계. 분류기 방식 — 문장 임베딩 + 로지스틱 회귀
# ---------------------------------------------------------------------------
print("\n[3단계] 분류기 방식: 학습 문장 84건으로 배우고 평가 문장에 붙인다")
print("-" * 78)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
X_tr = model.encode(X_tr_txt, show_progress_bar=False)
X_te = model.encode(X_te_txt, show_progress_bar=False)
clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_tr, y_tr)
pred_clf = clf.predict(X_te).tolist()
print(f"문장 하나 = 숫자 {X_tr.shape[1]}개, 학습 {len(X_tr)}건")

# ---------------------------------------------------------------------------
# 4단계. 정답 라벨과 대조
# ---------------------------------------------------------------------------
print("\n[4단계] 정답 라벨과 대조 (평가 문장 36건)")
print("-" * 78)

print(f"{'방법':<10}{'정확도':>8}{'매크로 F1':>11}{'카파(κ)':>10}")
for name, pred in [("사전", pred_lex), ("분류기", pred_clf)]:
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_te, pred)
    print(f"{name:<10}{acc:>10.3f}{f1:>12.3f}{kappa:>11.3f}")

for name, pred in [("사전", pred_lex), ("분류기", pred_clf)]:
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=ORDER),
                      index=[f"정답 {s}" for s in ORDER],
                      columns=[f"판정 {s}" for s in ORDER])
    print(f"\n{name} 방식의 대조표 (행 = 정답, 열 = 판정)")
    print(cm.to_string())

# ---------------------------------------------------------------------------
# 5단계. 부정 비율 — 2부에서 변수로 쓰는 숫자
# ---------------------------------------------------------------------------
print("\n[5단계] 평가 문장 36건의 부정 비율")
print("-" * 78)

n = len(y_te)
for name, pred in [("정답 라벨", y_te), ("사전 판정", pred_lex),
                   ("분류기 판정", pred_clf)]:
    k = sum(p == "부정" for p in pred)
    print(f"{name:<8}: 부정 {k}건 / {n}건 = {k / n * 100:.1f}%")

for name, pred in [("사전", pred_lex), ("분류기", pred_clf)]:
    wrong = [(t, y, p) for t, y, p in zip(X_te_txt, y_te, pred) if y != p]
    print(f"\n{name} 방식이 틀린 문장 {len(wrong)}건")
    for t, y, p in wrong:
        print(f"  정답 {y} → 판정 {p} | {t}")

out = pd.DataFrame({"민원문장": X_te_txt, "정답": y_te,
                    "사전판정": pred_lex, "분류기판정": pred_clf})
out_path = os.path.join(HERE, "10-4b-sentiment-results.csv")
out.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n결과표 저장: {os.path.basename(out_path)}")
