"""
실습 10-2. 정책 문서 100건을 카테고리로 나누기
==============================================

목적
  실제 CSV에 든 정책 문서 100건을 10개 카테고리로 나눈다.
  같은 분류기(로지스틱 회귀)에 수치화 방식만 바꿔 넣고,
  어느 쪽을 골라야 하는지 판정 기준을 만든다.

입력
  ../data/policy_sample_data.csv  (정책문서, 카테고리, 수집일)

출력
  10-2-document-classification.png
  10-2-results-table.csv

수정 이력
  2026-08-17  이전 파일명 10-2-bert-gpt-comparison.py를 대체함.
    - 문제: 이전 코드는 계산을 하나도 하지 않았다. 'BERT 정확도 88.5%',
      'GPT 정확도 85.7%' 같은 값을 소스에 문자열로 적어 두고 표로 찍었다.
      실행해도 데이터를 읽지 않으므로 출력 숫자에 근거가 없었다.
    - 조치: 문서 100건을 실제로 읽어 학습·평가하도록 새로 썼다.
      출력되는 모든 숫자는 이 스크립트가 계산한 값이다.
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sentence_transformers import SentenceTransformer

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "policy_sample_data.csv")
SEED = 42

print("=" * 78)
print("실습 10-2. 정책 문서 100건을 카테고리로 나누기")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1단계. 데이터
# ---------------------------------------------------------------------------
print("\n[1단계] 데이터 확인")
print("-" * 78)

df = pd.read_csv(DATA)
docs = df["정책문서"].tolist()
labels = df["카테고리"].tolist()
cats = sorted(df["카테고리"].unique())

print(f"문서 수: {len(df)}")
print(f"카테고리 수: {len(cats)}")
print(f"카테고리당 문서 수: {df['카테고리'].value_counts().min()}~"
      f"{df['카테고리'].value_counts().max()}")
print(f"문서 길이(글자): 평균 {df['정책문서'].str.len().mean():.0f}, "
      f"최소 {df['정책문서'].str.len().min()}, 최대 {df['정책문서'].str.len().max()}")
print(f"\n첫 문서 앞 60자: {docs[0][:60]}…")
print(f"그 문서의 카테고리: {labels[0]}")

X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
    docs, labels, test_size=0.3, random_state=SEED, stratify=labels)
print(f"\n학습 {len(X_tr_txt)}건 / 평가 {len(X_te_txt)}건")

# ---------------------------------------------------------------------------
# 2단계. 수치화 방식 1 — 단어 빈도(TF-IDF)
# ---------------------------------------------------------------------------
print("\n[2단계] 수치화 1 — 단어 빈도(TF-IDF)")
print("-" * 78)

tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
Xtr_tf = tfidf.fit_transform(X_tr_txt)
Xte_tf = tfidf.transform(X_te_txt)
print(f"어휘 사전 크기: {len(tfidf.get_feature_names_out())}개")
print(f"학습 행렬 크기: {Xtr_tf.shape[0]}행 x {Xtr_tf.shape[1]}열")
print(f"0이 아닌 칸의 비율: {Xtr_tf.nnz / (Xtr_tf.shape[0] * Xtr_tf.shape[1]):.3f}")

clf_tf = LogisticRegression(max_iter=2000, random_state=SEED)
clf_tf.fit(Xtr_tf, y_tr)
pred_tf = clf_tf.predict(Xte_tf)
acc_tf = accuracy_score(y_te, pred_tf)
f1_tf = f1_score(y_te, pred_tf, average="macro", zero_division=0)
print(f"\n정확도: {acc_tf:.3f}  ({int(round(acc_tf * len(y_te)))}/{len(y_te)}건 적중)")
print(f"매크로 F1: {f1_tf:.3f}")

# ---------------------------------------------------------------------------
# 3단계. 수치화 방식 2 — 문장 임베딩
# ---------------------------------------------------------------------------
print("\n[3단계] 수치화 2 — 문장 임베딩")
print("-" * 78)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
Xtr_em = model.encode(X_tr_txt, show_progress_bar=False)
Xte_em = model.encode(X_te_txt, show_progress_bar=False)
print(f"학습 행렬 크기: {Xtr_em.shape[0]}행 x {Xtr_em.shape[1]}열")
print("모든 칸이 0이 아닌 실수다. 어휘 사전이 없으므로 새 단어가 나와도 열이 늘지 않는다.")

clf_em = LogisticRegression(max_iter=2000, random_state=SEED)
clf_em.fit(Xtr_em, y_tr)
pred_em = clf_em.predict(Xte_em)
acc_em = accuracy_score(y_te, pred_em)
f1_em = f1_score(y_te, pred_em, average="macro", zero_division=0)
print(f"\n정확도: {acc_em:.3f}  ({int(round(acc_em * len(y_te)))}/{len(y_te)}건 적중)")
print(f"매크로 F1: {f1_em:.3f}")

# ---------------------------------------------------------------------------
# 4단계. 결과표
# ---------------------------------------------------------------------------
print("\n[4단계] 결과표")
print("-" * 78)

res = pd.DataFrame([
    {"수치화 방식": "단어 빈도(TF-IDF)", "열 개수": Xtr_tf.shape[1],
     "정확도": round(acc_tf, 3), "매크로 F1": round(f1_tf, 3),
     "틀린 건수": int(len(y_te) - round(acc_tf * len(y_te))),
     "새 단어가 나오면": "열이 없어 무시된다"},
    {"수치화 방식": "문장 임베딩", "열 개수": Xtr_em.shape[1],
     "정확도": round(acc_em, 3), "매크로 F1": round(f1_em, 3),
     "틀린 건수": int(len(y_te) - round(acc_em * len(y_te))),
     "새 단어가 나오면": "열 개수가 그대로다"},
])
print(res.to_string(index=False))
res.to_csv(os.path.join(HERE, "10-2-results-table.csv"),
           index=False, encoding="utf-8-sig")
print("\n결과표 저장: 10-2-results-table.csv")

# ---------------------------------------------------------------------------
# 5단계. 어디서 틀렸는가
# ---------------------------------------------------------------------------
print("\n[5단계] 어디서 틀렸는가 (문장 임베딩 기준)")
print("-" * 78)

wrong = [(t, p, x) for t, p, x in zip(y_te, pred_em, X_te_txt) if t != p]
if wrong:
    print(f"틀린 {len(wrong)}건:")
    for t, p, x in wrong:
        print(f"  정답 {t} → 예측 {p}")
        print(f"    앞 50자: {x[:50]}…")
else:
    print("평가 30건을 모두 맞혔다.")

cm = confusion_matrix(y_te, pred_em, labels=cats)

print("\n카테고리별 적중 (문장 임베딩)")
per_cat = pd.DataFrame({
    "카테고리": cats,
    "평가 건수": cm.sum(axis=1),
    "맞힌 건수": np.diag(cm),
})
per_cat["적중률"] = (per_cat["맞힌 건수"] / per_cat["평가 건수"]).round(2)
print(per_cat.to_string(index=False))

# ---------------------------------------------------------------------------
# 6단계. 학습 문서 수를 줄이면
# ---------------------------------------------------------------------------
print("\n[6단계] 카테고리당 학습 문서 수를 줄이면")
print("-" * 78)

rng = np.random.RandomState(SEED)
tr_df = pd.DataFrame({"txt": X_tr_txt, "y": y_tr, "emb": list(Xtr_em)})
curve = []
for k in [1, 2, 3, 5, 7]:
    idx = tr_df.groupby("y", group_keys=False).apply(
        lambda g: g.sample(n=min(k, len(g)), random_state=SEED)).index
    sub = tr_df.loc[idx]
    m = LogisticRegression(max_iter=2000, random_state=SEED)
    m.fit(np.vstack(sub["emb"].values), sub["y"].values)
    a = accuracy_score(y_te, m.predict(Xte_em))
    curve.append({"카테고리당 학습 문서": k, "총 학습 문서": len(sub),
                  "평가 정확도": round(a, 3)})
curve_df = pd.DataFrame(curve)
print(curve_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 7단계. 그림
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.bar(["단어 빈도", "문장 임베딩"], [acc_tf, acc_em],
       color=["#f0a868", "#5b8dd6"], edgecolor="black")
for i, v in enumerate([acc_tf, acc_em]):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.12)
ax.set_ylabel("평가 정확도", fontsize=11)
ax.set_title("(a) 수치화 방식별 평가 정확도", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=90, fontsize=8)
ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=8)
for i in range(len(cats)):
    for j in range(len(cats)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
ax.set_xlabel("예측", fontsize=10); ax.set_ylabel("정답", fontsize=10)
ax.set_title("(b) 혼동 행렬 (문장 임베딩)", fontsize=12, fontweight="bold")

ax = axes[2]
ax.plot(curve_df["카테고리당 학습 문서"], curve_df["평가 정확도"],
        marker="o", linewidth=2, color="#5b8dd6")
for _, r in curve_df.iterrows():
    ax.annotate(f"{r['평가 정확도']:.2f}",
                (r["카테고리당 학습 문서"], r["평가 정확도"]),
                textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
ax.set_xlabel("카테고리당 학습 문서 수", fontsize=11)
ax.set_ylabel("평가 정확도", fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title("(c) 학습 문서 수와 정확도", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(HERE, "10-2-document-classification.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n그림 저장: {os.path.basename(out)}")

print("\n" + "=" * 78)
print("정리")
print("=" * 78)
print(f"평가 30건에서 단어 빈도 {acc_tf:.3f}, 문장 임베딩 {acc_em:.3f}.")
print(f"학습 문서를 카테고리당 {curve_df.iloc[0]['카테고리당 학습 문서']}건까지 줄여도 "
      f"정확도가 {curve_df.iloc[0]['평가 정확도']:.3f}이다.")
print("=" * 78)
