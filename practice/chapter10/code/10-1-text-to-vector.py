"""
실습 10-1. 문장을 숫자로 바꾸는 두 가지 방법
============================================

목적
  짧은 한국어 문장 네 개를 (1) 단어 세기 벡터와 (2) 문장 임베딩으로 바꾸고,
  두 방식이 문장 사이 거리를 어떻게 다르게 재는지 숫자로 확인한다.

출력
  10-1-text-to-vector.png

수정 이력
  2026-08-17  이전 파일명 10-1-nlp-comparison.py를 대체함.
    - 문제 1: 이전 코드는 카테고리 이름을 본문에 그대로 넣은 문장을 만들어
      분류했다. 라벨이 입력에 들어 있으므로 정확도가 과제 난이도를 재지 못했다.
    - 문제 2: 결과를 'TF-IDF + Naive Bayes vs SVM 성능 순위표'로 출력했다.
      강의노트 작성규칙에서 성능 순위표를 싣지 않기로 했다.
    - 조치: 과제를 '문장을 숫자로 바꾼다'로 바꾸고, 벡터 값을 직접 출력해
      학생이 손으로 따라 계산할 수 있게 했다. 순위표는 없앴다.
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
np.set_printoptions(precision=3, suppress=True)

SENTENCES = [
    "민원 처리 속도가 너무 느립니다",       # A
    "행정 절차가 오래 걸려 불편합니다",     # B  A와 뜻이 가깝고 겹치는 단어가 없다
    "민원 처리 속도가 빨라져서 만족합니다",  # C  A와 단어가 겹치고 뜻이 반대다
    "오늘 서울 낮 최고기온은 31도입니다",    # D  주제가 전혀 다르다
]
TAGS = ["A", "B", "C", "D"]

print("=" * 78)
print("실습 10-1. 문장을 숫자로 바꾸는 두 가지 방법")
print("=" * 78)

print("\n[1단계] 원문 네 문장")
print("-" * 78)
for tag, s in zip(TAGS, SENTENCES):
    print(f"  {tag}: {s}")

# ---------------------------------------------------------------------------
# 2단계. 단어 세기 벡터
# ---------------------------------------------------------------------------
print("\n[2단계] 방법 1 — 단어 세기 벡터")
print("-" * 78)

cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
counts = cv.fit_transform(SENTENCES).toarray()
vocab = cv.get_feature_names_out()

print(f"어휘 사전 크기: {len(vocab)}개")
print(f"어휘: {' / '.join(vocab)}")
print(f"\n각 문장은 {len(vocab)}칸짜리 숫자 목록이 된다.")

count_df = pd.DataFrame(counts, index=TAGS, columns=vocab)
print("\n단어 세기 행렬 (행=문장, 열=어휘, 값=등장 횟수)")
print(count_df.to_string())

print("\n문장 A의 벡터:")
print(f"  {counts[0].tolist()}")

sim_count = cosine_similarity(counts)
print("\n코사인 유사도 (단어 세기 벡터)")
print(pd.DataFrame(sim_count, index=TAGS, columns=TAGS).round(3).to_string())

print("\n손으로 확인하기 — A와 C의 코사인 유사도")
a, c = counts[0].astype(float), counts[2].astype(float)
dot = float(np.dot(a, c))
na, nc = float(np.linalg.norm(a)), float(np.linalg.norm(c))
print(f"  내적           = {dot:.0f}   (두 문장에 함께 나온 단어 수)")
print(f"  A의 길이       = sqrt({int(na**2)}) = {na:.3f}")
print(f"  C의 길이       = sqrt({int(nc**2)}) = {nc:.3f}")
print(f"  코사인 유사도  = {dot:.0f} / ({na:.3f} x {nc:.3f}) = {dot / (na * nc):.3f}")

# ---------------------------------------------------------------------------
# 3단계. 문장 임베딩
# ---------------------------------------------------------------------------
print("\n[3단계] 방법 2 — 문장 임베딩")
print("-" * 78)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
emb = model.encode(SENTENCES, show_progress_bar=False)

print(f"임베딩 차원: {emb.shape[1]}칸")
print("어휘 사전과 무관하게 문장 길이에 상관없이 항상 같은 칸 수가 나온다.")
fmt = lambda v: "[" + ", ".join(f"{x:+.3f}" for x in v) + ", ...]"
print("\n문장 A 임베딩의 앞 8칸:")
print(f"  {fmt(emb[0][:8])}")
print("\n문장 B 임베딩의 앞 8칸:")
print(f"  {fmt(emb[1][:8])}")

sim_emb = cosine_similarity(emb)
print("\n코사인 유사도 (문장 임베딩)")
print(pd.DataFrame(sim_emb, index=TAGS, columns=TAGS).round(3).to_string())

# ---------------------------------------------------------------------------
# 4단계. 두 방식이 갈리는 지점
# ---------------------------------------------------------------------------
print("\n[4단계] 두 방식이 갈리는 지점")
print("-" * 78)

pairs = [(0, 1, "뜻이 가깝고 겹치는 단어가 없다"),
         (0, 2, "단어가 겹치고 뜻이 반대다"),
         (0, 3, "주제가 다르다")]

rows = []
for i, j, note in pairs:
    rows.append({
        "문장 쌍": f"{TAGS[i]}-{TAGS[j]}",
        "단어 세기": round(float(sim_count[i, j]), 3),
        "문장 임베딩": round(float(sim_emb[i, j]), 3),
        "두 문장의 관계": note,
    })
table = pd.DataFrame(rows)
print(table.to_string(index=False))

print(f"\nA-B: 단어 세기 {sim_count[0,1]:.3f} → 문장 임베딩 {sim_emb[0,1]:.3f}")
print(f"A-C: 단어 세기 {sim_count[0,2]:.3f} → 문장 임베딩 {sim_emb[0,2]:.3f}")
print(f"A-D: 단어 세기 {sim_count[0,3]:.3f} → 문장 임베딩 {sim_emb[0,3]:.3f}")

table.to_csv(os.path.join(HERE, "10-1-similarity-table.csv"),
             index=False, encoding="utf-8-sig")
print("\n표 저장: 10-1-similarity-table.csv")

# ---------------------------------------------------------------------------
# 5단계. 그림
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

ax = axes[0]
im = ax.imshow(counts, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(vocab)))
ax.set_xticklabels(vocab, rotation=90, fontsize=8)
ax.set_yticks(range(4))
ax.set_yticklabels([f"{t}: {s[:12]}…" for t, s in zip(TAGS, SENTENCES)], fontsize=9)
ax.set_title("(a) 단어 세기 행렬", fontsize=12, fontweight="bold")

for ax, mat, title in [(axes[1], sim_count, "(b) 유사도 — 단어 세기"),
                       (axes[2], sim_emb, "(c) 유사도 — 문장 임베딩")]:
    im = ax.imshow(mat, cmap="OrRd", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(TAGS, fontsize=11)
    ax.set_yticks(range(4)); ax.set_yticklabels(TAGS, fontsize=11)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=11,
                    color="white" if mat[i, j] > 0.6 else "black")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
out = os.path.join(HERE, "10-1-text-to-vector.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"그림 저장: {os.path.basename(out)}")

print("\n" + "=" * 78)
print("정리")
print("=" * 78)
print(f"단어 세기 벡터는 겹치는 단어가 없으면 0을 준다 (A-B = {sim_count[0,1]:.3f}).")
print(f"문장 임베딩은 같은 상황을 다른 단어로 써도 값을 준다 (A-B = {sim_emb[0,1]:.3f}).")
print("어느 쪽도 A와 C의 반대되는 뜻은 구분하지 못한다. 감정은 따로 재야 한다.")
print("=" * 78)
