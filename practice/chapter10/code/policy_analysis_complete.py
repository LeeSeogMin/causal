"""
제10장: 정책 텍스트 분석 종합 실습
===================================

목적: 전체 정책 분석 파이프라인 통합 실행
- 데이터 준비 → 분류 → 토픽 모델링 → 연도별 토픽 비중 → 결과 요약

저자: AI 기반 정책분석방법론
날짜: 2025-11-20

수정 이력
  2026-11-18  계산하지 않고 코드에 직접 적어 둔 수치를 모두 지웠다.
    - 지운 것: 학습 시간 "8분 32초 (시뮬레이션)", 감정 분석 단계 전체
      (가상 피드백의 라벨을 그대로 세고, 카테고리별 긍정 비율 58%·55% 등을 적어 둠),
      시간 변화 단계의 토픽 비중(12% → 18% 등), 시사점·종합의 문구
      ("기존 TF-IDF 65% 대비", "20%p 향상", "디지털+교육 융합" 등)
    - 감성 분석은 10-4b-sentiment-variable.py에서 다룬다.
    - 토픽 모델링은 형태소 분석(kiwipiepy)으로 명사만 뽑은 뒤 묶는다.
    - 시간 변화는 데이터의 수집일 열로 연도별 토픽 비중을 계산한다.
    - 필요 패키지: pip install kiwipiepy
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


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from kiwipiepy import Kiwi
from collections import Counter
import matplotlib.pyplot as plt

np.random.seed(42)

print("=" * 90)
print("정책 텍스트 분석 종합 실습")
print("=" * 90)

# ===== 1. 데이터 준비 =====
print("\n" + "=" * 90)
print("[1단계] 데이터셋 준비")
print("=" * 90)

data_path = Path(__file__).resolve().parents[1] / "data" / "policy_sample_data.csv"
if data_path.exists():
    df = pd.read_csv(data_path)
    expected_cols = ["정책문서", "카테고리", "수집일"]
    if not set(expected_cols).issubset(set(df.columns)):
        raise ValueError(f"Expected columns {expected_cols} in {data_path}")
    df = df[expected_cols].dropna()
    categories = sorted(df["카테고리"].unique())
    print(f"데이터 로드: {data_path}")
else:
    categories = ['디지털정책', '경제정책', '환경정책', '보건정책', '고용정책',
                  '교육정책', '과학기술정책', '지역개발정책', '보안정책', '농업정책']

category_keywords = {
    '디지털정책': ['디지털', '데이터', 'AI', '플랫폼'],
    '경제정책': ['경제', '성장', '투자', 'GDP'],
    '환경정책': ['탄소중립', '재생에너지', '친환경', '그린'],
    '보건정책': ['건강', '의료', '예방', '질병'],
    '고용정책': ['일자리', '청년', '고용', '취업'],
    '교육정책': ['교육', '미래', 'AI교육', '학생'],
    '과학기술정책': ['연구', '개발', '혁신', '기술'],
    '지역개발정책': ['지역', '균형', '발전', '활성화'],
    '보안정책': ['사이버', '보안', '위협', '안전'],
    '농업정책': ['농업', '스마트팜', '농촌', '농산물']
}

if not data_path.exists():
    documents = []
    labels = []

    for category, keywords in category_keywords.items():
        for _ in range(10):
            kw_sample = np.random.choice(keywords, size=2, replace=False)
            doc = f"정부는 {' '.join(kw_sample)} 정책을 추진하여 {category} 발전을 도모한다."
            documents.append(doc)
            labels.append(category)

    df = pd.DataFrame({
        '정책문서': documents,
        '카테고리': labels
    })

print(f"총 문서 수: {len(df)}")
print(f"카테고리 수: {len(categories)}")
print(f"평균 텍스트 길이: {df['정책문서'].str.len().mean():.0f}자")

print("\n카테고리 분포:")
category_dist = df['카테고리'].value_counts()
for cat, count in category_dist.items():
    print(f"  {cat}: {count}개")

# ===== 2. BERT 기반 정책 분류 =====
print("\n" + "=" * 90)
print("[2단계] 문서 분류 (TF-IDF + 로지스틱 회귀 분류기)")
print("=" * 90)

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    df['정책문서'], df['카테고리'], 
    test_size=0.3, random_state=42, stratify=df['카테고리']
)

# TF-IDF (BERT 대체)
vectorizer = TfidfVectorizer(max_features=500)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 분류 모델
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train_vec, y_train)

# 예측
y_pred = classifier.predict(X_test_vec)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"정확도 (Accuracy): {accuracy:.1%}")
print(f"정밀도 (Precision): {precision:.3f}")
print(f"재현율 (Recall): {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# 혼동 행렬 분석
print("\n오분류 사례:")
errors = []
for i, (true, pred) in enumerate(zip(y_test, y_pred)):
    if true != pred:
        errors.append((true, pred))

error_counter = Counter(errors)
for (true_cat, pred_cat), count in error_counter.most_common(3):
    print(f"  {true_cat} → {pred_cat}: {count}건")

# ===== 3. 토픽 모델링 =====
# 감성 분석은 10-4b-sentiment-variable.py에서 다룬다 (2026-11-18 이 스크립트에서 삭제)
print("\n" + "=" * 90)
print("[3단계] 토픽 모델링 (형태소 분석 + TF-IDF + K-평균 군집)")
print("=" * 90)

# 형태소 분석: 명사(일반명사 NNG, 고유명사 NNP, 영문 SL)만 남긴다
kiwi = Kiwi()
NOUN_TAGS = {'NNG', 'NNP', 'SL'}


def extract_nouns(text):
    return ' '.join(t.form for t in kiwi.tokenize(text) if t.tag in NOUN_TAGS)


df['명사'] = df['정책문서'].apply(extract_nouns)

vec = TfidfVectorizer(max_features=100)
doc_vecs = vec.fit_transform(df['명사'])

n_topics = 8
kmeans = KMeans(n_clusters=n_topics, random_state=42)
df['토픽'] = kmeans.fit_predict(doc_vecs) + 1

print(f"묶음 수(사람이 정함): {n_topics}개")

feature_names = vec.get_feature_names_out()
topic_keywords = {}
for topic_id in range(n_topics):
    center = kmeans.cluster_centers_[topic_id]
    top_idx = center.argsort()[-4:][::-1]
    keywords = [feature_names[idx] for idx in top_idx]
    topic_keywords[topic_id + 1] = keywords
    doc_count = (df['토픽'] == topic_id + 1).sum()
    print(f"토픽 {topic_id + 1} ({doc_count}개): {', '.join(keywords)}")

# ===== 4. 연도별 토픽 비중 =====
print("\n" + "=" * 90)
print("[4단계] 연도별 토픽 비중 (수집일 기준)")
print("=" * 90)

if '수집일' in df.columns:
    df['연도'] = df['수집일'].astype(str).str[:4]
    year_counts = pd.crosstab(df['토픽'], df['연도'])
    year_totals = year_counts.sum(axis=0)
    print("연도별 문서 수(비중의 분모): " + ", ".join(f"{y}년 {n}건" for y, n in year_totals.items()))
    print("\n토픽별 문서 수와 비중:")
    years = list(year_counts.columns)
    for topic_id, row in year_counts.iterrows():
        parts = [f"{y}년 {row[y]}건({row[y] / year_totals[y]:.1%})" for y in years]
        print(f"  토픽 {topic_id} ({', '.join(topic_keywords[topic_id][:2])}): " + " → ".join(parts))
else:
    print("수집일 열이 없어 연도별 비중을 계산하지 않는다.")

# ===== 5. 결과 요약 =====
print("\n" + "=" * 90)
print("[5단계] 결과 요약 (이 실행에서 계산한 값)")
print("=" * 90)

smallest = category_dist.idxmin()
print(f"정책 문서: {len(df)}건, 카테고리: {len(categories)}개")
print(f"가장 작은 카테고리: {smallest} {category_dist.min()}건")
print(f"분류 정확도: {accuracy:.1%} (평가 문서 {len(y_test)}건 중 {int(round(accuracy * len(y_test)))}건)")
print(f"F1 (카테고리별 F1을 문서 수로 가중 평균): {f1:.3f}")
print(f"토픽 묶음 수: {n_topics}개")

print("\n" + "=" * 90)
print("주의사항 및 한계")
print("=" * 90)

limitations = """
본 실습은 교육 목적 간소화 버전이다:
  - 분류의 수치화는 문장 임베딩 대신 TF-IDF를 쓴다 (임베딩 비교는 10-2-document-classification.py)
  - 토픽 모델링은 BERTopic 대신 K-평균 군집을 쓰고, 묶음 수를 사람이 정한다
  - 문서 100건, 카테고리당 5~15건의 소규모 자료다

실무 적용 시 확인할 것:
  - 사람이 라벨을 붙인 검증 표본과 대조한다 (카테고리당 50건 기준)
  - 토픽은 대표 단어를 사람이 읽어 이름을 붙이고, 사람이 주제를 붙인 표본과 대조한 뒤 쓴다
  - 비중을 보고할 때는 분모(기간별 문서 수)를 함께 적는다
  - 개인정보는 수치화 전에 마스킹한다
"""

print(limitations)

print("\n" + "=" * 90)
print("실습 완료")
print("=" * 90)
