"""
제10장: 정책 분석 종합 툴킷  
===========================

목적: 실전 딥러닝 정책 분석 기법 통합
- 문서 분류, 감정 분석, 유사도 분석, 요약, QA

저자: AI 기반 정책분석방법론
날짜: 2025-11-20
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re

np.random.seed(42)

print("=" * 80)
print("정책 분석 종합 툴킷")
print("=" * 80)

# 1. 샘플 정책 데이터 생성
print("\n[1단계] 샘플 정책 데이터 생성")
print("-" * 80)

categories = ['디지털정책', '경제정책', '환경정책', '보건정책', '고용정책', 
              '교육정책', '과학기술정책', '지역개발정책', '보안정책', '농업정책']

sample_policies = []
for i, category in enumerate(categories):
    for j in range(10):
        policy = {
            'id': i*10 + j,
            'category': category,
            'text': f"{category} 관련 정책 제안서 {j+1}. {category}을 통해 국가 발전을 도모한다.",
            'year': 2024 + (j % 2)
        }
        sample_policies.append(policy)

df_policies = pd.DataFrame(sample_policies)
print(f"총 정책 문서: {len(df_policies)}개")
print(f"카테고리: {len(categories)}개")

# 2. 정책 문서 자동 분류
print("\n[2단계] 정책 문서 자동 분류")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    df_policies['text'], df_policies['category'], 
    test_size=0.3, random_state=42, stratify=df_policies['category']
)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train_vec, y_train)

y_pred = classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"분류 정확도: {accuracy:.1%}")
print(f"F1 Score: {classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']:.3f}")

# 3. 시민 피드백 감정 분석
print("\n[3단계] 시민 피드백 감정 분석")
print("-" * 80)

feedbacks = [
    "이 정책은 정말 훌륭합니다",
    "정책 효과가 의문입니다",
    "좋은 방향이지만 실현 가능성이 낮아요",
    "적극 찬성합니다",
    "반대합니다. 비현실적입니다",
    "괜찮은 정책인 것 같습니다",
    "전혀 도움이 되지 않습니다",
    "기대됩니다",
    "우려스럽습니다",
    "나쁘지 않네요"
]

# 간단한 감정 분석 (키워드 기반)
positive_words = ['훌륭', '좋은', '찬성', '괜찮', '기대', '않네요']
negative_words = ['의문', '반대', '비현실', '우려', '않습니다']

sentiments = []
for feedback in feedbacks:
    pos_count = sum(1 for word in positive_words if word in feedback)
    neg_count = sum(1 for word in negative_words if word in feedback)
    
    if pos_count > neg_count:
        sentiment = '긍정'
    elif neg_count > pos_count:
        sentiment = '부정'
    else:
        sentiment = '중립'
    
    sentiments.append(sentiment)

sentiment_df = pd.DataFrame({
    '피드백': feedbacks,
    '감정': sentiments
})

print(sentiment_df.to_string(index=False))

sentiment_counts = pd.Series(sentiments).value_counts()
print(f"\n감정 분포:")
for sent, count in sentiment_counts.items():
    print(f"  {sent}: {count}개 ({count/len(feedbacks):.1%})")

# 4. 정책 간 의미적 유사도 분석
print("\n[4단계] 정책 간 의미적 유사도 분석")
print("-" * 80)

# 일부 정책 선택
sample_texts = df_policies['text'].head(5).tolist()
sample_labels = df_policies['category'].head(5).tolist()

# TF-IDF 벡터화
vec = TfidfVectorizer()
vectors = vec.fit_transform(sample_texts)

# 유사도 계산
similarity_matrix = cosine_similarity(vectors)

print("정책 간 코사인 유사도:")
for i in range(len(sample_labels)):
    for j in range(i+1, len(sample_labels)):
        sim = similarity_matrix[i, j]
        if sim > 0.1:  # 임계값 이상인 경우만 출력
            print(f"  {sample_labels[i]} <-> {sample_labels[j]}: {sim:.3f}")

# 5. 자동 요약 (추출적 요약)
print("\n[5단계] 정책 문서 자동 요약")
print("-" * 80)

long_policy = """
정부는 디지털 전환 정책을 통해 국민의 삶의 질을 향상시키고자 한다.
디지털 뉴딜 정책은 5G 네트워크 확대, 데이터 센터 구축, AI 산업 육성을 포함한다.
총 50조 원의 예산이 투입되며, 2025년까지 단계적으로 추진된다.
디지털 교육 강화를 통해 국민 모두가 디지털 기술을 활용할 수 있도록 지원한다.
공공 데이터 개방을 확대하여 민간 혁신을 촉진한다.
"""

# 문장 분리
sentences = [s.strip() for s in long_policy.split('.') if s.strip()]

# 간단한 중요도 계산 (문장 길이 + 핵심 키워드 포함)
keywords = ['정책', '예산', '지원', '추진']
scores = []
for sent in sentences:
    score = len(sent) / 50  # 길이 점수
    score += sum(2 for kw in keywords if kw in sent)  # 키워드 점수
    scores.append(score)

# 상위 2개 문장 선택
top_indices = np.argsort(scores)[-2:]
summary = '. '.join([sentences[i] for i in sorted(top_indices)]) + '.'

print(f"원문 ({len(sentences)}개 문장):")
print(long_policy)
print(f"\n요약 (2개 문장):")
print(summary)

# 6. 질의응답 (QA) 시스템
print("\n[6단계] 정책 문서 질의응답")
print("-" * 80)

context = "2025년 디지털 뉴딜 정책의 예산은 5조 원이며, 3월부터 시행된다."
questions = [
    "디지털 뉴딜 정책의 예산은 얼마인가?",
    "정책 시행 시기는 언제인가?",
    "정책 명칭은 무엇인가?"
]

# 간단한 QA (키워드 매칭 기반)
for q in questions:
    print(f"\nQ: {q}")
    
    # 예산 질문
    if '예산' in q:
        match = re.search(r'(\d+조\s*원)', context)
        if match:
            print(f"A: {match.group(1)}")
    # 시기 질문
    elif '시기' in q or '언제' in q:
        match = re.search(r'(\d+월)', context)
        if match:
            print(f"A: {match.group(1)}")
    # 명칭 질문
    elif '명칭' in q or '무엇' in q:
        match = re.search(r'(\S+\s정책)', context)
        if match:
            print(f"A: {match.group(1)}")
    else:
        print(f"A: 답변을 찾을 수 없습니다.")

# 7. 통합 분석 결과
print("\n" + "=" * 80)
print("정책 분석 종합 결과")
print("=" * 80)

results_summary = f"""
✓ 정책 문서 분류: {len(df_policies)}개 문서, 정확도 {accuracy:.1%}
✓ 감정 분석: {len(feedbacks)}개 피드백 분석
  - 긍정: {sentiment_counts.get('긍정', 0)}개
  - 부정: {sentiment_counts.get('부정', 0)}개
  - 중립: {sentiment_counts.get('중립', 0)}개
✓ 유사도 분석: 정책 간 중복 가능성 탐지
✓ 자동 요약: 긴 문서를 핵심 문장으로 압축
✓ 질의응답: 정책 정보 자동 추출
"""

print(results_summary)

print("\n실무 통합 워크플로우:")
print("  1) 신규 정책 제안서 → 자동 분류 → 담당 부서 배정")
print("  2) 유사도 분석 → 관련 정책 발견 → 통합 검토")
print("  3) 요약 모델 → 핵심 내용 추출 → 보고서 작성")
print("  4) QA 시스템 → 정보 검색 → 의사결정 지원")
print("  5) 시민 의견 수집 → 감정 분석 → 정책 피드백")

print("\n=" * 80)
print("주의: 본 구현은 교육 목적 간소화 버전")
print("실무 적용시 BERT/GPT 기반 고도화 필요")
print("=" * 80)
