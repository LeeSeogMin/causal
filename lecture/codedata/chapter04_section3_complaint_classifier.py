"""
제4장 4.3절: 민원 분류 AI 만들기 실습
실제 민원 데이터를 사용하여 텍스트 분류 모델을 구현하고 평가
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    """민원 데이터 로드 및 탐색적 분석"""
    print("📁 민원 데이터 로딩 중...")
    df = pd.read_csv('../data/complaints_data.csv')
    
    # 한국어 민원만 필터링
    korean_df = df[df['language'] == 'ko'].copy()
    
    print(f"✅ 총 {len(korean_df)}건의 한국어 민원 데이터 로드 완료")
    print(f"📊 카테고리 분포:")
    print(korean_df['category'].value_counts())
    
    return korean_df

def clean_text(text):
    """텍스트 전처리 함수"""
    if pd.isna(text):
        return ""
    
    # 1. 불필요한 공백 제거
    text = text.strip()
    
    # 2. 여러 공백을 하나로
    text = ' '.join(text.split())
    
    # 3. 특수문자 일부 제거 (선택사항)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    
    # 4. 다시 공백 정리
    text = ' '.join(text.split())
    
    return text

def preprocess_data(df):
    """데이터 전처리"""
    print("🧹 텍스트 전처리 중...")
    
    # 텍스트 정리
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # 빈 텍스트 제거
    df = df[df['cleaned_text'].str.len() > 0].copy()
    
    # 텍스트 길이 분석
    df['text_length'] = df['cleaned_text'].str.len()
    
    print(f"✅ 전처리 완료. 최종 데이터: {len(df)}건")
    print(f"📏 평균 텍스트 길이: {df['text_length'].mean():.1f}자")
    
    return df

def classify_complaint_simple(text):
    """키워드 기반 간단 분류기"""
    
    # 카테고리별 키워드
    keywords = {
        '교통': ['도로', '신호등', '주차', '교통', '차량', '포트홀', '버스', '지하철'],
        '환경': ['쓰레기', '소음', '오염', '청소', '재활용', '미세먼지', '공원', '녹지'],
        '안전': ['가로등', '치안', 'CCTV', '위험', '사고', '범죄', '안전시설'],
        '복지': ['놀이터', '복지관', '어린이집', '경로당', '시설', '기초연금', '지원금'],
        '행정': ['민원처리', '공무원', '서류발급', '온라인', '행정절차', '시스템']
    }
    
    # 각 카테고리별 점수 계산
    scores = {}
    for category, words in keywords.items():
        score = sum(1 for word in words if word in text)
        scores[category] = score
    
    # 가장 높은 점수의 카테고리 반환
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    else:
        return '기타'

def simple_keyword_classifier(df):
    """키워드 기반 분류기 실행 및 평가"""
    print("🏷️ 키워드 기반 분류기 실행 중...")
    
    # 분류 실행
    df['predicted_simple'] = df['cleaned_text'].apply(classify_complaint_simple)
    
    # 정확도 계산
    accuracy = accuracy_score(df['category'], df['predicted_simple'])
    
    print(f"✅ 키워드 기반 분류기 정확도: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return df, accuracy

def tfidf_classifier(df):
    """TF-IDF + 머신러닝 분류기"""
    print("🤖 TF-IDF 기반 분류기 학습 중...")
    
    # 데이터 분할
    X = df['cleaned_text']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 여러 분류기 학습
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        # 모델 학습
        clf.fit(X_train_tfidf, y_train)
        
        # 예측
        y_pred = clf.predict(X_test_tfidf)
        
        # 정확도 계산
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"✅ {name} 정확도: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return results, vectorizer

def visualize_classification_results(simple_accuracy, ml_results):
    """분류 결과를 시각화하는 함수"""
    
    # 모든 결과 정리
    all_results = {'키워드 기반': simple_accuracy}
    for name, result in ml_results.items():
        all_results[name] = result['accuracy']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 모델별 정확도 비교
    models = list(all_results.keys())
    accuracies = [acc * 100 for acc in all_results.values()]
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    
    bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('정확도 (%)', fontsize=13, weight='bold')
    ax1.set_title('분류 모델별 성능 비교', fontsize=14, weight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, weight='bold')
    
    # 2. 최고 성능 모델의 혼동 행렬
    best_model_name = max(ml_results.keys(), key=lambda x: ml_results[x]['accuracy'])
    best_result = ml_results[best_model_name]
    
    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
    categories = sorted(best_result['y_test'].unique())
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories, ax=ax2)
    ax2.set_title(f'{best_model_name} 혼동 행렬', fontsize=14, weight='bold')
    ax2.set_xlabel('예측 카테고리', fontsize=12)
    ax2.set_ylabel('실제 카테고리', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../output/classification_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model_name

def analyze_misclassifications(df, ml_results):
    """오분류 사례 분석"""
    print("🔍 오분류 사례 분석 중...")
    
    # 최고 성능 모델 선택
    best_model_name = max(ml_results.keys(), key=lambda x: ml_results[x]['accuracy'])
    best_result = ml_results[best_model_name]
    
    # 오분류 사례 찾기
    test_indices = df.index[-len(best_result['y_test']):]  # 테스트 데이터 인덱스
    misclassified = []
    
    for i, (true_label, pred_label) in enumerate(zip(best_result['y_test'], best_result['y_pred'])):
        if true_label != pred_label:
            original_idx = test_indices[i]
            misclassified.append({
                'text': df.loc[original_idx, 'cleaned_text'],
                'true_category': true_label,
                'predicted_category': pred_label
            })
    
    print(f"📊 총 {len(misclassified)}건의 오분류 발견")
    
    # 오분류 패턴 분석
    error_patterns = Counter()
    for error in misclassified:
        pattern = f"{error['true_category']} → {error['predicted_category']}"
        error_patterns[pattern] += 1
    
    print("🔄 주요 오분류 패턴:")
    for pattern, count in error_patterns.most_common(5):
        print(f"   {pattern}: {count}건")
    
    # 오분류 사례 샘플 출력
    print("\n📝 오분류 사례 샘플:")
    for i, error in enumerate(misclassified[:3]):
        print(f"\n{i+1}. 텍스트: {error['text'][:50]}...")
        print(f"   실제: {error['true_category']} | 예측: {error['predicted_category']}")
    
    return misclassified

def find_topics_simple(texts, n_topics=5):
    """단어 빈도 기반 간단 토픽 찾기"""
    
    # 모든 텍스트 합치기
    all_text = ' '.join(texts)
    
    # 단어 분리
    words = all_text.split()
    
    # 불용어 제거 (간단 버전)
    stopwords = ['이', '가', '을', '를', '에', '에서', '으로', '와', '과', '는', '은', '의', '도', '만', '부터', '까지', '하고', '그리고']
    words = [w for w in words if w not in stopwords and len(w) > 1]
    
    # 상위 빈출 단어 찾기
    word_counts = Counter(words)
    top_words = word_counts.most_common(n_topics)
    
    return top_words

def visualize_topics_and_trends(df):
    """토픽 분석 및 트렌드 시각화"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 주요 키워드 분석
    topics = find_topics_simple(df['cleaned_text'].tolist())
    words = [w for w, c in topics]
    counts = [c for w, c in topics]
    
    colors = plt.cm.Set3(range(len(words)))
    ax1.bar(words, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_title('민원 주요 키워드 (토픽)', fontsize=14, weight='bold')
    ax1.set_xlabel('키워드', fontsize=12)
    ax1.set_ylabel('등장 횟수', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 카테고리별 감성 분포
    sentiment_by_category = pd.crosstab(df['category'], df['sentiment'])
    sentiment_by_category.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#FFE66D', '#4ECDC4'])
    ax2.set_title('카테고리별 감성 분포', fontsize=14, weight='bold')
    ax2.set_xlabel('카테고리', fontsize=12)
    ax2.set_ylabel('민원 수', fontsize=12)
    ax2.legend(['부정적', '중립적', '긍정적'])
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 텍스트 길이 분포
    ax3.hist(df['text_length'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(df['text_length'].mean(), color='red', linestyle='--', linewidth=2, label=f'평균: {df["text_length"].mean():.0f}자')
    ax3.set_title('민원 텍스트 길이 분포', fontsize=14, weight='bold')
    ax3.set_xlabel('텍스트 길이 (글자 수)', fontsize=12)
    ax3.set_ylabel('빈도', fontsize=12)
    ax3.legend()
    
    # 4. 처리 상태별 분포
    status_counts = df['status'].value_counts()
    colors_status = ['#FF9999', '#99CCFF', '#99FF99']
    ax4.pie(status_counts.values, labels=status_counts.index, colors=colors_status, 
           autopct='%1.1f%%', startangle=90)
    ax4.set_title('민원 처리 상태 분포', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('../output/topics_and_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_insights_report(df, simple_accuracy, ml_results, misclassified):
    """분석 결과 인사이트 리포트 생성"""
    
    best_model_name = max(ml_results.keys(), key=lambda x: ml_results[x]['accuracy'])
    best_accuracy = ml_results[best_model_name]['accuracy']
    
    report = f"""
=== 민원 분류 AI 분석 리포트 ===

📊 데이터 개요:
• 총 민원 수: {len(df):,}건
• 카테고리 수: {df['category'].nunique()}개
• 평균 텍스트 길이: {df['text_length'].mean():.1f}자

🎯 모델 성능:
• 키워드 기반 분류기: {simple_accuracy*100:.1f}%
• 최고 성능 모델: {best_model_name} ({best_accuracy*100:.1f}%)
• 성능 향상: +{(best_accuracy - simple_accuracy)*100:.1f}%p

🔍 주요 발견사항:
• 가장 많은 민원 카테고리: {df['category'].value_counts().index[0]} ({df['category'].value_counts().iloc[0]}건)
• 부정적 감성 비율: {len(df[df['sentiment']=='negative'])/len(df)*100:.1f}%
• 오분류 건수: {len(misclassified)}건 ({len(misclassified)/len(df)*100:.1f}%)

💡 개선 제안:
1. 더 많은 학습 데이터 수집 (현재 {len(df)}건 → 목표 5,000건+)
2. 한국어 특화 전처리 강화 (형태소 분석 도입)
3. KoBERT 등 사전학습 모델 활용
4. 앙상블 모델 구성으로 성능 향상

🚀 실무 적용 방안:
• 실시간 민원 자동 분류 시스템 구축
• 민원 처리 우선순위 자동 결정
• 반복 민원 패턴 자동 감지
• 정책 이슈 트렌드 모니터링
"""
    
    print(report)
    
    # 리포트를 파일로 저장
    with open('../output/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

def main():
    """메인 실행 함수"""
    print("💻 제4장 4.3절: 민원 분류 AI 만들기 실습 시작!")
    print("=" * 60)
    
    # 1. 데이터 로드 및 탐색
    print("1️⃣ 데이터 로드 및 탐색...")
    df = load_and_explore_data()
    
    # 2. 데이터 전처리
    print("\n2️⃣ 데이터 전처리...")
    df = preprocess_data(df)
    
    # 3. 키워드 기반 분류기
    print("\n3️⃣ 키워드 기반 분류기 실행...")
    df, simple_accuracy = simple_keyword_classifier(df)
    
    # 4. TF-IDF 기반 머신러닝 분류기
    print("\n4️⃣ TF-IDF 기반 분류기 학습...")
    ml_results, vectorizer = tfidf_classifier(df)
    
    # 5. 결과 시각화
    print("\n5️⃣ 분류 결과 시각화...")
    best_model_name = visualize_classification_results(simple_accuracy, ml_results)
    
    # 6. 오분류 분석
    print("\n6️⃣ 오분류 사례 분석...")
    misclassified = analyze_misclassifications(df, ml_results)
    
    # 7. 토픽 및 트렌드 분석
    print("\n7️⃣ 토픽 및 트렌드 분석...")
    visualize_topics_and_trends(df)
    
    # 8. 인사이트 리포트 생성
    print("\n8️⃣ 분석 리포트 생성...")
    report = generate_insights_report(df, simple_accuracy, ml_results, misclassified)
    
    print("\n✅ 4.3절 실습 완료!")
    print("📁 결과 파일들이 ../output/ 폴더에 저장되었습니다.")
    print(f"🏆 최고 성능 모델: {best_model_name} ({ml_results[best_model_name]['accuracy']*100:.1f}%)")

if __name__ == "__main__":
    main()
