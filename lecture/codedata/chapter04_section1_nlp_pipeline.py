"""
제4장 4.1절: NLP 파이프라인 시각화 및 기본 개념 실습
자연어 처리의 기본 과정을 시각화하고 이해하기
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def draw_nlp_pipeline():
    """NLP 처리 과정을 단계별로 시각화"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 단계별 박스 그리기
    stages = [
        ("입력 텍스트\n'환경 정책이 필요해요'", 1, '#FFE5B4'),
        ("토큰화\n['환경', '정책', '이', '필요', '해요']", 2, '#B4E5FF'),
        ("벡터 변환\n[[0.2, -0.5], [0.8, 0.3], ...]", 3, '#C8FFB4'),
        ("모델 처리\nBERT/GPT", 4, '#FFB4E5'),
        ("결과 출력\n카테고리: 환경", 5, '#E5B4FF')
    ]
    
    for i, (text, pos, color) in enumerate(stages):
        # 각 단계를 박스로 표현
        box = FancyBboxPatch((pos-0.4, 0.3), 0.8, 0.4,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(pos, 0.5, text, ha='center', va='center', fontsize=12, weight='bold')
        
        # 화살표 그리기
        if i < len(stages) - 1:
            ax.arrow(pos + 0.5, 0.5, 0.4, 0, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('🔄 NLP 처리 과정 - 텍스트가 컴퓨터 언어로 변환되는 과정', fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('output/nlp_pipeline.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_complaint_categories():
    """민원 카테고리별 분포 시각화"""
    # 실제 데이터 로드
    df = pd.read_csv('../data/complaints_data.csv')
    
    # 한국어 민원만 필터링
    korean_df = df[df['language'] == 'ko']
    
    categories = korean_df['category'].value_counts()
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 막대 그래프
    bars = ax1.bar(categories.index, categories.values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_title('📊 한국 민원 카테고리별 분포', fontsize=14, weight='bold')
    ax1.set_xlabel('카테고리', fontsize=12)
    ax1.set_ylabel('민원 수', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # 각 막대 위에 숫자 표시
    for bar, count in zip(bars, categories.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{count}건', ha='center', fontsize=11, weight='bold')
    
    # 파이 차트
    explode = (0.1, 0, 0, 0, 0)  # 첫 번째 조각 강조
    ax2.pie(categories.values, labels=categories.index, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=explode, shadow=True)
    ax2.set_title('🥧 한국 민원 비율 분석', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('../output/complaint_categories.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return categories

def compare_korean_english():
    """한국어와 영어 처리 차이 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 영어 토큰화
    ax1.text(0.5, 0.9, "영어: 'I love you'", ha='center', fontsize=16, weight='bold')
    eng_tokens = ['I', 'love', 'you']
    for i, token in enumerate(eng_tokens):
        rect = patches.Rectangle((0.2 + i*0.25, 0.5), 0.2, 0.2,
                                linewidth=2, edgecolor='blue', facecolor='lightblue')
        ax1.add_patch(rect)
        ax1.text(0.3 + i*0.25, 0.6, token, ha='center', va='center', fontsize=14, weight='bold')
    
    ax1.arrow(0.5, 0.4, 0, -0.15, head_width=0.05, head_length=0.03, fc='red', ec='red')
    ax1.text(0.5, 0.15, "3개 토큰 (단순)", ha='center', fontsize=12, color='blue', weight='bold')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('🔤 영어 처리', fontsize=15, weight='bold')
    
    # 한국어 토큰화
    ax2.text(0.5, 0.9, "한국어: '사랑해요'", ha='center', fontsize=16, weight='bold')
    kor_tokens = ['사랑', '해', '요']
    kor_detail = ['명사', '동사', '존칭']
    
    for i, (token, detail) in enumerate(zip(kor_tokens, kor_detail)):
        rect = patches.Rectangle((0.2 + i*0.25, 0.5), 0.2, 0.2,
                                linewidth=2, edgecolor='red', facecolor='#FFE5E5')
        ax2.add_patch(rect)
        ax2.text(0.3 + i*0.25, 0.65, token, ha='center', va='center', fontsize=14, weight='bold')
        ax2.text(0.3 + i*0.25, 0.55, f'({detail})', ha='center', va='center', fontsize=10, style='italic')
    
    ax2.arrow(0.5, 0.4, 0, -0.15, head_width=0.05, head_length=0.03, fc='red', ec='red')
    ax2.text(0.5, 0.15, "형태소 분석 필요 (복잡)", ha='center', fontsize=12, color='red', weight='bold')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('🇰🇷 한국어 처리', fontsize=15, weight='bold')
    
    plt.tight_layout()
    plt.savefig('../output/korean_vs_english_processing.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_tokenization():
    """다양한 토큰화 방식 시각화"""
    sentence = "안녕하세요"
    
    tokenizations = {
        '원본': ['안녕하세요'],
        '글자': ['안', '녕', '하', '세', '요'],
        '형태소': ['안녕', '하', '세요'],
        '서브워드': ['안녕', '##하세요']
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#FFE5B4', '#B4E5FF', '#C8FFB4', '#FFB4E5']
    
    for idx, (method, tokens) in enumerate(tokenizations.items()):
        ax = axes[idx]
        
        # 토큰 박스 그리기
        total_width = len(tokens) * 0.8 + (len(tokens)-1) * 0.1
        start_x = (3 - total_width) / 2
        
        for i, token in enumerate(tokens):
            x = start_x + i * 0.9
            box = FancyBboxPatch((x, 0.3), 0.8, 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=colors[idx], 
                                 edgecolor='black', linewidth=2)
            ax.add_patch(box)
            ax.text(x + 0.4, 0.5, token, ha='center', va='center', 
                   fontsize=14, weight='bold')
        
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'{method} 토큰화', fontsize=15, weight='bold')
        ax.text(1.5, 0.1, f'토큰 수: {len(tokens)}개', 
               ha='center', fontsize=12, style='italic', weight='bold')
    
    plt.suptitle('🔪 토큰화 방식 비교 - "안녕하세요"를 자르는 방법들', 
                fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('../output/tokenization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_word_embedding():
    """단어 임베딩 2D 공간 시각화"""
    # 예시 단어들과 가상의 2D 좌표
    words = {
        '강아지': (2, 3),
        '고양이': (2.5, 2.8),
        '개': (1.8, 3.2),
        '애완동물': (2.2, 2.5),
        '자동차': (-2, -1),
        '버스': (-2.3, -0.8),
        '택시': (-1.8, -1.2),
        '교통수단': (-2, -1.5),
        '사과': (1, -2),
        '바나나': (1.2, -2.3),
        '과일': (0.8, -2.1)
    }
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 카테고리별 색상
    categories = {
        'animal': ['강아지', '고양이', '개', '애완동물'],
        'vehicle': ['자동차', '버스', '택시', '교통수단'],
        'fruit': ['사과', '바나나', '과일']
    }
    
    colors = {'animal': '#FFB4B4', 'vehicle': '#B4D4FF', 'fruit': '#B4FFB4'}
    
    for category, word_list in categories.items():
        for word in word_list:
            if word in words:
                x, y = words[word]
                ax.scatter(x, y, s=800, c=colors[category], edgecolor='black', 
                          linewidth=2, alpha=0.8)
                ax.annotate(word, (x, y), ha='center', va='center', 
                           fontsize=13, weight='bold')
    
    # 유사도 선 그리기
    similar_pairs = [('강아지', '개'), ('자동차', '버스'), ('사과', '바나나')]
    for word1, word2 in similar_pairs:
        x1, y1 = words[word1]
        x2, y2 = words[word2]
        ax.plot([x1, x2], [y1, y2], 'gray', linestyle='--', alpha=0.6, linewidth=2)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 4)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('차원 1', fontsize=14, weight='bold')
    ax.set_ylabel('차원 2', fontsize=14, weight='bold')
    ax.set_title('📍 단어 임베딩 공간 - 비슷한 의미는 가까이 모여있어요!', 
                fontsize=16, weight='bold')
    
    # 범례
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['animal'], label='동물'),
                       Patch(facecolor=colors['vehicle'], label='교통'),
                       Patch(facecolor=colors['fruit'], label='과일')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../output/word_embedding_space.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_real_complaint_data():
    """실제 민원 데이터 분석 및 시각화"""
    # 데이터 로드
    df = pd.read_csv('../data/complaints_data.csv')
    
    print("=== 실제 민원 데이터 분석 결과 ===")
    print(f"총 민원 수: {len(df):,}건")
    print(f"언어별 분포: 한국어 {len(df[df['language']=='ko'])}건, 영어 {len(df[df['language']=='en'])}건")
    
    # 한국어 민원 분석
    korean_df = df[df['language'] == 'ko']
    
    # 감성 분포 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 감성 분포
    sentiment_counts = korean_df['sentiment'].value_counts()
    colors_sentiment = ['#FF6B6B', '#FFE66D', '#4ECDC4']
    ax1.pie(sentiment_counts.values, labels=['부정적', '중립적', '긍정적'], 
           colors=colors_sentiment, autopct='%1.1f%%', startangle=90)
    ax1.set_title('한국 민원 감성 분포', fontsize=14, weight='bold')
    
    # 2. 처리 상태 분포
    status_counts = korean_df['status'].value_counts()
    ax2.bar(status_counts.index, status_counts.values, 
           color=['#FF9999', '#99CCFF', '#99FF99'], edgecolor='black')
    ax2.set_title('민원 처리 상태 분포', fontsize=14, weight='bold')
    ax2.set_ylabel('민원 수')
    
    # 3. 월별 민원 접수 현황
    korean_df['date'] = pd.to_datetime(korean_df['date'])
    korean_df['month'] = korean_df['date'].dt.to_period('M')
    monthly_counts = korean_df['month'].value_counts().sort_index()
    
    ax3.plot(range(len(monthly_counts)), monthly_counts.values, 
            marker='o', linewidth=2, markersize=6, color='#FF6B6B')
    ax3.set_title('월별 민원 접수 현황', fontsize=14, weight='bold')
    ax3.set_ylabel('민원 수')
    ax3.set_xlabel('월')
    ax3.grid(True, alpha=0.3)
    
    # 4. 지역별 민원 현황 (상위 10개)
    location_counts = korean_df['location'].value_counts().head(10)
    ax4.barh(range(len(location_counts)), location_counts.values, color='#4ECDC4')
    ax4.set_yticks(range(len(location_counts)))
    ax4.set_yticklabels(location_counts.index, fontsize=10)
    ax4.set_title('지역별 민원 현황 (상위 10개)', fontsize=14, weight='bold')
    ax4.set_xlabel('민원 수')
    
    plt.tight_layout()
    plt.savefig('../output/complaint_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return korean_df

def main():
    """메인 실행 함수"""
    print("🚀 제4장 4.1절: NLP 파이프라인 시각화 실습 시작!")
    print("=" * 60)
    
    # 1. NLP 파이프라인 시각화
    print("1️⃣ NLP 처리 과정 시각화...")
    draw_nlp_pipeline()
    
    # 2. 민원 카테고리 분포 시각화
    print("2️⃣ 민원 카테고리 분포 분석...")
    categories = visualize_complaint_categories()
    
    # 3. 한국어 vs 영어 처리 비교
    print("3️⃣ 한국어와 영어 처리 차이 시각화...")
    compare_korean_english()
    
    # 4. 토큰화 방식 비교
    print("4️⃣ 토큰화 방식 비교 시각화...")
    visualize_tokenization()
    
    # 5. 단어 임베딩 공간 시각화
    print("5️⃣ 단어 임베딩 공간 시각화...")
    visualize_word_embedding()
    
    # 6. 실제 민원 데이터 분석
    print("6️⃣ 실제 민원 데이터 분석...")
    korean_df = analyze_real_complaint_data()
    
    print("\n✅ 4.1절 실습 완료!")
    print("📁 결과 이미지들이 ../output/ 폴더에 저장되었습니다.")
    
    # 분석 결과 요약
    print("\n📊 분석 결과 요약:")
    print(f"• 가장 많은 민원 카테고리: {categories.index[0]} ({categories.iloc[0]}건)")
    print(f"• 전체 한국어 민원 중 부정적 감성: {len(korean_df[korean_df['sentiment']=='negative'])/len(korean_df)*100:.1f}%")
    print(f"• 처리 완료된 민원 비율: {len(korean_df[korean_df['status']=='완료'])/len(korean_df)*100:.1f}%")

if __name__ == "__main__":
    main()
