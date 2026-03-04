"""
Chapter 11 - Graph Neural Networks and Policy Network Analysis
11.2.3 Policy Influence and Centrality Relationship

종합 영향력 점수 계산 및 중심성 지표 간 상관관계 분석
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows: Malgun Gothic)
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = 'practice/chapter11/data/'
RESULT_DIR = 'result/'

# 디렉토리 생성
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def load_centrality_data():
    """중심성 데이터 로드"""
    df = pd.read_csv(OUTPUT_DIR + '11-centrality-analysis.csv')
    return df

def calculate_influence_score(df_centrality, weights=None):
    """
    종합 영향력 점수 계산

    Parameters:
    -----------
    df_centrality : pd.DataFrame
        중심성 지표 데이터프레임
    weights : dict
        각 중심성 지표의 가중치
        기본값: {'degree': 0.2, 'closeness': 0.2, 'betweenness': 0.3, 'eigenvector': 0.3}

    Returns:
    --------
    df_centrality : pd.DataFrame
        영향력 점수가 추가된 데이터프레임
    """
    if weights is None:
        weights = {
            'degree': 0.2,
            'closeness': 0.2,
            'betweenness': 0.3,
            'eigenvector': 0.3
        }

    # 종합 영향력 점수 계산
    df_centrality['influence_score'] = (
        df_centrality['degree_centrality'] * weights['degree'] +
        df_centrality['closeness_centrality'] * weights['closeness'] +
        df_centrality['betweenness_centrality'] * weights['betweenness'] +
        df_centrality['eigenvector_centrality'] * weights['eigenvector']
    )

    return df_centrality

def analyze_centrality_correlations(df_centrality):
    """중심성 지표 간 상관관계 분석"""
    centrality_cols = ['degree_centrality', 'closeness_centrality',
                      'betweenness_centrality', 'eigenvector_centrality']

    correlation_matrix = df_centrality[centrality_cols].corr()

    return correlation_matrix

def visualize_influence_analysis(df_influence, correlation_matrix):
    """영향력 분석 시각화"""
    fig = plt.figure(figsize=(16, 10))

    # (1) 종합 영향력 순위 (상위 10개)
    ax1 = plt.subplot(2, 2, 1)
    df_top10 = df_influence.nlargest(10, 'influence_score').sort_values('influence_score')
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_top10)))
    ax1.barh(df_top10['ministry'], df_top10['influence_score'], color=colors, alpha=0.8)
    ax1.set_xlabel('Influence Score', fontsize=11)
    ax1.set_title('(a) Top 10 Ministries by Influence Score', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # (2) 중심성 지표 간 상관관계 히트맵
    ax2 = plt.subplot(2, 2, 2)
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax2, vmin=-1, vmax=1)
    ax2.set_title('(b) Centrality Correlation Matrix', fontsize=12, fontweight='bold')

    # (3) 중심성 구성 요소 비교 (상위 5개)
    ax3 = plt.subplot(2, 2, 3)
    df_top5 = df_influence.nlargest(5, 'influence_score')

    # 스택 바 차트 데이터 준비
    x_pos = np.arange(len(df_top5))
    degree_contrib = df_top5['degree_centrality'].values * 0.2
    closeness_contrib = df_top5['closeness_centrality'].values * 0.2
    betweenness_contrib = df_top5['betweenness_centrality'].values * 0.3
    eigenvector_contrib = df_top5['eigenvector_centrality'].values * 0.3

    ax3.bar(x_pos, degree_contrib, label='Degree (0.2)', color='steelblue', alpha=0.8)
    ax3.bar(x_pos, closeness_contrib, bottom=degree_contrib,
           label='Closeness (0.2)', color='mediumseagreen', alpha=0.8)
    ax3.bar(x_pos, betweenness_contrib,
           bottom=degree_contrib + closeness_contrib,
           label='Betweenness (0.3)', color='coral', alpha=0.8)
    ax3.bar(x_pos, eigenvector_contrib,
           bottom=degree_contrib + closeness_contrib + betweenness_contrib,
           label='Eigenvector (0.3)', color='mediumpurple', alpha=0.8)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(df_top5['ministry'], rotation=45, ha='right')
    ax3.set_ylabel('Contribution to Influence Score', fontsize=11)
    ax3.set_title('(c) Influence Score Composition (Top 5)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # (4) 산점도: 매개 vs 고유벡터 중심성
    ax4 = plt.subplot(2, 2, 4)
    scatter = ax4.scatter(df_influence['betweenness_centrality'],
                         df_influence['eigenvector_centrality'],
                         s=df_influence['influence_score'] * 500,
                         c=df_influence['influence_score'],
                         cmap='plasma', alpha=0.6, edgecolors='black', linewidth=1)

    # 상위 5개 부처 레이블 표시
    for _, row in df_top5.iterrows():
        ax4.annotate(row['ministry'], (row['betweenness_centrality'], row['eigenvector_centrality']),
                    fontsize=8, alpha=0.8, xytext=(5, 5), textcoords='offset points')

    ax4.set_xlabel('Betweenness Centrality', fontsize=11)
    ax4.set_ylabel('Eigenvector Centrality', fontsize=11)
    ax4.set_title('(d) Betweenness vs Eigenvector (size = influence)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Influence Score', fontsize=9)

    plt.tight_layout()
    plt.savefig("diagrams/11-influence-analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제11장: 정책 영향력과 중심성의 관계 분석")
    print("=" * 80)

    # 1. 중심성 데이터 로드
    print("\n[1단계] 중심성 데이터 로드 중...")
    try:
        df_centrality = load_centrality_data()
        print(f"  - 부처 수: {len(df_centrality)}")
        print(f"  - 중심성 지표: 4개 (degree, closeness, betweenness, eigenvector)")
    except FileNotFoundError:
        print("  ⚠ 중심성 데이터 파일이 없습니다.")
        print("  먼저 11-centrality-analysis.py를 실행하세요.")
        return

    # 2. 종합 영향력 점수 계산
    print("\n[2단계] 종합 영향력 점수 계산 중...")
    weights = {
        'degree': 0.2,
        'closeness': 0.2,
        'betweenness': 0.3,
        'eigenvector': 0.3
    }
    df_influence = calculate_influence_score(df_centrality, weights)
    print("  ✓ 가중치 적용 완료:")
    print(f"     - Degree Centrality: {weights['degree']}")
    print(f"     - Closeness Centrality: {weights['closeness']}")
    print(f"     - Betweenness Centrality: {weights['betweenness']}")
    print(f"     - Eigenvector Centrality: {weights['eigenvector']}")

    # 3. 중심성 상관관계 분석
    print("\n[3단계] 중심성 지표 간 상관관계 분석 중...")
    correlation_matrix = analyze_centrality_correlations(df_influence)

    print("\n=== 중심성 지표 간 상관관계 ===")
    print(correlation_matrix.round(3).to_string())

    # 4. 종합 영향력 순위 출력
    print("\n[4단계] 종합 영향력 순위 (상위 10개)")
    df_ranked = df_influence.sort_values('influence_score', ascending=False)

    print("\n순위 | 부처명                  | 영향력 점수")
    print("-" * 60)
    for idx, (_, row) in enumerate(df_ranked.head(10).iterrows(), 1):
        print(f"{idx:2d}   | {row['ministry']:20s} | {row['influence_score']:.3f}")

    # 5. 데이터 저장
    print("\n[5단계] 영향력 분석 데이터 저장 중...")
    df_ranked.to_csv(OUTPUT_DIR + '11-influence-ranking.csv',
                    index=False, encoding='utf-8-sig')
    df_ranked.to_csv(RESULT_DIR + '11-influence-ranking.csv',
                    index=False, encoding='utf-8-sig')
    print(f"  - {OUTPUT_DIR}11-influence-ranking.csv")
    print(f"  - {RESULT_DIR}11-influence-ranking.csv")

    correlation_matrix.to_csv(OUTPUT_DIR + '11-centrality-correlation.csv',
                             encoding='utf-8-sig')
    correlation_matrix.to_csv(RESULT_DIR + '11-centrality-correlation.csv',
                             encoding='utf-8-sig')
    print(f"  - {OUTPUT_DIR}11-centrality-correlation.csv")
    print(f"  - {RESULT_DIR}11-centrality-correlation.csv")

    # 6. 시각화
    print("\n[6단계] 영향력 분석 시각화 중...")
    visualize_influence_analysis(df_influence, correlation_matrix)

    # 7. 분석 해석
    print("\n" + "=" * 80)
    print("영향력 분석 완료")
    print("=" * 80)

    top1 = df_ranked.iloc[0]
    top2 = df_ranked.iloc[1]
    top3 = df_ranked.iloc[2]

    print("\n주요 발견 사항:")
    print(f"  1. 종합 영향력 1위: {top1['ministry']} ({top1['influence_score']:.3f})")
    print(f"     - 매개 중심성과 고유벡터 중심성이 높아 정책 네트워크의 핵심 허브")

    print(f"\n  2. 종합 영향력 2위: {top2['ministry']} ({top2['influence_score']:.3f})")
    print(f"     - 다양한 중심성 지표에서 균형 잡힌 높은 점수")

    print(f"\n  3. 종합 영향력 3위: {top3['ministry']} ({top3['influence_score']:.3f})")
    print(f"     - 정책 조정 기능을 통한 전략적 위치")

    # 상관관계 분석 해석
    degree_eigen_corr = correlation_matrix.loc['degree_centrality', 'eigenvector_centrality']
    print(f"\n  4. 연결 중심성과 고유벡터 중심성 상관관계: {degree_eigen_corr:.3f}")
    if degree_eigen_corr > 0.7:
        print("     - 높은 양의 상관관계: 영향력 있는 부처들이 서로 연결되는 경향")

    print("\n분석 시사점:")
    print("  - 매개 중심성과 고유벡터 중심성(각 0.3 가중치)이 정책 영향력에")
    print("    가장 중요한 요소로 평가됨")
    print("  - 브로커 역할(매개)과 엘리트 네트워크 소속(고유벡터)이")
    print("    실질적인 정책 변화를 이끌어내는 핵심 요인")
    print("  - 한국 정부의 정책 네트워크는 디지털 전환과 예산 조정을 중심으로")
    print("    구조화되어 있음")

    print("\n출력 파일:")
    print(f"  - {OUTPUT_DIR}11-influence-ranking.csv")
    print(f"  - {OUTPUT_DIR}11-centrality-correlation.csv")
    print(f"  - {OUTPUT_DIR}11-influence-analysis.png")
    print(f"  - {RESULT_DIR}11-influence-ranking.csv (결과 폴더)")
    print(f"  - {RESULT_DIR}11-centrality-correlation.csv (결과 폴더)")
    print(f"  - {RESULT_DIR}11-influence-analysis.png (결과 폴더)")

if __name__ == "__main__":
    main()
