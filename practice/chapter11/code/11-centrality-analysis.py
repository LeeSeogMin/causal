"""
Chapter 11 - Graph Neural Networks and Policy Network Analysis
11.2.2 Centrality Analysis

네트워크 중심성 지표 계산과 정책 영향력 측정
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
matplotlib.rc('font', family='Arial')
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = 'practice/chapter11/data/'
RESULT_DIR = 'result/'

# 디렉토리 생성
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def load_government_network():
    """저장된 네트워크 데이터 로드"""
    # 엣지 데이터 로드
    df_edges = pd.read_csv(OUTPUT_DIR + '11-government-network-edges.csv')

    # 그래프 생성
    G = nx.DiGraph()

    for _, row in df_edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])

    return G

def calculate_all_centralities(G):
    """
    모든 중심성 지표 계산

    Parameters:
    -----------
    G : nx.DiGraph
        분석할 그래프

    Returns:
    --------
    df_centrality : pd.DataFrame
        중심성 지표가 포함된 데이터프레임
    """
    # 무방향 그래프로 변환 (일부 중심성 계산용)
    G_undirected = G.to_undirected()

    centrality_data = []

    for node in G.nodes():
        data = {
            'ministry': node,
            # 연결 중심성 (degree centrality)
            'degree_centrality': nx.degree_centrality(G_undirected)[node],
            'in_degree_centrality': nx.in_degree_centrality(G)[node],
            'out_degree_centrality': nx.out_degree_centrality(G)[node],
            # 근접 중심성 (closeness centrality)
            'closeness_centrality': nx.closeness_centrality(G_undirected, distance='weight')[node],
            # 매개 중심성 (betweenness centrality)
            'betweenness_centrality': nx.betweenness_centrality(G_undirected, weight='weight')[node],
            # 고유벡터 중심성 (eigenvector centrality)
            'eigenvector_centrality': nx.eigenvector_centrality(G_undirected, weight='weight', max_iter=1000)[node]
        }
        centrality_data.append(data)

    df_centrality = pd.DataFrame(centrality_data)

    return df_centrality

def visualize_centrality_comparison(df_centrality):
    """중심성 지표 비교 시각화"""
    # 상위 5개 부처만 선택
    df_top = df_centrality.nlargest(5, 'degree_centrality')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 연결 중심성
    ax1 = axes[0, 0]
    df_plot = df_top.sort_values('degree_centrality', ascending=True)
    ax1.barh(df_plot['ministry'], df_plot['degree_centrality'], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Degree Centrality', fontsize=11)
    ax1.set_title('(a) Degree Centrality (Top 5)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # (2) 매개 중심성
    ax2 = axes[0, 1]
    df_plot = df_top.sort_values('betweenness_centrality', ascending=True)
    ax2.barh(df_plot['ministry'], df_plot['betweenness_centrality'], color='coral', alpha=0.7)
    ax2.set_xlabel('Betweenness Centrality', fontsize=11)
    ax2.set_title('(b) Betweenness Centrality (Top 5)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # (3) 근접 중심성
    ax3 = axes[1, 0]
    df_plot = df_top.sort_values('closeness_centrality', ascending=True)
    ax3.barh(df_plot['ministry'], df_plot['closeness_centrality'], color='mediumseagreen', alpha=0.7)
    ax3.set_xlabel('Closeness Centrality', fontsize=11)
    ax3.set_title('(c) Closeness Centrality (Top 5)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # (4) 고유벡터 중심성
    ax4 = axes[1, 1]
    df_plot = df_top.sort_values('eigenvector_centrality', ascending=True)
    ax4.barh(df_plot['ministry'], df_plot['eigenvector_centrality'], color='mediumpurple', alpha=0.7)
    ax4.set_xlabel('Eigenvector Centrality', fontsize=11)
    ax4.set_title('(d) Eigenvector Centrality (Top 5)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('diagrams/11-centrality-comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제11장: 네트워크 중심성 분석과 정책 영향력 측정")
    print("=" * 80)

    # 1. 네트워크 로드
    print("\n[1단계] 정부 부처 네트워크 로드 중...")
    try:
        G = load_government_network()
        print(f"  - 부처 수: {G.number_of_nodes()}")
        print(f"  - 협업 관계 수: {G.number_of_edges()}")
    except FileNotFoundError:
        print("  ⚠ 네트워크 데이터 파일이 없습니다.")
        print("  먼저 11-government-network.py를 실행하세요.")
        return

    # 2. 중심성 지표 계산
    print("\n[2단계] 중심성 지표 계산 중...")
    df_centrality = calculate_all_centralities(G)

    print("  ✓ 연결 중심성 (Degree Centrality)")
    print("  ✓ 근접 중심성 (Closeness Centrality)")
    print("  ✓ 매개 중심성 (Betweenness Centrality)")
    print("  ✓ 고유벡터 중심성 (Eigenvector Centrality)")

    # 3. 상위 5개 부처 출력
    print("\n[3단계] 중심성 분석 결과 (상위 5개 부처)")

    print("\n=== 연결 중심성 (Degree Centrality) ===")
    df_degree_top = df_centrality.nlargest(5, 'degree_centrality')
    for idx, row in df_degree_top.iterrows():
        print(f"  {row['ministry']:20s}: {row['degree_centrality']:.3f}")

    print("\n=== 매개 중심성 (Betweenness Centrality) ===")
    df_between_top = df_centrality.nlargest(5, 'betweenness_centrality')
    for idx, row in df_between_top.iterrows():
        print(f"  {row['ministry']:20s}: {row['betweenness_centrality']:.3f}")

    print("\n=== 근접 중심성 (Closeness Centrality) ===")
    df_close_top = df_centrality.nlargest(5, 'closeness_centrality')
    for idx, row in df_close_top.iterrows():
        print(f"  {row['ministry']:20s}: {row['closeness_centrality']:.3f}")

    print("\n=== 고유벡터 중심성 (Eigenvector Centrality) ===")
    df_eigen_top = df_centrality.nlargest(5, 'eigenvector_centrality')
    for idx, row in df_eigen_top.iterrows():
        print(f"  {row['ministry']:20s}: {row['eigenvector_centrality']:.3f}")

    # 4. 데이터 저장
    print("\n[4단계] 중심성 데이터 저장 중...")
    df_centrality.to_csv(OUTPUT_DIR + '11-centrality-analysis.csv',
                        index=False, encoding='utf-8-sig')
    df_centrality.to_csv(RESULT_DIR + '11-centrality-analysis.csv',
                        index=False, encoding='utf-8-sig')
    print(f"  - {OUTPUT_DIR}11-centrality-analysis.csv")
    print(f"  - {RESULT_DIR}11-centrality-analysis.csv")

    # 5. 시각화
    print("\n[5단계] 중심성 비교 시각화 중...")
    visualize_centrality_comparison(df_centrality)

    # 6. 분석 해석
    print("\n" + "=" * 80)
    print("중심성 분석 완료")
    print("=" * 80)

    # 연결 중심성 1위 찾기
    top_degree = df_centrality.nlargest(1, 'degree_centrality').iloc[0]
    top_between = df_centrality.nlargest(1, 'betweenness_centrality').iloc[0]
    top_close = df_centrality.nlargest(1, 'closeness_centrality').iloc[0]
    top_eigen = df_centrality.nlargest(1, 'eigenvector_centrality').iloc[0]

    print("\n주요 발견 사항:")
    print(f"  1. 연결 중심성 1위: {top_degree['ministry']}")
    print(f"     → 가장 많은 부처와 직접 협력 관계를 맺고 있음")

    print(f"\n  2. 매개 중심성 1위: {top_between['ministry']}")
    print(f"     → 서로 다른 정책 영역을 연결하는 브로커 역할 수행")

    print(f"\n  3. 근접 중심성 1위: {top_close['ministry']}")
    print(f"     → 전체 네트워크에 대한 접근성이 가장 높음")

    print(f"\n  4. 고유벡터 중심성 1위: {top_eigen['ministry']}")
    print(f"     → 영향력 있는 부처들과 연결되어 있어 높은 지위를 보유")

    print("\n분석 시사점:")
    print("  - 높은 연결 중심성은 직접적 영향력 행사 능력을 나타냄")
    print("  - 높은 매개 중심성은 정보 흐름 통제 능력을 의미")
    print("  - 높은 근접 중심성은 정책 조정자 역할의 중요성을 시사")
    print("  - 높은 고유벡터 중심성은 네트워크 내 엘리트 위치를 반영")

    print("\n출력 파일:")
    print(f"  - {OUTPUT_DIR}11-centrality-analysis.csv")
    print(f"  - {OUTPUT_DIR}11-centrality-comparison.png")
    print(f"  - {RESULT_DIR}11-centrality-analysis.csv (결과 폴더)")
    print(f"  - {RESULT_DIR}11-centrality-comparison.png (결과 폴더)")

if __name__ == "__main__":
    main()
