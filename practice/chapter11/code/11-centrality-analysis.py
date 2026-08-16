"""
Chapter 11 - 그래프 신경망과 조직 네트워크 분석
11.2 네트워크 중심성 분석

네 가지 중심성 지표를 계산해 "누가 중요한 부처인가"를 지표별로 비교한다.

수정 이력
---------
2026-08-17
1. 경로 오류: 프로젝트 루트 기준 상대경로여서 code 폴더에서 실행하면
   입력 CSV를 못 찾고 "먼저 11-government-network.py를 실행하세요"만 찍고 끝났다.
   → __file__ 기준 절대경로로 교체.
2. 폰트 오류: matplotlib.rc('font', family='Arial')이라 그래프의 한글 부처명이
   전부 네모(tofu)로 나왔다. → Malgun Gothic으로 교체.
3. plt.show() 때문에 배치 실행이 멈췄다. → Agg 백엔드 + close().
4. 그림 저장 위치를 diagrams/(개념도 폴더)에서 code 폴더로 옮겼다.
5. 근접·매개 중심성에 distance/weight='weight'를 그대로 넘겨
   '협업이 강할수록 거리가 멀다'로 계산되고 있었다.
   가중치 0.8은 강한 협업이므로 거리로는 짧아야 한다.
   → 거리 = 1/weight 를 별도 속성으로 만들어 근접 중심성에 쓴다.
6. result/ 폴더 중복 저장 삭제.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data') + os.sep
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # 거리 속성 만들기: 협업 강도가 셀수록 두 부처 사이는 '가깝다'
    # weight 0.8 -> distance 1.25, weight 0.5 -> distance 2.0
    for u, v, d in G_undirected.edges(data=True):
        d['distance'] = 1.0 / d['weight']

    deg = nx.degree_centrality(G_undirected)
    indeg = nx.in_degree_centrality(G)
    outdeg = nx.out_degree_centrality(G)
    clo = nx.closeness_centrality(G_undirected, distance='distance')
    btw = nx.betweenness_centrality(G_undirected, weight='distance', normalized=True)
    eig = nx.eigenvector_centrality(G_undirected, weight='weight', max_iter=1000)

    centrality_data = []
    for node in G.nodes():
        centrality_data.append({
            'ministry': node,
            'degree_centrality': deg[node],
            'in_degree_centrality': indeg[node],
            'out_degree_centrality': outdeg[node],
            'closeness_centrality': clo[node],
            'betweenness_centrality': btw[node],
            'eigenvector_centrality': eig[node],
        })

    df_centrality = pd.DataFrame(centrality_data)

    return df_centrality

def visualize_centrality_comparison(df_centrality):
    """중심성 지표 비교 시각화

    지표마다 상위 5개를 따로 뽑는다. 지표가 바뀌면 명단도 바뀐다는 것이
    이 그림의 요점이다.
    """
    specs = [
        ('degree_centrality', '연결 중심성', 'steelblue', '(a) 연결 중심성 상위 5'),
        ('betweenness_centrality', '매개 중심성', 'coral', '(b) 매개 중심성 상위 5'),
        ('closeness_centrality', '근접 중심성', 'mediumseagreen', '(c) 근접 중심성 상위 5'),
        ('eigenvector_centrality', '고유벡터 중심성', 'mediumpurple', '(d) 고유벡터 중심성 상위 5'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, (col, xlab, color, title) in zip(axes.ravel(), specs):
        df_plot = df_centrality.nlargest(5, col).sort_values(col, ascending=True)
        bars = ax.barh(df_plot['ministry'], df_plot[col], color=color, alpha=0.8)
        for bar, val in zip(bars, df_plot[col]):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f' {val:.3f}',
                    va='center', fontsize=9)
        ax.set_xlabel(xlab, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, df_plot[col].max() * 1.30)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '11-centrality-analysis.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

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

    # 3-1. 지표별 순위가 얼마나 다른지
    print("\n[3-1단계] 연결 중심성 순위 vs 매개 중심성 순위")
    df_rank = df_centrality.copy()
    df_rank['연결순위'] = df_rank['degree_centrality'].rank(ascending=False, method='min').astype(int)
    df_rank['매개순위'] = df_rank['betweenness_centrality'].rank(ascending=False, method='min').astype(int)
    df_rank['순위차'] = df_rank['매개순위'] - df_rank['연결순위']
    df_rank = df_rank.sort_values('연결순위')
    print(f"  {'부처':<20}{'연결순위':>8}{'매개순위':>8}{'차이':>6}")
    for _, r in df_rank.head(8).iterrows():
        print(f"  {r['ministry']:<20}{r['연결순위']:>8}{r['매개순위']:>8}{r['순위차']:>+6}")
    corr_dg_bt = df_centrality['degree_centrality'].corr(
        df_centrality['betweenness_centrality'])
    corr_dg_ev = df_centrality['degree_centrality'].corr(
        df_centrality['eigenvector_centrality'])
    print(f"\n  연결-매개 상관: {corr_dg_bt:.3f}")
    print(f"  연결-고유벡터 상관: {corr_dg_ev:.3f}")

    # 4. 데이터 저장
    print("\n[4단계] 중심성 데이터 저장 중...")
    df_centrality.to_csv(OUTPUT_DIR + '11-centrality-analysis.csv',
                        index=False, encoding='utf-8-sig')
    print(f"  - ../data/11-centrality-analysis.csv")

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
    print("  - ../data/11-centrality-analysis.csv")
    print("  - 11-centrality-analysis.png")

if __name__ == "__main__":
    main()
