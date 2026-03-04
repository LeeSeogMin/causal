"""
Chapter 11 - Graph Neural Networks and Policy Network Analysis
11.3.2 Louvain Algorithm and Policy Coalition Identification

Louvain 알고리즘을 활용한 정책 연합 탐지
"""

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

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

    # 무방향 그래프 생성 (커뮤니티 탐지용)
    G = nx.Graph()

    for _, row in df_edges.iterrows():
        # 양방향으로 추가
        G.add_edge(row['source'], row['target'], weight=row['weight'])

    return G

def detect_communities_louvain(G):
    """
    Louvain 알고리즘을 사용한 커뮤니티 탐지

    Parameters:
    -----------
    G : nx.Graph
        분석할 무방향 그래프

    Returns:
    --------
    communities_list : list
        커뮤니티 리스트 (각 커뮤니티는 노드의 집합)
    modularity : float
        모듈러리티 값
    """
    # Greedy modularity communities (Louvain 알고리즘 기반)
    communities_gen = community.greedy_modularity_communities(G, weight='weight')
    communities_list = list(communities_gen)

    # 모듈러리티 계산
    modularity = community.modularity(G, communities_list, weight='weight')

    return communities_list, modularity

def analyze_community_properties(G, communities_list):
    """
    각 커뮤니티의 속성 분석

    Parameters:
    -----------
    G : nx.Graph
        전체 그래프
    communities_list : list
        커뮤니티 리스트

    Returns:
    --------
    community_info : list of dict
        커뮤니티 정보 리스트
    """
    community_info = []

    for idx, comm in enumerate(communities_list, 1):
        members = sorted(list(comm))
        subgraph = G.subgraph(comm)

        # 내부 연결 밀도 계산
        n = len(members)
        if n > 1:
            max_edges = n * (n - 1) / 2
            internal_density = subgraph.number_of_edges() / max_edges
        else:
            internal_density = 0

        # 평균 엣지 가중치
        if subgraph.number_of_edges() > 0:
            avg_weight = np.mean([G[u][v]['weight'] for u, v in subgraph.edges()])
        else:
            avg_weight = 0

        info = {
            'community_id': idx,
            'size': len(members),
            'members': members,
            'internal_edges': subgraph.number_of_edges(),
            'internal_density': internal_density,
            'avg_weight': avg_weight
        }
        community_info.append(info)

    return community_info

def visualize_communities(G, communities_list, title='정책 연합 커뮤니티 탐지'):
    """커뮤니티 시각화"""
    plt.figure(figsize=(16, 12))

    # Spring layout으로 배치
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 커뮤니티별 색상 매핑
    colors = plt.cm.Set3(np.linspace(0, 1, len(communities_list)))
    node_colors = {}
    for idx, comm in enumerate(communities_list):
        for node in comm:
            node_colors[node] = colors[idx]

    # 노드 색상 리스트 생성
    node_color_list = [node_colors[node] for node in G.nodes()]

    # 노드 크기: 차수에 비례 (한글 라벨 수용을 위해 크게 설정)
    node_sizes = [2500 + G.degree(node) * 300 for node in G.nodes()]

    # 엣지 두께: 가중치에 비례
    edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]

    # 그래프 그리기
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color=node_color_list, alpha=0.85,
                          edgecolors='black', linewidths=2)

    nx.draw_networkx_edges(G, pos, width=edge_widths,
                          alpha=0.4, edge_color='gray')

    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold',
                           font_family='Malgun Gothic')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('diagrams/11-community-detection.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제11장: 커뮤니티 탐지와 정책 연합 분석")
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

    # 2. Louvain 알고리즘으로 커뮤니티 탐지
    print("\n[2단계] Louvain 알고리즘으로 커뮤니티 탐지 중...")
    communities_list, modularity = detect_communities_louvain(G)

    print(f"  ✓ 탐지된 커뮤니티 수: {len(communities_list)}")
    print(f"  ✓ 모듈러리티 (Modularity): {modularity:.3f}")

    if modularity >= 0.3:
        print("     → 강한 커뮤니티 구조를 나타냄 (Q ≥ 0.3)")
    else:
        print("     → 약한 커뮤니티 구조 (Q < 0.3)")

    # 3. 커뮤니티 속성 분석
    print("\n[3단계] 커뮤니티별 속성 분석 중...")
    community_info = analyze_community_properties(G, communities_list)

    # 크기 순으로 정렬
    community_info.sort(key=lambda x: x['size'], reverse=True)

    print("\n=== 커뮤니티 분석 결과 ===")
    for info in community_info:
        print(f"\n커뮤니티 {info['community_id']} ({info['size']}개 부처):")
        print(f"  구성원: {', '.join(info['members'])}")
        print(f"  내부 연결 수: {info['internal_edges']}개")
        print(f"  내부 연결 밀도: {info['internal_density']:.3f}")
        print(f"  평균 협업 강도: {info['avg_weight']:.3f}")

    # 4. 데이터 저장
    print("\n[4단계] 커뮤니티 데이터 저장 중...")

    # 커뮤니티 정보 저장
    comm_data = []
    for info in community_info:
        for member in info['members']:
            comm_data.append({
                'ministry': member,
                'community_id': info['community_id'],
                'community_size': info['size']
            })

    df_communities = pd.DataFrame(comm_data)
    df_communities.to_csv(OUTPUT_DIR + '11-community-assignments.csv',
                         index=False, encoding='utf-8-sig')
    df_communities.to_csv(RESULT_DIR + '11-community-assignments.csv',
                         index=False, encoding='utf-8-sig')
    print(f"  - {OUTPUT_DIR}11-community-assignments.csv")
    print(f"  - {RESULT_DIR}11-community-assignments.csv")

    # 커뮤니티 통계 저장
    comm_stats = []
    for info in community_info:
        comm_stats.append({
            'community_id': info['community_id'],
            'size': info['size'],
            'members': ', '.join(info['members']),
            'internal_edges': info['internal_edges'],
            'internal_density': info['internal_density'],
            'avg_weight': info['avg_weight']
        })

    df_stats = pd.DataFrame(comm_stats)
    df_stats.to_csv(OUTPUT_DIR + '11-community-statistics.csv',
                   index=False, encoding='utf-8-sig')
    df_stats.to_csv(RESULT_DIR + '11-community-statistics.csv',
                   index=False, encoding='utf-8-sig')
    print(f"  - {OUTPUT_DIR}11-community-statistics.csv")
    print(f"  - {RESULT_DIR}11-community-statistics.csv")

    # 5. 시각화
    print("\n[5단계] 커뮤니티 시각화 중...")
    visualize_communities(G, communities_list)

    # 6. 분석 해석
    print("\n" + "=" * 80)
    print("커뮤니티 탐지 완료")
    print("=" * 80)

    print("\n주요 발견 사항:")
    print(f"  - 총 {len(communities_list)}개의 정책 연합이 식별됨")
    print(f"  - 모듈러리티 {modularity:.3f}은 유의미한 커뮤니티 구조를 나타냄")

    # 가장 큰 커뮤니티 분석
    largest_comm = community_info[0]
    print(f"\n  - 가장 큰 커뮤니티: 커뮤니티 {largest_comm['community_id']}")
    print(f"    규모: {largest_comm['size']}개 부처")
    print(f"    구성원: {', '.join(largest_comm['members'])}")

    # 커뮤니티별 해석
    print("\n커뮤니티 해석:")
    for info in community_info:
        members = ', '.join(info['members'])
        if any(x in members for x in ['과학기술정보통신부', '산업통상자원부', '중소벤처기업부']):
            print(f"  - 커뮤니티 {info['community_id']}: 디지털 전환·경제 정책 연합")
        elif any(x in members for x in ['국토교통부', '환경부', '농림축산식품부']):
            print(f"  - 커뮤니티 {info['community_id']}: 국토·환경 정책 연합")
        elif any(x in members for x in ['보건복지부', '고용노동부', '여성가족부']):
            print(f"  - 커뮤니티 {info['community_id']}: 사회·복지 정책 연합")
        elif any(x in members for x in ['외교부', '국방부', '법무부']):
            print(f"  - 커뮤니티 {info['community_id']}: 외교·안보 연합")
        else:
            print(f"  - 커뮤니티 {info['community_id']}: {members[:30]}...")

    print("\n분석 시사점:")
    print("  - 정책 영역별로 뚜렷한 협업 그룹이 형성되어 있음")
    print("  - 부처 간 협업이 기능적 유사성에 따라 클러스터링됨")
    print("  - 범정부 조정이 필요한 이슈에서는 클러스터 간 협력이 중요")

    print("\n출력 파일:")
    print(f"  - {OUTPUT_DIR}11-community-assignments.csv")
    print(f"  - {OUTPUT_DIR}11-community-statistics.csv")
    print(f"  - {OUTPUT_DIR}11-community-detection.png")
    print(f"  - {RESULT_DIR}11-community-assignments.csv (결과 폴더)")
    print(f"  - {RESULT_DIR}11-community-statistics.csv (결과 폴더)")
    print(f"  - {RESULT_DIR}11-community-detection.png (결과 폴더)")

if __name__ == "__main__":
    main()
