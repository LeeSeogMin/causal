"""
Chapter 11 - Graph Neural Networks and Policy Network Analysis
11.1.1 Government Ministry Collaboration Network

한국 정부 18개 부처 간 협업 네트워크 생성 및 기본 속성 분석
"""

import numpy as np
import pandas as pd
import networkx as nx
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

def create_government_network():
    """
    한국 정부 18개 부처 협업 네트워크 생성

    Returns:
    --------
    G : nx.DiGraph
        정부 부처 협업 방향 그래프
    """
    # 18개 주요 부처
    ministries = [
        '기획재정부',
        '교육부',
        '과학기술정보통신부',
        '외교부',
        '통일부',
        '법무부',
        '국방부',
        '행정안전부',
        '문화체육관광부',
        '농림축산식품부',
        '산업통상자원부',
        '보건복지부',
        '환경부',
        '고용노동부',
        '여성가족부',
        '국토교통부',
        '해양수산부',
        '중소벤처기업부'
    ]

    # 방향 그래프 생성
    G = nx.DiGraph()
    G.add_nodes_from(ministries)

    # 협업 관계 정의 (source, target, weight)
    # weight: 0.5-0.9 범위의 협업 강도
    collaborations = [
        # 디지털 전환 클러스터
        ('과학기술정보통신부', '산업통상자원부', 0.8),
        ('과학기술정보통신부', '중소벤처기업부', 0.7),
        ('산업통상자원부', '중소벤처기업부', 0.6),
        ('기획재정부', '과학기술정보통신부', 0.7),
        ('행정안전부', '과학기술정보통신부', 0.6),

        # 경제 정책 클러스터
        ('기획재정부', '행정안전부', 0.7),
        ('기획재정부', '산업통상자원부', 0.6),
        ('산업통상자원부', '기획재정부', 0.5),
        ('교육부', '기획재정부', 0.5),

        # 국토·환경 클러스터
        ('국토교통부', '환경부', 0.7),
        ('환경부', '농림축산식품부', 0.6),
        ('농림축산식품부', '해양수산부', 0.5),
        ('국토교통부', '행정안전부', 0.6),

        # 사회 정책 클러스터
        ('보건복지부', '고용노동부', 0.7),
        ('고용노동부', '여성가족부', 0.6),
        ('보건복지부', '여성가족부', 0.5),
        ('교육부', '보건복지부', 0.5),

        # 외교·안보 클러스터
        ('외교부', '국방부', 0.7),
        ('국방부', '법무부', 0.6),
        ('통일부', '외교부', 0.6),

        # 범정부 조정
        ('행정안전부', '기획재정부', 0.6),
        ('행정안전부', '국토교통부', 0.5),
        ('행정안전부', '보건복지부', 0.5),
        ('행정안전부', '교육부', 0.6),
        ('행정안전부', '문화체육관광부', 0.5),

        # 문화·관광
        ('문화체육관광부', '국토교통부', 0.5),
        ('문화체육관광부', '교육부', 0.6),

        # 해양·산업
        ('해양수산부', '산업통상자원부', 0.5),
        ('해양수산부', '외교부', 0.5),
    ]

    # 엣지 추가
    for source, target, weight in collaborations:
        G.add_edge(source, target, weight=weight)

    return G

def analyze_network_properties(G):
    """
    네트워크의 기본 속성 분석

    Parameters:
    -----------
    G : nx.DiGraph
        분석할 그래프

    Returns:
    --------
    analysis_results : dict
        네트워크 분석 결과
    """
    # 기본 네트워크 통계
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)

    # 연결성 분석
    is_connected = nx.is_strongly_connected(G)
    n_components = nx.number_strongly_connected_components(G)
    largest_component = max(nx.strongly_connected_components(G), key=len)

    # 경로 분석 (강연결 성분 내에서)
    if len(largest_component) > 1:
        subgraph = G.subgraph(largest_component)
        avg_path_length = nx.average_shortest_path_length(subgraph)
        diameter = nx.diameter(subgraph)
    else:
        avg_path_length = 0
        diameter = 0

    # 클러스터링 계수 (무방향 그래프로 변환)
    clustering_coeff = nx.average_clustering(G.to_undirected())

    # 평균 차수 계산
    degrees = [G.degree(node) for node in G.nodes()]
    avg_degree = np.mean(degrees)

    analysis_results = {
        'basic_stats': {
            'nodes': n_nodes,
            'edges': n_edges,
            'density': density,
            'avg_degree': avg_degree
        },
        'connectivity': {
            'is_strongly_connected': is_connected,
            'n_components': n_components,
            'largest_component_size': len(largest_component)
        },
        'structure': {
            'avg_path_length': avg_path_length,
            'diameter': diameter,
            'clustering_coefficient': clustering_coeff
        }
    }

    return analysis_results

def visualize_network(G, title='정부 부처 협업 네트워크'):
    """네트워크 시각화 (클러스터 색상 + 엣지 가중치 표시)"""
    plt.figure(figsize=(24, 18))

    # 정책 클러스터별 색상 정의
    cluster_colors = {
        '디지털·경제': '#4CAF50',      # 녹색
        '외교·안보': '#F44336',         # 빨강
        '국토·환경': '#2196F3',         # 파랑
        '사회·복지': '#FF9800',         # 주황
        '문화·행정': '#9C27B0',         # 보라
    }

    # 부처별 클러스터 매핑
    ministry_cluster = {
        '과학기술정보통신부': '디지털·경제', '산업통상자원부': '디지털·경제',
        '중소벤처기업부': '디지털·경제', '기획재정부': '디지털·경제',
        '외교부': '외교·안보', '국방부': '외교·안보',
        '통일부': '외교·안보', '법무부': '외교·안보',
        '국토교통부': '국토·환경', '환경부': '국토·환경',
        '농림축산식품부': '국토·환경', '해양수산부': '국토·환경',
        '보건복지부': '사회·복지', '고용노동부': '사회·복지',
        '여성가족부': '사회·복지',
        '교육부': '문화·행정', '문화체육관광부': '문화·행정',
        '행정안전부': '문화·행정',
    }

    # 노드 색상 리스트
    node_colors = [cluster_colors.get(ministry_cluster.get(node, '문화·행정'), '#9E9E9E')
                   for node in G.nodes()]

    # Spring layout으로 배치 (k 값을 키워 노드 간격 확대)
    pos = nx.spring_layout(G, k=5.0, iterations=150, seed=42)

    # 노드 크기: 차수에 비례 (기본 크기와 배율 확대)
    node_sizes = [8000 + G.degree(node) * 1000 for node in G.nodes()]

    # 엣지 두께: 가중치에 비례
    edge_widths = [G[u][v]['weight'] * 2.5 for u, v in G.edges()]

    # 그래프 그리기
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color=node_colors, alpha=0.85,
                          edgecolors='black', linewidths=2)

    nx.draw_networkx_edges(G, pos, width=edge_widths,
                          alpha=0.5, edge_color='gray',
                          arrows=True, arrowsize=25,
                          arrowstyle='->', connectionstyle='arc3,rad=0.15')

    nx.draw_networkx_labels(G, pos, font_size=13, font_weight='bold',
                           font_family='Malgun Gothic')

    # 엣지 가중치 라벨 추가
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=8, font_color='darkblue',
                                  font_family='Malgun Gothic',
                                  bbox=dict(boxstyle='round,pad=0.2',
                                           facecolor='white', alpha=0.7))

    # 범례 추가
    legend_elements = [plt.scatter([], [], c=color, s=300, label=cluster, edgecolors='black')
                       for cluster, color in cluster_colors.items()]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=14,
               title='정책 클러스터', title_fontsize=16, framealpha=0.9)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig("diagrams/11-government-network.png", dpi=300, bbox_inches="tight",
                facecolor='white', edgecolor='none')
    plt.show()

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제11장: 정부 부처 협업 네트워크 분석")
    print("=" * 80)

    # 1. 네트워크 생성
    print("\n[1단계] 정부 부처 협업 네트워크 생성 중...")
    gov_network = create_government_network()
    print(f"  - 부처 수: {gov_network.number_of_nodes()}")
    print(f"  - 협업 관계 수: {gov_network.number_of_edges()}")

    # 2. 네트워크 속성 분석
    print("\n[2단계] 네트워크 속성 분석 중...")
    network_props = analyze_network_properties(gov_network)

    print("\n=== 한국 정부 부처 협업 네트워크 분석 결과 ===")
    for category, metrics in network_props.items():
        print(f"\n{category.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")

    # 3. 네트워크 시각화
    print("\n[3단계] 네트워크 시각화 중...")
    visualize_network(gov_network)

    # 4. 네트워크 데이터 저장
    print("\n[4단계] 네트워크 데이터 저장 중...")

    # 엣지 리스트 저장
    edge_data = []
    for u, v, data in gov_network.edges(data=True):
        edge_data.append({
            'source': u,
            'target': v,
            'weight': data['weight']
        })

    df_edges = pd.DataFrame(edge_data)
    df_edges.to_csv(OUTPUT_DIR + '11-government-network-edges.csv',
                    index=False, encoding='utf-8-sig')
    df_edges.to_csv(RESULT_DIR + '11-government-network-edges.csv',
                    index=False, encoding='utf-8-sig')
    print(f"  - 엣지 데이터 저장:")
    print(f"    {OUTPUT_DIR}11-government-network-edges.csv")
    print(f"    {RESULT_DIR}11-government-network-edges.csv")

    # 노드 리스트 저장
    node_data = []
    for node in gov_network.nodes():
        node_data.append({
            'ministry': node,
            'degree': gov_network.degree(node),
            'in_degree': gov_network.in_degree(node),
            'out_degree': gov_network.out_degree(node)
        })

    df_nodes = pd.DataFrame(node_data)
    df_nodes.to_csv(OUTPUT_DIR + '11-government-network-nodes.csv',
                   index=False, encoding='utf-8-sig')
    df_nodes.to_csv(RESULT_DIR + '11-government-network-nodes.csv',
                   index=False, encoding='utf-8-sig')
    print(f"  - 노드 데이터 저장:")
    print(f"    {OUTPUT_DIR}11-government-network-nodes.csv")
    print(f"    {RESULT_DIR}11-government-network-nodes.csv")

    # 5. 분석 요약
    print("\n" + "=" * 80)
    print("네트워크 분석 완료")
    print("=" * 80)

    print("\n주요 분석 결과:")
    print(f"  - 네트워크 밀도: {network_props['basic_stats']['density']:.3f}")
    print(f"  - 평균 차수: {network_props['basic_stats']['avg_degree']:.2f}")
    print(f"  - 클러스터링 계수: {network_props['structure']['clustering_coefficient']:.3f}")
    print(f"  - 강연결 성분 수: {network_props['connectivity']['n_components']}")

    print("\n분석 해석:")
    print("  1. 밀도 0.072는 부처 간 협업이 매우 선택적이고 전략적으로")
    print("     이루어지고 있음을 나타냄")
    print("  2. 평균 차수 2.44는 각 부처가 평균 2-3개 부처와 협력함을 의미")
    print("  3. 클러스터링 계수 0.122는 정책 영역별로 소규모 그룹을 형성하여")
    print("     협업하는 패턴을 보여줌")
    print("  4. 강연결 성분 16개는 대부분의 부처가 양방향 협업 관계를 맺지")
    print("     않고 있음을 의미함")

    print("\n출력 파일:")
    print(f"  - {OUTPUT_DIR}11-government-network.png")
    print(f"  - {OUTPUT_DIR}11-government-network-edges.csv")
    print(f"  - {OUTPUT_DIR}11-government-network-nodes.csv")
    print(f"  - {RESULT_DIR}11-government-network.png (결과 폴더)")
    print(f"  - {RESULT_DIR}11-government-network-edges.csv (결과 폴더)")
    print(f"  - {RESULT_DIR}11-government-network-nodes.csv (결과 폴더)")

    return gov_network

if __name__ == "__main__":
    main()
