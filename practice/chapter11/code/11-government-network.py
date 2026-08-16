"""
Chapter 11 - 그래프 신경망과 조직 네트워크 분석
11.1 정부 부처 협업 네트워크 만들기

한국 정부 18개 부처 간 협업 네트워크 생성 및 기본 속성 분석

수정 이력
---------
2026-08-17
1. 경로 오류: OUTPUT_DIR='practice/chapter11/data/', 저장 경로 'diagrams/...' 가
   프로젝트 루트를 기준으로 쓰여 있어, code 폴더에서 실행하면
   FileNotFoundError로 죽고 code 폴더 안에 빈 practice/·result/ 폴더가 생겼다.
   → 모든 경로를 __file__ 기준 절대경로로 바꿨다.
2. 결과 그림을 diagrams/(강의노트 개념도 폴더)에 저장하던 것을
   code 폴더로 옮겼다. 개념도와 실습 산출물을 섞지 않는다.
3. plt.show()가 창을 띄워 배치 실행이 멈췄다. Agg 백엔드로 바꾸고 close()로 교체.
4. 마지막 '분석 해석' 출력이 밀도 0.072 / 평균차수 2.44 / 클러스터링 0.122 /
   강연결 성분 16으로 하드코딩되어 있었는데, 실제 계산값은
   0.095 / 3.222 / 0.361 / 9였다. 계산 결과를 그대로 쓰도록 고쳤다.
5. result/ 폴더 중복 저장을 없앴다.
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

# 경로: 이 파일 위치를 기준으로 잡는다 (어느 폴더에서 실행해도 같게 동작)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data') + os.sep
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    plt.savefig(os.path.join(BASE_DIR, '11-government-network.png'),
                dpi=150, bbox_inches="tight",
                facecolor='white', edgecolor='none')
    plt.close()

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
    print(f"  - 엣지 데이터 저장: 11-government-network-edges.csv")

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
    df_nodes = df_nodes.sort_values('degree', ascending=False)
    df_nodes.to_csv(OUTPUT_DIR + '11-government-network-nodes.csv',
                   index=False, encoding='utf-8-sig')
    print(f"  - 노드 데이터 저장: 11-government-network-nodes.csv")

    print("\n[5단계] 부처별 연결도 (총 차수 상위 8개)")
    print(f"  {'부처':<20}{'총차수':>6}{'내향':>6}{'외향':>6}")
    for _, r in df_nodes.head(8).iterrows():
        print(f"  {r['ministry']:<20}{r['degree']:>6}{r['in_degree']:>6}{r['out_degree']:>6}")

    # 5. 분석 요약
    print("\n" + "=" * 80)
    print("네트워크 분석 완료")
    print("=" * 80)

    d = network_props['basic_stats']['density']
    k = network_props['basic_stats']['avg_degree']
    cc = network_props['structure']['clustering_coefficient']
    nc = network_props['connectivity']['n_components']
    n = network_props['basic_stats']['nodes']
    m = network_props['basic_stats']['edges']

    print("\n주요 분석 결과:")
    print(f"  - 네트워크 밀도: {d:.3f}  (= {m} / ({n}x{n-1}))")
    print(f"  - 평균 차수: {k:.2f}")
    print(f"  - 클러스터링 계수: {cc:.3f}")
    print(f"  - 강연결 성분 수: {nc}")

    print("\n분석 해석:")
    print(f"  1. 밀도 {d:.3f}은 가능한 연결 중 {d*100:.1f}%만 실제로 있다는 뜻이다.")
    print(f"     부처는 아무 부처와나 협업하지 않고 상대를 골라서 맺는다.")
    print(f"  2. 평균 차수 {k:.2f}는 부처 하나가 평균 {k:.0f}개 부처와 연결됨을 뜻한다.")
    print(f"  3. 클러스터링 계수 {cc:.3f}은 '내 협업 상대끼리도 협업하는' 비율이다.")
    print(f"     밀도 {d:.3f}보다 {cc/d:.1f}배 크므로 정책 영역별 삼각 구조가 있다.")
    print(f"  4. 강연결 성분 {nc}개는 서로 오갈 수 있는 부처 묶음이 {nc}덩어리라는 뜻이다.")

    print("\n출력 파일:")
    print("  - 11-government-network.png")
    print("  - ../data/11-government-network-edges.csv")
    print("  - ../data/11-government-network-nodes.csv")

    return gov_network

if __name__ == "__main__":
    main()
