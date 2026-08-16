"""
Chapter 11 - 그래프 신경망과 조직 네트워크 분석
11.4 GNN으로 노드 임베딩 학습하기

GATv2 레이어 두 층으로 부처마다 벡터(임베딩)를 학습하고,
그 벡터를 K-Means로 묶어 커뮤니티를 얻는다. Louvain 결과와 비교한다.

수정 이력
---------
2026-08-17
1. 학습 손실 함수가 틀렸다. 원래 코드는
       loss = F.mse_loss(embeddings, torch.zeros_like(embeddings))
   였다. 이 식은 "모든 임베딩을 0으로 만들어라"는 뜻이므로,
   학습할수록 부처 사이 구분이 사라진다. 주석에는 '재구성 손실'이라
   적혀 있었지만 재구성과 아무 관계가 없었다.
   → 그래프 자기부호화기(graph autoencoder)의 링크 예측 손실로 교체했다.
     연결된 두 부처의 임베딩 내적은 크게, 연결 안 된 두 부처의 내적은
     작게 만드는 이진 교차엔트로피다. 이제 손실이 줄면 실제로
     연결 구조가 임베딩에 담긴다.
2. 난수 시드가 없어 실행할 때마다 클러스터가 달라졌다.
   → torch.manual_seed(42), np.random.seed(42) 추가.
3. 경로 오류: 프로젝트 루트 기준 상대경로여서 code 폴더에서 실행하면
   입력 CSV를 못 찾았다. → __file__ 기준 절대경로로 교체.
4. 폰트 오류: Arial이라 그래프의 한글 부처명이 네모로 나왔다.
   → Malgun Gothic으로 교체.
5. plt.show() 때문에 배치 실행이 멈췄다. → Agg 백엔드 + close().
6. 그림 저장 위치를 diagrams/에서 code 폴더로 옮겼다.
7. result/ 폴더 중복 저장 삭제.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings('ignore')

# 재현성: 시드를 고정하지 않으면 실행할 때마다 클러스터가 달라진다
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 한글 폰트 설정 (Windows: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data') + os.sep
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PolicyGNN(torch.nn.Module):
    """
    정책 네트워크 분석용 Graph Attention Network (GAT) 모델

    Parameters:
    -----------
    num_features : int
        입력 노드 특성 차원
    embedding_dim : int
        출력 임베딩 차원
    """
    def __init__(self, num_features, embedding_dim):
        super(PolicyGNN, self).__init__()
        # 첫 번째 GAT 레이어: multi-head attention (4 heads)
        self.conv1 = GATv2Conv(num_features, 32, heads=4, concat=True)
        # 두 번째 GAT 레이어: single-head attention
        self.conv2 = GATv2Conv(32*4, embedding_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        """
        순전파

        Parameters:
        -----------
        x : torch.Tensor
            노드 특성 행렬 (num_nodes, num_features)
        edge_index : torch.Tensor
            엣지 인덱스 (2, num_edges)

        Returns:
        --------
        embeddings : torch.Tensor
            노드 임베딩 (num_nodes, embedding_dim)
        """
        # 첫 번째 GAT 레이어 + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 두 번째 GAT 레이어
        x = self.conv2(x, edge_index)

        return x

def load_government_network():
    """저장된 네트워크 데이터 로드"""
    # 엣지 데이터 로드
    df_edges = pd.read_csv(OUTPUT_DIR + '11-government-network-edges.csv')

    # 무방향 그래프 생성 (GNN용)
    G = nx.Graph()

    for _, row in df_edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])

    return G

def prepare_pytorch_geometric_data(G):
    """
    NetworkX 그래프를 PyTorch Geometric Data 객체로 변환

    Parameters:
    -----------
    G : nx.Graph
        NetworkX 그래프

    Returns:
    --------
    data : torch_geometric.data.Data
        PyTorch Geometric 데이터 객체
    node_to_idx : dict
        노드 이름 → 인덱스 매핑
    idx_to_node : dict
        인덱스 → 노드 이름 매핑
    """
    # 노드 매핑 생성
    nodes = sorted(list(G.nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    # 엣지 인덱스 생성
    edge_index = []
    edge_weights = []

    for u, v, data in G.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        weight = data['weight']

        # 무방향 그래프이므로 양방향 엣지 추가
        edge_index.append([u_idx, v_idx])
        edge_index.append([v_idx, u_idx])
        edge_weights.append(weight)
        edge_weights.append(weight)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    # 노드 특성: 중심성 지표 사용
    df_centrality = pd.read_csv(OUTPUT_DIR + '11-centrality-analysis.csv')

    # 노드 순서대로 특성 정렬
    node_features = []
    for node in nodes:
        row = df_centrality[df_centrality['ministry'] == node].iloc[0]
        features = [
            row['degree_centrality'],
            row['closeness_centrality'],
            row['betweenness_centrality'],
            row['eigenvector_centrality']
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # PyTorch Geometric Data 객체 생성
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights)

    return data, node_to_idx, idx_to_node

def train_gnn_and_cluster(data, num_clusters=5, embedding_dim=16, epochs=200):
    """
    GNN 모델 학습 및 클러스터링

    Parameters:
    -----------
    data : torch_geometric.data.Data
        PyTorch Geometric 데이터
    num_clusters : int
        클러스터 수
    embedding_dim : int
        임베딩 차원
    epochs : int
        학습 에포크 수

    Returns:
    --------
    embeddings : np.ndarray
        노드 임베딩
    cluster_labels : np.ndarray
        클러스터 레이블
    """
    # 모델 초기화
    num_features = data.x.size(1)
    model = PolicyGNN(num_features, embedding_dim)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"\n[GNN 모델 학습]")
    print(f"  - 입력 특성 차원: {num_features}")
    print(f"  - 임베딩 차원: {embedding_dim}")
    print(f"  - 학습 에포크: {epochs}")
    print(f"  - 손실: 링크 예측 손실(연결된 쌍의 내적은 크게, 안 된 쌍은 작게)")

    num_nodes = data.num_nodes
    pos_edge = data.edge_index

    # 학습: 그래프 자기부호화기(GAE)의 링크 예측 손실
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        z = model(data.x, data.edge_index)

        # 연결된 쌍(양성)과 연결 안 된 쌍(음성)을 같은 수만큼 뽑는다
        neg_edge = negative_sampling(edge_index=pos_edge,
                                     num_nodes=num_nodes,
                                     num_neg_samples=pos_edge.size(1))

        pos_score = (z[pos_edge[0]] * z[pos_edge[1]]).sum(dim=1)
        neg_score = (z[neg_edge[0]] * z[neg_edge[1]]).sum(dim=1)

        loss = (F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
                + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score)))

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 최종 임베딩 추출
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        embeddings_np = embeddings.numpy()
    print(f"  - 임베딩 표준편차: {embeddings_np.std():.4f}"
          f"  (0에 가까우면 학습이 무너진 것이다)")

    # K-Means 클러스터링
    print(f"\n[K-Means 클러스터링]")
    print(f"  - 클러스터 수: {num_clusters}")

    kmeans = KMeans(n_clusters=num_clusters, random_state=SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_np)

    return embeddings_np, cluster_labels

def compare_with_louvain(cluster_labels_gnn, idx_to_node):
    """
    GNN 기반 커뮤니티와 Louvain 알고리즘 결과 비교

    Parameters:
    -----------
    cluster_labels_gnn : np.ndarray
        GNN 기반 클러스터 레이블
    idx_to_node : dict
        인덱스 → 노드 이름 매핑

    Returns:
    --------
    comparison_results : dict
        비교 결과
    """
    # Louvain 결과 로드
    df_louvain = pd.read_csv(OUTPUT_DIR + '11-community-assignments.csv')

    # GNN 결과와 매칭
    cluster_labels_louvain = []
    for idx in sorted(idx_to_node.keys()):
        node = idx_to_node[idx]
        louvain_comm = df_louvain[df_louvain['ministry'] == node]['community_id'].values[0]
        cluster_labels_louvain.append(louvain_comm)

    cluster_labels_louvain = np.array(cluster_labels_louvain)

    # 평가 지표 계산
    ari = adjusted_rand_score(cluster_labels_louvain, cluster_labels_gnn)
    nmi = normalized_mutual_info_score(cluster_labels_louvain, cluster_labels_gnn)

    comparison_results = {
        'ari': ari,
        'nmi': nmi,
        'louvain_labels': cluster_labels_louvain,
        'gnn_labels': cluster_labels_gnn
    }

    return comparison_results

def visualize_gnn_clusters(embeddings, cluster_labels, idx_to_node):
    """GNN 임베딩과 클러스터 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 2D 임베딩 시각화 (PCA 또는 처음 2차원)
    from sklearn.decomposition import PCA

    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    # 클러스터별 색상
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(cluster_labels))))

    ax1 = axes[0]
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[cluster_id]], s=200, alpha=0.7,
                   label=f'Community {cluster_id+1}', edgecolors='black', linewidths=1.5)

    # 노드 레이블 추가
    for idx, (x, y) in enumerate(embeddings_2d):
        node_name = idx_to_node[idx]
        ax1.text(x, y, node_name, fontsize=8, ha='center', va='center')

    ax1.set_xlabel('임베딩 1축 (PCA)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('임베딩 2축 (PCA)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) GNN 임베딩을 2차원으로 줄여 그린 것', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 클러스터 크기 비교
    ax2 = axes[1]
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_names = [f'Community {i+1}' for i in cluster_sizes.index]

    bars = ax2.bar(cluster_names, cluster_sizes.values, color=colors[:len(cluster_sizes)])
    ax2.set_xlabel('커뮤니티', fontsize=12, fontweight='bold')
    ax2.set_ylabel('부처 수', fontsize=12, fontweight='bold')
    ax2.set_title('(b) GNN 커뮤니티별 부처 수', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 막대 위에 숫자 표시
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '11-gnn-community.png'),
                dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제11장: GNN 기반 커뮤니티 탐지")
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

    # 2. PyTorch Geometric 데이터 준비
    print("\n[2단계] PyTorch Geometric 데이터 준비 중...")
    data, node_to_idx, idx_to_node = prepare_pytorch_geometric_data(G)
    print(f"  ✓ 노드 수: {data.num_nodes}")
    print(f"  ✓ 엣지 수: {data.num_edges // 2}개 (양방향)")
    print(f"  ✓ 노드 특성 차원: {data.num_node_features}")

    # 3. GNN 학습 및 클러스터링
    print("\n[3단계] GNN 학습 및 클러스터링 중...")
    embeddings, cluster_labels = train_gnn_and_cluster(data, num_clusters=5, embedding_dim=16, epochs=200)

    # 4. 클러스터 분석
    print("\n[4단계] 클러스터 분석 중...")

    # 클러스터별 구성원
    print("\n=== GNN 기반 커뮤니티 탐지 결과 ===")
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        members = [idx_to_node[idx] for idx in np.where(mask)[0]]
        print(f"\nCommunity {cluster_id+1} ({len(members)}개 부처):")
        print(f"  구성원: {', '.join(sorted(members))}")

    # 5. Louvain 알고리즘과 비교
    print("\n[5단계] Louvain 알고리즘과 비교 중...")
    comparison = compare_with_louvain(cluster_labels, idx_to_node)

    print(f"\n=== GNN vs Louvain 비교 결과 ===")
    print(f"  - Adjusted Rand Index (ARI): {comparison['ari']:.3f}")
    print(f"  - Normalized Mutual Information (NMI): {comparison['nmi']:.3f}")

    if comparison['ari'] >= 0.5:
        print("     → GNN과 Louvain이 유사한 커뮤니티를 탐지함")
    elif comparison['ari'] >= 0.3:
        print("     → GNN과 Louvain이 부분적으로 일치하는 커뮤니티를 탐지함")
    else:
        print("     → GNN과 Louvain이 서로 다른 관점에서 커뮤니티를 탐지함")

    # 5-1. 두 분할의 모듈러리티를 같은 그래프 위에서 비교
    from networkx.algorithms import community as nx_comm

    def to_sets(labels):
        out = {}
        for idx, lab in enumerate(labels):
            out.setdefault(int(lab), set()).add(idx_to_node[idx])
        return list(out.values())

    q_gnn = nx_comm.modularity(G, to_sets(cluster_labels), weight='weight')
    q_louvain = nx_comm.modularity(G, to_sets(comparison['louvain_labels']), weight='weight')
    print(f"\n  모듈러리티 Q (같은 그래프, 같은 가중치 기준)")
    print(f"    Louvain 분할: {q_louvain:.3f}")
    print(f"    GNN 분할:     {q_gnn:.3f}")

    # 5-2. 두 방법의 배정을 나란히 (커뮤니티 번호 자체는 서로 뜻이 다르다)
    print("\n  Louvain 배정과 GNN 배정 나란히 보기")
    rows = [(idx_to_node[i], int(comparison['louvain_labels'][i]),
             int(cluster_labels[i]) + 1) for i in sorted(idx_to_node.keys())]
    rows.sort(key=lambda r: (r[1], r[2]))
    print(f"    {'부처':<20}{'Louvain':>8}{'GNN':>6}")
    for name, lv, gn in rows:
        print(f"    {name:<20}{('C' + str(lv)):>8}{('C' + str(gn)):>6}")

    # 6. 데이터 저장
    print("\n[6단계] GNN 커뮤니티 데이터 저장 중...")

    # 커뮤니티 할당 저장
    gnn_data = []
    for idx in sorted(idx_to_node.keys()):
        node = idx_to_node[idx]
        gnn_data.append({
            'ministry': node,
            'gnn_community_id': int(cluster_labels[idx]) + 1,
            'embedding_dim1': embeddings[idx, 0],
            'embedding_dim2': embeddings[idx, 1] if embeddings.shape[1] > 1 else 0
        })

    df_gnn = pd.DataFrame(gnn_data)
    df_gnn.to_csv(OUTPUT_DIR + '11-gnn-community-assignments.csv', index=False, encoding='utf-8-sig')
    print("  - ../data/11-gnn-community-assignments.csv")

    # 비교 결과 저장
    comparison_df = pd.DataFrame({
        'Metric': ['ARI', 'NMI'],
        'Score': [comparison['ari'], comparison['nmi']]
    })
    comparison_df.to_csv(OUTPUT_DIR + '11-gnn-louvain-comparison.csv', index=False, encoding='utf-8-sig')
    print("  - ../data/11-gnn-louvain-comparison.csv")

    # 7. 시각화
    print("\n[7단계] GNN 클러스터 시각화 중...")
    visualize_gnn_clusters(embeddings, cluster_labels, idx_to_node)

    # 8. 분석 해석
    print("\n" + "=" * 80)
    print("GNN 기반 커뮤니티 탐지 완료")
    print("=" * 80)

    print("\n주요 발견 사항:")
    print(f"  - GNN을 통해 총 {len(np.unique(cluster_labels))}개의 정책 연합이 식별됨")
    print(f"  - Louvain 알고리즘과 ARI={comparison['ari']:.3f}, NMI={comparison['nmi']:.3f} 일치")

    print("\n분석 시사점:")
    print("  - GNN은 노드 특성(중심성)과 구조를 모두 고려하여 커뮤니티 탐지")
    print("  - Graph Attention 메커니즘이 부처 간 협업 강도를 학습")
    print("  - 비지도 학습 방식으로 정책 네트워크의 잠재 구조 발견")

    print("\n출력 파일:")
    print("  - ../data/11-gnn-community-assignments.csv")
    print("  - ../data/11-gnn-louvain-comparison.csv")
    print("  - 11-gnn-community.png")

if __name__ == "__main__":
    main()
