"""
11장 용어 그림 — 작은 예시 그래프로 용어 여덟 개를 그린다
==========================================================

출력
  diagrams/11-terms.png             (강의자료 lecture/ch11.md 용어 표 아래에 넣는 그림)
  diagrams/11-centrality-path.png   (11.3 표 11-3 위에 넣는 다섯 사람 중심성 그림)
  diagrams/11-louvain-steps.png     (11.4 Louvain 방법의 원리)
  diagrams/11-gnn-steps.png         (11.5 GNN이 연결과 속성을 섞는 원리)

각 칸의 숫자는 이 스크립트가 networkx로 계산해 그림에 적는다.
"""
import os
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# 한글 폰트 설정 (mac: AppleGothic, Windows: Malgun Gothic, 그 밖: NanumGothic)
KOREAN_FONT = {'Darwin': 'AppleGothic', 'Windows': 'Malgun Gothic'}.get(platform.system(), 'NanumGothic')
plt.rcParams['font.family'] = KOREAN_FONT
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIAGRAM_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'diagrams')
OUT_PATH = os.path.join(DIAGRAM_DIR, '11-terms.png')
CENTRALITY_PATH = os.path.join(DIAGRAM_DIR, '11-centrality-path.png')
LOUVAIN_PATH = os.path.join(DIAGRAM_DIR, '11-louvain-steps.png')
GNN_PATH = os.path.join(DIAGRAM_DIR, '11-gnn-steps.png')

GRAY, BLUE, RED, GREEN, ORANGE = '#BDBDBD', '#64B5F6', '#E57373', '#81C784', '#FFB74D'
NODE_KW = dict(node_size=700, edgecolors='black', linewidths=1.2)


def draw(ax, G, pos, colors=None, edge_colors='gray', widths=1.5, labels=True, **kw):
    """작은 그래프 하나를 그린다"""
    colors = colors or [BLUE] * G.number_of_nodes()
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=widths, **kw)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, **NODE_KW)
    if labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_family=KOREAN_FONT, font_size=10)


def panel_node_edge(ax):
    G = nx.DiGraph()
    G.add_edge('A', 'B', weight=0.8)
    G.add_edge('B', 'C', weight=0.5)
    pos = {'A': (0, 0), 'B': (1, 0.6), 'C': (2, 0)}
    draw(ax, G, pos, arrows=True, arrowsize=18, node_size=700,
         widths=[4 * d['weight'] for _, _, d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=9,
                                 edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)})
    ax.text(1, -0.45, '노드(점) = 부처, 엣지(선) = 협업\n화살표 = 방향, 숫자·굵기 = 협업 세기',
            ha='center', fontsize=9)
    ax.set_ylim(-0.8, 0.9)
    ax.set_title('(a) 노드·엣지·방향·가중치')


def panel_density(ax):
    G = nx.Graph([('A', 'B'), ('B', 'C'), ('C', 'D')])
    pos = {'A': (0, 1), 'B': (1, 1), 'C': (1, 0), 'D': (0, 0)}
    possible = nx.complete_graph(['A', 'B', 'C', 'D'])
    missing = [e for e in possible.edges() if not G.has_edge(*e)]
    nx.draw_networkx_edges(possible, pos, ax=ax, edgelist=missing, style='dashed', edge_color=GRAY)
    draw(ax, G, pos, widths=2.5)
    d = nx.density(G)
    ax.text(0.5, -0.45, f'실선 {G.number_of_edges()}개 ÷ 가능한 선 {possible.number_of_edges()}개'
            f' = 밀도 {d:.2f}', ha='center', fontsize=9)
    ax.set_title('(b) 밀도')


def panel_clustering(ax):
    G = nx.Graph([('나', '가'), ('나', '다'), ('나', '라'), ('가', '다')])
    pos = {'나': (0.5, 0.4), '가': (0, 1), '다': (1, 1), '라': (0.5, -0.4)}
    colors = [ORANGE if n == '나' else BLUE for n in G.nodes()]
    edge_colors = [RED if set(e) == {'가', '다'} else 'gray' for e in G.edges()]
    draw(ax, G, pos, colors=colors, edge_colors=edge_colors, widths=2.5)
    c = nx.clustering(G, '나')
    ax.text(0.5, -0.95, f'"나"의 상대 3명이 만드는 쌍 3개 중\n서로 연결된 쌍 1개(빨간 선) → {c:.2f}',
            ha='center', fontsize=9)
    ax.set_title('(c) 클러스터링 계수')


def panel_centrality(ax):
    names = ['가', '나', '다', '라', '마']
    G = nx.path_graph(names)
    pos = {n: (i, 0) for i, n in enumerate(names)}
    colors = [RED if n == '다' else BLUE for n in names]
    draw(ax, G, pos, colors=colors, widths=2.5)
    btw = nx.betweenness_centrality(G)
    ax.text(2, -0.5, f'"다"가 빠지면 둘로 끊긴다\n매개 중심성: 다 {btw["다"]:.2f}, 가 {btw["가"]:.2f}',
            ha='center', fontsize=9)
    ax.set_ylim(-0.9, 0.5)
    ax.set_title('(d) 중심성 — 자리가 중요한 노드')


def panel_community(ax):
    left, right = ['A', 'B', 'C'], ['D', 'E', 'F']
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C'), ('D', 'E'), ('E', 'F'), ('D', 'F'), ('C', 'D')])
    pos = {'A': (0, 1), 'B': (0, 0), 'C': (0.8, 0.5), 'D': (1.7, 0.5), 'E': (2.5, 1), 'F': (2.5, 0)}
    colors = [GREEN if n in left else ORANGE for n in G.nodes()]
    edge_colors = [RED if set(e) == {'C', 'D'} else 'gray' for e in G.edges()]
    draw(ax, G, pos, colors=colors, edge_colors=edge_colors, widths=2.5)
    q = nx.community.modularity(G, [left, right])
    ax.text(1.25, -0.6, f'안쪽 선 6개, 사이 선 1개(빨간 선)\n두 묶음으로 나눈 모듈러리티 Q = {q:.2f}',
            ha='center', fontsize=9)
    ax.set_title('(e) 커뮤니티와 모듈러리티')


def panel_gnn(ax):
    G = nx.Graph([('나', '가'), ('나', '다'), ('나', '라'), ('라', '마')])
    pos = {'나': (0.5, 0.5), '가': (0, 1.1), '다': (1, 1.1), '라': (0.5, -0.2), '마': (1.2, -0.6)}
    colors = [ORANGE if n == '나' else (GRAY if n == '마' else BLUE) for n in G.nodes()]
    draw(ax, G, pos, colors=colors, widths=1.2)
    for src in ['가', '다', '라']:
        ax.annotate('', xy=pos['나'], xytext=pos[src],
                    arrowprops=dict(arrowstyle='->', color=RED, lw=2, shrinkA=14, shrinkB=14))
    ax.set_ylim(-1.5, 1.4)
    ax.text(0.5, -1.2, '1번째: 파란 이웃의 숫자를 모아 "나"를 고친다\n2번째: 회색 "마"의 정보도 "라"를 거쳐 들어온다',
            ha='center', fontsize=9)
    ax.set_title('(f) GNN — 이웃 정보 모으기')


def panel_threshold(ax):
    G = nx.Graph([('나', '가'), ('나', '다'), ('나', '라'), ('나', '마')])
    pos = {'나': (0.5, 0.4), '가': (0, 1), '다': (1, 1), '라': (0, -0.2), '마': (1, -0.2)}
    adopted = {'가', '다'}
    colors = [ORANGE if n == '나' else (GREEN if n in adopted else GRAY) for n in G.nodes()]
    draw(ax, G, pos, colors=colors, widths=2)
    ratio = len(adopted) / G.degree('나')
    ax.text(0.5, -0.85, f'이웃 4곳 중 2곳 채택(초록) → 비율 {ratio:.2f}\n'
            f'임계값 0.25면 "나"도 채택, 0.60이면 채택 안 함', ha='center', fontsize=9)
    ax.set_title('(g) 임계값')


def panel_spillover(ax):
    G = nx.Graph([('처치', '통제1'), ('통제2', '통제3')])
    pos = {'처치': (0, 0.5), '통제1': (1, 0.5), '통제2': (0, -0.4), '통제3': (1, -0.4)}
    colors = [RED if n == '처치' else BLUE for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=1100, edgecolors='black', linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_family=KOREAN_FONT, font_size=9)
    ax.annotate('', xy=(0.85, 0.62), xytext=(0.15, 0.62),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2.5, connectionstyle='arc3,rad=-0.4'))
    ax.text(0.5, 0.95, '효과가 샌다', ha='center', fontsize=9, color=RED)
    ax.set_ylim(-1.3, 1.2)
    ax.set_xlim(-0.4, 1.4)
    ax.text(0.5, -1.0, '"통제1"은 처치를 안 받았지만 연결 때문에 영향을 받는다\n"통제2·3"은 연결이 없어 영향이 없다',
            ha='center', fontsize=9)
    ax.set_title('(h) 파급효과')


def figure_centrality_path():
    """가-나-다-라-마 한 줄 관계망: (a) 다에서 각 사람까지 거리 (b) 다를 지나는 최단 경로 4개 (c) 네 가지 중심성"""
    names = ['가', '나', '다', '라', '마']
    G = nx.path_graph(names)
    pos = {n: (i, 0) for i, n in enumerate(names)}
    deg = nx.degree_centrality(G)
    clo = nx.closeness_centrality(G)
    btw = nx.betweenness_centrality(G)
    eig = nx.eigenvector_centrality_numpy(G)
    dist = nx.single_source_shortest_path_length(G, '다')

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    colors = [RED if n == '다' else BLUE for n in names]

    # (a) 근접: 다에서 각 사람까지 거리
    ax = axes[0]
    draw(ax, G, pos, colors=colors, widths=2.5)
    for n in names:
        if n != '다':
            ax.text(pos[n][0], 0.35, f'거리 {dist[n]}', ha='center', fontsize=10)
    ax.text(2, -0.55, f'"다"의 거리 합 = 1+1+2+2 = 6 → 근접 중심성 4÷6 = {clo["다"]:.2f}\n'
            f'"가"의 거리 합 = 1+2+3+4 = 10 → 4÷10 = {clo["가"]:.2f}', ha='center', fontsize=10)
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.9, 0.7)
    ax.set_title('(a) 근접 중심성 — "다"는 모두에게 가깝다')

    # (b) 매개: 가·나 ↔ 라·마 경로 4개가 모두 다를 지난다
    ax = axes[1]
    draw(ax, G, pos, colors=colors, widths=2.5)
    pairs = [('가', '라'), ('가', '마'), ('나', '라'), ('나', '마')]
    for k, (a, b) in enumerate(pairs):
        ax.annotate('', xy=(pos[b][0], -0.15), xytext=(pos[a][0], -0.15),
                    arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=1.6,
                                    connectionstyle=f'arc3,rad={0.18 + 0.07 * k}'))
    ax.text(2, -0.95, f'가·나 ↔ 라·마를 오가는 최단 경로 4개가 모두 "다"를 지난다\n'
            f'"다"가 빠지면 둘로 끊긴다 → 매개 중심성 {btw["다"]:.2f}', ha='center', fontsize=10)
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-1.25, 0.5)
    ax.set_title('(b) 매개 중심성 — "다"는 통로에 서 있다')

    # (c) 네 가지 중심성 막대
    ax = axes[2]
    measures = [('연결', deg), ('근접', clo), ('매개', btw), ('고유벡터', eig)]
    width = 0.15
    for i, n in enumerate(names):
        vals = [m[n] for _, m in measures]
        xs = [j + (i - 2) * width for j in range(len(measures))]
        group = {'가': '가·마', '마': '가·마', '나': '나·라', '라': '나·라', '다': '다'}[n]
        ax.bar(xs, vals, width, color={'가·마': GRAY, '나·라': GREEN, '다': RED}[group],
               edgecolor='black', linewidth=0.5, label=group if n in ('가', '나', '다') else None)
    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels(['연결\n(degree)', '근접\n(closeness)', '매개\n(betweenness)', '고유벡터\n(eigenvector)'])
    ax.set_ylim(0, 0.8)
    ax.legend(title='막대 순서: 가 나 다 라 마 (좌우 대칭이라 가=마, 나=라)', fontsize=9, title_fontsize=9,
              ncol=3, loc='upper left', frameon=False)
    ax.set_ylim(0, 0.95)
    ax.set_title('(c) 네 가지 중심성 — 연결은 나=다, 나머지는 다가 가장 높다')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for a in axes[:2]:
        a.axis('off')
    plt.tight_layout()
    fig.savefig(CENTRALITY_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'그림 저장: {os.path.normpath(CENTRALITY_PATH)}')

# 11.4·11.5 원리 그림에 쓰는 작은 그래프: 삼각형 둘을 선 하나(C-D)가 잇는다
TOY_EDGES = [('A', 'B'), ('B', 'C'), ('A', 'C'), ('D', 'E'), ('E', 'F'), ('D', 'F'), ('C', 'D')]
TOY_POS = {'A': (0, 1), 'B': (0, 0), 'C': (0.8, 0.5), 'D': (1.7, 0.5), 'E': (2.5, 1), 'F': (2.5, 0)}


def figure_louvain_steps():
    """Louvain: 각자 묶음 → 옮겨서 Q 올리기 → 묶음을 점으로 접기 → 더 오르지 않으면 멈춤"""
    G = nx.Graph(TOY_EDGES)
    Q = lambda parts: nx.community.modularity(G, parts)
    palette = [BLUE, GREEN, ORANGE, RED, '#BA68C8', GRAY]

    steps = [
        ('(a) 시작: 모두 각자 묶음', [{n} for n in 'ABCDEF']),
        ('(b) B를 A 묶음으로 옮김', [{'A', 'B'}, {'C'}, {'D'}, {'E'}, {'F'}]),
        ('(c) 1단계 끝: 더 옮겨도 Q가 오르지 않음', [{'A', 'B', 'C'}, {'D', 'E', 'F'}]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (title, parts) in zip(axes[:3], steps):
        color = {n: palette[k] for k, part in enumerate(parts) for n in part}
        draw(ax, G, TOY_POS, colors=[color[n] for n in G.nodes()], widths=2)
        ax.text(1.25, -0.45, f'Q = {Q(parts):.2f}', ha='center', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=12)

    # (d) 묶음을 점 하나로 접는다: 안쪽 선 3개는 점의 무게, 사이 선 1개는 두 점을 잇는다
    ax = axes[3]
    H = nx.Graph([('ABC', 'DEF')])
    pos = {'ABC': (0.4, 0.5), 'DEF': (2.1, 0.5)}
    nx.draw_networkx_edges(H, pos, ax=ax, width=2, edge_color='gray')
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=[BLUE, GREEN], node_size=2200,
                           edgecolors='black', linewidths=1.2)
    nx.draw_networkx_labels(H, pos, ax=ax, font_family=KOREAN_FONT, font_size=11)
    ax.text(1.25, 0.62, '사이 선 1개', ha='center', fontsize=10)
    ax.text(1.25, -0.25, f'두 점을 더 합치면 Q = {Q([set("ABCDEF")]):.2f}로 내려감', ha='center', fontsize=11)
    ax.text(1.25, -0.45, f'→ 멈춤, 최종 Q = {Q(steps[2][1]):.2f}', ha='center', fontsize=13, fontweight='bold')
    ax.set_title('(d) 묶음을 점 하나로 접고 반복', fontsize=12)

    for ax in axes:
        ax.set_xlim(-0.4, 2.9)
        ax.set_ylim(-0.6, 1.3)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(LOUVAIN_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'그림 저장: {os.path.normpath(LOUVAIN_PATH)}')


def figure_gnn_steps():
    """GNN 원리를 평균으로 단순화: 속성(숫자)을 연결을 따라 섞으면 가까운 노드끼리 값이 닮는다"""
    import numpy as np
    from matplotlib import cm, colors as mcolors
    G = nx.Graph(TOY_EDGES)
    x = {'A': 0.9, 'B': 0.8, 'C': 0.6, 'D': 0.3, 'E': 0.1, 'F': 0.2}   # 노드 속성(예: 표준화 전 점수)
    rounds = [x]
    for _ in range(2):
        prev = rounds[-1]
        rounds.append({n: float(np.mean([prev[n]] + [prev[m] for m in G[n]])) for n in G})

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('coolwarm')
    titles = ['(a) 입력: 연결 + 속성(숫자 1칸)', '(b) 1번 모으기: 나와 이웃의 평균', '(c) 2번 모으기 → 값이 닮은 것끼리 묶음']
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for k, (ax, vals, title) in enumerate(zip(axes, rounds, titles)):
        nx.draw_networkx_edges(G, TOY_POS, ax=ax, width=2, edge_color='gray')
        nx.draw_networkx_nodes(G, TOY_POS, ax=ax, node_size=1300, edgecolors='black', linewidths=1.2,
                               node_color=[cmap(norm(vals[n])) for n in G.nodes()])
        nx.draw_networkx_labels(G, TOY_POS, ax=ax, font_size=10, font_family=KOREAN_FONT,
                                labels={n: f'{n}\n{vals[n]:.2f}' for n in G})
        ax.set_title(title, fontsize=12)
    # (b) C가 무엇을 모으는지 화살표로 표시
    for src in ['B', 'A', 'D']:
        axes[1].annotate('', xy=TOY_POS['C'], xytext=TOY_POS[src],
                         arrowprops=dict(arrowstyle='->', color=RED, lw=2, shrinkA=18, shrinkB=18,
                                         connectionstyle='arc3,rad=0.25'))
    axes[1].text(1.25, -0.45, 'C = (C + A + B + D) ÷ 4', ha='center', fontsize=11, color=RED)
    axes[0].text(1.25, -0.45, '숫자 = 노드 속성, 선 = 연결', ha='center', fontsize=11)
    for part, cx in [('ABC', 0.27), ('DEF', 2.23)]:
        axes[2].add_patch(plt.Circle((cx, 0.5), 0.72, fill=False, ls='--', lw=1.5, color='black'))
    axes[2].text(1.25, -0.5, '두 번 모으면 두 칸 떨어진 노드 정보까지 섞인다', ha='center', fontsize=11)
    for ax in axes:
        ax.set_xlim(-0.7, 3.2)
        ax.set_ylim(-0.65, 1.45)
        ax.set_aspect('equal')
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(GNN_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'그림 저장: {os.path.normpath(GNN_PATH)}')


def main():
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    panels = [panel_node_edge, panel_density, panel_clustering, panel_centrality,
              panel_community, panel_gnn, panel_threshold, panel_spillover]
    for ax, panel in zip(axes.flat, panels):
        ax.margins(0.2)
        panel(ax)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'그림 저장: {os.path.normpath(OUT_PATH)}')
    figure_centrality_path()
    figure_louvain_steps()
    figure_gnn_steps()


if __name__ == '__main__':
    main()
