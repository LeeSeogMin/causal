"""
Chapter 11 - 네트워크 분석과 그래프 신경망
11.7 시간적 네트워크 — 연결이 분기마다 바뀐다

11.2~11.6은 1년치 협업을 한 장의 그래프로 합쳐 썼다. 실제 협업은 분기마다
생기고 사라진다. 이 스크립트는 분기별 엣지 목록(언제, 누가, 누구에게)을 만들고
(1) 분기마다 그래프의 모양이 어떻게 바뀌는지,
(2) 시간 순서를 지키면 과기정통부의 소식이 어디까지 닿는지,
(3) 11.6의 임계값 확산을 분기별 연결 위에서 돌리면 결과가 어떻게 달라지는지 계산한다.

자료: 11-government-network-edges.csv의 협업 29개를 바탕으로 분기마다
각 관계가 활성(그 분기에 협조 공문이 기준 건수 이상 오감)인지를 시드 42로 정한
연습용 자료다. 2025년 1분기부터는 'AI 행정 TF'로 과기정통부와 세 부처의
새 협업이 생긴다고 가정했다.

출력
  data/11-temporal-edges.csv   분기별 엣지 목록 (quarter, source, target)
  results/11-temporal-network.log, results/11-temporal-network.png
"""

# == RESULTS LOG (auto): 콘솔 출력과 그림을 results/ 에 저장 ==
import sys as _sys
from pathlib import Path as _Path

_results_dir = _Path(__file__).resolve().parent.parent / "results"
_results_dir.mkdir(exist_ok=True)

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for _s in self._streams:
            _s.write(data)
            _s.flush()
    def flush(self):
        for _s in self._streams:
            _s.flush()

_sys.stdout = _Tee(_sys.stdout, open(_results_dir / (_Path(__file__).stem + ".log"), "w", encoding="utf-8"))

# 그림 저장을 results/ 로 돌린다 (강의자료용 diagrams/ 경로는 그대로 둔다)
try:
    from matplotlib.figure import Figure as _Fig
    if not getattr(_Fig, "_results_patched", False):
        _orig_savefig = _Fig.savefig
        def _savefig_to_results(self, fname, *a, **k):
            try:
                _p = _Path(fname)
                if "diagrams" not in [s.lower() for s in _p.parts]:
                    fname = _results_dir / _p.name
                    print(f"[그림 저장 → {fname}]")
            except TypeError:
                pass
            return _orig_savefig(self, fname, *a, **k)
        _Fig.savefig = _savefig_to_results
        _Fig._results_patched = True
except ImportError:
    pass
# == /RESULTS LOG ==


import os
import platform
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KOREAN_FONT = {'Darwin': 'AppleGothic', 'Windows': 'Malgun Gothic'}.get(platform.system(), 'NanumGothic')
plt.rcParams['font.family'] = KOREAN_FONT
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data') + os.sep

QUARTERS = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1', '2025Q2', '2025Q3', '2025Q4']
ACTIVE_PROB = 0.6                      # 한 관계가 어느 분기에 활성일 확률
TF_START = '2025Q1'                    # AI 행정 TF 시작 분기
TF_EDGES = [('과학기술정보통신부', '보건복지부'),
            ('과학기술정보통신부', '국토교통부'),
            ('과학기술정보통신부', '교육부')]
START = '과학기술정보통신부'
THRESHOLD = 0.25


def build_temporal_edges():
    """협업 29개를 바탕으로 분기별 엣지 목록을 만든다"""
    base = pd.read_csv(OUTPUT_DIR + '11-government-network-edges.csv')
    rng = np.random.default_rng(SEED)
    rows = []
    for q in QUARTERS:
        active = rng.random(len(base)) < ACTIVE_PROB
        for (_, r), on in zip(base.iterrows(), active):
            if on:
                rows.append((q, r['source'], r['target']))
        if q >= TF_START:
            rows += [(q, s, t) for s, t in TF_EDGES]
    return pd.DataFrame(rows, columns=['quarter', 'source', 'target'])


def quarter_graph(df, q, nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(df.loc[df['quarter'] == q, ['source', 'target']].itertuples(index=False))
    return G


def threshold_step(G, adopted):
    """11.6과 같은 규칙: 이웃 가운데 채택 비율이 임계값 이상이면 채택"""
    new = set()
    for n in G.nodes():
        if n in adopted or G.degree(n) == 0:
            continue
        ratio = sum(m in adopted for m in G[n]) / G.degree(n)
        if ratio >= THRESHOLD:
            new.add(n)
    return adopted | new


def main():
    print('=' * 70)
    print('제11장: 시간적 네트워크 — 분기별 협업과 확산')
    print('=' * 70)

    df = build_temporal_edges()
    df.to_csv(OUTPUT_DIR + '11-temporal-edges.csv', index=False, encoding='utf-8-sig')
    nodes = sorted(set(pd.read_csv(OUTPUT_DIR + '11-government-network-edges.csv')[['source', 'target']].values.ravel()))
    print(f'\n[1단계] 분기별 엣지 목록: {len(df)}행 ({len(QUARTERS)}개 분기)')
    print(df.head(4).to_string(index=False))

    print('\n[2단계] 분기별 그래프 모양')
    print(f'  {"분기":<8}{"엣지":>5}{"밀도":>8}{"과기정통부 연결 수":>14}')
    stats = []
    for q in QUARTERS:
        D = nx.DiGraph()
        D.add_nodes_from(nodes)
        D.add_edges_from(df.loc[df['quarter'] == q, ['source', 'target']].itertuples(index=False))
        G = D.to_undirected()
        stats.append((q, D.number_of_edges(), nx.density(D), G.degree(START)))
        print(f'  {q:<8}{D.number_of_edges():>5}{nx.density(D):>8.3f}{G.degree(START):>14}')

    print('\n[3단계] 과기정통부에서 시작한 확산 (임계값 0.25, 한 분기에 한 단계)')
    static = nx.Graph()
    static.add_nodes_from(nodes)
    static.add_edges_from(df[['source', 'target']].itertuples(index=False))   # 8개 분기를 합친 그래프
    adopted_static, adopted_temporal = {START}, {START}
    curve_static, curve_temporal = [], []
    print(f'  {"분기":<8}{"합친 그래프":>10}{"분기별 그래프":>12}')
    for q in QUARTERS:
        G_q = quarter_graph(df, q, nodes)                       # 그 분기의 연결만
        adopted_static = threshold_step(static, adopted_static)  # 8개 분기를 합친 그래프
        adopted_temporal = threshold_step(G_q, adopted_temporal)
        curve_static.append(len(adopted_static))
        curve_temporal.append(len(adopted_temporal))
        print(f'  {q:<8}{len(adopted_static):>10}{len(adopted_temporal):>12}')
    missing = sorted(set(nodes) - adopted_temporal)
    print(f'  2025Q4까지 분기별 그래프에서 채택하지 않은 부처 {len(missing)}개: {", ".join(missing)}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].bar(QUARTERS, [s[1] for s in stats], color='#64B5F6', edgecolor='black')
    axes[0].axvline(QUARTERS.index(TF_START) - 0.5, ls='--', color='gray')
    top = max(s[1] for s in stats)
    axes[0].set_ylim(0, top + 6)
    axes[0].text(QUARTERS.index(TF_START) - 0.4, top + 3, 'AI 행정 TF 시작 →', fontsize=9)
    axes[0].set_title('(a) 분기별 활성 협업 수')
    axes[0].set_ylabel('엣지 수')
    axes[1].plot(QUARTERS, curve_static, 'o-', label='8개 분기를 합친 그래프', color='gray')
    axes[1].plot(QUARTERS, curve_temporal, 'o-', label='분기별 그래프(시간 순서를 지킴)', color='#E57373')
    axes[1].set_ylim(0, 19)
    axes[1].set_yticks(range(0, 19, 3))
    axes[1].set_title('(b) 채택한 부처 수 (18개 중, 임계값 0.25)')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, '11-temporal-network.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
