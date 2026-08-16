"""
Chapter 11 - 그래프 신경망과 조직 네트워크 분석
11.4 임계값 모델로 보는 파급효과

이웃 중 몇 %가 이미 채택했으면 나도 채택한다는 규칙(임계값 모델)으로
정책 확산을 돌린다. 인과추론에서 말하는 간섭(interference)을
눈으로 보게 만드는 실습이다.

수정 이력
---------
2026-08-17
1. 경로 오류: 프로젝트 루트 기준 상대경로여서 code 폴더에서 실행하면
   FileNotFoundError로 죽었다. → __file__ 기준 절대경로로 교체.
2. plt.show() 때문에 배치 실행이 멈췄다. → Agg 백엔드 + close().
3. 그림 저장 위치를 diagrams/(개념도 폴더)에서 code 폴더로 옮겼다.
   result/ 폴더 중복 저장도 삭제.
4. (d) 티핑 포인트 그림에 '티핑 포인트 θ≈0.35' 세로선이 좌표 0.35에
   고정 하드코딩되어 있었다. 실제로 채택률이 꺾이는 지점을
   계산해 그 위치에 선을 긋도록 바꿨다.
5. 마지막 요약의 "다중 시작이 더 빠르고 넓은 확산 달성" 문장이
   실제 출력(단일 5단계, 다중 5단계로 같음)과 어긋났다.
   → 두 값을 비교해 결과대로 쓰도록 고쳤다.
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


class PolicyDiffusionModel:
    """정책 확산 모델 시뮬레이션"""

    def __init__(self, G, diffusion_type='threshold'):
        self.G = G
        self.diffusion_type = diffusion_type
        self.states = {node: 'susceptible' for node in G.nodes()}
        self.adoption_history = []

    def set_initial_adopters(self, adopters):
        """초기 채택자 설정"""
        for node in adopters:
            if node in self.states:
                self.states[node] = 'adopted'
        self.adoption_history.append({
            'step': 0,
            'adopters': list(adopters),
            'total_adopted': len(adopters),
            'adoption_rate': len(adopters) / len(self.G.nodes())
        })

    def threshold_diffusion(self, threshold=0.3):
        """임계값 모델 기반 확산 - 한 스텝"""
        new_adopters = []
        for node in self.G.nodes():
            if self.states[node] == 'susceptible':
                neighbors = list(self.G.neighbors(node))
                if neighbors:
                    adopted_ratio = sum(1 for n in neighbors
                                       if self.states[n] == 'adopted') / len(neighbors)
                    if adopted_ratio >= threshold:
                        new_adopters.append(node)
        return new_adopters

    def run_simulation(self, threshold=0.3, max_steps=20):
        """시뮬레이션 실행"""
        step = 1
        while step <= max_steps:
            new_adopters = self.threshold_diffusion(threshold)
            if not new_adopters:
                break

            for node in new_adopters:
                self.states[node] = 'adopted'

            total_adopted = sum(1 for s in self.states.values() if s == 'adopted')
            self.adoption_history.append({
                'step': step,
                'adopters': new_adopters,
                'total_adopted': total_adopted,
                'adoption_rate': total_adopted / len(self.G.nodes())
            })
            step += 1

        return self.adoption_history

    def get_final_adopters(self):
        """최종 채택자 목록"""
        return [node for node, state in self.states.items() if state == 'adopted']


def create_government_network():
    """정부 부처 협업 네트워크 생성"""
    ministries = [
        '기획재정부', '교육부', '과학기술정보통신부', '외교부', '통일부',
        '법무부', '국방부', '행정안전부', '문화체육관광부', '농림축산식품부',
        '산업통상자원부', '보건복지부', '환경부', '고용노동부', '여성가족부',
        '국토교통부', '해양수산부', '중소벤처기업부'
    ]

    G = nx.Graph()
    G.add_nodes_from(ministries)

    collaborations = [
        ('과학기술정보통신부', '산업통상자원부', 0.8),
        ('과학기술정보통신부', '중소벤처기업부', 0.7),
        ('산업통상자원부', '중소벤처기업부', 0.6),
        ('기획재정부', '과학기술정보통신부', 0.7),
        ('행정안전부', '과학기술정보통신부', 0.6),
        ('기획재정부', '행정안전부', 0.7),
        ('기획재정부', '산업통상자원부', 0.6),
        ('교육부', '기획재정부', 0.5),
        ('국토교통부', '환경부', 0.7),
        ('환경부', '농림축산식품부', 0.6),
        ('농림축산식품부', '해양수산부', 0.5),
        ('국토교통부', '행정안전부', 0.6),
        ('보건복지부', '고용노동부', 0.7),
        ('고용노동부', '여성가족부', 0.6),
        ('보건복지부', '여성가족부', 0.5),
        ('교육부', '보건복지부', 0.5),
        ('외교부', '국방부', 0.7),
        ('국방부', '법무부', 0.6),
        ('통일부', '외교부', 0.6),
        ('행정안전부', '국토교통부', 0.5),
        ('행정안전부', '보건복지부', 0.5),
        ('행정안전부', '교육부', 0.6),
        ('행정안전부', '문화체육관광부', 0.5),
        ('문화체육관광부', '국토교통부', 0.5),
        ('문화체육관광부', '교육부', 0.6),
        ('해양수산부', '산업통상자원부', 0.5),
        ('해양수산부', '외교부', 0.5),
    ]

    for source, target, weight in collaborations:
        G.add_edge(source, target, weight=weight)

    return G


def run_diffusion_scenarios(G):
    """다양한 시나리오로 확산 시뮬레이션 실행"""
    results = []

    # 시나리오 1: 고중심성 부처에서 시작 (과기정통부)
    model1 = PolicyDiffusionModel(G, 'threshold')
    model1.set_initial_adopters(['과학기술정보통신부'])
    history1 = model1.run_simulation(threshold=0.25)
    results.append({
        'scenario': '고중심성 시작 (과기정통부)',
        'initial_adopter': '과학기술정보통신부',
        'threshold': 0.25,
        'final_adoption_rate': history1[-1]['adoption_rate'],
        'steps_to_saturation': len(history1) - 1,
        'final_adopters': model1.get_final_adopters(),
        'history': history1
    })

    # 시나리오 2: 중간 중심성 부처에서 시작 (교육부)
    model2 = PolicyDiffusionModel(G, 'threshold')
    model2.set_initial_adopters(['교육부'])
    history2 = model2.run_simulation(threshold=0.25)
    results.append({
        'scenario': '중간중심성 시작 (교육부)',
        'initial_adopter': '교육부',
        'threshold': 0.25,
        'final_adoption_rate': history2[-1]['adoption_rate'],
        'steps_to_saturation': len(history2) - 1,
        'final_adopters': model2.get_final_adopters(),
        'history': history2
    })

    # 시나리오 3: 저중심성 부처에서 시작 (통일부)
    model3 = PolicyDiffusionModel(G, 'threshold')
    model3.set_initial_adopters(['통일부'])
    history3 = model3.run_simulation(threshold=0.25)
    results.append({
        'scenario': '저중심성 시작 (통일부)',
        'initial_adopter': '통일부',
        'threshold': 0.25,
        'final_adoption_rate': history3[-1]['adoption_rate'],
        'steps_to_saturation': len(history3) - 1,
        'final_adopters': model3.get_final_adopters(),
        'history': history3
    })

    # 시나리오 4: 다중 초기 채택자 (과기정통부 + 행안부)
    model4 = PolicyDiffusionModel(G, 'threshold')
    model4.set_initial_adopters(['과학기술정보통신부', '행정안전부'])
    history4 = model4.run_simulation(threshold=0.25)
    results.append({
        'scenario': '다중 시작 (과기정통부+행안부)',
        'initial_adopter': '과학기술정보통신부, 행정안전부',
        'threshold': 0.25,
        'final_adoption_rate': history4[-1]['adoption_rate'],
        'steps_to_saturation': len(history4) - 1,
        'final_adopters': model4.get_final_adopters(),
        'history': history4
    })

    # 시나리오 5: 높은 임계값 (0.4)
    model5 = PolicyDiffusionModel(G, 'threshold')
    model5.set_initial_adopters(['과학기술정보통신부'])
    history5 = model5.run_simulation(threshold=0.4)
    results.append({
        'scenario': '높은 임계값 (θ=0.4)',
        'initial_adopter': '과학기술정보통신부',
        'threshold': 0.4,
        'final_adoption_rate': history5[-1]['adoption_rate'],
        'steps_to_saturation': len(history5) - 1,
        'final_adopters': model5.get_final_adopters(),
        'history': history5
    })

    return results


def visualize_diffusion(results, G):
    """확산 과정 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # (a) 확산 곡선 비교
    ax1 = axes[0, 0]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    for i, result in enumerate(results[:4]):
        steps = [h['step'] for h in result['history']]
        rates = [h['adoption_rate'] * 100 for h in result['history']]
        ax1.plot(steps, rates, 'o-', color=colors[i], linewidth=2,
                markersize=8, label=result['scenario'])
    ax1.set_xlabel('확산 단계 (Step)', fontsize=12)
    ax1.set_ylabel('채택률 (%)', fontsize=12)
    ax1.set_title('(a) 초기 채택자 위치별 확산 곡선', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # (b) 최종 채택률 비교
    ax2 = axes[0, 1]
    scenarios = [r['scenario'][:10] + '...' if len(r['scenario']) > 10 else r['scenario']
                 for r in results]
    final_rates = [r['final_adoption_rate'] * 100 for r in results]
    bars = ax2.bar(range(len(results)), final_rates, color=colors)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(['고중심성', '중간중심성', '저중심성', '다중시작', '높은임계값'],
                        rotation=45, ha='right')
    ax2.set_ylabel('최종 채택률 (%)', fontsize=12)
    ax2.set_title('(b) 시나리오별 최종 채택률', fontsize=14, fontweight='bold')
    for i, (bar, rate) in enumerate(zip(bars, final_rates)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')

    # (c) 확산 단계별 네트워크 (시나리오 1)
    ax3 = axes[1, 0]
    history = results[0]['history']
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # 채택 순서에 따른 색상
    adoption_order = {}
    for record in history:
        for adopter in record['adopters']:
            if adopter not in adoption_order:
                adoption_order[adopter] = record['step']

    node_colors = []
    for node in G.nodes():
        if node in adoption_order:
            step = adoption_order[node]
            if step == 0:
                node_colors.append('#F44336')  # 초기 채택자: 빨강
            elif step == 1:
                node_colors.append('#FF9800')  # 1단계: 주황
            elif step == 2:
                node_colors.append('#FFEB3B')  # 2단계: 노랑
            else:
                node_colors.append('#4CAF50')  # 3단계 이후: 초록
        else:
            node_colors.append('#9E9E9E')  # 미채택: 회색

    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors,
                          alpha=0.85, edgecolors='black', linewidths=1.5, ax=ax3)
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=1.5, ax=ax3)
    nx.draw_networkx_labels(G, pos, font_size=7, font_family='Malgun Gothic', ax=ax3)

    # 범례
    legend_elements = [
        plt.scatter([], [], c='#F44336', s=150, label='초기 채택 (Step 0)', edgecolors='black'),
        plt.scatter([], [], c='#FF9800', s=150, label='1단계 확산', edgecolors='black'),
        plt.scatter([], [], c='#FFEB3B', s=150, label='2단계 확산', edgecolors='black'),
        plt.scatter([], [], c='#4CAF50', s=150, label='3단계+ 확산', edgecolors='black'),
        plt.scatter([], [], c='#9E9E9E', s=150, label='미채택', edgecolors='black'),
    ]
    ax3.legend(handles=legend_elements, loc='upper left', fontsize=9)
    ax3.set_title('(c) 확산 단계별 채택 현황 (과기정통부 시작)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # (d) 티핑 포인트 분석
    ax4 = axes[1, 1]
    thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    final_rates_by_threshold = []

    for th in thresholds:
        model = PolicyDiffusionModel(G, 'threshold')
        model.set_initial_adopters(['과학기술정보통신부'])
        history = model.run_simulation(threshold=th)
        final_rates_by_threshold.append(history[-1]['adoption_rate'] * 100)

    # 채택률이 실제로 꺾이는 지점을 계산해서 그 자리에 선을 긋는다
    drops = [final_rates_by_threshold[i] - final_rates_by_threshold[i + 1]
             for i in range(len(thresholds) - 1)]
    j = int(np.argmax(drops))
    tipping = (thresholds[j] + thresholds[j + 1]) / 2

    ax4.plot(thresholds, final_rates_by_threshold, 'o-', color='#2196F3',
            linewidth=2.5, markersize=10)
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% 채택 기준선')
    ax4.axvline(x=tipping, color='green', linestyle='--', alpha=0.7,
                label=f'티핑 포인트 (θ≈{tipping:.3f})')
    ax4.fill_between(thresholds, final_rates_by_threshold, alpha=0.2, color='#2196F3')
    print(f"  - 임계값별 최종 채택률: "
          + ", ".join(f"θ={t:.2f}:{r:.1f}%"
                      for t, r in zip(thresholds, final_rates_by_threshold)))
    print(f"  - 채택률이 가장 크게 떨어지는 구간: "
          f"θ={thresholds[j]:.2f} → {thresholds[j+1]:.2f} "
          f"({final_rates_by_threshold[j]:.1f}% → {final_rates_by_threshold[j+1]:.1f}%)")
    ax4.set_xlabel('임계값 (θ)', fontsize=12)
    ax4.set_ylabel('최종 채택률 (%)', fontsize=12)
    ax4.set_title('(d) 임계값에 따른 최종 채택률 (티핑 포인트 분석)', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '11-policy-diffusion.png'),
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def main():
    print("=" * 80)
    print("제11장: 정책 확산과 네트워크 영향 모델")
    print("=" * 80)

    # 1. 네트워크 생성
    print("\n[1단계] 정부 부처 협업 네트워크 로드 중...")
    G = create_government_network()
    print(f"  - 부처 수: {G.number_of_nodes()}")
    print(f"  - 협업 관계 수: {G.number_of_edges()}")

    # 2. 확산 시뮬레이션
    print("\n[2단계] 정책 확산 시뮬레이션 실행 중...")
    results = run_diffusion_scenarios(G)

    # 3. 결과 출력
    print("\n" + "=" * 80)
    print("정책 확산 시뮬레이션 결과")
    print("=" * 80)

    print("\n=== 시나리오별 확산 결과 ===\n")
    print(f"{'시나리오':<30} {'초기 채택자':<25} {'임계값':>8} {'최종채택률':>10} {'확산단계':>8}")
    print("-" * 85)
    for r in results:
        print(f"{r['scenario']:<30} {r['initial_adopter']:<25} {r['threshold']:>8.2f} "
              f"{r['final_adoption_rate']*100:>9.1f}% {r['steps_to_saturation']:>8}")

    # 4. 상세 확산 과정 (시나리오 1)
    print("\n=== 고중심성 시작 시나리오 상세 확산 과정 ===\n")
    for record in results[0]['history']:
        adopters_str = ', '.join(record['adopters'][:3])
        if len(record['adopters']) > 3:
            adopters_str += f" 외 {len(record['adopters'])-3}개"
        print(f"  Step {record['step']}: {adopters_str} "
              f"(누적 {record['total_adopted']}개, {record['adoption_rate']*100:.1f}%)")

    # 5. 시각화
    print("\n[3단계] 확산 시각화 생성 중...")
    visualize_diffusion(results, G)

    # 6. 데이터 저장
    print("\n[4단계] 결과 데이터 저장 중...")

    # 시나리오 결과 저장
    df_results = pd.DataFrame([{
        'scenario': r['scenario'],
        'initial_adopter': r['initial_adopter'],
        'threshold': r['threshold'],
        'final_adoption_rate': r['final_adoption_rate'],
        'steps_to_saturation': r['steps_to_saturation'],
        'final_adopter_count': len(r['final_adopters'])
    } for r in results])

    df_results.to_csv(OUTPUT_DIR + '11-diffusion-results.csv',
                      index=False, encoding='utf-8-sig')

    print("  - ../data/11-diffusion-results.csv")
    print("  - 11-policy-diffusion.png")

    # 7. 주요 발견
    print("\n" + "=" * 80)
    print("주요 발견 사항")
    print("=" * 80)

    print("\n  1. 초기 채택자 위치의 중요성:")
    print(f"     - 고중심성 부처(과기정통부) 시작: "
          f"{results[0]['final_adoption_rate']*100:.1f}% 채택, "
          f"{results[0]['steps_to_saturation']}단계")
    print(f"     - 저중심성 부처(통일부) 시작: "
          f"{results[2]['final_adoption_rate']*100:.1f}% 채택, "
          f"{results[2]['steps_to_saturation']}단계")
    print(f"     → 최종 채택률은 같고 걸리는 단계 수만 "
          f"{results[2]['steps_to_saturation'] - results[0]['steps_to_saturation']}단계 차이 난다")

    print("\n  2. 다중 초기 채택자 효과:")
    s1, s4 = results[0]['steps_to_saturation'], results[3]['steps_to_saturation']
    print(f"     - 단일 시작: {results[0]['final_adoption_rate']*100:.1f}%, {s1}단계")
    print(f"     - 다중 시작: {results[3]['final_adoption_rate']*100:.1f}%, {s4}단계")
    if s4 < s1:
        print(f"     → 다중 시작이 {s1-s4}단계 빠르다")
    elif s4 == s1:
        print(f"     → 단계 수가 같다. 초기 채택자를 늘려도 빨라지지 않았다")
    else:
        print(f"     → 다중 시작이 오히려 {s4-s1}단계 느리다")

    print("\n  3. 임계값(티핑 포인트) 효과:")
    print(f"     - 낮은 임계값(0.25): {results[0]['final_adoption_rate']*100:.1f}% 채택")
    print(f"     - 높은 임계값(0.40): {results[4]['final_adoption_rate']*100:.1f}% 채택")
    print(f"     → 임계값이 조금 올라가면 확산이 통째로 멈춘다")

    return results


if __name__ == "__main__":
    main()
