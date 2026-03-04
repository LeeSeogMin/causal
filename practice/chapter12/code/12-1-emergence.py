"""
제12장: 복잡계 이론과 정책 시뮬레이션
12.1 창발성 시뮬레이션: 개별 에이전트의 단순 규칙이 복잡한 패턴 생성

이 코드는 단순한 국지적 상호작용 규칙이 어떻게 시스템 수준의
창발적 패턴을 만들어내는지 보여줍니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def simulate_emergence(n_agents=100, steps=50, threshold=0.3, seed=42):
    """
    창발성 시뮬레이션

    Parameters:
    -----------
    n_agents : int, 에이전트 수
    steps : int, 시뮬레이션 단계
    threshold : float, 상태 변화 임계값
    seed : int, 랜덤 시드

    Returns:
    --------
    history : list, 각 단계별 평균 상태값
    state_history : array, 전체 상태 이력
    """
    np.random.seed(seed)

    # 초기 상태: 무작위 (0 또는 1)
    states = np.random.choice([0, 1], n_agents)
    history = []
    state_history = []

    for step in range(steps):
        new_states = states.copy()

        # 각 에이전트의 상태 업데이트
        for i in range(n_agents):
            # 이웃 에이전트들의 상태 확인
            left_neighbor = states[(i-1) % n_agents]
            right_neighbor = states[(i+1) % n_agents]
            neighbors = [left_neighbor, right_neighbor]

            # 이웃들의 평균 상태가 임계값을 넘으면 활성화
            if np.mean(neighbors) > threshold:
                new_states[i] = 1
            else:
                new_states[i] = 0

        states = new_states
        history.append(np.mean(states))
        state_history.append(states.copy())

    return history, np.array(state_history)

def visualize_emergence(state_history, history):
    """창발 패턴 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 시공간 패턴 (Space-Time Diagram)
    ax1.imshow(state_history.T, cmap='binary', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Agent ID', fontsize=12)
    ax1.set_title('Emergent Spatiotemporal Pattern', fontsize=14, fontweight='bold')

    # 시스템 수준 행동
    ax2.plot(history, 'b-', linewidth=2, label='System State')
    ax2.fill_between(range(len(history)), 0, history, alpha=0.3)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Average State', fontsize=12)
    ax2.set_title('System-Level Behavior', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    # 이미지 저장하지 않음 (CLAUDE.md 원칙)
    plt.show()

def analyze_sensitivity(n_agents=100, steps=50):
    """임계값 변화에 대한 민감도 분석"""
    thresholds = np.linspace(0.1, 0.9, 9)
    final_states = []
    convergence_times = []

    for threshold in thresholds:
        history, _ = simulate_emergence(n_agents, steps, threshold)
        final_states.append(history[-1])

        # 수렴 시간 계산 (변화율이 0.01 이하가 되는 시점)
        convergence_time = steps
        for i in range(10, steps):
            if abs(history[i] - history[i-1]) < 0.01:
                convergence_time = i
                break
        convergence_times.append(convergence_time)

    # 결과 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(thresholds, final_states, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Final System State', fontsize=12)
    ax1.set_title('Threshold Sensitivity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.bar(thresholds, convergence_times, width=0.08, color='skyblue', edgecolor='navy')
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Convergence Time', fontsize=12)
    ax2.set_title('Time to Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    # 이미지 저장하지 않음 (CLAUDE.md 원칙)
    plt.show()

    return thresholds, final_states, convergence_times

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("복잡계 창발성 시뮬레이션")
    print("=" * 60)

    # 기본 시뮬레이션 실행
    print("\n1. 기본 창발 패턴 생성 중...")
    history, state_history = simulate_emergence(n_agents=100, steps=50, threshold=0.3)

    print(f"   - 초기 시스템 상태: {history[0]:.3f}")
    print(f"   - 최종 시스템 상태: {history[-1]:.3f}")
    print(f"   - 상태 변화 범위: {max(history) - min(history):.3f}")

    # 시각화
    print("\n2. 창발 패턴 시각화...")
    visualize_emergence(state_history, history)

    # 민감도 분석
    print("\n3. 임계값 민감도 분석 중...")
    thresholds, final_states, convergence_times = analyze_sensitivity()

    print("\n임계값별 분석 결과:")
    print("-" * 40)
    for t, fs, ct in zip(thresholds, final_states, convergence_times):
        print(f"임계값: {t:.1f} | 최종상태: {fs:.3f} | 수렴시간: {ct}")

    print("\n" + "=" * 60)
    print("시뮬레이션 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()