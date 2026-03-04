#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제12장: 에이전트 기반 모델링 - 정책 확산 시뮬레이션
교육 목적의 시뮬레이션 예제
"""

import numpy as np
import random


class PolicyAgent:
    """정책 확산 에이전트"""

    def __init__(self, agent_id, innovation_threshold=0.5):
        self.id = agent_id
        self.adopted = False
        self.threshold = innovation_threshold
        self.neighbors = []

    def update(self):
        """이웃의 채택률에 따라 정책 채택 결정"""
        if not self.adopted and len(self.neighbors) > 0:
            adoption_rate = sum(n.adopted for n in self.neighbors) / len(self.neighbors)
            if adoption_rate > self.threshold:
                self.adopted = True
                return True
        return False


def create_network(num_agents, network_type='random', avg_degree=4):
    """네트워크 생성"""
    agents = [PolicyAgent(i) for i in range(num_agents)]

    if network_type == 'random':
        # 무작위 네트워크
        for agent in agents:
            num_connections = np.random.poisson(avg_degree)
            neighbors = random.sample(agents, min(num_connections, num_agents-1))
            agent.neighbors = [n for n in neighbors if n.id != agent.id]

    elif network_type == 'small-world':
        # 작은 세계 네트워크 (간단한 근사)
        k = avg_degree
        for i, agent in enumerate(agents):
            # 이웃 연결
            for j in range(1, k//2 + 1):
                neighbor_idx = (i + j) % num_agents
                if agents[neighbor_idx] not in agent.neighbors:
                    agent.neighbors.append(agents[neighbor_idx])
                if agent not in agents[neighbor_idx].neighbors:
                    agents[neighbor_idx].neighbors.append(agent)
            # 일부 재배선
            if random.random() < 0.1:
                random_neighbor = random.choice(agents)
                if random_neighbor not in agent.neighbors and random_neighbor.id != agent.id:
                    agent.neighbors.append(random_neighbor)

    elif network_type == 'scale-free':
        # 척도 없는 네트워크 (간단한 근사)
        # 선호적 연결 (preferential attachment)
        for i, new_agent in enumerate(agents):
            if i == 0:
                continue
            # 기존 노드 중 연결이 많은 노드에 우선 연결
            num_new_connections = min(avg_degree, i)
            degrees = [(agent, len(agent.neighbors)) for agent in agents[:i]]
            total_degree = sum(d for _, d in degrees) + len(degrees)  # +1 for each to avoid zero
            probabilities = [(d+1)/total_degree for _, d in degrees]

            selected = np.random.choice(agents[:i], size=num_new_connections,
                                        replace=False, p=probabilities)
            for neighbor in selected:
                new_agent.neighbors.append(neighbor)
                neighbor.neighbors.append(new_agent)

    return agents


def simulate_diffusion(agents, initial_adopters=5, max_steps=50):
    """정책 확산 시뮬레이션"""
    # 초기 채택자 설정
    initial = random.sample(agents, initial_adopters)
    for agent in initial:
        agent.adopted = True

    # 확산 과정 추적
    adoption_history = []

    for step in range(max_steps):
        adopted_count = sum(agent.adopted for agent in agents)
        adoption_rate = adopted_count / len(agents)
        adoption_history.append((step, adopted_count, adoption_rate))

        # 모든 에이전트 업데이트
        newly_adopted = 0
        for agent in agents:
            if agent.update():
                newly_adopted += 1

        # 더 이상 확산이 없으면 종료
        if newly_adopted == 0:
            break

    return adoption_history


def run_network_comparison():
    """네트워크 구조별 정책 확산 비교"""
    print("=" * 60)
    print("정책 확산 시뮬레이션 - 네트워크 구조 비교")
    print("=" * 60)

    num_agents = 100
    network_types = ['random', 'small-world', 'scale-free']
    results = {}

    for net_type in network_types:
        print(f"\n네트워크 유형: {net_type}")
        print("-" * 40)

        # 에이전트 생성 및 네트워크 구성
        agents = create_network(num_agents, net_type, avg_degree=4)

        # 중도적 임계값 설정 (0.5)
        for agent in agents:
            agent.threshold = 0.5

        # 시뮬레이션 실행
        history = simulate_diffusion(agents, initial_adopters=5, max_steps=50)

        # 결과 분석
        final_adoption = history[-1][2] * 100
        steps_to_50 = next((step for step, count, rate in history if rate >= 0.5), None)
        steps_to_90 = next((step for step, count, rate in history if rate >= 0.9), None)

        print(f"최종 채택률: {final_adoption:.1f}%")
        print(f"50% 도달: {steps_to_50 if steps_to_50 else '미달성'}단계")
        print(f"90% 도달: {steps_to_90 if steps_to_90 else '미달성'}단계")

        results[net_type] = {
            'final_rate': final_adoption,
            'steps_to_50': steps_to_50 if steps_to_50 else 999,
            'steps_to_90': steps_to_90 if steps_to_90 else 999
        }

    print("\n\n" + "=" * 60)
    print("네트워크 구조별 비교 요약")
    print("=" * 60)
    print(f"{'네트워크 유형':^15} | {'최종 채택률':>10} | {'50% 시간':>8} | {'90% 시간':>8}")
    print("-" * 60)

    for net_type, res in results.items():
        print(f"{net_type:^15} | {res['final_rate']:>9.0f}% | {res['steps_to_50']:>7}단계 | {res['steps_to_90']:>7}단계")

    print("=" * 60)
    return results


def run_threshold_comparison():
    """혁신 성향별 정책 채택 비교"""
    print("\n\n" + "=" * 60)
    print("정책 확산 시뮬레이션 - 혁신 성향 비교")
    print("=" * 60)

    num_agents = 100
    thresholds = {'보수적': 0.7, '중도적': 0.5, '진보적': 0.3}
    results = {}

    for label, threshold in thresholds.items():
        print(f"\n혁신 성향: {label} (임계값 {threshold})")
        print("-" * 40)

        # 작은 세계 네트워크 사용
        agents = create_network(num_agents, 'small-world', avg_degree=4)

        for agent in agents:
            agent.threshold = threshold

        history = simulate_diffusion(agents, initial_adopters=5, max_steps=50)

        final_adoption = history[-1][2] * 100
        completion_step = len(history) - 1

        print(f"최종 채택률: {final_adoption:.1f}%")
        print(f"확산 완료: {completion_step}단계")

        results[label] = {
            'threshold': threshold,
            'final_rate': final_adoption,
            'completion': completion_step
        }

    print("\n\n" + "=" * 60)
    print("혁신 성향별 비교 요약")
    print("=" * 60)
    print(f"{'혁신 성향':^10} | {'임계값':>8} | {'최종 채택률':>10} | {'완료 시간':>8}")
    print("-" * 60)

    for label, res in results.items():
        print(f"{label:^10} | {res['threshold']:>8.1f} | {res['final_rate']:>9.0f}% | {res['completion']:>7}단계")

    print("=" * 60)
    return results


if __name__ == "__main__":
    # 네트워크 구조 비교
    net_results = run_network_comparison()

    # 혁신 성향 비교
    threshold_results = run_threshold_comparison()

    print("\n\n※ 이 시뮬레이션은 정책 확산의 네트워크 동학을 이해하기 위한 교육 목적의 예제입니다.")
    print("   실제 정책 분석 시에는 지방자치단체 간 실제 교류 네트워크, 정책 특성,")
    print("   지역 여건 등을 반영한 모델 구축이 필요합니다.")
