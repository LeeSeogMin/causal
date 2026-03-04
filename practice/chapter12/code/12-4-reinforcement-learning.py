#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제12장: 강화학습 - 정책 최적화 시뮬레이션
교육 목적의 시뮬레이션 예제
"""

import numpy as np


class PolicyEnvironment:
    """경제 정책 환경 시뮬레이터"""

    def __init__(self):
        self.gdp = 100
        self.welfare = 50
        self.tax_rate = 0.3
        self.step_count = 0
        self.max_steps = 20

    def reset(self):
        """환경 초기화"""
        self.gdp = 100
        self.welfare = 50
        self.tax_rate = 0.3
        self.step_count = 0
        return (self.gdp, self.welfare, self.tax_rate)

    def step(self, action):
        """
        행동 실행
        Args:
            action: -0.05~0.05 범위의 세율 조정
        Returns:
            (state, reward, done)
        """
        # 세율 조정 (0.1~0.5 범위로 제한)
        self.tax_rate = np.clip(self.tax_rate + action, 0.1, 0.5)

        # 경제 성장률 계산 (세율이 낮을수록 성장률 높음)
        growth = 0.05 * (1 - self.tax_rate) - 0.02

        # GDP 업데이트
        self.gdp *= (1 + growth)

        # 복지 수준 (세율 × GDP)
        self.welfare = self.tax_rate * self.gdp

        # 보상 계산 (성장과 복지의 가중 평균)
        reward = 0.6 * growth + 0.4 * (self.welfare / self.gdp)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return (self.gdp, self.welfare, self.tax_rate), reward, done


class QLearningAgent:
    """Q-Learning 에이전트"""

    def __init__(self, tax_rate_bins=20, action_bins=5, learning_rate=0.1,
                 discount_factor=0.95, epsilon=0.1):
        self.tax_rate_bins = tax_rate_bins
        self.action_bins = action_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Q-테이블 초기화
        self.q_table = np.zeros((tax_rate_bins, action_bins))

        # 행동 공간 정의 (세율 조정 범위)
        self.actions = np.linspace(-0.05, 0.05, action_bins)

    def discretize_state(self, tax_rate):
        """연속 상태를 이산 상태로 변환"""
        bin_index = int((tax_rate - 0.1) / 0.4 * (self.tax_rate_bins - 1))
        return np.clip(bin_index, 0, self.tax_rate_bins - 1)

    def choose_action(self, state):
        """ε-greedy 정책으로 행동 선택"""
        if np.random.random() < self.epsilon:
            # 탐색: 무작위 행동
            action_idx = np.random.randint(self.action_bins)
        else:
            # 활용: 최선의 행동
            action_idx = np.argmax(self.q_table[state])
        return action_idx

    def update(self, state, action_idx, reward, next_state):
        """Q-값 업데이트"""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action_idx]
        self.q_table[state, action_idx] += self.learning_rate * td_error

    def get_action_value(self, action_idx):
        """행동 인덱스를 실제 행동 값으로 변환"""
        return self.actions[action_idx]


def train_agent(num_episodes=1000):
    """에이전트 학습"""
    env = PolicyEnvironment()
    agent = QLearningAgent()

    episode_rewards = []
    episode_tax_rates = []

    for episode in range(num_episodes):
        state_tuple = env.reset()
        state = agent.discretize_state(state_tuple[2])  # tax_rate

        total_reward = 0
        final_tax_rate = 0

        while True:
            # 행동 선택
            action_idx = agent.choose_action(state)
            action = agent.get_action_value(action_idx)

            # 행동 실행
            next_state_tuple, reward, done = env.step(action)
            next_state = agent.discretize_state(next_state_tuple[2])

            # Q-테이블 업데이트
            agent.update(state, action_idx, reward, next_state)

            total_reward += reward
            state = next_state

            if done:
                final_tax_rate = next_state_tuple[2]
                break

        episode_rewards.append(total_reward)
        episode_tax_rates.append(final_tax_rate)

    return agent, episode_rewards, episode_tax_rates


def evaluate_performance(episode_rewards, episode_tax_rates):
    """학습 성능 평가"""
    print("=" * 60)
    print("Q-Learning 에이전트 학습 결과")
    print("=" * 60)

    # 초기, 중기, 후기 단계 구분
    stages = [
        ("초기 (1-200)", 0, 200),
        ("중기 (201-600)", 200, 600),
        ("후기 (601-1000)", 600, 1000)
    ]

    print(f"\n{'학습 단계':^15} | {'평균 보상':>10} | {'최적 세율':>10} | {'특징':^15}")
    print("-" * 65)

    results = []
    for stage_name, start, end in stages:
        avg_reward = np.mean(episode_rewards[start:end])
        avg_tax = np.mean(episode_tax_rates[start:end])
        std_tax = np.std(episode_tax_rates[start:end])

        if start == 0:
            feature = "무작위 탐색"
        elif start == 200:
            feature = "정책 개선"
        else:
            feature = "최적 정책 수렴"

        print(f"{stage_name:^15} | {avg_reward:>10.2f} | {avg_tax:>6.2f}±{std_tax:.2f} | {feature:^15}")

        results.append({
            'stage': stage_name,
            'avg_reward': avg_reward,
            'avg_tax': avg_tax,
            'std_tax': std_tax,
            'feature': feature
        })

    print("=" * 60)

    # 학습 통찰
    print("\n주요 발견 사항:")
    print("-" * 60)
    print(f"1. 학습 초기 평균 보상: {results[0]['avg_reward']:.2f}")
    print(f"2. 학습 후기 평균 보상: {results[2]['avg_reward']:.2f}")
    print(f"3. 보상 개선: {results[2]['avg_reward'] - results[0]['avg_reward']:.2f}")
    print(f"4. 최종 학습된 세율: {results[2]['avg_tax']:.2f} (변동성: ±{results[2]['std_tax']:.2f})")

    print("\n학습 통찰:")
    print("- 에이전트는 탐색-활용 균형을 통해 최적 정책 학습")
    print("- 세율 정책의 장기 효과를 고려한 의사결정 학습")
    print("- 경제 성장과 사회 복지의 균형점 발견")

    print("=" * 60)

    return results


if __name__ == "__main__":
    # 학습 실행
    print("강화학습 기반 정책 최적화 시작...")
    print("학습 중... (1000 에피소드)\n")

    agent, rewards, tax_rates = train_agent(num_episodes=1000)

    # 결과 평가
    results = evaluate_performance(rewards, tax_rates)

    print("\n※ 이 시뮬레이션은 RL 기반 정책 최적화의 개념을 이해하기 위한")
    print("   교육 목적의 단순화된 예제입니다. 실제 경제 정책은 수백 개의")
    print("   상태 변수, 복잡한 비선형 동학, 확률적 충격을 포함하므로,")
    print("   실용적 적용을 위해서는 고차원 상태 공간을 다루는 Deep RL과")
    print("   실증 데이터 기반 환경 모델링이 필요합니다.")
