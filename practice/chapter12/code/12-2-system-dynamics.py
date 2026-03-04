#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제12장: 시스템 다이내믹스 - 정책 피드백 시뮬레이션
교육 목적의 시뮬레이션 예제
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
import warnings
warnings.filterwarnings('ignore')


class PolicyFeedbackModel:
    """정책 피드백 시뮬레이션 모델"""

    def __init__(self, innovation_capacity='high'):
        rates = {
            'high': (0.3, 0.2, 0.05),
            'low': (0.05, 0.02, 0.4)
        }
        self.innovation_rate, self.efficiency_gain, self.dependency_rate = rates[innovation_capacity]
        self.capacity = innovation_capacity

    def dynamics(self, state, t, policy_support):
        health, innovation, efficiency = state
        innovation_fb = self.innovation_rate * health * (1 - innovation)
        dependency_fb = -self.dependency_rate * policy_support * innovation
        
        dHealth = self.efficiency_gain * efficiency + policy_support - 0.1 * health
        dInnovation = innovation_fb + dependency_fb
        dEfficiency = 0.15 * innovation * health - 0.05 * efficiency
        
        return [dHealth, dInnovation, dEfficiency]

    def simulate(self, policy_support, time_span=50):
        initial_state = [1.0, 0.1, 0.1]
        t = np.linspace(0, time_span, 200)
        solution = odeint(self.dynamics, initial_state, t, args=(policy_support,))
        return solution[-1, 2]


def run_simulation():
    print("=" * 60)
    print("정책 피드백 시뮬레이션 - 고혁신 vs 저혁신 기업")
    print("=" * 60)

    policy_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    high_model = PolicyFeedbackModel('high')
    low_model = PolicyFeedbackModel('low')

    results = {
        'policy_level': [],
        'high_efficiency': [],
        'low_efficiency': [],
        'gap': []
    }

    print("\n정책 지원 수준별 장기 경제 효율성 비교:")
    print("-" * 60)
    print(f"{'정책 강도':^10} | {'고혁신 효율':>12} | {'저혁신 효율':>12} | {'격차':>12}")
    print("-" * 60)

    for policy in policy_levels:
        high_eff = high_model.simulate(policy)
        low_eff = low_model.simulate(policy)
        gap = high_eff - low_eff

        results['policy_level'].append(policy)
        results['high_efficiency'].append(high_eff)
        results['low_efficiency'].append(low_eff)
        results['gap'].append(gap)

        print(f"{policy:^10.1f} | {high_eff:>12,.2f} | {low_eff:>12,.2f} | {gap:>12,.2f}")

    print("-" * 60)
    print("\n주요 발견 사항:")
    print("=" * 60)

    max_high = max(results['high_efficiency'])
    max_low = max(results['low_efficiency'])
    max_gap = max(results['gap'])

    print(f"1. 고혁신 기업 최대 효율성: {max_high:,.2f}")
    print(f"2. 저혁신 기업 최대 효율성: {max_low:,.2f}")
    print(f"3. 최대 효율성 격차: {max_gap:,.2f} ({max_gap/max_low:.1f}배)")

    if len(results['high_efficiency']) >= 4:
        growth_1 = results['high_efficiency'][4] / results['high_efficiency'][3]
        growth_2 = results['high_efficiency'][5] / results['high_efficiency'][4]

        print(f"\n4. 정책 효과의 비선형성 (고혁신 기업):")
        print(f"   - 지원 0.6→0.8: {growth_1:.1f}배 증가")
        print(f"   - 지원 0.8→1.0: {growth_2:.1f}배 증가")

    print("\n※ 교육 목적의 시뮬레이션 예제입니다.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_simulation()
