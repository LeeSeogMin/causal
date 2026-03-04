"""
제12장: 복잡계 이론과 정책 시뮬레이션
12.5 복잡계 시뮬레이션의 검증과 한계: 불확실성 정량화

모델 검증, 불확실성 분석, 민감도 분석을 수행합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, uniform
import seaborn as sns
from typing import Dict, List, Tuple, Callable

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class PolicySimulationModel:
    """정책 시뮬레이션 예시 모델"""

    def __init__(self, params: Dict[str, float]):
        """
        Parameters:
        -----------
        params : dict, 모델 매개변수
        """
        self.params = params

    def simulate(self, time_steps: int = 100) -> np.ndarray:
        """
        간단한 정책 효과 시뮬레이션

        Returns:
        --------
        outcomes : array, 시간별 정책 효과
        """
        # 초기 조건
        state = self.params.get('initial_state', 0.5)
        outcomes = []

        for t in range(time_steps):
            # 정책 효과 (비선형 동역학)
            policy_effect = self.params['policy_strength'] * (1 - state)

            # 외부 충격 (확률적 요소)
            external_shock = np.random.normal(0, self.params['volatility'])

            # 피드백 효과
            feedback = self.params['feedback_strength'] * state * (1 - state)

            # 상태 업데이트
            state = state + policy_effect + feedback + external_shock
            state = np.clip(state, 0, 1)

            outcomes.append(state)

        return np.array(outcomes)

def monte_carlo_simulation(model_func: Callable,
                          param_ranges: Dict[str, Tuple[float, float]],
                          n_samples: int = 1000) -> Dict:
    """
    몬테카를로 불확실성 분석

    Parameters:
    -----------
    model_func : callable, 시뮬레이션 모델 함수
    param_ranges : dict, 매개변수 범위
    n_samples : int, 샘플 수

    Returns:
    --------
    results : dict, 분석 결과
    """
    all_outcomes = []
    sampled_params = []

    for _ in range(n_samples):
        # 매개변수 샘플링 (균등 분포)
        params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            params[param_name] = np.random.uniform(min_val, max_val)
        sampled_params.append(params)

        # 모델 실행
        model = model_func(params)
        outcome = model.simulate()
        all_outcomes.append(outcome)

    all_outcomes = np.array(all_outcomes)

    # 통계 계산
    mean_trajectory = np.mean(all_outcomes, axis=0)
    std_trajectory = np.std(all_outcomes, axis=0)
    percentiles = np.percentile(all_outcomes, [5, 25, 50, 75, 95], axis=0)

    return {
        'outcomes': all_outcomes,
        'params': sampled_params,
        'mean': mean_trajectory,
        'std': std_trajectory,
        'percentiles': percentiles
    }

def sensitivity_analysis(model_func: Callable,
                        base_params: Dict[str, float],
                        param_ranges: Dict[str, Tuple[float, float]],
                        n_samples: int = 100) -> Dict:
    """
    전역 민감도 분석 (Sobol indices 간소화 버전)

    Parameters:
    -----------
    model_func : callable, 모델 함수
    base_params : dict, 기본 매개변수
    param_ranges : dict, 매개변수 범위
    n_samples : int, 샘플 수

    Returns:
    --------
    sensitivity : dict, 민감도 지표
    """
    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    # 첫 번째 차수 민감도 (간소화된 방법)
    first_order_indices = {}
    total_variance = 0

    # 전체 분산 계산
    base_outcomes = []
    for _ in range(n_samples):
        params = base_params.copy()
        for param, (min_val, max_val) in param_ranges.items():
            params[param] = np.random.uniform(min_val, max_val)
        model = model_func(params)
        outcome = model.simulate()
        base_outcomes.append(np.mean(outcome[-20:]))  # 마지막 20 시점 평균
    total_variance = np.var(base_outcomes)

    # 각 매개변수별 민감도
    for param_name in param_names:
        conditional_variances = []

        # 매개변수 고정하고 다른 매개변수 변동
        for fixed_value in np.linspace(param_ranges[param_name][0],
                                      param_ranges[param_name][1], 10):
            outcomes_fixed = []
            for _ in range(n_samples // 10):
                params = base_params.copy()
                for p, (min_val, max_val) in param_ranges.items():
                    if p == param_name:
                        params[p] = fixed_value
                    else:
                        params[p] = np.random.uniform(min_val, max_val)
                model = model_func(params)
                outcome = model.simulate()
                outcomes_fixed.append(np.mean(outcome[-20:]))
            conditional_variances.append(np.var(outcomes_fixed))

        # 민감도 지수 계산
        first_order_indices[param_name] = 1 - np.mean(conditional_variances) / total_variance

    return first_order_indices

def validation_analysis(model_outcomes: np.ndarray,
                       observed_data: np.ndarray) -> Dict:
    """
    모델 검증 메트릭 계산

    Parameters:
    -----------
    model_outcomes : array, 모델 예측값
    observed_data : array, 관측값

    Returns:
    --------
    metrics : dict, 검증 메트릭
    """
    # 평균 절대 오차 (MAE)
    mae = np.mean(np.abs(model_outcomes - observed_data))

    # 평균 제곱근 오차 (RMSE)
    rmse = np.sqrt(np.mean((model_outcomes - observed_data) ** 2))

    # 결정계수 (R²)
    ss_tot = np.sum((observed_data - np.mean(observed_data)) ** 2)
    ss_res = np.sum((observed_data - model_outcomes) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Theil's U statistic
    numerator = np.sqrt(np.mean((model_outcomes - observed_data) ** 2))
    denominator = np.sqrt(np.mean(model_outcomes ** 2)) + np.sqrt(np.mean(observed_data ** 2))
    theil_u = numerator / denominator

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R_squared': r_squared,
        'Theil_U': theil_u
    }

def visualize_uncertainty(mc_results: Dict):
    """불확실성 분석 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    time_steps = range(len(mc_results['mean']))

    # 1. 불확실성 범위와 평균 궤적
    axes[0, 0].plot(time_steps, mc_results['mean'], 'b-', linewidth=2, label='Mean')
    axes[0, 0].fill_between(time_steps,
                           mc_results['percentiles'][0],  # 5%
                           mc_results['percentiles'][4],  # 95%
                           alpha=0.2, color='blue', label='90% CI')
    axes[0, 0].fill_between(time_steps,
                           mc_results['percentiles'][1],  # 25%
                           mc_results['percentiles'][3],  # 75%
                           alpha=0.3, color='blue', label='50% CI')
    axes[0, 0].set_xlabel('Time', fontsize=12)
    axes[0, 0].set_ylabel('Policy Outcome', fontsize=12)
    axes[0, 0].set_title('Uncertainty Propagation', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 최종 결과 분포
    final_outcomes = mc_results['outcomes'][:, -1]
    axes[0, 1].hist(final_outcomes, bins=30, density=True,
                   alpha=0.7, color='skyblue', edgecolor='navy')
    axes[0, 1].axvline(np.mean(final_outcomes), color='red',
                      linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].axvline(np.median(final_outcomes), color='green',
                      linestyle='--', linewidth=2, label='Median')
    axes[0, 1].set_xlabel('Final Outcome', fontsize=12)
    axes[0, 1].set_ylabel('Probability Density', fontsize=12)
    axes[0, 1].set_title('Distribution of Final States', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. 시간별 분산
    variance_over_time = np.var(mc_results['outcomes'], axis=0)
    axes[1, 0].plot(time_steps, variance_over_time, 'r-', linewidth=2)
    axes[1, 0].fill_between(time_steps, 0, variance_over_time, alpha=0.3, color='red')
    axes[1, 0].set_xlabel('Time', fontsize=12)
    axes[1, 0].set_ylabel('Variance', fontsize=12)
    axes[1, 0].set_title('Uncertainty Growth Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 샘플 궤적
    n_trajectories = min(50, len(mc_results['outcomes']))
    for i in range(n_trajectories):
        axes[1, 1].plot(time_steps, mc_results['outcomes'][i],
                       alpha=0.3, color='gray', linewidth=0.5)
    axes[1, 1].plot(time_steps, mc_results['mean'],
                   'b-', linewidth=2, label='Mean Trajectory')
    axes[1, 1].set_xlabel('Time', fontsize=12)
    axes[1, 1].set_ylabel('Policy Outcome', fontsize=12)
    axes[1, 1].set_title(f'Sample Trajectories (n={n_trajectories})',
                        fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Model Uncertainty Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # 이미지 저장하지 않음 (CLAUDE.md 원칙)
    plt.show()

def visualize_sensitivity(sensitivity_indices: Dict):
    """민감도 분석 결과 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 막대 그래프
    params = list(sensitivity_indices.keys())
    indices = list(sensitivity_indices.values())
    colors = plt.cm.RdYlBu_r(np.array(indices) / max(indices))

    bars = ax1.bar(range(len(params)), indices, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(params)))
    ax1.set_xticklabels(params, rotation=45, ha='right')
    ax1.set_ylabel('Sensitivity Index', fontsize=12)
    ax1.set_title('Parameter Sensitivity Analysis', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 값 표시
    for bar, val in zip(bars, indices):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')

    # 2. 파이 차트
    # 음수 값 처리 (0으로 설정)
    positive_indices = [max(0, idx) for idx in indices]
    total = sum(positive_indices)

    if total > 0:
        sizes = [idx/total * 100 for idx in positive_indices]
        explode = [0.1 if idx == max(positive_indices) else 0 for idx in positive_indices]

        ax2.pie(sizes, labels=params, autopct='%1.1f%%',
               explode=explode, colors=colors)
        ax2.set_title('Relative Importance', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No positive indices', ha='center', va='center')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

    plt.tight_layout()
    # 이미지 저장하지 않음 (CLAUDE.md 원칙)
    plt.show()

def model_comparison():
    """다중 모델 비교 및 앙상블"""
    # 세 가지 다른 모델 구조
    models = {
        'Conservative': {'policy_strength': 0.1, 'feedback_strength': 0.05,
                        'volatility': 0.02, 'initial_state': 0.5},
        'Moderate': {'policy_strength': 0.2, 'feedback_strength': 0.1,
                    'volatility': 0.05, 'initial_state': 0.5},
        'Aggressive': {'policy_strength': 0.3, 'feedback_strength': 0.15,
                      'volatility': 0.08, 'initial_state': 0.5}
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    all_trajectories = {}
    for model_name, params in models.items():
        model = PolicySimulationModel(params)
        trajectories = []
        for _ in range(100):
            trajectories.append(model.simulate())
        all_trajectories[model_name] = np.array(trajectories)

        mean_traj = np.mean(all_trajectories[model_name], axis=0)
        std_traj = np.std(all_trajectories[model_name], axis=0)

        # 평균 궤적
        axes[0].plot(mean_traj, label=model_name, linewidth=2)
        axes[0].fill_between(range(len(mean_traj)),
                           mean_traj - std_traj,
                           mean_traj + std_traj,
                           alpha=0.2)

    # 앙상블 평균
    ensemble_mean = np.mean([np.mean(traj, axis=0)
                            for traj in all_trajectories.values()], axis=0)
    axes[0].plot(ensemble_mean, 'k--', linewidth=2, label='Ensemble Mean')

    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Policy Outcome', fontsize=12)
    axes[0].set_title('Model Comparison', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 모델 간 차이
    time_points = [25, 50, 75, 99]
    model_names = list(models.keys())
    x = np.arange(len(time_points))
    width = 0.25

    for i, model_name in enumerate(model_names):
        values = [np.mean(all_trajectories[model_name][:, t])
                 for t in time_points]
        axes[1].bar(x + i * width, values, width, label=model_name)

    axes[1].set_xlabel('Time Point', fontsize=12)
    axes[1].set_ylabel('Average Outcome', fontsize=12)
    axes[1].set_title('Outcomes at Different Time Points', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([f't={t}' for t in time_points])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    # 이미지 저장하지 않음 (CLAUDE.md 원칙)
    plt.show()

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("복잡계 시뮬레이션 검증과 불확실성 분석")
    print("=" * 60)

    # 1. 몬테카를로 불확실성 분석
    print("\n1. 몬테카를로 시뮬레이션 (1000회 실행)...")
    param_ranges = {
        'policy_strength': (0.05, 0.3),
        'feedback_strength': (0.02, 0.15),
        'volatility': (0.01, 0.1),
        'initial_state': (0.3, 0.7)
    }

    mc_results = monte_carlo_simulation(PolicySimulationModel,
                                       param_ranges,
                                       n_samples=1000)

    print(f"   - 최종 상태 평균: {np.mean(mc_results['outcomes'][:, -1]):.3f}")
    print(f"   - 최종 상태 표준편차: {np.std(mc_results['outcomes'][:, -1]):.3f}")
    print(f"   - 95% 신뢰구간: [{np.percentile(mc_results['outcomes'][:, -1], 2.5):.3f}, "
          f"{np.percentile(mc_results['outcomes'][:, -1], 97.5):.3f}]")

    # 2. 민감도 분석
    print("\n2. 전역 민감도 분석...")
    base_params = {
        'policy_strength': 0.15,
        'feedback_strength': 0.08,
        'volatility': 0.05,
        'initial_state': 0.5
    }

    sensitivity_indices = sensitivity_analysis(PolicySimulationModel,
                                              base_params,
                                              param_ranges,
                                              n_samples=500)

    print("   민감도 지수:")
    for param, index in sorted(sensitivity_indices.items(),
                              key=lambda x: x[1], reverse=True):
        print(f"   - {param}: {index:.3f}")

    # 3. 시각화
    print("\n3. 결과 시각화...")
    visualize_uncertainty(mc_results)
    visualize_sensitivity(sensitivity_indices)

    # 4. 모델 비교
    print("\n4. 다중 모델 비교...")
    model_comparison()

    # 5. 검증 예시 (가상 데이터)
    print("\n5. 모델 검증 메트릭 (가상 관측 데이터)...")
    # 가상의 관측 데이터 생성
    np.random.seed(42)
    observed = mc_results['mean'] + np.random.normal(0, 0.05, len(mc_results['mean']))

    validation_metrics = validation_analysis(mc_results['mean'], observed)
    print("   검증 메트릭:")
    for metric, value in validation_metrics.items():
        print(f"   - {metric}: {value:.4f}")

    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()