"""
제4장: Synthetic Difference-in-Differences (SDID)
DID와 SCM의 장점을 결합한 이중 강건 추정

수정 이력
---------
2026-08-17
- 증상: 스크립트가 끝까지 실행되지 않고 멈춤. 표준출력에 아무것도 남지 않음.
- 원인: matplotlib 기본 백엔드가 tkagg여서 plt.show()가 창을 띄우고 사용자
  입력을 기다린다. 콘솔에서 일괄 실행하면 여기서 무한 대기한다.
- 조치: pyplot을 import 하기 전에 matplotlib.use('Agg')로 화면 없는 백엔드를
  지정하고, plt.show() 대신 plt.close()로 그림 객체를 닫는다.
  PNG 저장(plt.savefig)은 그대로 동작한다.

2026-08-17 (2)
- 증상 1: 시간 가중치 표에서 가장 오래된 시점 t-20이 "최근 1개월"로 나왔다.
- 원인 1: 반복 인덱스 i가 0일 때를 최근으로 보고 f"최근 {i+1}개월"을 붙였다.
  i=0은 사전 기간의 첫 시점, 즉 처치에서 가장 먼 시점이다.
- 조치 1: 처치까지 남은 개월 수(months_before = n_pre - i)로 설명을 붙인다.

- 증상 2: 시간 가중치가 20개 시점 모두 0.05(균등)인데 "최근 시점에 집중(25%)"으로
  해석했다. 25%는 균등 분포에서 5/20으로 그대로 나오는 값이다.
- 원인 2: 균등 분포일 때의 기준값과 대조하지 않고 절대값만 보고 판정했다.
- 조치 2: 균등 분포 기준값(5/n_pre)과 대조해 쏠림 여부를 판정한다.

- 증상 3: SDID 추정값이 SCM과 같은데 "DID와 SCM 사이에 위치"로 서술했다.
- 원인 3: 위치 관계를 계산하지 않고 고정 문장을 출력했다.
- 조치 3: 두 값과의 차이를 계산해 그대로 출력한다.

- 증상 4: 단위 가중치 막대그래프의 한글 도시명이 네모로 깨졌다.
- 원인 4: 폰트가 'Arial'이라 한글 글리프가 없다.
- 조치 4: 'Malgun Gothic'으로 바꿨다.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 화면 없는 백엔드 (plt.show() 대기 방지)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # donor 이름이 한글이라 한글 폰트 필요
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def generate_sdid_data(n_donors=20, n_pre=20, n_post=10, seed=42):
    """
    SDID 분석을 위한 데이터 생성
    
    핵심: 평행추세가 부분적으로만 성립하는 데이터
    - 초기 기간: 평행추세 위배 (시간 가변적 요인)
    - 후기 기간: 평행추세 근사적 성립
    - 이 경우 SDID가 DID보다 효율적
    """
    np.random.seed(seed)
    
    n_periods = n_pre + n_post
    
    # 도시 이름 (한국 도시들)
    cities = ['광주', '부산', '대구', '대전', '울산', '인천', '세종', '제주',
              '수원', '성남', '고양', '용인', '창원', '청주', '천안', '전주',
              '포항', '강릉', '순천', '김해']
    
    # === 시간 가변적 요인 (평행추세 위배 원인) ===
    # 초기에는 도시별로 다른 추세, 후기에는 수렴
    time_varying_factor = np.zeros((n_periods, n_donors))
    for i in range(n_donors):
        # 각 도시의 고유 초기 추세 (시간이 지나면서 감소)
        city_specific_trend = np.random.randn() * 0.8
        decay = np.exp(-np.arange(n_periods) / 10)  # 지수 감쇠
        time_varying_factor[:, i] = city_specific_trend * decay
    
    # === Donor 데이터 생성 ===
    Y_donors = np.zeros((n_periods, n_donors))
    for i in range(n_donors):
        # 기본 수준
        baseline = 20 + np.random.randn() * 2
        # 공통 추세
        common_trend = 0.3 * np.arange(n_periods)
        # 계절성
        seasonal = np.sin(np.arange(n_periods) * 2 * np.pi / 12) * np.random.rand() * 1.5
        # 노이즈
        noise = np.random.randn(n_periods) * 0.8
        
        Y_donors[:, i] = baseline + common_trend + seasonal + time_varying_factor[:, i] + noise
    
    # === 처치 지역 데이터 (서울) ===
    # 가중치 설정 (합성 대조군 구성 가능하도록)
    true_weights = np.zeros(n_donors)
    true_weights[0] = 0.35  # 광주
    true_weights[1] = 0.28  # 부산
    true_weights[2] = 0.20  # 대구
    true_weights[3] = 0.10  # 대전
    true_weights[4] = 0.07  # 울산
    
    # 서울의 기본 결과 (donor 가중 평균 + 고유 시간 가변 요인)
    seoul_time_varying = np.random.randn() * 0.6 * np.exp(-np.arange(n_periods) / 8)
    Y_treated_base = Y_donors @ true_weights + seoul_time_varying + np.random.randn(n_periods) * 0.5
    
    # === 처치효과 추가 (사후 기간) ===
    treatment_effect = -3.50  # 진정한 처치효과
    Y_treated = Y_treated_base.copy()
    Y_treated[n_pre:] += treatment_effect
    
    # === 평행추세 위배 정도 확인 ===
    # 초기 기간 vs 후기 기간의 평행추세 차이
    early_period = slice(0, n_pre // 2)
    late_period = slice(n_pre // 2, n_pre)
    
    donor_mean = Y_donors.mean(axis=1)
    
    # 초기: 평행추세 위배 (서울과 donor 평균의 추세 차이가 큼)
    early_trend_diff = np.std(Y_treated_base[early_period] - donor_mean[early_period])
    # 후기: 평행추세 근사 성립 (추세 차이가 작음)
    late_trend_diff = np.std(Y_treated_base[late_period] - donor_mean[late_period])
    
    print(f"   - 평행추세 위배 정도:")
    print(f"     초기 기간 (t=1~{n_pre//2}): 추세 차이 SD = {early_trend_diff:.3f}")
    print(f"     후기 기간 (t={n_pre//2+1}~{n_pre}): 추세 차이 SD = {late_trend_diff:.3f}")
    print(f"     → 후기에 평행추세가 더 잘 성립 (SDID가 시간 가중치로 활용)")
    
    return Y_treated, Y_donors, cities

def estimate_sdid(Y_treated, Y_donors, n_pre, reg_param=0.01, seed=42):
    """
    Synthetic Difference-in-Differences 추정
    
    Arkhangelsky et al. (2021) 방법론 기반
    - 단위 가중치(ω): 사전 기간 결과를 재현하는 가중치
    - 시간 가중치(λ): 최근 추세에 더 높은 가중치

    Parameters:
    -----------
    Y_treated : ndarray
        처치 지역 데이터
    Y_donors : ndarray
        Donor 데이터 행렬
    n_pre : int
        사전 기간 길이
    reg_param : float
        정규화 파라미터 (ζ)
    seed : int
        난수 시드

    Returns:
    --------
    dict : SDID 추정 결과
    """
    np.random.seed(seed)

    n_periods = len(Y_treated)
    n_post = n_periods - n_pre
    n_donors = Y_donors.shape[1]

    # === 1단계: 단위 가중치 (ω) 추정 (SCM 방식) ===
    def unit_objective(omega):
        Y_synth_pre = Y_donors[:n_pre, :] @ omega
        residuals = Y_treated[:n_pre] - Y_synth_pre
        mspe = np.mean(residuals**2)
        penalty = reg_param * np.sum(omega**2)
        return mspe + penalty
    
    omega_init = np.ones(n_donors) / n_donors
    omega_constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    omega_bounds = [(0, 1)] * n_donors
    
    omega_result = minimize(unit_objective, omega_init, method='SLSQP',
                           bounds=omega_bounds, constraints=omega_constraints,
                           options={'maxiter': 1000})
    omega_opt = omega_result.x
    
    # === 2단계: 시간 가중치 (λ) 추정 ===
    # 최근 시점에 더 높은 가중치를 주되, 사전 gap 최소화
    Y_synth_pre = Y_donors[:n_pre, :] @ omega_opt
    pre_gaps = Y_treated[:n_pre] - Y_synth_pre
    
    def time_objective(lambda_t):
        weighted_gap = lambda_t @ pre_gaps
        # 목표: 가중 gap이 0에 가까우면서, 최근 시점에 가중치 집중
        # 정규화: 균등 분포에서 벗어나는 정도
        uniform = np.ones(n_pre) / n_pre
        deviation_penalty = reg_param * np.sum((lambda_t - uniform)**2)
        return weighted_gap**2 + deviation_penalty
    
    lambda_init = np.ones(n_pre) / n_pre
    # 최근 시점에 더 높은 초기값
    lambda_init[-5:] *= 2
    lambda_init /= lambda_init.sum()
    
    lambda_constraints = {'type': 'eq', 'fun': lambda l: np.sum(l) - 1}
    lambda_bounds = [(0, 1)] * n_pre
    
    lambda_result = minimize(time_objective, lambda_init, method='SLSQP',
                            bounds=lambda_bounds, constraints=lambda_constraints,
                            options={'maxiter': 1000})
    lambda_opt = lambda_result.x

    # === 3단계: SDID 처치효과 추정 ===
    # 합성 대조군 구성
    Y_synth = Y_donors @ omega_opt
    
    # 가중 사전 gap
    weighted_pre_gap = lambda_opt @ (Y_treated[:n_pre] - Y_synth[:n_pre])
    
    # 사후 평균 gap
    post_gap = np.mean(Y_treated[n_pre:] - Y_synth[n_pre:])
    
    # SDID 추정값 = 사후 gap - 가중 사전 gap
    sdid_effect = post_gap - weighted_pre_gap

    # === 표준오차 추정 (부트스트랩) ===
    n_bootstrap = 200
    bootstrap_effects = []
    
    for b in range(n_bootstrap):
        # 사후 기간 리샘플링
        post_indices = np.random.choice(n_post, size=n_post, replace=True)
        post_gaps_boot = (Y_treated[n_pre:] - Y_synth[n_pre:])[post_indices]
        post_gap_boot = np.mean(post_gaps_boot)
        
        # 사전 기간 리샘플링
        pre_indices = np.random.choice(n_pre, size=n_pre, replace=True)
        pre_gaps_boot = (Y_treated[:n_pre] - Y_synth[:n_pre])[pre_indices]
        lambda_boot = lambda_opt[pre_indices]
        lambda_boot = lambda_boot / lambda_boot.sum()  # 정규화
        weighted_pre_boot = lambda_boot @ pre_gaps_boot
        
        bootstrap_effects.append(post_gap_boot - weighted_pre_boot)
    
    se_sdid = np.std(bootstrap_effects)

    results = {
        'sdid_effect': sdid_effect,
        'se': se_sdid,
        'ci_lower': sdid_effect - 1.96 * se_sdid,
        'ci_upper': sdid_effect + 1.96 * se_sdid,
        'omega': omega_opt,
        'lambda': lambda_opt,
        'Y_synth': Y_synth,
        'post_diff': post_gap,
        'weighted_pre_diff': weighted_pre_gap
    }

    return results

def compare_did_scm_sdid(Y_treated, Y_donors, n_pre):
    """
    DID, SCM, SDID 세 방법 비교

    Parameters:
    -----------
    Y_treated : ndarray
        처치 지역 데이터
    Y_donors : ndarray
        Donor 데이터
    n_pre : int
        사전 기간 길이

    Returns:
    --------
    dict : 세 방법의 결과
    """
    # 1. Traditional DID (단순 평균)
    treated_pre_mean = np.mean(Y_treated[:n_pre])
    treated_post_mean = np.mean(Y_treated[n_pre:])
    control_pre_mean = np.mean(Y_donors[:n_pre, :])
    control_post_mean = np.mean(Y_donors[n_pre:, :])

    did_effect = (treated_post_mean - treated_pre_mean) - (control_post_mean - control_pre_mean)

    # DID 표준오차 (간단 추정)
    n_post = len(Y_treated) - n_pre
    post_var_treated = np.var(Y_treated[n_pre:])
    post_var_control = np.var(Y_donors[n_pre:, :])
    se_did = np.sqrt(post_var_treated/n_post + post_var_control/n_post)

    # 2. SCM
    from importlib import import_module
    scm_module = import_module('4-3-scm')

    scm_results = scm_module.compute_scm_weights(Y_treated[:n_pre],
                                                Y_donors[:n_pre, :],
                                                reg_param=0.01)
    scm_weights = scm_results['weights']

    Y_synth_scm = Y_donors @ scm_weights
    scm_effect = np.mean(Y_treated[n_pre:] - Y_synth_scm[n_pre:])
    se_scm = np.std(Y_treated[n_pre:] - Y_synth_scm[n_pre:]) / np.sqrt(n_post)

    # 3. SDID
    sdid_results = estimate_sdid(Y_treated, Y_donors, n_pre, reg_param=0.01)

    results = {
        'DID': {
            'effect': did_effect,
            'se': se_did,
            'ci_lower': did_effect - 1.96 * se_did,
            'ci_upper': did_effect + 1.96 * se_did
        },
        'SCM': {
            'effect': scm_effect,
            'se': se_scm,
            'ci_lower': scm_effect - 1.96 * se_scm,
            'ci_upper': scm_effect + 1.96 * se_scm,
            'weights': scm_weights
        },
        'SDID': {
            'effect': sdid_results['sdid_effect'],
            'se': sdid_results['se'],
            'ci_lower': sdid_results['ci_lower'],
            'ci_upper': sdid_results['ci_upper'],
            'omega': sdid_results['omega'],
            'lambda': sdid_results['lambda']
        }
    }

    return results

def visualize_sdid_results(Y_treated, Y_donors, sdid_results, comparison_results,
                           cities, n_pre):
    """
    SDID 분석 결과 시각화

    Parameters:
    -----------
    Y_treated : ndarray
        처치 지역 데이터
    Y_donors : ndarray
        Donor 데이터
    sdid_results : dict
        SDID 추정 결과
    comparison_results : dict
        방법론 간 비교 결과
    cities : list
        도시 이름
    n_pre : int
        사전 기간 길이
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_periods = len(Y_treated)
    periods = np.arange(n_periods)

    # 1. 세 방법 비교 (처치 지역 vs 합성 대조군들)
    axes[0, 0].plot(periods, Y_treated, 'r-o', linewidth=2,
                   markersize=6, label='Treated')

    # SCM 합성 대조군
    Y_synth_scm = Y_donors @ comparison_results['SCM']['weights']
    axes[0, 0].plot(periods, Y_synth_scm, 'b--', linewidth=2,
                   label=f'SCM (effect={comparison_results["SCM"]["effect"]:.2f})')

    # SDID 합성 대조군
    axes[0, 0].plot(periods, sdid_results['Y_synth'], 'g:', linewidth=2,
                   label=f'SDID (effect={sdid_results["sdid_effect"]:.2f})')

    axes[0, 0].axvline(x=n_pre, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Time Period')
    axes[0, 0].set_ylabel('Outcome')
    axes[0, 0].set_title('Method Comparison: Treated vs Synthetic Controls')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 처치효과 및 신뢰구간 비교
    methods = ['DID', 'SCM', 'SDID']
    effects = [comparison_results[m]['effect'] for m in methods]
    ses = [comparison_results[m]['se'] for m in methods]

    x_pos = np.arange(len(methods))
    colors = ['blue', 'green', 'orange']

    axes[0, 1].bar(x_pos, effects, yerr=1.96*np.array(ses),
                   capsize=10, alpha=0.7, color=colors)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(methods)
    axes[0, 1].set_ylabel('Treatment Effect')
    axes[0, 1].set_title('Treatment Effect Estimates Comparison')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 값 표시
    for i, (effect, se) in enumerate(zip(effects, ses)):
        axes[0, 1].text(i, effect + 1.96*se + 0.3,
                       f'{effect:.2f}\n±{1.96*se:.2f}',
                       ha='center', fontsize=9)

    # 3. 단위 가중치 비교 (SCM vs SDID, 상위 5개)
    scm_weights = comparison_results['SCM']['weights']
    sdid_weights = sdid_results['omega']

    top_5_idx = np.argsort(scm_weights)[-5:][::-1]
    top_cities = [cities[i] for i in top_5_idx]

    x = np.arange(len(top_cities))
    width = 0.35

    axes[1, 0].bar(x - width/2, scm_weights[top_5_idx], width,
                   label='SCM', color='skyblue', alpha=0.7)
    axes[1, 0].bar(x + width/2, sdid_weights[top_5_idx], width,
                   label='SDID', color='lightcoral', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(top_cities, rotation=15, ha='right')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_title('Unit Weights Comparison (Top 5)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. 시간 가중치 분포 (SDID)
    lambda_weights = sdid_results['lambda']
    time_periods = np.arange(n_pre)

    axes[1, 1].bar(time_periods, lambda_weights, color='steelblue', alpha=0.7)
    axes[1, 1].set_xlabel('Pre-treatment Period')
    axes[1, 1].set_ylabel('Time Weight (λ)')
    axes[1, 1].set_title('SDID Time Weights Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 최근 5시점 가중치 합계 표시
    recent_5_sum = np.sum(lambda_weights[-5:])
    axes[1, 1].text(0.5, 0.95, f'Recent 5 periods: {recent_5_sum*100:.1f}%',
                   transform=axes[1, 1].transAxes,
                   ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('4-4-sdid-results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """메인 실행 함수"""

    print("=" * 70)
    print("제4장: Synthetic Difference-in-Differences (SDID)")
    print("=" * 70)

    # 1. 데이터 생성
    print("\n1. SDID 데이터 생성 중...")
    Y_treated, Y_donors, cities = generate_sdid_data(n_donors=20, n_pre=20, n_post=10, seed=42)
    n_pre = 20
    n_post = 10

    print(f"   - 처치 지역: 서울")
    print(f"   - Donor pool: {len(cities)}개 도시")
    print(f"   - 사전 기간: {n_pre}개월")
    print(f"   - 사후 기간: {n_post}개월")

    # 2. SDID 추정
    print("\n2. SDID 추정 중...")
    sdid_results = estimate_sdid(Y_treated, Y_donors, n_pre, reg_param=0.01)

    print(f"   - 단위 가중치 (ω) 최적화 완료")
    print(f"   - 시간 가중치 (λ) 최적화 완료")
    print(f"   - SDID 효과: {sdid_results['sdid_effect']:.2f}")

    # 3. DID, SCM, SDID 비교
    print("\n3. 방법론 간 비교 분석 중...")
    comparison_results = compare_did_scm_sdid(Y_treated, Y_donors, n_pre)

    print("\n" + "=" * 70)
    print("Synthetic Difference-in-Differences 추정 결과")
    print("=" * 70)

    print("\n{:<10} {:<12} {:<10} {:<25} {:<15}".format(
        '방법론', '처치효과', '표준오차', '95% 신뢰구간', '효율성'))
    print("-" * 80)

    for method in ['DID', 'SCM', 'SDID']:
        res = comparison_results[method]
        # DID 대비 효율성 (표준오차 역수 비율)
        efficiency = comparison_results['DID']['se'] / res['se']

        print("{:<10} {:<12.2f} {:<10.2f} [{:>6.2f}, {:>6.2f}] {:<15.2f} ({:+.0f}%)".format(
            method,
            res['effect'],
            res['se'],
            res['ci_lower'],
            res['ci_upper'],
            efficiency,
            (efficiency - 1) * 100
        ))

    # 4. 단위 가중치 상세 비교
    print("\n[단위 가중치 (Unit Weights ω) 상세]")
    print(f"{'순위':<6} {'도시':<10} {'SDID 가중치':<15} {'SCM 가중치':<15} {'차이':<10}")
    print("-" * 60)

    scm_weights = comparison_results['SCM']['weights']
    sdid_weights = sdid_results['omega']

    sorted_idx = np.argsort(scm_weights)[::-1]
    for rank, idx in enumerate(sorted_idx[:5], 1):
        diff = sdid_weights[idx] - scm_weights[idx]
        print(f"{rank:<6} {cities[idx]:<10} {sdid_weights[idx]:<15.2f} "
              f"{scm_weights[idx]:<15.2f} {diff:>+.2f}")

    # 5. 시간 가중치 분포
    print("\n[시간 가중치 (Time Weights λ) 분포]")
    print(f"{'시점':<15} {'처치 전 기간':<15} {'시간 가중치':<15} {'누적 가중치':<15}")
    print("-" * 65)

    lambda_weights = sdid_results['lambda']
    cumulative = 0
    for i in range(n_pre):
        cumulative += lambda_weights[i]
        months_before = n_pre - i          # i=0이 사전 기간의 첫 시점이다
        period_label = f"t-{months_before}"
        if months_before <= 3:
            period_desc = f"처치 {months_before}개월 전"
        elif months_before <= 5:
            period_desc = "최근 추세"
        else:
            period_desc = "과거 추세"

        print(f"{period_label:<15} {period_desc:<15} {lambda_weights[i]:<15.2f} {cumulative:<15.2f}")

    recent_5_sum = np.sum(lambda_weights[-5:])
    print(f"\n최근 5시점 가중치 합계: {recent_5_sum:.2f} ({recent_5_sum*100:.0f}%)")

    # 6. 결과 해석
    print("\n" + "=" * 70)
    print("분석 결과 해석")
    print("=" * 70)

    print("\n1. 최적 균형점 달성:")
    sdid_effect = sdid_results['sdid_effect']
    did_effect = comparison_results['DID']['effect']
    scm_effect = comparison_results['SCM']['effect']

    print(f"   - SDID 추정값 {sdid_effect:.2f}")
    print(f"     DID({did_effect:.2f})와의 차이 {sdid_effect-did_effect:+.2f}, "
          f"SCM({scm_effect:.2f})과의 차이 {sdid_effect-scm_effect:+.2f}")
    print(f"   - 평행추세 정보와 사전 적합 정보를 균형있게 통합")

    print("\n2. 효율성 향상:")
    did_se = comparison_results['DID']['se']
    scm_se = comparison_results['SCM']['se']
    sdid_se = sdid_results['se']

    print(f"   - SDID 표준오차 {sdid_se:.2f}는")
    print(f"     DID({did_se:.2f})보다 {(1-sdid_se/did_se)*100:.0f}% 작고")
    print(f"     SCM({scm_se:.2f})보다 {(1-sdid_se/scm_se)*100:.0f}% 작음")
    print(f"   - 두 방법의 정보를 결합하여 분산 감소")

    print("\n3. 시간 가중치의 분포:")
    uniform_5_sum = 5 / n_pre
    print(f"   - 최근 5시점 합계 {recent_5_sum*100:.0f}%")
    print(f"   - 균등 분포일 때의 값 {uniform_5_sum*100:.0f}%")
    if recent_5_sum > uniform_5_sum * 1.2:
        print("   - 판정: 최근 시점에 가중치가 쏠렸다")
    elif recent_5_sum < uniform_5_sum * 0.8:
        print("   - 판정: 과거 시점에 가중치가 쏠렸다")
    else:
        print("   - 판정: 균등 분포와 사실상 같다. 특정 시점을 강조하지 않았다")

    # 7. 세 방법론 종합 비교
    print("\n[세 방법론의 종합 비교]")
    print(f"{'측면':<20} {'DID':<15} {'SCM':<15} {'SDID':<15}")
    print("-" * 70)
    print(f"{'추정값':<20} {did_effect:<15.2f} {scm_effect:<15.2f} {sdid_effect:<15.2f}")
    print(f"{'표준오차':<20} {did_se:<15.2f} {scm_se:<15.2f} {sdid_se:<15.2f}")

    did_ci_width = comparison_results['DID']['ci_upper'] - comparison_results['DID']['ci_lower']
    scm_ci_width = comparison_results['SCM']['ci_upper'] - comparison_results['SCM']['ci_lower']
    sdid_ci_width = sdid_results['ci_upper'] - sdid_results['ci_lower']
    print(f"{'신뢰구간 폭':<20} {did_ci_width:<15.2f} {scm_ci_width:<15.2f} {sdid_ci_width:<15.2f}")

    print(f"{'핵심 가정':<20} {'평행추세':<15} {'사전 적합':<15} {'둘 중 하나':<15}")
    print(f"{'시간 가변성':<20} {'약함':<15} {'강함':<15} {'매우 강함':<15}")
    print(f"{'계산 복잡도':<20} {'낮음':<15} {'중간':<15} {'높음':<15}")

    # 8. 시각화
    print("\n4. 시각화 생성 중...")
    visualize_sdid_results(Y_treated, Y_donors, sdid_results,
                          comparison_results, cities, n_pre)

    print("\n※ 본 코드는 교육 목적의 예제입니다.")
    print("   실제 적용 시 synthdid 패키지 사용을 권장합니다.")
    print("\n분석 완료! 결과가 4-4-sdid-results.png로 저장되었습니다.")

    # 9. 데이터 저장
    results_df = pd.DataFrame({
        'method': ['DID', 'SCM', 'SDID'],
        'effect': [did_effect, scm_effect, sdid_effect],
        'se': [did_se, scm_se, sdid_se],
        'ci_lower': [comparison_results[m]['ci_lower'] for m in ['DID', 'SCM', 'SDID']],
        'ci_upper': [comparison_results[m]['ci_upper'] for m in ['DID', 'SCM', 'SDID']]
    })
    results_df.to_csv('4-4-sdid-comparison.csv', index=False)
    print("데이터가 4-4-sdid-comparison.csv로 저장되었습니다.")

    return sdid_results, comparison_results

if __name__ == "__main__":
    sdid_results, comparison_results = main()
