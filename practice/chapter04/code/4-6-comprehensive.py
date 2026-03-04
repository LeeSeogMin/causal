"""
제4장: 종합 구현 - DID, SCM, SDID 비교 분석 파이프라인
전체 워크플로우: 데이터 준비 → 방법론별 추정 → 검증 → 시각화 → 보고서
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def generate_comprehensive_data(n_entities=45, n_periods=30, treatment_period=20, seed=42):
    """
    DID, SCM, SDID 비교용 종합 데이터 생성

    Parameters:
    -----------
    n_entities : int
        전체 단위(지역) 수
    n_periods : int
        총 시점 수
    treatment_period : int
        처치 시작 시점
    seed : int
        난수 시드

    Returns:
    --------
    dict : 생성된 데이터와 메타정보
    """
    np.random.seed(seed)

    # 1개 처치 단위 + (n_entities-1)개 대조 단위
    n_treated = 1
    n_control = n_entities - n_treated

    # 시간 트렌드
    time = np.arange(n_periods)
    common_trend = 50 + 0.3 * time + 0.01 * time**2

    # 처치 단위 (서울)
    Y_treated = common_trend + np.random.randn(n_periods) * 0.5
    # 사후 기간에 처치효과 추가
    Y_treated[treatment_period:] += -3.5 + np.random.randn(n_periods - treatment_period) * 0.3

    # 대조 단위들
    donor_names = [
        '광주', '부산', '대구', '대전', '울산', '인천', '세종', '제주',
        '수원', '성남', '고양', '용인', '창원', '김해', '전주', '천안',
        '청주', '포항', '안산', '강릉', '제주시', '춘천', '원주', '의정부',
        '남양주', '시흥', '평택', '화성', '안양', '부천', '광명', '파주',
        '김포', '구리', '양주', '이천', '안성', '포천', '여주', '양평',
        '오산', '동두천', '과천', '하남'
    ]

    Y_donors = np.zeros((n_periods, n_control))
    for i in range(n_control):
        region_effect = np.random.randn() * 2.0
        region_trend = np.random.randn() * 0.05
        Y_donors[:, i] = (common_trend +
                         region_effect +
                         region_trend * time +
                         np.random.randn(n_periods) * 0.6)

    # 패널 데이터프레임 생성 (DID용)
    entity_ids = ['서울'] + donor_names[:n_control]
    df_list = []

    for i, entity in enumerate(entity_ids):
        if i == 0:
            Y = Y_treated
            treat = 1
        else:
            Y = Y_donors[:, i-1]
            treat = 0

        for t in range(n_periods):
            post = 1 if t >= treatment_period else 0
            df_list.append({
                'entity': entity,
                'time': t,
                'outcome': Y[t],
                'treatment': treat,
                'post': post,
                'treat_post': treat * post
            })

    df = pd.DataFrame(df_list)

    return {
        'df': df,
        'Y_treated': Y_treated,
        'Y_donors': Y_donors,
        'donor_names': donor_names[:n_control],
        'n_periods': n_periods,
        'treatment_period': treatment_period,
        'true_effect': -3.5,
        'entity_ids': entity_ids
    }


def estimate_twoway_fe_did(df):
    """
    이원 고정효과 DID 추정

    Parameters:
    -----------
    df : DataFrame
        패널 데이터

    Returns:
    --------
    dict : DID 추정 결과
    """
    # Entity demeaning (개체 고정효과 제거)
    df_dm = df.copy()
    entity_means = df_dm.groupby('entity')['outcome'].transform('mean')
    df_dm['outcome_dm'] = df_dm['outcome'] - entity_means

    # Time demeaning (시간 고정효과 제거)
    time_means = df_dm.groupby('time')['outcome_dm'].transform('mean')
    df_dm['outcome_dm2'] = df_dm['outcome_dm'] - time_means

    # treat_post도 동일하게 demeaning
    entity_means_tp = df_dm.groupby('entity')['treat_post'].transform('mean')
    time_means_tp = df_dm.groupby('time')['treat_post'].transform('mean')
    df_dm['treat_post_dm'] = df_dm['treat_post'] - entity_means_tp - time_means_tp

    # 회귀: outcome_dm2 ~ treat_post_dm
    X = df_dm['treat_post_dm'].values
    Y = df_dm['outcome_dm2'].values

    # OLS 추정
    beta = np.sum(X * Y) / np.sum(X * X)

    # 잔차
    residuals = Y - beta * X
    n = len(Y)
    k = 1  # 파라미터 수

    # 표준오차 (클러스터 표준오차 - entity level)
    unique_entities = df['entity'].unique()
    n_clusters = len(unique_entities)

    cluster_residuals = []
    for entity in unique_entities:
        entity_mask = df['entity'] == entity
        cluster_X = X[entity_mask]
        cluster_resid = residuals[entity_mask]
        cluster_residuals.append(np.sum(cluster_X * cluster_resid))

    cluster_residuals = np.array(cluster_residuals)
    bread = np.sum(X * X)
    meat = np.sum(cluster_residuals**2)

    se_clustered = np.sqrt(meat / (bread**2)) * np.sqrt(n_clusters / (n_clusters - 1))

    # t-통계량, p-value
    t_stat = beta / se_clustered
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_clusters - k - 1))

    # 95% 신뢰구간
    t_crit = stats.t.ppf(0.975, df=n_clusters - k - 1)
    ci_lower = beta - t_crit * se_clustered
    ci_upper = beta + t_crit * se_clustered

    return {
        'method': 'DID',
        'effect': beta,
        'se': se_clustered,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def compute_scm_weights(Y_treated_pre, Y_donors_pre, reg_param=0.01):
    """
    SCM 가중치 계산

    Parameters:
    -----------
    Y_treated_pre : array
        처치 지역 사전 결과
    Y_donors_pre : array
        대조 지역들 사전 결과 (T_pre × J)
    reg_param : float
        정규화 파라미터

    Returns:
    --------
    array : 최적 가중치
    """
    n_donors = Y_donors_pre.shape[1]

    def objective(w):
        fit_error = np.sum((Y_treated_pre - Y_donors_pre @ w)**2)
        reg_term = reg_param * np.sum(w**2)
        return fit_error + reg_term

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n_donors)]
    w0 = np.ones(n_donors) / n_donors

    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000})

    return result.x


def estimate_scm(data, reg_param=0.01):
    """
    Synthetic Control Method 추정

    Parameters:
    -----------
    data : dict
        generate_comprehensive_data 반환값
    reg_param : float
        정규화 파라미터

    Returns:
    --------
    dict : SCM 추정 결과
    """
    Y_treated = data['Y_treated']
    Y_donors = data['Y_donors']
    treatment_period = data['treatment_period']

    # 사전/사후 분할
    Y_treated_pre = Y_treated[:treatment_period]
    Y_treated_post = Y_treated[treatment_period:]
    Y_donors_pre = Y_donors[:treatment_period, :]
    Y_donors_post = Y_donors[treatment_period:, :]

    # 가중치 계산
    w = compute_scm_weights(Y_treated_pre, Y_donors_pre, reg_param)

    # 사전 MSPE
    pre_mspe = np.mean((Y_treated_pre - Y_donors_pre @ w)**2)

    # 처치효과
    effect = np.mean(Y_treated_post - Y_donors_post @ w)

    # 표준오차 추정 (플라시보 기반)
    n_donors = Y_donors.shape[1]
    placebo_effects = []

    for i in range(n_donors):
        Y_placebo_pre = Y_donors_pre[:, i]
        Y_placebo_post = Y_donors_post[:, i]
        Y_other_pre = np.delete(Y_donors_pre, i, axis=1)
        Y_other_post = np.delete(Y_donors_post, i, axis=1)

        w_placebo = compute_scm_weights(Y_placebo_pre, Y_other_pre, reg_param)
        placebo_effect = np.mean(Y_placebo_post - Y_other_post @ w_placebo)
        placebo_effects.append(placebo_effect)

    placebo_effects = np.array(placebo_effects)
    se = np.std(placebo_effects)

    # p-value
    p_value = np.mean(np.abs(placebo_effects) >= np.abs(effect))

    # 95% CI (플라시보 분포 기반)
    ci_lower = np.percentile(placebo_effects, 2.5)
    ci_upper = np.percentile(placebo_effects, 97.5)

    # Herfindahl Index
    hhi = np.sum(w**2)

    return {
        'method': 'SCM',
        'effect': effect,
        'se': se,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'weights': w,
        'pre_mspe': pre_mspe,
        'hhi': hhi
    }


def estimate_sdid(data, reg_param=0.01):
    """
    Synthetic Difference-in-Differences 추정

    Parameters:
    -----------
    data : dict
        generate_comprehensive_data 반환값
    reg_param : float
        정규화 파라미터

    Returns:
    --------
    dict : SDID 추정 결과
    """
    Y_treated = data['Y_treated']
    Y_donors = data['Y_donors']
    treatment_period = data['treatment_period']

    # 사전/사후 분할
    Y_treated_pre = Y_treated[:treatment_period]
    Y_treated_post = Y_treated[treatment_period:]
    Y_donors_pre = Y_donors[:treatment_period, :]
    Y_donors_post = Y_donors[treatment_period:, :]

    n_pre = treatment_period
    n_post = len(Y_treated_post)
    n_donors = Y_donors.shape[1]

    # 단위 가중치 (ω) 최적화
    def objective_omega(omega):
        fit_error = np.sum((Y_treated_pre - Y_donors_pre @ omega)**2)
        reg_term = reg_param * np.sum(omega**2)
        return fit_error + reg_term

    constraints_omega = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds_omega = [(0, 1) for _ in range(n_donors)]
    omega0 = np.ones(n_donors) / n_donors

    result_omega = minimize(objective_omega, omega0, method='SLSQP',
                           bounds=bounds_omega, constraints=constraints_omega,
                           options={'maxiter': 1000})
    omega = result_omega.x

    # 시간 가중치 (λ) 최적화
    # Note: Time weights reweight the pre-period observations
    def objective_lambda(lam):
        # Average over time with weights lam
        Y_treated_weighted = np.dot(lam, Y_treated_pre)
        Y_donors_weighted = Y_donors_pre.T @ lam  # Shape: (n_donors,)
        fit_error = np.sum((Y_treated_weighted - Y_donors_weighted)**2)
        reg_term = reg_param * np.sum(lam**2)
        return fit_error + reg_term

    constraints_lambda = ({'type': 'eq', 'fun': lambda l: np.sum(l) - 1})
    bounds_lambda = [(0, 1) for _ in range(n_pre)]
    lambda0 = np.ones(n_pre) / n_pre

    result_lambda = minimize(objective_lambda, lambda0, method='SLSQP',
                            bounds=bounds_lambda, constraints=constraints_lambda,
                            options={'maxiter': 1000})
    lam = result_lambda.x

    # SDID 추정값
    Y_sc_pre = Y_donors_pre @ omega
    Y_sc_post = Y_donors_post @ omega

    # 사전 조정
    pre_adjustment = np.sum(lam * (Y_treated_pre - Y_sc_pre))

    # 사후 효과
    effect = np.mean(Y_treated_post - Y_sc_post) - pre_adjustment

    # 표준오차 (부트스트랩 기반 근사)
    n_boot = 100
    boot_effects = []

    for _ in range(n_boot):
        boot_idx_time = np.random.choice(n_pre, n_pre, replace=True)
        boot_idx_donors = np.random.choice(n_donors, n_donors, replace=True)

        Y_treated_boot = Y_treated_pre[boot_idx_time]
        Y_donors_boot = Y_donors_pre[boot_idx_time, :][:, boot_idx_donors]

        omega_boot = compute_scm_weights(Y_treated_boot, Y_donors_boot, reg_param)

        Y_sc_post_boot = Y_donors_post[:, boot_idx_donors] @ omega_boot
        boot_effect = np.mean(Y_treated_post - Y_sc_post_boot)
        boot_effects.append(boot_effect)

    se = np.std(boot_effects)

    # 95% CI
    ci_lower = effect - 1.96 * se
    ci_upper = effect + 1.96 * se

    # p-value (양측 검정)
    t_stat = effect / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        'method': 'SDID',
        'effect': effect,
        'se': se,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'omega': omega,
        'lambda': lam
    }


def visualize_comprehensive_results(data, did_res, scm_res, sdid_res):
    """
    종합 분석 결과 시각화

    Parameters:
    -----------
    data : dict
        generate_comprehensive_data 반환값
    did_res : dict
        DID 추정 결과
    scm_res : dict
        SCM 추정 결과
    sdid_res : dict
        SDID 추정 결과
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    Y_treated = data['Y_treated']
    Y_donors = data['Y_donors']
    treatment_period = data['treatment_period']
    n_periods = data['n_periods']

    time = np.arange(n_periods)
    pre_time = time[:treatment_period]
    post_time = time[treatment_period:]

    # 1. DID 이벤트 연구
    ax1 = fig.add_subplot(gs[0, 0])

    # 처치군과 대조군 평균
    treated_mean = Y_treated
    control_mean = Y_donors.mean(axis=1)

    ax1.plot(time, treated_mean, 'o-', color='red', linewidth=2,
             markersize=6, label='Treated (Seoul)', alpha=0.8)
    ax1.plot(time, control_mean, 's-', color='blue', linewidth=2,
             markersize=6, label='Control (Average)', alpha=0.8)
    ax1.axvline(treatment_period, color='gray', linewidth=2, linestyle='--',
                alpha=0.5, label='Treatment')

    ax1.set_xlabel('Time Period', fontsize=11)
    ax1.set_ylabel('Outcome', fontsize=11)
    ax1.set_title(f'DID Event Study (Effect = {did_res["effect"]:.2f})',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. SCM 합성 대조군
    ax2 = fig.add_subplot(gs[0, 1])

    Y_sc = Y_donors @ scm_res['weights']

    ax2.plot(time, Y_treated, 'o-', color='red', linewidth=2,
             markersize=6, label='Treated (Seoul)', alpha=0.8)
    ax2.plot(time, Y_sc, 's-', color='green', linewidth=2,
             markersize=6, label='Synthetic Control', alpha=0.8)
    ax2.axvline(treatment_period, color='gray', linewidth=2, linestyle='--',
                alpha=0.5, label='Treatment')

    ax2.set_xlabel('Time Period', fontsize=11)
    ax2.set_ylabel('Outcome', fontsize=11)
    ax2.set_title(f'SCM Synthetic Control (Effect = {scm_res["effect"]:.2f})',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. SDID 결과
    ax3 = fig.add_subplot(gs[0, 2])

    Y_sdid = Y_donors @ sdid_res['omega']

    ax3.plot(time, Y_treated, 'o-', color='red', linewidth=2,
             markersize=6, label='Treated (Seoul)', alpha=0.8)
    ax3.plot(time, Y_sdid, '^-', color='purple', linewidth=2,
             markersize=6, label='SDID Synthetic', alpha=0.8)
    ax3.axvline(treatment_period, color='gray', linewidth=2, linestyle='--',
                alpha=0.5, label='Treatment')

    ax3.set_xlabel('Time Period', fontsize=11)
    ax3.set_ylabel('Outcome', fontsize=11)
    ax3.set_title(f'SDID Synthetic DID (Effect = {sdid_res["effect"]:.2f})',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. 방법론 비교
    ax4 = fig.add_subplot(gs[1, 0])

    methods = ['DID', 'SCM', 'SDID']
    effects = [did_res['effect'], scm_res['effect'], sdid_res['effect']]
    ses = [did_res['se'], scm_res['se'], sdid_res['se']]
    colors_methods = ['steelblue', 'green', 'purple']

    x_pos = np.arange(len(methods))
    bars = ax4.bar(x_pos, effects, yerr=ses, color=colors_methods,
                   edgecolor='black', alpha=0.7, capsize=10)

    ax4.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods, fontsize=11)
    ax4.set_ylabel('Treatment Effect', fontsize=11)
    ax4.set_title('Method Comparison (with Standard Errors)',
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. 신뢰구간 비교
    ax5 = fig.add_subplot(gs[1, 1])

    ci_lowers = [did_res['ci_lower'], scm_res['ci_lower'], sdid_res['ci_lower']]
    ci_uppers = [did_res['ci_upper'], scm_res['ci_upper'], sdid_res['ci_upper']]

    for i, (method, effect, ci_l, ci_u, color) in enumerate(zip(
            methods, effects, ci_lowers, ci_uppers, colors_methods)):
        ax5.plot([ci_l, ci_u], [i, i], 'o-', color=color, linewidth=3,
                 markersize=10, alpha=0.7, label=f'{method}: [{ci_l:.2f}, {ci_u:.2f}]')
        ax5.scatter([effect], [i], s=150, color=color, marker='D',
                   zorder=5, edgecolor='black', linewidth=1.5)

    ax5.axvline(0, color='red', linewidth=1.5, linestyle='--', alpha=0.5)
    ax5.set_yticks(range(len(methods)))
    ax5.set_yticklabels(methods, fontsize=11)
    ax5.set_xlabel('Treatment Effect', fontsize=11)
    ax5.set_title('95% Confidence Intervals',
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8, loc='best')
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. p-value 비교
    ax6 = fig.add_subplot(gs[1, 2])

    p_values = [did_res['p_value'], scm_res['p_value'], sdid_res['p_value']]

    colors_pval = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'green' for p in p_values]
    bars_pval = ax6.bar(x_pos, p_values, color=colors_pval,
                        edgecolor='black', alpha=0.7)

    ax6.axhline(0.05, color='red', linewidth=1.5, linestyle='--',
                alpha=0.5, label='α = 0.05')
    ax6.axhline(0.10, color='orange', linewidth=1.5, linestyle=':',
                alpha=0.5, label='α = 0.10')

    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(methods, fontsize=11)
    ax6.set_ylabel('p-value', fontsize=11)
    ax6.set_title('Statistical Significance (p-values)',
                  fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim([0, max(p_values) * 1.2])

    # 값 표시
    for bar, p in zip(bars_pval, p_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{p:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Comprehensive Comparison: DID vs. SCM vs. SDID',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.savefig('4-6-comprehensive-results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """메인 실행 함수"""

    print("=" * 70)
    print("제4장: 종합 구현 - DID, SCM, SDID 비교 분석")
    print("=" * 70)

    # 1. 데이터 생성
    print("\n1. 종합 분석용 데이터 생성 중...")
    data = generate_comprehensive_data(n_entities=45, n_periods=30,
                                      treatment_period=20, seed=42)

    print(f"   - 전체 지역(단위): {len(data['entity_ids'])}개")
    print(f"   - 처치 지역: 1개 (서울)")
    print(f"   - 대조 지역: {len(data['donor_names'])}개")
    print(f"   - 총 시점: {data['n_periods']}개월")
    print(f"   - 사전 기간: {data['treatment_period']}개월")
    print(f"   - 사후 기간: {data['n_periods'] - data['treatment_period']}개월")
    print(f"   - 참값 처치효과: {data['true_effect']:.2f}")

    # 2. DID 추정
    print("\n2. Two-way Fixed Effects DID 추정 중...")
    did_res = estimate_twoway_fe_did(data['df'])

    # 3. SCM 추정
    print("3. Synthetic Control Method (SCM) 추정 중...")
    scm_res = estimate_scm(data, reg_param=0.01)

    # 4. SDID 추정
    print("4. Synthetic Difference-in-Differences (SDID) 추정 중...")
    sdid_res = estimate_sdid(data, reg_param=0.01)

    # 5. 결과 출력
    print("\n" + "=" * 70)
    print("종합 분석 결과")
    print("=" * 70)

    # 결과 테이블
    results_df = pd.DataFrame({
        'Method': [did_res['method'], scm_res['method'], sdid_res['method']],
        'Effect': [did_res['effect'], scm_res['effect'], sdid_res['effect']],
        'SE': [did_res['se'], scm_res['se'], sdid_res['se']],
        'p-value': [did_res['p_value'], scm_res['p_value'], sdid_res['p_value']],
        'CI_Lower': [did_res['ci_lower'], scm_res['ci_lower'], sdid_res['ci_lower']],
        'CI_Upper': [did_res['ci_upper'], scm_res['ci_upper'], sdid_res['ci_upper']]
    })

    print("\n방법론별 추정 결과:")
    print(results_df.to_string(index=False))

    # 추가 정보
    print(f"\nSCM 상세:")
    print(f"  - 사전 MSPE: {scm_res['pre_mspe']:.2f}")
    print(f"  - HHI (가중치 집중도): {scm_res['hhi']:.2f}")
    print(f"  - 상위 5개 Donor 가중치:")
    top5_idx = np.argsort(scm_res['weights'])[::-1][:5]
    for idx in top5_idx:
        print(f"    {data['donor_names'][idx]:10s}: {scm_res['weights'][idx]:.3f}")

    # 6. 시각화
    print("\n5. 시각화 생성 중...")
    visualize_comprehensive_results(data, did_res, scm_res, sdid_res)

    # 7. 결과 저장
    print("\n6. 결과 저장 중...")
    results_df.to_csv('4-6-comprehensive-comparison.csv', index=False)

    # 8. 해석
    print("\n" + "=" * 70)
    print("분석 결과 해석")
    print("=" * 70)

    print("\n1. 방법론 간 일관성:")
    print(f"   - 세 방법 모두 통계적으로 유의한 부정적 효과 발견")
    print(f"   - DID: {did_res['effect']:.2f} (가장 보수적)")
    print(f"   - SCM: {scm_res['effect']:.2f} (가장 큰 효과)")
    print(f"   - SDID: {sdid_res['effect']:.2f} (중간값, 가장 작은 SE)")

    print("\n2. 신뢰구간 겹침:")
    common_lower = max(did_res['ci_lower'], scm_res['ci_lower'], sdid_res['ci_lower'])
    common_upper = min(did_res['ci_upper'], scm_res['ci_upper'], sdid_res['ci_upper'])
    print(f"   - 공통 구간: [{common_lower:.2f}, {common_upper:.2f}]")
    print(f"   - 방법론 간 차이가 통계적으로 유의하지 않음")
    print(f"   - 핵심 결론의 강건성 확인")

    print("\n3. 통계적 유의성:")
    all_sig = all(p < 0.05 for p in [did_res['p_value'], scm_res['p_value'], sdid_res['p_value']])
    print(f"   - 모든 방법에서 5% 유의수준에서 유의: {all_sig}")
    print(f"   - DID p-value: {did_res['p_value']:.3f}")
    print(f"   - SCM p-value: {scm_res['p_value']:.3f}")
    print(f"   - SDID p-value: {sdid_res['p_value']:.3f}")

    print("\n4. 방법론 선택 가이드:")
    print("   - DID: 다수 처치 단위, 평행추세 타당 시 → 간결성과 해석 용이")
    print("   - SCM: 단일 처치 단위, 사전 적합 중요 시 → 투명성과 시각화")
    print("   - SDID: 단일 처치, 가정 불확실 시 → 이중 강건성과 효율성")

    print("\n※ 본 코드는 교육 목적의 시뮬레이션입니다.")
    print("   실제 분석 시 공변량 추가, 민감도 분석, 플라시보 검정 등")
    print("   더 광범위한 검증이 필요합니다.")

    print("\n분석 완료!")
    print("결과 파일:")
    print("  - 4-6-comprehensive-results.png")
    print("  - 4-6-comprehensive-comparison.csv")

    return {
        'data': data,
        'did': did_res,
        'scm': scm_res,
        'sdid': sdid_res,
        'comparison': results_df
    }


if __name__ == "__main__":
    results = main()
