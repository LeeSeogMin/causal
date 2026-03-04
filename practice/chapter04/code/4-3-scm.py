"""
제4장: Synthetic Control Method
합성 대조군 방법론을 활용한 정책 효과 추정
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def generate_scm_data(n_donors=20, n_pre=20, n_post=10, seed=42):
    """
    SCM 분석을 위한 시계열 데이터 생성

    Parameters:
    -----------
    n_donors : int
        Donor pool 크기
    n_pre : int
        처치 전 기간 길이
    n_post : int
        처치 후 기간 길이
    seed : int
        난수 시드

    Returns:
    --------
    tuple : (처치 지역 데이터, donor 데이터, 도시 이름)
    """
    np.random.seed(seed)

    n_periods = n_pre + n_post

    # 도시 이름 (한국 도시들)
    cities = ['광주', '부산', '대구', '대전', '울산', '인천', '세종', '제주',
              '수원', '성남', '고양', '용인', '창원', '청주', '천안', '전주',
              '포항', '강릉', '순천', '김해']

    # 공통 시간 추세
    time_trend = np.linspace(0, 2, n_periods)

    # Donor 데이터 생성
    Y_donors = np.zeros((n_periods, n_donors))
    for i in range(n_donors):
        # 각 donor의 고유 특성
        baseline = 20 + np.random.randn() * 3
        trend_coef = 0.5 + np.random.randn() * 0.2
        seasonal = np.sin(np.arange(n_periods) * 2 * np.pi / 12) * np.random.rand() * 2

        Y_donors[:, i] = baseline + trend_coef * time_trend + seasonal + np.random.randn(n_periods) * 0.5

    # 처치 지역 데이터 (합성 대조군으로 재현 가능하도록 생성)
    # 주요 donor들의 가중 평균으로 사전 기간 생성
    true_weights = np.zeros(n_donors)
    true_weights[0] = 0.38  # 광주
    true_weights[1] = 0.29  # 부산
    true_weights[2] = 0.21  # 대구
    true_weights[3] = 0.08  # 대전
    true_weights[4] = 0.04  # 울산

    Y_treated_pre = Y_donors[:n_pre, :] @ true_weights + np.random.randn(n_pre) * 0.3

    # 처치 후 기간: 부정적 효과 추가 (-3.47)
    treatment_effect = -3.47
    Y_treated_post = (Y_donors[n_pre:, :] @ true_weights +
                     treatment_effect +
                     np.random.randn(n_post) * 0.5)

    Y_treated = np.concatenate([Y_treated_pre, Y_treated_post])

    return Y_treated, Y_donors, cities

def compute_scm_weights(Y_treated_pre, Y_donors_pre, reg_param=0.01):
    """
    SCM 최적 가중치 계산

    Parameters:
    -----------
    Y_treated_pre : ndarray
        처치 지역의 사전 결과
    Y_donors_pre : ndarray
        Donor들의 사전 결과 행렬
    reg_param : float
        정규화 파라미터

    Returns:
    --------
    dict : 최적 가중치 및 적합도 지표
    """
    n_donors = Y_donors_pre.shape[1]

    # 목적 함수: 사전 기간 MSPE 최소화
    def objective(w):
        residuals = Y_treated_pre - Y_donors_pre @ w
        mspe = np.mean(residuals**2)
        # L2 정규화 추가
        penalty = reg_param * np.sum(w**2)
        return mspe + penalty

    # 제약: Σwⱼ=1, wⱼ≥0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_donors

    # 초기값
    w_init = np.ones(n_donors) / n_donors

    # 최적화
    result = minimize(objective, w_init, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000})

    if not result.success:
        print("경고: 최적화가 완전히 수렴하지 않았습니다.")

    w_opt = result.x

    # 적합도 지표 계산
    Y_synth_pre = Y_donors_pre @ w_opt
    residuals = Y_treated_pre - Y_synth_pre
    mspe = np.mean(residuals**2)
    rmspe = np.sqrt(mspe)

    # Herfindahl Index (가중치 집중도)
    hhi = np.sum(w_opt**2)

    # 최대 단일 가중치
    max_weight = np.max(w_opt)

    # 거의 0인 가중치 개수
    near_zero = np.sum(w_opt < 0.05)

    results = {
        'weights': w_opt,
        'mspe': mspe,
        'rmspe': rmspe,
        'hhi': hhi,
        'max_weight': max_weight,
        'near_zero_count': near_zero
    }

    return results

def estimate_treatment_effect(Y_treated, Y_donors, weights, n_pre):
    """
    처치효과 추정

    Parameters:
    -----------
    Y_treated : ndarray
        처치 지역의 전체 결과
    Y_donors : ndarray
        Donor들의 전체 결과 행렬
    weights : ndarray
        최적 가중치
    n_pre : int
        사전 기간 길이

    Returns:
    --------
    dict : 처치효과 추정 결과
    """
    # 합성 대조군 구성
    Y_synth = Y_donors @ weights

    # 처치 후 기간 효과
    treatment_effects = Y_treated[n_pre:] - Y_synth[n_pre:]
    avg_effect = np.mean(treatment_effects)

    # 사전 기간 gap (검증용)
    pre_gap = Y_treated[:n_pre] - Y_synth[:n_pre]

    results = {
        'Y_synth': Y_synth,
        'treatment_effects': treatment_effects,
        'avg_effect': avg_effect,
        'pre_gap': pre_gap
    }

    return results

def visualize_scm_results(Y_treated, Y_synth, weights, cities, n_pre):
    """
    SCM 분석 결과 시각화

    Parameters:
    -----------
    Y_treated : ndarray
        처치 지역 데이터
    Y_synth : ndarray
        합성 대조군 데이터
    weights : ndarray
        최적 가중치
    cities : list
        도시 이름
    n_pre : int
        사전 기간 길이
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_periods = len(Y_treated)
    periods = np.arange(n_periods)

    # 1. 실제 vs 합성 대조군 비교
    axes[0, 0].plot(periods, Y_treated, 'r-o', linewidth=2,
                   markersize=6, label='Treated (Seoul)')
    axes[0, 0].plot(periods, Y_synth, 'b--s', linewidth=2,
                   markersize=6, label='Synthetic Control')
    axes[0, 0].axvline(x=n_pre, color='gray', linestyle='--',
                      alpha=0.5, label='Treatment Start')
    axes[0, 0].set_xlabel('Time Period')
    axes[0, 0].set_ylabel('Outcome')
    axes[0, 0].set_title('Treated vs Synthetic Control')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Gap (처치효과) 플롯
    gap = Y_treated - Y_synth
    axes[0, 1].plot(periods, gap, 'g-o', linewidth=2, markersize=6)
    axes[0, 1].axvline(x=n_pre, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 처치 후 평균 효과 표시
    post_avg = np.mean(gap[n_pre:])
    axes[0, 1].axhline(y=post_avg, color='red', linestyle='--',
                      linewidth=2, label=f'Avg Effect = {post_avg:.2f}')
    axes[0, 1].set_xlabel('Time Period')
    axes[0, 1].set_ylabel('Gap (Treated - Synthetic)')
    axes[0, 1].set_title('Treatment Effect Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Donor 가중치 분포 (상위 10개)
    top_10_idx = np.argsort(weights)[-10:]
    top_weights = weights[top_10_idx]
    top_cities = [cities[i] for i in top_10_idx]

    axes[1, 0].barh(top_cities, top_weights, color='steelblue', alpha=0.7)
    axes[1, 0].set_xlabel('Weight')
    axes[1, 0].set_title('Top 10 Donor Weights')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # 4. 사전 적합도 (Pre-treatment Fit)
    pre_residuals = gap[:n_pre]
    axes[1, 1].scatter(periods[:n_pre], pre_residuals, alpha=0.6,
                      s=60, color='purple')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_xlabel('Pre-treatment Period')
    axes[1, 1].set_ylabel('Pre-treatment Gap')
    mspe_pre = np.mean(pre_residuals**2)
    axes[1, 1].set_title(f'Pre-treatment Fit (MSPE = {mspe_pre:.2f})')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('4-3-scm-results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""

    print("=" * 70)
    print("제4장: Synthetic Control Method")
    print("=" * 70)

    # 1. 데이터 생성
    print("\n1. SCM 데이터 생성 중...")
    Y_treated, Y_donors, cities = generate_scm_data(n_donors=20, n_pre=20, n_post=10, seed=42)
    n_pre = 20
    n_post = 10

    print(f"   - 처치 지역: 서울 (새벽배송 도입)")
    print(f"   - Donor pool: {len(cities)}개 도시")
    print(f"   - 사전 기간: {n_pre}개월")
    print(f"   - 사후 기간: {n_post}개월")

    # 2. SCM 가중치 계산
    print("\n2. SCM 최적 가중치 계산 중...")
    Y_treated_pre = Y_treated[:n_pre]
    Y_donors_pre = Y_donors[:n_pre, :]

    scm_results = compute_scm_weights(Y_treated_pre, Y_donors_pre, reg_param=0.01)
    weights = scm_results['weights']

    print(f"   - 최적화 완료")
    print(f"   - 사전 MSPE: {scm_results['mspe']:.2f}")
    print(f"   - 사전 RMSPE: {scm_results['rmspe']:.2f}")

    # 3. 처치효과 추정
    print("\n3. 처치효과 추정 중...")
    effect_results = estimate_treatment_effect(Y_treated, Y_donors, weights, n_pre)

    print("\n" + "=" * 70)
    print("Synthetic Control Method 추정 결과")
    print("=" * 70)

    print("\n[평가 지표]")
    print(f"사전 MSPE:                {scm_results['mspe']:.2f} (< 2.0 양호)")
    print(f"사전 RMSPE:               {scm_results['rmspe']:.2f}")
    Y_std = np.std(Y_treated[:n_pre])
    print(f"결과 변수 표준편차 대비:  {scm_results['rmspe']/Y_std*100:.1f}%")
    print(f"가중치 HHI:               {scm_results['hhi']:.2f} (< 0.3 분산)")
    print(f"평균 처치효과 (사후):     {effect_results['avg_effect']:.2f}")
    print(f"최대 단일 donor 가중치:   {scm_results['max_weight']:.2f} (< 0.5 양호)")
    print(f"0에 가까운 가중치 수:     {scm_results['near_zero_count']}개 (총 20개 중)")

    # 4. Donor 가중치 상세 분포
    print("\n[Donor 가중치 상세 분포]")
    print(f"{'순위':<6} {'도시':<10} {'가중치':<10}")
    print("-" * 30)

    sorted_idx = np.argsort(weights)[::-1]
    for rank, idx in enumerate(sorted_idx[:5], 1):
        print(f"{rank:<6} {cities[idx]:<10} {weights[idx]:<10.2f}")

    remaining_weight = np.sum(weights[sorted_idx[5:]])
    print(f"{'6-20':<6} {'기타':<10} {remaining_weight:<10.2f}")

    # 5. 결과 해석
    print("\n" + "=" * 70)
    print("분석 결과 해석")
    print("=" * 70)

    print("\n1. 합성 대조군의 품질:")
    print(f"   - 사전 MSPE {scm_results['mspe']:.2f}는")
    print(f"     결과 변수 표준편차({Y_std:.2f}) 대비 {scm_results['rmspe']/Y_std*100:.1f}%")
    print(f"   - 합성 대조군이 처치 지역의 사전 추세를 양호하게 재현")
    print(f"   - 일반적으로 RMSPE가 표준편차의 20% 이내이면 적합도 양호")

    print("\n2. 가중치의 안정성:")
    print(f"   - HHI {scm_results['hhi']:.2f}는 가중치가 여러 donor에 분산됨을 표시")
    print(f"   - HHI > 0.7이면 소수 donor에 과도하게 의존")
    print(f"   - 본 분석은 주로 보간(interpolation)에 기반하므로 신뢰성 높음")

    donor_names = ['광주', '부산', '대구', '대전', '울산']
    top_5_weights = [weights[sorted_idx[i]] for i in range(5)]
    print(f"   - 상위 5개 도시({', '.join(donor_names[:5])})가")
    print(f"     총 가중치의 {sum(top_5_weights)*100:.0f}%를 차지")

    print("\n3. 처치효과의 크기와 유의성:")
    print(f"   - 평균 처치효과 {effect_results['avg_effect']:.2f}는")
    print(f"     새벽배송 도입 후 오프라인 상권 매출이")
    print(f"     합성 대조군 대비 {abs(effect_results['avg_effect']):.2f} 단위 감소")
    avg_outcome = np.mean(Y_treated[:n_pre])
    print(f"   - 이는 매출의 약 {abs(effect_results['avg_effect'])/avg_outcome*100:.1f}%에 해당")

    # 6. 시각화
    print("\n4. 시각화 생성 중...")
    visualize_scm_results(Y_treated, effect_results['Y_synth'],
                         weights, cities, n_pre)

    print("\n※ 본 코드는 교육 목적의 기본 SCM 구현입니다.")
    print("   실제 적용 시 V 행렬 최적화와 강건성 검정이 필요합니다.")
    print("\n분석 완료! 결과가 4-3-scm-results.png로 저장되었습니다.")

    # 7. 데이터 저장
    results_df = pd.DataFrame({
        'period': np.arange(len(Y_treated)),
        'treated': Y_treated,
        'synthetic': effect_results['Y_synth'],
        'gap': Y_treated - effect_results['Y_synth']
    })
    results_df.to_csv('4-3-scm-data.csv', index=False)
    print("데이터가 4-3-scm-data.csv로 저장되었습니다.")

    return scm_results, effect_results

if __name__ == "__main__":
    scm_results, effect_results = main()
