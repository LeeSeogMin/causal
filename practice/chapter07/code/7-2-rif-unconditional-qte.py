"""
제7장: Unconditional QTE 추정
RIF Regression for Unconditional Quantile Treatment Effects

Firpo, Fortin, Lemieux(2009)의 RIF 회귀를 통한 무조건부 분위 처치효과 추정

수정 이력 (2026-08-17)
---------------------
1. "3. 조건부와 무조건부 효과의 차이" 절이 "조건부 QTE > 무조건부 QTE"라는
   문장을 값과 무관하게 그대로 찍었다. 실제 출력은 조건부 184 < 무조건부 511로
   반대였다.
   -> 두 값을 비교해 부등호와 해석 문장을 계산해서 찍도록 고쳤다.
2. 결과표의 Interpretation 열이 RIF 계수를 "소득 64만원 증가"로 적었다.
   RIF 회귀 계수는 훈련 참여 비율이 조금 늘 때 무조건부 τ분위수가 움직이는
   크기(UQPE)이지, τ분위 사람의 소득 증가분이 아니다.
   -> "참여율 1%p 상승 시 τ분위 소득 x.xx만원 상승"으로 고쳤다.
3. "해석: 중위소득층 수혜자가 교육 기회를 가장 효과적으로 활용" 등
   결과와 무관하게 박힌 해석 문장을 값에서 계산하도록 바꿨다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import gaussian_kde
from matplotlib import font_manager as fm

# 파일 저장용 백엔드 설정
import matplotlib
matplotlib.use('Agg')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# 시드 설정
np.random.seed(42)

def generate_training_data(n=3000):
    """
    직업훈련 프로그램 데이터 생성

    Parameters:
    -----------
    n : int
        샘플 크기

    Returns:
    --------
    pd.DataFrame : 훈련 데이터
    """
    # 공변량 생성
    education = np.random.uniform(10, 18, n)  # 교육 연수
    age = np.random.uniform(25, 55, n)  # 나이

    # 훈련 참여 확률 (교육과 나이에 의존)
    prob_training = 1 / (1 + np.exp(-(education - 14) * 0.3 - (age - 40) * 0.05))
    training = (np.random.rand(n) < prob_training).astype(int)

    # 기본 소득 (단위: 만원)
    base_income = 200 + 15 * education + 3 * age

    # 훈련 효과: 분위별로 다름 (역U자형 패턴)
    # 중위소득층에서 효과 극대, 극단에서 효과 작음
    epsilon = np.random.randn(n) * 50

    # 각 개인의 잠재적 위치 (분포상의 위치)
    latent_position = (base_income - base_income.min()) / (base_income.max() - base_income.min())

    # 역U자형 효과 함수: 중간(0.5)에서 최대
    effect_multiplier = 1 - 2 * (latent_position - 0.5)**2

    # 훈련 효과
    training_effect = training * (120 + 70 * effect_multiplier)

    # 최종 소득
    income = base_income + training_effect + epsilon

    # 음수 소득 방지
    income = np.maximum(income, 50)

    df = pd.DataFrame({
        'income': income,
        'training': training,
        'education': education,
        'age': age
    })

    return df

def compute_rif(y, tau):
    """
    Recentered Influence Function (RIF) 계산

    Parameters:
    -----------
    y : np.array
        결과 변수
    tau : float
        분위수

    Returns:
    --------
    np.array : RIF 값
    """
    # τ번째 표본 분위수 계산
    q_tau = np.quantile(y, tau)

    # 커널 밀도 추정으로 f_Y(q_τ) 계산
    # bandwidth는 Silverman's rule of thumb 사용
    kde = gaussian_kde(y, bw_method='silverman')
    f_q = kde.evaluate(q_tau)[0]

    # f_q가 너무 작으면 수치적 안정성을 위해 최소값 설정
    f_q = np.maximum(f_q, 1e-6)

    # RIF 계산
    # RIF(Y; q_τ) = q_τ + (τ - I(Y ≤ q_τ)) / f_Y(q_τ)
    indicator = (y <= q_tau).astype(float)
    rif = q_tau + (tau - indicator) / f_q

    return rif

def estimate_uqte_rif(df, quantiles):
    """
    RIF 회귀로 Unconditional QTE 추정

    Parameters:
    -----------
    df : pd.DataFrame
        데이터
    quantiles : list
        추정할 분위수 리스트

    Returns:
    --------
    dict : 분위별 추정 결과
    """
    results = {}

    for tau in quantiles:
        # RIF 계산
        df_temp = df.copy()
        df_temp['rif_income'] = compute_rif(df['income'].values, tau)

        # RIF 회귀
        mod = smf.ols('rif_income ~ training + education + age', data=df_temp)
        res = mod.fit(cov_type='HC3')  # 이분산성 강건 표준오차

        results[tau] = {
            'model': res,
            'uqte': res.params['training'],
            'se': res.bse['training'],
            'ci_lower': res.conf_int().loc['training', 0],
            'ci_upper': res.conf_int().loc['training', 1],
            'pvalue': res.pvalues['training']
        }

    return results

def estimate_conditional_qte(df, quantiles):
    """
    조건부 QTE 추정 (비교용)

    Parameters:
    -----------
    df : pd.DataFrame
        데이터
    quantiles : list
        추정할 분위수 리스트

    Returns:
    --------
    dict : 분위별 추정 결과
    """
    results = {}

    for tau in quantiles:
        mod = smf.quantreg('income ~ training + education + age', data=df)
        res = mod.fit(q=tau)

        results[tau] = {
            'model': res,
            'cqte': res.params['training'],
            'se': res.bse['training'],
            'ci_lower': res.conf_int().loc['training', 0],
            'ci_upper': res.conf_int().loc['training', 1],
            'pvalue': res.pvalues['training']
        }

    return results

def estimate_ate_ols(df):
    """
    평균 처치효과 (ATE) 추정

    Parameters:
    -----------
    df : pd.DataFrame
        데이터

    Returns:
    --------
    dict : OLS 결과
    """
    mod = smf.ols('income ~ training + education + age', data=df)
    res = mod.fit(cov_type='HC3')

    return {
        'ate': res.params['training'],
        'se': res.bse['training'],
        'ci_lower': res.conf_int().loc['training', 0],
        'ci_upper': res.conf_int().loc['training', 1],
        'pvalue': res.pvalues['training']
    }

def test_quantile_difference(uqte_results, tau1, tau2):
    """
    분위 간 효과 차이 검정

    Parameters:
    -----------
    uqte_results : dict
        UQTE 결과
    tau1, tau2 : float
        비교할 분위수

    Returns:
    --------
    dict : 검정 결과
    """
    uqte1 = uqte_results[tau1]['uqte']
    uqte2 = uqte_results[tau2]['uqte']

    se1 = uqte_results[tau1]['se']
    se2 = uqte_results[tau2]['se']

    diff = uqte1 - uqte2
    se_diff = np.sqrt(se1**2 + se2**2)

    # Wald 통계량
    wald_stat = (diff / se_diff)**2

    # p-value
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(wald_stat, df=1)

    return {
        'diff': diff,
        'se_diff': se_diff,
        'wald_statistic': wald_stat,
        'p_value': p_value
    }

def visualize_results(uqte_results, cqte_results, ate_result, quantiles):
    """
    결과 시각화 (흑백 출력용)

    Parameters:
    -----------
    uqte_results : dict
        무조건부 QTE 결과
    cqte_results : dict
        조건부 QTE 결과
    ate_result : dict
        ATE 결과
    quantiles : list
        분위수 리스트
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Unconditional QTE (흑백)
    ax = axes[0]

    uqtes = [uqte_results[q]['uqte'] for q in quantiles]
    uqte_ses = [uqte_results[q]['se'] for q in quantiles]

    ax.plot(quantiles, uqtes, 'o-', color='black', linewidth=2, markersize=8,
            label='Unconditional QTE (RIF)')
    ax.fill_between(quantiles,
                     np.array(uqtes) - 1.96 * np.array(uqte_ses),
                     np.array(uqtes) + 1.96 * np.array(uqte_ses),
                     alpha=0.3, color='gray')

    # ATE 참조선
    ax.axhline(ate_result['ate'], color='black', linestyle=':', linewidth=2,
               label=f'ATE (OLS): {ate_result["ate"]:.0f}')

    ax.set_xlabel('Quantile (tau)', fontsize=12)
    ax.set_ylabel('Training Effect (10,000 KRW)', fontsize=12)
    ax.set_title('Unconditional Quantile Treatment Effects', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Conditional vs Unconditional QTE 비교 (흑백)
    ax = axes[1]

    cqtes = [cqte_results[q]['cqte'] for q in quantiles]

    ax.plot(quantiles, uqtes, 'o-', color='black', linewidth=2, markersize=8,
            label='Unconditional QTE')
    ax.plot(quantiles, cqtes, 's--', color='gray', linewidth=2, markersize=8,
            label='Conditional QTE')

    ax.set_xlabel('Quantile (tau)', fontsize=12)
    ax.set_ylabel('Training Effect (10,000 KRW)', fontsize=12)
    ax.set_title('Conditional vs Unconditional QTE', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 그래프 저장
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, '7-2-rif-unconditional-qte.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"그래프 저장 완료: {output_path}")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제7장: RIF 회귀를 통한 Unconditional QTE 추정")
    print("=" * 80)

    # 데이터 로드
    print("\n[1단계] 직업훈련 프로그램 데이터 로드")
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../data/7-2-training-data.csv')
    df = pd.read_csv(data_path)
    print(f"샘플 크기: {len(df)}")
    print(f"훈련 참여율: {df['training'].mean():.1%}")
    print(f"소득 범위: {df['income'].min():.0f} ~ {df['income'].max():.0f} 만원")

    # Unconditional QTE 추정
    print("\n[2단계] RIF 회귀로 Unconditional QTE 추정")
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    uqte_results = estimate_uqte_rif(df, quantiles)

    # Conditional QTE 추정 (비교용)
    print("\n[3단계] 조건부 QTE 추정 (비교)")
    cqte_results = estimate_conditional_qte(df, quantiles)

    # ATE 추정
    print("\n[4단계] 평균 처치효과 (ATE) 추정")
    ate_result = estimate_ate_ols(df)

    # 결과 출력
    print("\n" + "=" * 80)
    print("RIF 회귀를 통한 Unconditional QTE 추정 결과 (단위: 만원)")
    print("=" * 80)

    results_table = []
    for q in quantiles:
        uqte = uqte_results[q]['uqte']
        se = uqte_results[q]['se']
        ci_lower = uqte_results[q]['ci_lower']
        ci_upper = uqte_results[q]['ci_upper']

        # 유의성 표시
        if uqte_results[q]['pvalue'] < 0.001:
            sig = '***'
        elif uqte_results[q]['pvalue'] < 0.01:
            sig = '**'
        elif uqte_results[q]['pvalue'] < 0.05:
            sig = '*'
        else:
            sig = ''

        results_table.append({
            'Quantile': f'{q:.2f}',
            'UQTE': f'{uqte:.0f}{sig}',
            'SE': f'{se:.0f}',
            'CI_95': f'[{ci_lower:.0f}, {ci_upper:.0f}]',
            'Interpretation': f'참여율 1%p 상승 -> {uqte/100:.2f}만원 상승'
        })

    # ATE 추가
    results_table.append({
        'Quantile': 'ATE (OLS)',
        'UQTE': f'{ate_result["ate"]:.0f}***',
        'SE': f'{ate_result["se"]:.0f}',
        'CI_95': f'[{ate_result["ci_lower"]:.0f}, {ate_result["ci_upper"]:.0f}]',
        'Interpretation': f'참여율 1%p 상승 -> 평균 {ate_result["ate"]/100:.2f}만원 상승'
    })

    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))

    # 분위 간 효과 차이 검정
    print("\n" + "=" * 80)
    print("분위 간 효과 차이 검정")
    print("=" * 80)
    diff_test = test_quantile_difference(uqte_results, 0.10, 0.90)
    print(f"H₀: UQTE₀.₁₀ = UQTE₀.₉₀")
    print(f"효과 차이: {diff_test['diff']:.0f} 만원")
    print(f"Wald 통계량: {diff_test['wald_statistic']:.2f}")
    print(f"p-value: {diff_test['p_value']:.3f}")
    if diff_test['p_value'] < 0.05:
        print("결론: 훈련 효과가 소득 분포 전반에서 유의하게 상이함")

    # 역U자형 패턴 분석
    print("\n" + "=" * 80)
    print("역U자형 효과 패턴 분석")
    print("=" * 80)

    max_effect_q = max(quantiles, key=lambda q: uqte_results[q]['uqte'])
    max_effect = uqte_results[max_effect_q]['uqte']

    min_effect_q = min(quantiles, key=lambda q: uqte_results[q]['uqte'])
    min_effect = uqte_results[min_effect_q]['uqte']

    print(f"효과 극대화 지점: τ={max_effect_q:.2f} (UQPE {max_effect:.0f})")
    print(f"효과 극소화 지점: τ={min_effect_q:.2f} (UQPE {min_effect:.0f})")
    print(f"최대/최소 비율: {max_effect/min_effect:.1f}배")

    # 조건부 vs 무조건부 비교 (τ=0.50)
    print("\n" + "=" * 80)
    print("조건부 vs 무조건부 QTE 비교 (τ=0.50)")
    print("=" * 80)

    cqte_50 = cqte_results[0.50]['cqte']
    cqte_se_50 = cqte_results[0.50]['se']
    uqte_50 = uqte_results[0.50]['uqte']
    uqte_se_50 = uqte_results[0.50]['se']

    print(f"Conditional QTE (분위 회귀): {cqte_50:.0f} (SE={cqte_se_50:.0f})")
    print(f"Unconditional QTE (RIF 회귀): {uqte_50:.0f} (SE={uqte_se_50:.0f})")
    print(f"차이 (무조건부 - 조건부): {uqte_50 - cqte_50:.0f}")
    print("두 값은 묻는 질문이 다르다. 조건부는 '같은 교육·나이인 사람 안에서'")
    print("훈련이 중위소득을 얼마나 올리는지, 무조건부는 '모집단 전체의 중위소득'이")
    print("훈련 참여율에 얼마나 반응하는지를 잰다.")

    # 정책적 함의
    print("\n" + "=" * 80)
    print("정책적 함의")
    print("=" * 80)

    print("\n1. 참여율 1%p 확대가 각 지점을 움직이는 크기:")
    print(f"   - 평균 소득: {ate_result['ate']/100:.2f}만원")
    print(f"   - 중위 소득: {uqte_results[0.50]['uqte']/100:.2f}만원")
    print(f"   - 하위 10% 지점: {uqte_results[0.10]['uqte']/100:.2f}만원")
    print(f"   - 상위 10% 지점: {uqte_results[0.90]['uqte']/100:.2f}만원")

    print("\n2. 불평등 지표가 움직이는 방향:")
    gap_reduction = uqte_results[0.10]['uqte'] - uqte_results[0.90]['uqte']
    print(f"   - 하위 10%와 상위 10%의 격차 변화: {gap_reduction/100:.2f}만원 (참여율 1%p당)")
    if abs(gap_reduction) < 2 * np.sqrt(uqte_results[0.10]['se']**2 + uqte_results[0.90]['se']**2):
        print(f"   - 이 값은 표준오차 범위 안이다. 90-10 격차는 사실상 안 움직인다")

    print("\n3. 조건부와 무조건부 효과의 차이 (τ=0.50):")
    if cqte_50 > uqte_50:
        print(f"   - 조건부 {cqte_50:.0f} > 무조건부 {uqte_50:.0f}")
    else:
        print(f"   - 조건부 {cqte_50:.0f} < 무조건부 {uqte_50:.0f}")
    print(f"   - 두 값을 같은 뜻으로 읽으면 안 된다. 정책 확대 효과를 보려면")
    print(f"     무조건부(RIF) 쪽을 본다")

    # 시각화
    print("\n[시각화] 결과 그래프 생성 중...")
    visualize_results(uqte_results, cqte_results, ate_result, quantiles)

if __name__ == "__main__":
    main()
