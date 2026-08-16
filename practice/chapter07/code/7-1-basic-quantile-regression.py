"""
제7장: 분위 회귀 기초
Basic Quantile Regression

조건부 분위수 추정과 OLS 비교

수정 이력 (2026-08-17)
---------------------
1. main()이 visualize_results()를 호출하지 않아 그림 파일이 하나도 만들어지지
   않았다. 또 visualize_results() 안에서 plt.show()를 불러 파일 저장이 없었다.
   -> Agg 백엔드를 지정하고 savefig로 7-1-basic-quantile-regression.png를 저장하며,
      main() 마지막에서 호출하도록 고쳤다.
2. 결과표의 유의성 표시 '***'가 p값과 무관하게 문자열로 박혀 있었다.
   -> 각 계수의 p값을 읽어 표시를 붙이도록 고쳤다.
3. 해석 출력이 계수를 "% 증가"로 적었다. 이 데이터의 income은 로그가 아니라
   수준 변수(33.87~302.05)이므로 계수의 단위는 %가 아니다.
   -> "소득 x.xx 증가(수준 단위)"로 고쳤다.
4. 교육과 경력의 변동폭을 부호 있는 값으로 비교해 "-0.06이 3.08보다 작다"는
   식으로 출력했다.
   -> 절댓값으로 비교하도록 고쳤다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# 그림은 영문 레이블로 그린다 (PDF 변환 시 폰트 문제 회피)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


def stars(pvalue):
    """p값에 따른 유의성 표시"""
    if pvalue < 0.001:
        return '***'
    if pvalue < 0.01:
        return '**'
    if pvalue < 0.05:
        return '*'
    return ''

def estimate_quantile_regression(df, quantiles):
    """
    분위 회귀 추정

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

    for q in quantiles:
        mod = smf.quantreg('income ~ education + experience', data=df)
        res = mod.fit(q=q)
        results[q] = res

    return results

def compare_with_ols(df):
    """
    OLS와 비교

    Parameters:
    -----------
    df : pd.DataFrame
        데이터

    Returns:
    --------
    결과 객체
    """
    ols_res = smf.ols('income ~ education + experience', data=df).fit()
    return ols_res

def test_heteroskedasticity(ols_res):
    """
    이분산성 검정 (Breusch-Pagan)

    Parameters:
    -----------
    ols_res : RegressionResults
        OLS 추정 결과

    Returns:
    --------
    dict : 검정 통계량
    """
    from statsmodels.stats.diagnostic import het_breuschpagan

    bp_test = het_breuschpagan(ols_res.resid, ols_res.model.exog)

    return {
        'lm_statistic': bp_test[0],
        'lm_pvalue': bp_test[1],
        'f_statistic': bp_test[2],
        'f_pvalue': bp_test[3]
    }

def test_quantile_equality(qr_results, quantiles, param_name='education'):
    """
    분위 간 계수 동일성 검정

    Parameters:
    -----------
    qr_results : dict
        분위 회귀 결과들
    quantiles : list
        검정할 분위수 쌍
    param_name : str
        검정할 파라미터 이름

    Returns:
    --------
    dict : 검정 결과
    """
    # 간단한 Wald 검정 시뮬레이션
    # 실제로는 부트스트랩이나 공분산 행렬 사용
    q_low = quantiles[0]
    q_high = quantiles[-1]

    coef_low = qr_results[q_low].params[param_name]
    coef_high = qr_results[q_high].params[param_name]

    se_low = qr_results[q_low].bse[param_name]
    se_high = qr_results[q_high].bse[param_name]

    # Wald 통계량 근사
    diff = coef_high - coef_low
    se_diff = np.sqrt(se_low**2 + se_high**2)
    wald_stat = (diff / se_diff)**2

    # 자유도 1인 카이제곱 분포
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(wald_stat, df=1)

    return {
        'wald_statistic': wald_stat,
        'p_value': p_value,
        'coef_diff': diff
    }

def visualize_results(df, qr_results, ols_res, quantiles):
    """
    결과 시각화

    Parameters:
    -----------
    df : pd.DataFrame
        원본 데이터
    qr_results : dict
        분위 회귀 결과
    ols_res : RegressionResults
        OLS 결과
    quantiles : list
        분위수 리스트
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 교육 계수의 분위별 변화
    ax = axes[0, 0]
    education_coefs = [qr_results[q].params['education'] for q in quantiles]
    education_ses = [qr_results[q].bse['education'] for q in quantiles]

    ax.plot(quantiles, education_coefs, 'o-', color='steelblue', linewidth=2, label='Quantile Regression')
    ax.fill_between(quantiles,
                     np.array(education_coefs) - 1.96 * np.array(education_ses),
                     np.array(education_coefs) + 1.96 * np.array(education_ses),
                     alpha=0.3, color='steelblue')

    ols_coef = ols_res.params['education']
    ax.axhline(ols_coef, color='red', linestyle='--', linewidth=2, label='OLS')

    ax.set_xlabel('Quantile (tau)', fontsize=12)
    ax.set_ylabel('Education Coefficient', fontsize=12)
    ax.set_title('Education Effect Across Income Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) 경력 계수의 분위별 변화
    ax = axes[0, 1]
    experience_coefs = [qr_results[q].params['experience'] for q in quantiles]
    experience_ses = [qr_results[q].bse['experience'] for q in quantiles]

    ax.plot(quantiles, experience_coefs, 'o-', color='darkorange', linewidth=2, label='Quantile Regression')
    ax.fill_between(quantiles,
                     np.array(experience_coefs) - 1.96 * np.array(experience_ses),
                     np.array(experience_coefs) + 1.96 * np.array(experience_ses),
                     alpha=0.3, color='darkorange')

    ols_coef_exp = ols_res.params['experience']
    ax.axhline(ols_coef_exp, color='red', linestyle='--', linewidth=2, label='OLS')

    ax.set_xlabel('Quantile (tau)', fontsize=12)
    ax.set_ylabel('Experience Coefficient', fontsize=12)
    ax.set_title('Experience Effect Across Income Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (3) 조건부 분위수 함수 (교육=15년 고정)
    ax = axes[1, 0]

    edu_fixed = 15
    exp_range = np.linspace(df['experience'].min(), df['experience'].max(), 100)

    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        pred = qr_results[q].params['Intercept'] + \
               qr_results[q].params['education'] * edu_fixed + \
               qr_results[q].params['experience'] * exp_range
        ax.plot(exp_range, pred, label=f'tau={q:.2f}', linewidth=2)

    # OLS 예측
    ols_pred = ols_res.params['Intercept'] + \
               ols_res.params['education'] * edu_fixed + \
               ols_res.params['experience'] * exp_range
    ax.plot(exp_range, ols_pred, 'k--', linewidth=2, label='OLS Mean')

    ax.set_xlabel('Experience (years)', fontsize=12)
    ax.set_ylabel('Income', fontsize=12)
    ax.set_title(f'Conditional Quantile Functions (Education={edu_fixed} years)',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (4) 잔차 분포 (이분산성 시각화)
    ax = axes[1, 1]

    df_plot = df.copy()
    df_plot['ols_resid'] = ols_res.resid
    df_plot['fitted'] = ols_res.fittedvalues

    ax.scatter(df_plot['fitted'], df_plot['ols_resid'], alpha=0.3, s=10)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('Fitted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('OLS Residuals vs Fitted (Heteroskedasticity Check)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, '7-1-basic-quantile-regression.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"그래프 저장 완료: {output_path}")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제7장: 분위 회귀 기초")
    print("=" * 80)

    # 데이터 로드
    print("\n[1단계] 임금 데이터 로드")
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../data/7-1-wage-data.csv')
    df = pd.read_csv(data_path)
    print(f"샘플 크기: {len(df)}")
    print(f"소득 범위: {df['income'].min():.2f} ~ {df['income'].max():.2f}")
    print(f"교육 범위: {df['education'].min():.2f} ~ {df['education'].max():.2f} 년")
    print(f"경력 범위: {df['experience'].min():.2f} ~ {df['experience'].max():.2f} 년")

    # 분위 회귀 추정
    print("\n[2단계] 분위 회귀 추정")
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    qr_results = estimate_quantile_regression(df, quantiles)

    # OLS 추정
    print("\n[3단계] OLS 회귀 추정 (비교)")
    ols_res = compare_with_ols(df)

    # 결과 출력
    print("\n" + "=" * 80)
    print("분위 회귀 vs OLS 비교 결과")
    print("=" * 80)

    results_table = []
    for q in quantiles:
        edu_coef = qr_results[q].params['education']
        edu_se = qr_results[q].bse['education']
        exp_coef = qr_results[q].params['experience']
        exp_se = qr_results[q].bse['experience']

        results_table.append({
            'Quantile': f'{q:.2f}',
            'Edu_Coef': f'{edu_coef:.2f}{stars(qr_results[q].pvalues["education"])}',
            'Edu_SE': f'({edu_se:.2f})',
            'Exp_Coef': f'{exp_coef:.2f}{stars(qr_results[q].pvalues["experience"])}',
            'Exp_SE': f'({exp_se:.2f})'
        })

    # OLS 추가
    results_table.append({
        'Quantile': 'OLS',
        'Edu_Coef': f'{ols_res.params["education"]:.2f}{stars(ols_res.pvalues["education"])}',
        'Edu_SE': f'({ols_res.bse["education"]:.2f})',
        'Exp_Coef': f'{ols_res.params["experience"]:.2f}{stars(ols_res.pvalues["experience"])}',
        'Exp_SE': f'({ols_res.bse["experience"]:.2f})'
    })

    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))

    # 이분산성 검정
    print("\n" + "=" * 80)
    print("이분산성 진단 (Breusch-Pagan Test)")
    print("=" * 80)
    bp_test = test_heteroskedasticity(ols_res)
    print(f"LM 통계량: {bp_test['lm_statistic']:.2f}")
    print(f"p-value: {bp_test['lm_pvalue']:.4f}")
    if bp_test['lm_pvalue'] < 0.001:
        print("결론: 이분산성이 유의하게 존재함 (p < 0.001)")
        print("       OLS 표준오차의 강건 조정 필요")
        print("       분위 회귀는 자연스럽게 이분산성에 강건")

    # 분위 간 계수 동일성 검정
    print("\n" + "=" * 80)
    print("분위 간 계수 차이 검정 (Wald Test)")
    print("=" * 80)
    wald_test = test_quantile_equality(qr_results, quantiles, param_name='education')
    print(f"H₀: β₀.₁₀ = β₀.₉₀ (10분위와 90분위 교육 계수 동일)")
    print(f"Wald 통계량: {wald_test['wald_statistic']:.2f}")
    print(f"p-value: {wald_test['p_value']:.4f}")
    print(f"계수 차이: {wald_test['coef_diff']:.2f}")
    if wald_test['p_value'] < 0.001:
        print("결론: 교육 효과가 소득 분포 전반에서 유의하게 상이함")

    # 해석
    print("\n" + "=" * 80)
    print("분석 결과 해석")
    print("=" * 80)

    edu_coef_010 = qr_results[0.10].params['education']
    edu_coef_090 = qr_results[0.90].params['education']
    edu_coef_ols = ols_res.params['education']

    print(f"\n1. 교육 수익률의 이질성 (income은 수준 변수, 단위 없음):")
    print(f"   - 저소득층(τ=0.10): 교육 1년당 소득 {edu_coef_010:.2f} 증가")
    print(f"   - 고소득층(τ=0.90): 교육 1년당 소득 {edu_coef_090:.2f} 증가")
    print(f"   - OLS 평균 효과: 교육 1년당 소득 {edu_coef_ols:.2f} 증가")
    print(f"   - 해석: 고소득층에서 교육 수익률이 {edu_coef_090/edu_coef_010:.1f}배 크다")

    exp_coef_010 = qr_results[0.10].params['experience']
    exp_coef_090 = qr_results[0.90].params['experience']
    edu_spread = abs(edu_coef_090 - edu_coef_010)
    exp_spread = abs(exp_coef_090 - exp_coef_010)

    print(f"\n2. 경력의 분포적 효과:")
    print(f"   - 저소득층(τ=0.10): 경력 1년당 소득 {exp_coef_010:.2f} 증가")
    print(f"   - 고소득층(τ=0.90): 경력 1년당 소득 {exp_coef_090:.2f} 증가")
    print(f"   - 분위 간 변동폭(절댓값): 교육 {edu_spread:.2f} vs 경력 {exp_spread:.2f}")
    print(f"   - 해석: 경력의 효과는 분위에 따라 거의 변하지 않는다")

    print(f"\n3. OLS의 한계:")
    print(f"   - OLS는 평균 효과 하나만 준다 (교육: {edu_coef_ols:.2f})")
    print(f"   - τ=0.10의 {edu_coef_010:.2f}와 τ=0.90의 {edu_coef_090:.2f}를 하나로 뭉갠다")
    print(f"   - 정책 대상이 저소득층이면 τ=0.10~0.25의 추정량을 본다")

    # 시각화
    print("\n[시각화] 결과 그래프 생성 중...")
    visualize_results(df, qr_results, ols_res, quantiles)

if __name__ == "__main__":
    main()
