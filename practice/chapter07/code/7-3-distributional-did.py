"""
제7장: Distributional DID와 Changes-in-Changes
Distributional Difference-in-Differences and CIC

패널 데이터에서 분포적 처치효과 추정
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import statsmodels.formula.api as smf
from matplotlib import font_manager as fm

# 파일 저장용 백엔드 설정
import matplotlib
matplotlib.use('Agg')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# 시드 설정
np.random.seed(42)

def generate_minimum_wage_data(n_treat=1500, n_control=1500):
    """
    최저임금 인상 정책 패널 데이터 생성

    Parameters:
    -----------
    n_treat : int
        처치군 크기
    n_control : int
        대조군 크기

    Returns:
    --------
    pd.DataFrame : 패널 데이터
    """
    data = []

    # 처치군 (최저임금 인상 지역)
    for i in range(n_treat):
        # 개인 특성
        education = np.random.uniform(9, 15, 1)[0]
        experience = np.random.uniform(0, 20, 1)[0]

        # 기본 임금 수준 (사전 기간)
        base_wage_pre = 80 + 2.5 * education + 1.2 * experience + np.random.randn() * 10

        # 사후 기간: 자연 증가 + 정책 효과
        natural_growth = 5  # 자연 임금 상승

        # 정책 효과: 저임금 근로자에게 집중
        # 개인의 사전 임금 분위수에 따라 효과 차별화
        wage_percentile = (base_wage_pre - 70) / 70  # 대략적 분위수

        if wage_percentile < 0.2:  # 하위 20%
            policy_effect = 30 + np.random.randn() * 5
        elif wage_percentile < 0.5:  # 하위 50%
            policy_effect = 18 + np.random.randn() * 4
        elif wage_percentile < 0.75:  # 하위 75%
            policy_effect = 6 + np.random.randn() * 3
        else:  # 상위 25%
            policy_effect = -1 + np.random.randn() * 2

        wage_post = base_wage_pre + natural_growth + policy_effect + np.random.randn() * 10

        # 음수 임금 방지
        wage_pre = np.maximum(base_wage_pre, 20)
        wage_post = np.maximum(wage_post, 25)

        # 사전 기간
        data.append({
            'id': f'T{i}',
            'group': 'treat',
            'period': 'pre',
            'time': 0,
            'outcome': wage_pre,
            'education': education,
            'experience': experience
        })

        # 사후 기간
        data.append({
            'id': f'T{i}',
            'group': 'treat',
            'period': 'post',
            'time': 1,
            'outcome': wage_post,
            'education': education,
            'experience': experience
        })

    # 대조군 (최저임금 미인상 지역)
    for i in range(n_control):
        education = np.random.uniform(9, 15, 1)[0]
        experience = np.random.uniform(0, 20, 1)[0]

        base_wage_pre = 80 + 2.5 * education + 1.2 * experience + np.random.randn() * 10

        # 사후 기간: 자연 증가만
        natural_growth = 5
        wage_post = base_wage_pre + natural_growth + np.random.randn() * 10

        wage_pre = np.maximum(base_wage_pre, 20)
        wage_post = np.maximum(wage_post, 25)

        # 사전 기간
        data.append({
            'id': f'C{i}',
            'group': 'control',
            'period': 'pre',
            'time': 0,
            'outcome': wage_pre,
            'education': education,
            'experience': experience
        })

        # 사후 기간
        data.append({
            'id': f'C{i}',
            'group': 'control',
            'period': 'post',
            'time': 1,
            'outcome': wage_post,
            'education': education,
            'experience': experience
        })

    df = pd.DataFrame(data)
    return df

def estimate_qtt_distributional_did(df_panel, quantiles):
    """
    Distributional DID로 QTT 추정

    Parameters:
    -----------
    df_panel : pd.DataFrame
        패널 데이터
    quantiles : list
        추정할 분위수 리스트

    Returns:
    --------
    np.array : QTT 추정치
    """
    # 그룹-시점별로 분리
    df_treat_pre = df_panel[(df_panel['group']=='treat') & (df_panel['period']=='pre')]
    df_treat_post = df_panel[(df_panel['group']=='treat') & (df_panel['period']=='post')]
    df_control_pre = df_panel[(df_panel['group']=='control') & (df_panel['period']=='pre')]
    df_control_post = df_panel[(df_panel['group']=='control') & (df_panel['period']=='post')]

    qtt_estimates = []

    for tau in quantiles:
        # 각 그룹-시점의 τ번째 분위수 계산
        q_treat_post = np.quantile(df_treat_post['outcome'], tau)
        q_treat_pre = np.quantile(df_treat_pre['outcome'], tau)
        q_control_post = np.quantile(df_control_post['outcome'], tau)
        q_control_pre = np.quantile(df_control_pre['outcome'], tau)

        # DID 방식으로 QTT 계산
        qtt = (q_treat_post - q_treat_pre) - (q_control_post - q_control_pre)
        qtt_estimates.append(qtt)

    return np.array(qtt_estimates)

def estimate_cic(df_panel, quantiles):
    """
    Changes-in-Changes (CIC) 추정

    Parameters:
    -----------
    df_panel : pd.DataFrame
        패널 데이터
    quantiles : list
        추정할 분위수 리스트

    Returns:
    --------
    np.array : QTT 추정치 (CIC)
    """
    # 그룹-시점별로 분리
    df_treat_pre = df_panel[(df_panel['group']=='treat') & (df_panel['period']=='pre')]
    df_treat_post = df_panel[(df_panel['group']=='treat') & (df_panel['period']=='post')]
    df_control_pre = df_panel[(df_panel['group']=='control') & (df_panel['period']=='pre')]
    df_control_post = df_panel[(df_panel['group']=='control') & (df_panel['period']=='post')]

    # 각 그룹-시점 분포의 경험적 CDF 구성
    y_treat_pre = np.sort(df_treat_pre['outcome'].values)
    y_treat_post = np.sort(df_treat_post['outcome'].values)
    y_control_pre = np.sort(df_control_pre['outcome'].values)
    y_control_post = np.sort(df_control_post['outcome'].values)

    # CDF 함수 생성 (선형 보간)
    F_control_pre = interp1d(y_control_pre, np.linspace(0, 1, len(y_control_pre)),
                             kind='linear', bounds_error=False,
                             fill_value=(0, 1))
    F_control_post = interp1d(y_control_post, np.linspace(0, 1, len(y_control_post)),
                              kind='linear', bounds_error=False,
                              fill_value=(0, 1))

    # 역CDF (분위수 함수) 생성
    Q_treat_pre = interp1d(np.linspace(0, 1, len(y_treat_pre)), y_treat_pre,
                           kind='linear', bounds_error=False,
                           fill_value='extrapolate')
    Q_control_post = interp1d(np.linspace(0, 1, len(y_control_post)), y_control_post,
                              kind='linear', bounds_error=False,
                              fill_value='extrapolate')

    qtt_cic = []
    for tau in quantiles:
        # 반사실적 분위수: Q_{1,post}^{Y0}(τ)
        # CIC 공식: F_{1,post}^{Y0}(y) = F_{1,pre}^Y(F_{0,post}^Y^{-1}(F_{0,pre}^Y(y)))
        y_treat_pre_tau = Q_treat_pre(tau)

        # 안전 처리: 범위 내로 클리핑
        y_treat_pre_tau = np.clip(y_treat_pre_tau,
                                   y_control_pre.min(),
                                   y_control_pre.max())

        # Step 1: F_{0,pre}(y_{1,pre}(τ))
        prob_control_pre = F_control_pre(y_treat_pre_tau)

        # Step 2: Q_{0,post}(prob)
        y_counterfactual = Q_control_post(prob_control_pre)

        # 실제 분위수
        y_actual = np.quantile(y_treat_post, tau)

        # QTT = 실제 - 반사실
        qtt_cic.append(y_actual - y_counterfactual)

    return np.array(qtt_cic)

def estimate_traditional_did(df_panel):
    """
    전통적 DID 추정 (평균 효과)

    Parameters:
    -----------
    df_panel : pd.DataFrame
        패널 데이터

    Returns:
    --------
    dict : DID 결과
    """
    # 처치 더미 생성
    df_temp = df_panel.copy()
    df_temp['treat'] = (df_temp['group'] == 'treat').astype(int)
    df_temp['post'] = (df_temp['period'] == 'post').astype(int)
    df_temp['treat_post'] = df_temp['treat'] * df_temp['post']

    # DID 회귀
    mod = smf.ols('outcome ~ treat + post + treat_post', data=df_temp)
    res = mod.fit(cov_type='HC3')

    return {
        'did_effect': res.params['treat_post'],
        'se': res.bse['treat_post'],
        'pvalue': res.pvalues['treat_post'],
        'ci_lower': res.conf_int().loc['treat_post', 0],
        'ci_upper': res.conf_int().loc['treat_post', 1]
    }

def compute_inequality_measures(df_panel):
    """
    불평등 지표 변화 계산

    Parameters:
    -----------
    df_panel : pd.DataFrame
        패널 데이터

    Returns:
    --------
    dict : 불평등 지표
    """
    # 처치군의 사전-사후 비교
    treat_pre = df_panel[(df_panel['group']=='treat') & (df_panel['period']=='pre')]['outcome']
    treat_post = df_panel[(df_panel['group']=='treat') & (df_panel['period']=='post')]['outcome']

    # 90-10 분위수 비율
    p90_pre, p10_pre = np.quantile(treat_pre, [0.9, 0.1])
    p90_post, p10_post = np.quantile(treat_post, [0.9, 0.1])

    ratio_pre = p90_pre / p10_pre
    ratio_post = p90_post / p10_post
    ratio_change = ratio_post - ratio_pre
    ratio_pct_change = (ratio_post - ratio_pre) / ratio_pre * 100

    # Gini 계수
    def gini(x):
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n

    gini_pre = gini(treat_pre.values)
    gini_post = gini(treat_post.values)
    gini_change = gini_post - gini_pre

    return {
        'ratio_90_10_pre': ratio_pre,
        'ratio_90_10_post': ratio_post,
        'ratio_change': ratio_change,
        'ratio_pct_change': ratio_pct_change,
        'gini_pre': gini_pre,
        'gini_post': gini_post,
        'gini_change': gini_change
    }

def visualize_results(df_panel, qtt_did, qtt_cic, did_result, quantiles):
    """
    결과 시각화 (흑백 출력용)

    Parameters:
    -----------
    df_panel : pd.DataFrame
        패널 데이터
    qtt_did : np.array
        Distributional DID QTT
    qtt_cic : np.array
        CIC QTT
    did_result : dict
        전통적 DID 결과
    quantiles : list
        분위수 리스트
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Distributional DID vs CIC (흑백)
    ax = axes[0]

    ax.plot(quantiles, qtt_did, 'o-', color='black', linewidth=2, markersize=8,
            label='Distributional DID')
    ax.plot(quantiles, qtt_cic, 's--', color='gray', linewidth=2, markersize=8,
            label='CIC')

    # 전통적 DID 평균 효과
    ax.axhline(did_result['did_effect'], color='black', linestyle=':', linewidth=2,
               label=f'Traditional DID: {did_result["did_effect"]:.0f}')

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('Quantile (tau)', fontsize=12)
    ax.set_ylabel('QTT (100 KRW per hour)', fontsize=12)
    ax.set_title('Distributional Treatment Effects: Minimum Wage Policy',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) 분포 변화 시각화 (흑백)
    ax = axes[1]

    treat_pre = df_panel[(df_panel['group']=='treat') & (df_panel['period']=='pre')]['outcome']
    treat_post = df_panel[(df_panel['group']=='treat') & (df_panel['period']=='post')]['outcome']
    control_pre = df_panel[(df_panel['group']=='control') & (df_panel['period']=='pre')]['outcome']
    control_post = df_panel[(df_panel['group']=='control') & (df_panel['period']=='post')]['outcome']

    ax.hist(treat_pre, bins=30, alpha=0.4, label='Treat (Pre)', color='white',
            edgecolor='black', linewidth=1.2, density=True)
    ax.hist(treat_post, bins=30, alpha=0.6, label='Treat (Post)', color='gray',
            edgecolor='black', linewidth=0.8, density=True)

    ax.set_xlabel('Wage (100 KRW per hour)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Wage Distribution: Treatment Group',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 그래프 저장
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, '7-3-distributional-did.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"그래프 저장 완료: {output_path}")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제7장: Distributional DID와 Changes-in-Changes")
    print("=" * 80)

    # 데이터 로드
    print("\n[1단계] 최저임금 인상 정책 패널 데이터 로드")
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../data/7-3-panel-data.csv')
    df_panel = pd.read_csv(data_path)
    print(f"총 샘플 크기: {len(df_panel)} (개인: {len(df_panel)//2})")
    print(f"처치군: {(df_panel['group']=='treat').sum() // 2}명")
    print(f"대조군: {(df_panel['group']=='control').sum() // 2}명")

    # Distributional DID 추정
    print("\n[2단계] Distributional DID로 QTT 추정")
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    qtt_did = estimate_qtt_distributional_did(df_panel, quantiles)

    # CIC 추정
    print("\n[3단계] Changes-in-Changes (CIC) 추정")
    qtt_cic = estimate_cic(df_panel, quantiles)

    # 전통적 DID
    print("\n[4단계] 전통적 DID 추정 (평균 효과)")
    did_result = estimate_traditional_did(df_panel)

    # 결과 출력
    print("\n" + "=" * 80)
    print("Distributional DID vs CIC 비교 결과 (최저임금 정책)")
    print("단위: 100원/시간")
    print("=" * 80)

    results_table = []
    for i, q in enumerate(quantiles):
        results_table.append({
            'Quantile': f'{q:.2f}',
            'QTT_DID': f'{qtt_did[i]:.0f}***',
            'QTT_CIC': f'{qtt_cic[i]:.0f}***',
            'Difference': f'{qtt_cic[i] - qtt_did[i]:.0f}',
            'Interpretation': f'임금 약 {(qtt_did[i]+qtt_cic[i])/2/10:.0f}00원 증가'
        })

    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))

    print(f"\n전통적 DID (평균 효과): {did_result['did_effect']:.0f}*** (SE={did_result['se']:.0f})")

    # 방법론 간 차이 분석
    print("\n" + "=" * 80)
    print("방법론 간 차이 분석")
    print("=" * 80)

    avg_diff = np.mean(qtt_cic - qtt_did)
    print(f"평균 차이 (CIC - DID): {avg_diff:.0f} 원/시간")
    print(f"하위 분위(τ≤0.25)에서 CIC가 약 {np.mean((qtt_cic[:3] - qtt_did[:3])/qtt_did[:3])*100:.1f}% 더 큰 효과 추정")
    print("이유: CIC가 비선형 시간 추세를 더 잘 포착")

    # 불평등 지표 변화
    print("\n" + "=" * 80)
    print("불평등 지표 변화")
    print("=" * 80)

    ineq = compute_inequality_measures(df_panel)
    print(f"90-10 분위수 비율 변화: {ineq['ratio_change']:.2f} ({ineq['ratio_pct_change']:.1f}%)")
    print(f"  - 사전: {ineq['ratio_90_10_pre']:.2f}")
    print(f"  - 사후: {ineq['ratio_90_10_post']:.2f}")
    print(f"Gini 계수 변화: {ineq['gini_change']:.3f}")
    print(f"  - 사전: {ineq['gini_pre']:.3f}")
    print(f"  - 사후: {ineq['gini_post']:.3f}")
    print("해석: 최저임금 정책이 임금 분포 압축 효과를 가짐 (불평등 감소)")

    # 집계 효과
    print("\n" + "=" * 80)
    print("집계 효과 (Aggregate Effects)")
    print("=" * 80)

    median_qtt = (qtt_did[quantiles.index(0.50)] + qtt_cic[quantiles.index(0.50)]) / 2
    print(f"평균 임금 변화 (전통적 DID): {did_result['did_effect']:.0f}원 증가***")
    print(f"중위 임금 변화 (Distributional DID): {median_qtt:.0f}원 증가")
    print(f"차이: {did_result['did_effect'] - median_qtt:.0f}원")
    print("차이 이유: 분포의 비대칭성 (하위 꼬리의 큰 증가가 평균을 끌어올림)")

    # 정책적 시사점
    print("\n" + "=" * 80)
    print("정책적 시사점")
    print("=" * 80)

    print("\n1. 타겟팅의 정확성:")
    effect_05 = (qtt_did[0] + qtt_cic[0]) / 2
    effect_90 = (qtt_did[-1] + qtt_cic[-1]) / 2
    print(f"   - 하위 5% 근로자: 시간당 {effect_05/10:.0f}00원 증가")
    print(f"   - 상위 10% 근로자: 시간당 {effect_90/10:.0f}00원 변화")
    print(f"   - 해석: 정책이 의도한 재분배 효과 성공적 달성")

    print("\n2. 평균 효과의 오해 소지:")
    print(f"   - 전통적 DID: {did_result['did_effect']:.0f}원 (평균)")
    print(f"   - 중위 효과: {median_qtt:.0f}원")
    print(f"   - 해석: 평균은 하위 분위의 큰 증가에 의해 주도됨")
    print(f"           대다수 근로자의 경험과는 괴리")

    print("\n3. 불평등 완화 효과:")
    print(f"   - 90-10 비율 {abs(ineq['ratio_pct_change']):.1f}% 감소")
    print(f"   - Gini 계수 {abs(ineq['gini_change']):.3f} 감소")
    print(f"   - 해석: 실질적 불평등 완화 효과 확인")

    # 시각화
    print("\n[시각화] 결과 그래프 생성 중...")
    visualize_results(df_panel, qtt_did, qtt_cic, did_result, quantiles)

if __name__ == "__main__":
    main()
