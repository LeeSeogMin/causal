"""
제7장: Distributional DID와 Changes-in-Changes
Distributional Difference-in-Differences and CIC

패널 데이터에서 분포적 처치효과 추정

수정 이력 (2026-08-17)
---------------------
1. 단위 환산이 틀렸다. outcome은 시간당 임금을 100원 단위로 담고 있는데,
   해석 문장은 값을 10으로 나눈 뒤 뒤에 "00원"을 붙였다. QTT=7.5(=750원)이
   "약 100원 증가"로 찍혔고, 전통적 DID 4(=400원)도 "4원"으로 찍혔다.
   -> 모든 해석 출력에서 100을 곱해 원 단위로 바꿨다.
2. 결과표의 '***'가 검정 없이 문자열로 박혀 있었다. 분위수 차이에는 닫힌 형태
   표준오차가 없다.
   -> 개인 단위 부트스트랩(B=500)으로 QTT의 표준오차와 95% 신뢰구간을 구하고,
      그 결과로 유의성 표시를 붙이도록 고쳤다.
3. 전통적 DID 회귀가 HC3만 썼다. 한 사람이 사전·사후 두 행으로 들어가므로
   개인 안에서 오차가 상관된다.
   -> id로 묶은 클러스터 강건 표준오차로 바꿨다.
4. "평균은 하위 분위의 큰 증가에 의해 주도됨", "CIC가 비선형 시간 추세를 더 잘
   포착", "정책이 의도한 재분배 효과 성공적 달성" 같은 문장이 값과 무관하게
   박혀 있었다.
   -> 값에서 계산하거나 삭제했다.
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

def bootstrap_qtt(df_panel, quantiles, n_boot=500, seed=42):
    """
    개인 단위 부트스트랩으로 QTT(DID)와 QTT(CIC)의 표준오차를 구한다.

    분위수의 차이에는 닫힌 형태 표준오차가 없다. 사람을 통째로 다시 뽑아
    (사전·사후 두 행을 함께 유지) 같은 계산을 반복하고, 추정치의 흩어짐을 잰다.

    Returns
    -------
    dict : QTT별 표준오차와 95% 신뢰구간
    """
    rng = np.random.default_rng(seed)

    treat_ids = df_panel[df_panel['group'] == 'treat']['id'].unique()
    control_ids = df_panel[df_panel['group'] == 'control']['id'].unique()
    indexed = df_panel.set_index('id')

    boot_did = np.zeros((n_boot, len(quantiles)))
    boot_cic = np.zeros((n_boot, len(quantiles)))

    for b in range(n_boot):
        pick_t = rng.choice(treat_ids, size=len(treat_ids), replace=True)
        pick_c = rng.choice(control_ids, size=len(control_ids), replace=True)
        sample = pd.concat([indexed.loc[pick_t], indexed.loc[pick_c]]).reset_index()

        boot_did[b] = estimate_qtt_distributional_did(sample, quantiles)
        boot_cic[b] = estimate_cic(sample, quantiles)

    return {
        'did_se': boot_did.std(axis=0, ddof=1),
        'did_lo': np.percentile(boot_did, 2.5, axis=0),
        'did_hi': np.percentile(boot_did, 97.5, axis=0),
        'cic_se': boot_cic.std(axis=0, ddof=1),
        'cic_lo': np.percentile(boot_cic, 2.5, axis=0),
        'cic_hi': np.percentile(boot_cic, 97.5, axis=0),
        'n_boot': n_boot
    }


def stars_from_ci(lo, hi):
    """부트스트랩 95% 신뢰구간이 0을 포함하지 않으면 * 를 붙인다."""
    return '*' if (lo > 0 or hi < 0) else ''


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

    # DID 회귀 (같은 사람이 두 행으로 들어가므로 id로 묶어 클러스터 강건 SE 사용)
    mod = smf.ols('outcome ~ treat + post + treat_post', data=df_temp)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': df_temp['id']})

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

    # 부트스트랩 표준오차
    print("\n[5단계] 개인 단위 부트스트랩으로 QTT 표준오차 계산 (B=500)")
    boot = bootstrap_qtt(df_panel, quantiles, n_boot=500)

    # 결과 출력
    print("\n" + "=" * 80)
    print("Distributional DID vs CIC 비교 결과 (최저임금 정책)")
    print("outcome 단위는 100원/시간. 아래 원 단위 열은 100을 곱한 값이다.")
    print("=" * 80)

    results_table = []
    for i, q in enumerate(quantiles):
        results_table.append({
            'Quantile': f'{q:.2f}',
            'QTT_DID': f'{qtt_did[i]:.1f}{stars_from_ci(boot["did_lo"][i], boot["did_hi"][i])}',
            'SE_DID': f'{boot["did_se"][i]:.2f}',
            'CI95_DID': f'[{boot["did_lo"][i]:.1f}, {boot["did_hi"][i]:.1f}]',
            'QTT_CIC': f'{qtt_cic[i]:.1f}{stars_from_ci(boot["cic_lo"][i], boot["cic_hi"][i])}',
            'Won_DID': f'{qtt_did[i]*100:.0f}원'
        })

    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))

    print(f"\n전통적 DID (평균 효과): {did_result['did_effect']:.2f} "
          f"(= {did_result['did_effect']*100:.0f}원/시간, 클러스터 SE={did_result['se']:.2f}, "
          f"p={did_result['pvalue']:.4f})")

    # 방법론 간 차이 분석
    print("\n" + "=" * 80)
    print("두 방법의 추정값 차이")
    print("=" * 80)

    avg_diff = np.mean(np.abs(qtt_cic - qtt_did))
    print(f"분위별 차이의 평균 절댓값: {avg_diff*100:.0f}원/시간")
    print(f"가장 큰 차이: τ={quantiles[int(np.argmax(np.abs(qtt_cic-qtt_did)))]:.2f}에서 "
          f"{np.max(np.abs(qtt_cic-qtt_did))*100:.0f}원")
    print(f"부트스트랩 SE(τ=0.05, DID): {boot['did_se'][0]*100:.0f}원")
    if np.max(np.abs(qtt_cic - qtt_did)) < np.max(boot['did_se']):
        print("가장 큰 차이도 부트스트랩 표준오차보다 작다. 두 방법의 결론은 갈리지 않는다.")
    else:
        print("차이가 부트스트랩 표준오차보다 크다. 어느 가정을 쓸지 밝혀야 한다.")

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

    # 집계 효과
    print("\n" + "=" * 80)
    print("집계 효과 (Aggregate Effects)")
    print("=" * 80)

    median_qtt = qtt_did[quantiles.index(0.50)]
    print(f"평균 임금 변화 (전통적 DID): {did_result['did_effect']*100:.0f}원 증가")
    print(f"중위 임금 변화 (Distributional DID): {median_qtt*100:.0f}원 증가")
    print(f"차이: {(did_result['did_effect'] - median_qtt)*100:.0f}원")

    # 정책적 시사점
    print("\n" + "=" * 80)
    print("정책적 시사점")
    print("=" * 80)

    print("\n1. 분위별 효과 크기:")
    print(f"   - 하위 5% 지점: 시간당 {qtt_did[0]*100:.0f}원 증가")
    print(f"   - 중위 지점: 시간당 {median_qtt*100:.0f}원 증가")
    print(f"   - 상위 10% 지점: 시간당 {qtt_did[-1]*100:.0f}원 변화")
    print(f"   - 하위와 상위의 비: {qtt_did[0]/qtt_did[-1]:.1f}배")

    print("\n2. 평균 하나만 보고했을 때 놓치는 것:")
    print(f"   - 전통적 DID(평균): {did_result['did_effect']*100:.0f}원")
    print(f"   - 분위별 범위: {qtt_did.min()*100:.0f}원 ~ {qtt_did.max()*100:.0f}원")
    print(f"   - 평균값 하나로는 이 {(qtt_did.max()-qtt_did.min())*100:.0f}원의 폭이 보이지 않는다")

    print("\n3. 불평등 지표 변화:")
    print(f"   - 90-10 비율: {ineq['ratio_90_10_pre']:.2f} -> {ineq['ratio_90_10_post']:.2f} "
          f"({ineq['ratio_pct_change']:.1f}%)")
    print(f"   - Gini 계수: {ineq['gini_pre']:.3f} -> {ineq['gini_post']:.3f} "
          f"({ineq['gini_change']:.3f})")

    # 시각화
    print("\n[시각화] 결과 그래프 생성 중...")
    visualize_results(df_panel, qtt_did, qtt_cic, did_result, quantiles)

if __name__ == "__main__":
    main()
