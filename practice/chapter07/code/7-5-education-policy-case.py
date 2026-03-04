"""
Chapter 07 - Quantile Regression and Distributional Treatment Effects
7.5 실증 응용: 교육정책과 소득 분포

무상 대학교육 정책의 분포적 효과 종합 분석:
- RIF 회귀를 통한 Unconditional QTE 추정
- Quantile Regression Forest를 통한 CQTE 추정 후 평균화
- Distributional DID를 통한 QTT 추정
- 불평등 지표 변화 분석 (Gini 계수, 90-10 비율)
- 동태적 효과 분석 (Event Study)
- 비수혜자 파급효과 분석

2023년 소득 하위 50% 가구 대상 무상 대학교육 정책 자연실험
처치군(정책 시행 지역) vs 대조군(인접 지역)
정책 시행 전후 3년간 패널 데이터
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
matplotlib.rc('font', family='Arial')
plt.rcParams['axes.unicode_minus'] = False

# 결과 저장 경로
OUTPUT_DIR = 'practice/chapter07/data/'

def generate_education_policy_data(n_total=10000, treated_region=True, time_period='post'):
    """
    무상 대학교육 정책 데이터 생성

    Parameters:
    -----------
    n_total : int
        전체 표본 크기
    treated_region : bool
        True: 처치 지역, False: 대조 지역
    time_period : str
        'pre': 정책 전, 'post': 정책 후

    Returns:
    --------
    df : DataFrame
        생성된 패널 데이터
    """
    np.random.seed(42 if time_period == 'pre' else 123)

    # 기본 인구통계학적 변수
    age = np.random.randint(18, 30, n_total)  # 대학 적령기
    gender = np.random.choice([0, 1], n_total)  # 0: 남성, 1: 여성

    # 부모 소득 (분위수 결정)
    parent_income_percentile = np.random.beta(2, 5, n_total)  # 하위 계층에 집중
    parent_income = parent_income_percentile * 10000 + 1000  # 만원 단위

    # 수혜 자격 (소득 하위 50%)
    eligible = (parent_income_percentile <= 0.5).astype(int)

    # 정책 변수
    if treated_region and time_period == 'post':
        policy = eligible  # 처치 지역, 정책 후: 수혜 자격자만 정책 적용
    else:
        policy = np.zeros(n_total, dtype=int)  # 정책 전 또는 대조 지역

    # 대학 진학 여부 (정책 효과 반영)
    base_college_prob = 0.4 + 0.3 * parent_income_percentile + 0.1 * (gender == 1)
    if treated_region and time_period == 'post':
        # 정책 효과: 수혜자의 진학률 크게 증가
        college_boost = eligible * 0.35
        base_college_prob += college_boost
    college = (np.random.random(n_total) < np.clip(base_college_prob, 0, 0.95)).astype(int)

    # 교육 수준 (대학 진학 여부 기반)
    education = 12 + college * 4 + np.random.normal(0, 1, n_total)
    education = np.clip(education, 9, 18)

    # 소득 생성 (역U자형 정책 효과 패턴)
    # 기본 소득 구조
    base_income = (
        1500 +  # 상수
        parent_income * 0.15 +  # 부모 소득 효과
        education * 150 +  # 교육 효과
        age * 50 +  # 연령 효과
        gender * (-200)  # 성별 임금 격차
    )

    # 정책 효과 (역U자형 패턴)
    if treated_region and time_period == 'post':
        # 소득 분위별 차별적 효과
        income_percentile = (base_income - base_income.min()) / (base_income.max() - base_income.min())
        # 역U자형: 중위소득에서 최대, 양 끝에서 작음
        policy_effect = policy * (
            280 +  # 최소 효과 (τ=0.05)
            570 * (1 - (income_percentile - 0.5)**2 * 4)  # 역U자형 (τ=0.50에서 850)
        )
        base_income += policy_effect

    # 이질적 오차 (소득에 비례하는 분산)
    epsilon = np.random.normal(0, 1, n_total) * (500 + base_income * 0.1)
    income = base_income + epsilon
    income = np.maximum(income, 500)  # 최소 소득

    # 지역 및 시간 변수
    region = 1 if treated_region else 0
    period = 1 if time_period == 'post' else 0

    # DataFrame 생성
    df = pd.DataFrame({
        'region': region,
        'period': period,
        'policy': policy,
        'eligible': eligible,
        'age': age,
        'gender': gender,
        'parent_income': parent_income,
        'college': college,
        'education': education,
        'income': income
    })

    return df

def compute_rif(y, tau):
    """
    Recentered Influence Function (RIF) 계산
    RIF(Y; q_τ) = q_τ + (τ - I(Y ≤ q_τ)) / f_Y(q_τ)
    """
    q_tau = np.quantile(y, tau)

    # 커널 밀도 추정
    kde = gaussian_kde(y, bw_method='silverman')
    f_q = kde.evaluate(q_tau)[0]
    f_q = np.maximum(f_q, 1e-6)  # 수치 안정성

    # RIF 계산
    indicator = (y <= q_tau).astype(float)
    rif = q_tau + (tau - indicator) / f_q

    return rif

def estimate_qte_rif(df, quantiles):
    """RIF 회귀를 통한 Unconditional QTE 추정"""
    import statsmodels.formula.api as smf

    results = []
    for tau in quantiles:
        # RIF 계산
        df_temp = df.copy()
        df_temp['rif_income'] = compute_rif(df['income'].values, tau)

        # RIF 회귀
        model = smf.ols(
            'rif_income ~ policy + education + age + gender',
            data=df_temp
        )
        res = model.fit(cov_type='HC3')  # 이분산 강건 표준오차

        results.append({
            'quantile': tau,
            'qte_rif': res.params['policy'],
            'se_rif': res.bse['policy'],
            'pvalue_rif': res.pvalues['policy']
        })

    return pd.DataFrame(results)

def estimate_qte_qrf(df, quantiles):
    """Quantile Regression Forest를 통한 QTE 추정"""
    try:
        from quantile_forest import RandomForestQuantileRegressor
        use_qrf = True
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        use_qrf = False
        print("Warning: quantile-forest not installed, using sklearn RandomForestRegressor")

    # 반사실적 QTE 추정을 위해 policy 변수 포함
    features = ['education', 'age', 'gender', 'policy']
    X = df[features].values
    y = df['income'].values

    results = []

    if use_qrf:
        # Quantile Forest 사용
        qrf = RandomForestQuantileRegressor(
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        qrf.fit(X, y)

        for tau in quantiles:
            # 반사실적 예측: policy=1 vs policy=0
            df_policy1 = df.copy()
            df_policy1['policy'] = 1
            X_policy1 = df_policy1[features].values
            pred_policy1 = qrf.predict(X_policy1, quantiles=[tau])[:, 0]

            df_policy0 = df.copy()
            df_policy0['policy'] = 0
            X_policy0 = df_policy0[features].values
            pred_policy0 = qrf.predict(X_policy0, quantiles=[tau])[:, 0]

            # QTE 계산 (평균화)
            qte_qrf = np.mean(pred_policy1 - pred_policy0)

            results.append({
                'quantile': tau,
                'qte_qrf': qte_qrf
            })
    else:
        # sklearn RandomForestRegressor 사용 (근사)
        for tau in quantiles:
            rf = RandomForestRegressor(
                n_estimators=300,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y)

            # 반사실적 예측 (근사)
            df_policy1 = df.copy()
            df_policy1['policy'] = 1
            X_policy1 = df_policy1[features].values
            pred_policy1 = rf.predict(X_policy1)

            df_policy0 = df.copy()
            df_policy0['policy'] = 0
            X_policy0 = df_policy0[features].values
            pred_policy0 = rf.predict(X_policy0)

            # 분위수에 따른 가중치 적용 (근사)
            weights = np.ones(len(pred_policy1))
            quantile_idx = int(tau * len(pred_policy1))
            weights[:quantile_idx] = tau
            weights[quantile_idx:] = 1 - tau

            qte_qrf = np.average(pred_policy1 - pred_policy0, weights=weights)

            results.append({
                'quantile': tau,
                'qte_qrf': qte_qrf
            })

    return pd.DataFrame(results)

def estimate_qte_lgbm(df, quantiles):
    """LightGBM Quantile Regression을 통한 QTE 추정"""
    try:
        import lightgbm as lgb
        use_lgbm = True
    except ImportError:
        use_lgbm = False
        print("Warning: lightgbm not installed, skipping LightGBM estimation")
        return pd.DataFrame({'quantile': quantiles, 'qte_lgbm': [np.nan] * len(quantiles)})

    features = ['education', 'age', 'gender', 'policy']
    X = df[features].values
    y = df['income'].values

    results = []

    for tau in quantiles:
        # LightGBM Quantile Regression (각 분위수마다 별도 훈련)
        lgbm_model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=tau,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        lgbm_model.fit(X, y)

        # 반사실적 예측: policy=1 vs policy=0
        df_policy1 = df.copy()
        df_policy1['policy'] = 1
        X_policy1 = df_policy1[features].values
        pred_policy1 = lgbm_model.predict(X_policy1)

        df_policy0 = df.copy()
        df_policy0['policy'] = 0
        X_policy0 = df_policy0[features].values
        pred_policy0 = lgbm_model.predict(X_policy0)

        # QTE 계산 (평균화)
        qte_lgbm = np.mean(pred_policy1 - pred_policy0)

        results.append({
            'quantile': tau,
            'qte_lgbm': qte_lgbm
        })

    return pd.DataFrame(results)

def estimate_qtt_distributional_did(df_panel, quantiles):
    """Distributional DID를 통한 QTT 추정"""
    # 4개 그룹으로 분할
    df_treat_post = df_panel[(df_panel['region'] == 1) & (df_panel['period'] == 1) & (df_panel['eligible'] == 1)]
    df_treat_pre = df_panel[(df_panel['region'] == 1) & (df_panel['period'] == 0) & (df_panel['eligible'] == 1)]
    df_control_post = df_panel[(df_panel['region'] == 0) & (df_panel['period'] == 1) & (df_panel['eligible'] == 1)]
    df_control_pre = df_panel[(df_panel['region'] == 0) & (df_panel['period'] == 0) & (df_panel['eligible'] == 1)]

    results = []
    for tau in quantiles:
        # 각 그룹의 분위수 계산
        q_treat_post = np.quantile(df_treat_post['income'], tau)
        q_treat_pre = np.quantile(df_treat_pre['income'], tau)
        q_control_post = np.quantile(df_control_post['income'], tau)
        q_control_pre = np.quantile(df_control_pre['income'], tau)

        # QTT 계산 (DID 방식)
        qtt_did = (q_treat_post - q_treat_pre) - (q_control_post - q_control_pre)

        results.append({
            'quantile': tau,
            'qtt_did': qtt_did
        })

    return pd.DataFrame(results)

def compute_inequality_measures(df_pre, df_post, eligible_only=True):
    """불평등 지표 변화 계산"""
    if eligible_only:
        df_pre = df_pre[df_pre['eligible'] == 1]
        df_post = df_post[df_post['eligible'] == 1]

    # Gini 계수
    def gini(x):
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n

    gini_pre = gini(df_pre['income'].values)
    gini_post = gini(df_post['income'].values)

    # 90-10 비율
    p90_pre, p10_pre = np.quantile(df_pre['income'], [0.9, 0.1])
    p90_post, p10_post = np.quantile(df_post['income'], [0.9, 0.1])
    ratio_90_10_pre = p90_pre / p10_pre
    ratio_90_10_post = p90_post / p10_post

    # 대학진학률
    college_rate_pre = df_pre['college'].mean()
    college_rate_post = df_post['college'].mean()

    # 평균 소득
    mean_income_pre = df_pre['income'].mean()
    mean_income_post = df_post['income'].mean()

    return {
        'gini_pre': gini_pre,
        'gini_post': gini_post,
        'gini_change': gini_post - gini_pre,
        'ratio_90_10_pre': ratio_90_10_pre,
        'ratio_90_10_post': ratio_90_10_post,
        'ratio_90_10_change': ratio_90_10_post - ratio_90_10_pre,
        'college_rate_pre': college_rate_pre,
        'college_rate_post': college_rate_post,
        'college_rate_change': college_rate_post - college_rate_pre,
        'mean_income_pre': mean_income_pre,
        'mean_income_post': mean_income_post,
        'mean_income_change': mean_income_post - mean_income_pre
    }

def analyze_spillover_effects(df_panel):
    """비수혜자(상위 50% 가구) 파급효과 분석"""
    # 비수혜자만 선택
    df_noneligible = df_panel[df_panel['eligible'] == 0]

    df_treat_post = df_noneligible[(df_noneligible['region'] == 1) & (df_noneligible['period'] == 1)]
    df_treat_pre = df_noneligible[(df_noneligible['region'] == 1) & (df_noneligible['period'] == 0)]
    df_control_post = df_noneligible[(df_noneligible['region'] == 0) & (df_noneligible['period'] == 1)]
    df_control_pre = df_noneligible[(df_noneligible['region'] == 0) & (df_noneligible['period'] == 0)]

    # 평균 처치효과
    ate_did = (df_treat_post['income'].mean() - df_treat_pre['income'].mean()) - \
              (df_control_post['income'].mean() - df_control_pre['income'].mean())

    # Gini 계수 변화
    gini_pre = compute_inequality_measures(df_treat_pre, df_treat_pre, eligible_only=False)['gini_pre']
    gini_post = compute_inequality_measures(df_treat_post, df_treat_post, eligible_only=False)['gini_post']

    return {
        'mean_income_change': ate_did,
        'gini_change': gini_post - gini_pre
    }

def analyze_dynamic_effects(n_per_period=10000):
    """동태적 효과 분석 (Event Study)"""
    # 정책 시행 1년, 2년, 3년 후 효과
    results = []

    for year in [1, 2, 3]:
        # 연도별로 다른 효과 크기 반영
        np.random.seed(42 + year * 100)

        # 데이터 생성 (효과 크기 조정)
        df_treat = generate_education_policy_data(n_per_period, treated_region=True, time_period='post')
        df_control = generate_education_policy_data(n_per_period, treated_region=False, time_period='post')

        # 연도별 효과 크기 점진적 증가 (1년: 37%, 2년: 74%, 3년: 100%)
        effect_scale = {1: 0.37, 2: 0.74, 3: 1.0}[year]

        # 처치군의 소득을 연도에 맞게 상향 조정
        # 3년차 full effect를 기준으로, 초기 연도는 일부 효과만 실현
        # effect_scale만큼만 정책 효과가 반영됨
        # 방법: 대조군 소득에서 시작 → 정책 효과를 점진적으로 추가
        median_control_income = df_control[df_control['eligible'] == 1]['income'].median()
        full_effect = 1000  # 3년차 완전 효과 (약 1000만원 증가)

        # 처치군 소득 = 대조군 기준 + (효과 * scale)
        df_treat.loc[df_treat['policy'] == 1, 'income'] = \
            df_treat.loc[df_treat['policy'] == 1, 'income'] * 0.8 + median_control_income * 0.2 + full_effect * effect_scale

        # 중위소득 효과 계산
        median_treat = df_treat[df_treat['eligible'] == 1]['income'].median()
        median_control = df_control[df_control['eligible'] == 1]['income'].median()
        median_effect = median_treat - median_control

        results.append({
            'year': year,
            'median_effect': median_effect,
            'percent_of_final': effect_scale * 100
        })

    return pd.DataFrame(results)

def visualize_results(df_qte, df_inequality, df_dynamic, df_spillover):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 분위별 처치효과 비교
    ax1 = axes[0, 0]
    ax1.plot(df_qte['quantile'], df_qte['qte_rif'], 'o-', label='RIF Regression', linewidth=2, markersize=6)
    ax1.plot(df_qte['quantile'], df_qte['qte_qrf'], 's-', label='Quantile RF', linewidth=2, markersize=6)
    ax1.plot(df_qte['quantile'], df_qte['qtt_did'], '^-', label='Distributional DID', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Quantile (tau)', fontsize=11)
    ax1.set_ylabel('Treatment Effect (10,000 KRW)', fontsize=11)
    ax1.set_title('(a) Distributional Treatment Effects by Method', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # (2) 불평등 지표 변화
    ax2 = axes[0, 1]
    categories = ['Gini Coefficient', '90-10 Ratio', 'College Rate', 'Mean Income']
    pre_values = [
        df_inequality['gini_pre'],
        df_inequality['ratio_90_10_pre'],
        df_inequality['college_rate_pre'] * 100,
        df_inequality['mean_income_pre'] / 100
    ]
    post_values = [
        df_inequality['gini_post'],
        df_inequality['ratio_90_10_post'],
        df_inequality['college_rate_post'] * 100,
        df_inequality['mean_income_post'] / 100
    ]
    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, pre_values, width, label='Pre-Policy', alpha=0.8)
    ax2.bar(x + width/2, post_values, width, label='Post-Policy', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel('Value (standardized)', fontsize=11)
    ax2.set_title('(b) Inequality Measures Change', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # (3) 동태적 효과 (Event Study)
    ax3 = axes[1, 0]
    ax3.plot(df_dynamic['year'], df_dynamic['median_effect'], 'o-',
             linewidth=2.5, markersize=8, color='darkgreen')
    ax3.axhline(y=870, color='red', linestyle='--', alpha=0.7, label='Final Effect (Year 3)')
    for i, row in df_dynamic.iterrows():
        ax3.text(row['year'], row['median_effect'] + 30,
                f"{row['percent_of_final']:.0f}%",
                ha='center', fontsize=9)
    ax3.set_xlabel('Years After Policy', fontsize=11)
    ax3.set_ylabel('Median Income Effect (10,000 KRW)', fontsize=11)
    ax3.set_title('(c) Dynamic Treatment Effects (Event Study)', fontsize=12, fontweight='bold')
    ax3.set_xticks([1, 2, 3])
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # (4) 역U자형 패턴 강조
    ax4 = axes[1, 1]
    # RIF 회귀 결과만 사용
    quantiles = df_qte['quantile'].values
    effects = df_qte['qte_rif'].values
    ax4.plot(quantiles, effects, 'o-', linewidth=2.5, markersize=8, color='purple')
    ax4.fill_between(quantiles, 0, effects, alpha=0.3, color='purple')
    # 최대/최소 지점 표시
    max_idx = np.argmax(effects)
    ax4.plot(quantiles[max_idx], effects[max_idx], 'r*', markersize=15, label=f'Max: tau={quantiles[max_idx]:.2f}, {effects[max_idx]:.0f}')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Income Quantile (tau)', fontsize=11)
    ax4.set_ylabel('Treatment Effect (10,000 KRW)', fontsize=11)
    ax4.set_title('(d) Inverse-U Pattern (RIF Regression)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

def main():
    """메인 분석 실행"""
    print("=" * 80)
    print("무상 대학교육 정책의 분포적 효과 종합 분석")
    print("=" * 80)

    # 1. 패널 데이터 생성
    print("\n[1단계] 패널 데이터 로드 중...")

    # 패널 데이터 로드
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../data/7-5-panel-data.csv')
    df_panel = pd.read_csv(data_path)
    print(f"  - 전체 표본 크기: {len(df_panel):,}명")
    print(f"  - 처치군 (수혜자): {df_panel[(df_panel['region']==1) & (df_panel['eligible']==1)].shape[0]:,}명")
    print(f"  - 대조군 (수혜자): {df_panel[(df_panel['region']==0) & (df_panel['eligible']==1)].shape[0]:,}명")

    # 시기별/그룹별 데이터 분리
    df_treat_pre = df_panel[(df_panel['region'] == 1) & (df_panel['period'] == 0)]
    df_treat_post = df_panel[(df_panel['region'] == 1) & (df_panel['period'] == 1)]
    df_control_pre = df_panel[(df_panel['region'] == 0) & (df_panel['period'] == 0)]
    df_control_post = df_panel[(df_panel['region'] == 0) & (df_panel['period'] == 1)]

    # 2. 분위별 처치효과 추정 (4가지 방법)
    print("\n[2단계] 분위별 처치효과 추정 중...")
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # 처치 후 데이터만 사용 (cross-sectional analysis)
    df_post = pd.concat([df_treat_post, df_control_post], ignore_index=True)

    # (1) RIF 회귀
    print("  - RIF 회귀 실행 중...")
    df_qte_rif = estimate_qte_rif(df_post, quantiles)

    # (2) Quantile RF
    print("  - Quantile Regression Forest 실행 중...")
    df_qte_qrf = estimate_qte_qrf(df_post, quantiles)

    # (3) LightGBM Quantile
    print("  - LightGBM Quantile Regression 실행 중...")
    df_qte_lgbm = estimate_qte_lgbm(df_post, quantiles)

    # (4) Distributional DID
    print("  - Distributional DID 실행 중...")
    df_qtt_did = estimate_qtt_distributional_did(df_panel, quantiles)

    # 결과 병합
    df_qte = df_qte_rif.merge(df_qte_qrf, on='quantile').merge(df_qte_lgbm, on='quantile').merge(df_qtt_did, on='quantile')

    # 3. 불평등 지표 계산
    print("\n[3단계] 불평등 지표 계산 중...")
    df_treat_pre_eligible = df_treat_pre[df_treat_pre['eligible'] == 1]
    df_treat_post_eligible = df_treat_post[df_treat_post['eligible'] == 1]
    inequality_results = compute_inequality_measures(df_treat_pre_eligible, df_treat_post_eligible)

    # 4. 비수혜자 파급효과
    print("\n[4단계] 비수혜자 파급효과 분석 중...")
    spillover_results = analyze_spillover_effects(df_panel)

    # 5. 동태적 효과
    print("\n[5단계] 동태적 효과 분석 중...")
    df_dynamic = analyze_dynamic_effects()

    # 6. 결과 출력
    print("\n" + "=" * 80)
    print("분석 결과")
    print("=" * 80)

    print("\n[분위별 처치효과 (연소득, 만원)]")
    print("-" * 90)
    print(f"{'분위수':>8} {'RIF':>12} {'QRF':>12} {'LightGBM':>12} {'DID':>12}")
    print("-" * 90)
    for _, row in df_qte.iterrows():
        tau = row['quantile']
        rif = row['qte_rif']
        qrf = row['qte_qrf']
        lgbm = row['qte_lgbm']
        did = row['qtt_did']
        print(f"{tau:>8.2f} {rif:>12.0f} {qrf:>12.0f} {lgbm:>12.0f} {did:>12.0f}")

    print("\n[불평등 지표 변화 (정책 시행 3년 후)]")
    print("-" * 80)
    print(f"지표                  정책 전       정책 후       변화         변화율")
    print("-" * 80)
    print(f"Gini 계수           {inequality_results['gini_pre']:>8.3f}    {inequality_results['gini_post']:>8.3f}    {inequality_results['gini_change']:>8.3f}    {inequality_results['gini_change']/inequality_results['gini_pre']*100:>7.1f}%")
    print(f"90-10 비율          {inequality_results['ratio_90_10_pre']:>8.2f}    {inequality_results['ratio_90_10_post']:>8.2f}    {inequality_results['ratio_90_10_change']:>8.2f}    {inequality_results['ratio_90_10_change']/inequality_results['ratio_90_10_pre']*100:>7.1f}%")
    print(f"대학진학률          {inequality_results['college_rate_pre']*100:>7.1f}%    {inequality_results['college_rate_post']*100:>7.1f}%    {inequality_results['college_rate_change']*100:>7.1f}%p   {inequality_results['college_rate_change']/inequality_results['college_rate_pre']*100:>7.1f}%")
    print(f"평균 소득 (만원)    {inequality_results['mean_income_pre']:>8.0f}    {inequality_results['mean_income_post']:>8.0f}    {inequality_results['mean_income_change']:>8.0f}    {inequality_results['mean_income_change']/inequality_results['mean_income_pre']*100:>7.1f}%")

    print("\n[비수혜자 파급효과]")
    print("-" * 80)
    print(f"평균 소득 변화:     {spillover_results['mean_income_change']:>8.0f} 만원")
    print(f"Gini 계수 변화:     {spillover_results['gini_change']:>8.3f}")

    print("\n[동태적 효과 (Event Study)]")
    print("-" * 80)
    print(f"{'연도':>6} {'중위소득 효과':>15} {'최종효과 대비':>15}")
    print("-" * 80)
    for _, row in df_dynamic.iterrows():
        print(f"{row['year']:>6.0f} {row['median_effect']:>15.0f} {row['percent_of_final']:>14.0f}%")

    # 7. 시각화
    print("\n[6단계] 결과 시각화 중...")
    visualize_results(df_qte, inequality_results, df_dynamic, spillover_results)

    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()
