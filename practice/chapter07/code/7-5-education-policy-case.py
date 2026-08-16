"""
Chapter 07 - Quantile Regression and Distributional Treatment Effects
7.5 실증 응용: 교육정책과 소득 분포

무상 대학교육 정책의 분포적 효과 종합 분석:
- RIF 회귀를 통한 Unconditional QTE 추정
- Quantile Regression Forest를 통한 CQTE 추정 후 평균화
- LightGBM 분위 회귀를 통한 CQTE 추정 후 평균화
- Distributional DID를 통한 QTT 추정
- 불평등 지표 변화 분석 (Gini 계수, 90-10 비율)
- 비수혜자 파급효과 분석

소득 하위 50% 가구 대상 무상 대학교육 정책 (가상 데이터)
처치군(정책 시행 지역) vs 대조군(인접 지역)
정책 시행 전후 패널 데이터

수정 이력 (2026-08-17)
---------------------
1. visualize_results()가 그림을 그리기만 하고 savefig도 show도 하지 않아
   PNG 파일이 하나도 생기지 않았다.
   -> Agg 백엔드를 지정하고 7-5-education-policy-case.png로 저장한다.
2. analyze_dynamic_effects()가 effect_scale = {1: 0.37, 2: 0.74, 3: 1.0}을
   직접 넣어 처치군 소득을 그만큼 올린 뒤, 그 결과를 "1년차 37% -> 3년차 100%"
   라는 발견처럼 출력했다. 넣은 값을 그대로 다시 읽는 순환 구조다.
   -> 함수와 출력, 해당 그림 패널을 모두 삭제했다. 그 자리에는 실제 데이터의
      처치군 사전/사후 소득 분포를 그린다.
3. analyze_spillover_effects()의 Gini 변화가 처치 지역의 사전-사후 차이만
   계산하면서 이름은 '파급효과'였다. 대조 지역 변화를 빼지 않아 시간 추세가
   섞인다.
   -> 대조 지역의 변화를 뺀 이중차분 형태로 고쳤다.
4. 불평등 지표 그림에서 Gini(0.12)와 평균 소득(5214)을 같은 축에 그려 Gini
   막대가 보이지 않았다.
   -> 정책 전을 100으로 놓은 지수로 바꿔 네 지표를 나란히 비교한다.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

matplotlib.rc('font', family='Arial')
plt.rcParams['axes.unicode_minus'] = False

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

    # Gini 계수 변화도 이중차분으로 계산한다 (대조 지역의 변화를 뺀다)
    def gini(x):
        sorted_x = np.sort(x)
        n = len(x)
        return (2 * np.sum(np.arange(1, n + 1) * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n

    gini_did = (gini(df_treat_post['income'].values) - gini(df_treat_pre['income'].values)) - \
               (gini(df_control_post['income'].values) - gini(df_control_pre['income'].values))

    return {
        'mean_income_change': ate_did,
        'gini_change': gini_did
    }

def visualize_results(df_qte, df_inequality, df_panel, df_spillover):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 분위별 처치효과 비교
    ax1 = axes[0, 0]
    ax1.plot(df_qte['quantile'], df_qte['qte_rif'], 'o-', label='RIF Regression', linewidth=2, markersize=6)
    ax1.plot(df_qte['quantile'], df_qte['qte_qrf'], 's-', label='Quantile RF', linewidth=2, markersize=6)
    ax1.plot(df_qte['quantile'], df_qte['qte_lgbm'], 'v-', label='LightGBM', linewidth=2, markersize=6)
    ax1.plot(df_qte['quantile'], df_qte['qtt_did'], '^-', label='Distributional DID', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Quantile (tau)', fontsize=11)
    ax1.set_ylabel('Treatment Effect (10,000 KRW)', fontsize=11)
    ax1.set_title('(a) Distributional Treatment Effects by Method', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # (2) 불평등 지표 변화 (정책 전 = 100 지수)
    ax2 = axes[0, 1]
    categories = ['Gini', '90-10 Ratio', 'College Rate', 'Mean Income']
    index_values = [
        df_inequality['gini_post'] / df_inequality['gini_pre'] * 100,
        df_inequality['ratio_90_10_post'] / df_inequality['ratio_90_10_pre'] * 100,
        df_inequality['college_rate_post'] / df_inequality['college_rate_pre'] * 100,
        df_inequality['mean_income_post'] / df_inequality['mean_income_pre'] * 100
    ]
    x = np.arange(len(categories))
    ax2.bar(x, index_values, 0.55, color='gray', edgecolor='black')
    ax2.axhline(100, color='black', linestyle='--', linewidth=1.5, label='Pre-Policy = 100')
    for xi, v in zip(x, index_values):
        ax2.text(xi, v + 2, f'{v:.0f}', ha='center', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel('Index (Pre-Policy = 100)', fontsize=11)
    ax2.set_title('(b) Inequality Measures, Beneficiary Group', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # (3) 수혜 집단의 소득 분포 변화 (실제 데이터)
    ax3 = axes[1, 0]
    pre = df_panel[(df_panel['region'] == 1) & (df_panel['period'] == 0) &
                   (df_panel['eligible'] == 1)]['income']
    post = df_panel[(df_panel['region'] == 1) & (df_panel['period'] == 1) &
                    (df_panel['eligible'] == 1)]['income']
    bins = np.linspace(min(pre.min(), post.min()), max(pre.quantile(0.99), post.quantile(0.99)), 40)
    ax3.hist(pre, bins=bins, alpha=1.0, label='Pre-Policy', color='white',
             edgecolor='black', linewidth=1.2, density=True)
    ax3.hist(post, bins=bins, alpha=0.55, label='Post-Policy', color='gray',
             edgecolor='black', linewidth=0.8, density=True)
    ax3.set_xlabel('Annual Income (10,000 KRW)', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('(c) Income Distribution, Beneficiary Group', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # (4) RIF 회귀 결과와 95% 신뢰구간
    ax4 = axes[1, 1]
    quantiles = df_qte['quantile'].values
    effects = df_qte['qte_rif'].values
    ses = df_qte['se_rif'].values
    ax4.plot(quantiles, effects, 'o-', linewidth=2.5, markersize=8, color='black')
    ax4.fill_between(quantiles, effects - 1.96 * ses, effects + 1.96 * ses,
                     alpha=0.3, color='gray')
    max_idx = int(np.argmax(effects))
    ax4.plot(quantiles[max_idx], effects[max_idx], 'k*', markersize=16,
             label=f'Max: tau={quantiles[max_idx]:.2f}, {effects[max_idx]:.0f}')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Income Quantile (tau)', fontsize=11)
    ax4.set_ylabel('UQPE (10,000 KRW)', fontsize=11)
    ax4.set_title('(d) RIF Regression with 95% CI', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, '7-5-education-policy-case.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"그래프 저장 완료: {output_path}")

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

    # 5. 결과 출력
    print("\n" + "=" * 80)
    print("분석 결과")
    print("=" * 80)

    print("\n[분위별 처치효과 (연소득, 만원)]")
    print("주의: RIF는 UQPE(수혜 비율 1 단위 변화에 대한 무조건부 분위수 반응),")
    print("      QRF/LightGBM은 조건부 분위수 차이의 평균, DID는 QTT다.")
    print("      같은 열에 놓았다고 같은 양을 재는 것이 아니다.")
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

    print("\n[불평등 지표 변화 (수혜 집단, 정책 전 -> 정책 후)]")
    print("-" * 80)
    print(f"지표                  정책 전       정책 후       변화         변화율")
    print("-" * 80)
    print(f"Gini 계수           {inequality_results['gini_pre']:>8.3f}    {inequality_results['gini_post']:>8.3f}    {inequality_results['gini_change']:>8.3f}    {inequality_results['gini_change']/inequality_results['gini_pre']*100:>7.1f}%")
    print(f"90-10 비율          {inequality_results['ratio_90_10_pre']:>8.2f}    {inequality_results['ratio_90_10_post']:>8.2f}    {inequality_results['ratio_90_10_change']:>8.2f}    {inequality_results['ratio_90_10_change']/inequality_results['ratio_90_10_pre']*100:>7.1f}%")
    print(f"대학진학률          {inequality_results['college_rate_pre']*100:>7.1f}%    {inequality_results['college_rate_post']*100:>7.1f}%    {inequality_results['college_rate_change']*100:>7.1f}%p   {inequality_results['college_rate_change']/inequality_results['college_rate_pre']*100:>7.1f}%")
    print(f"평균 소득 (만원)    {inequality_results['mean_income_pre']:>8.0f}    {inequality_results['mean_income_post']:>8.0f}    {inequality_results['mean_income_change']:>8.0f}    {inequality_results['mean_income_change']/inequality_results['mean_income_pre']*100:>7.1f}%")

    print("\n[비수혜자(상위 50% 가구) 파급효과, 이중차분]")
    print("-" * 80)
    print(f"평균 소득 변화:     {spillover_results['mean_income_change']:>8.0f} 만원")
    print(f"Gini 계수 변화:     {spillover_results['gini_change']:>8.3f}")

    # 6. 시각화
    print("\n[6단계] 결과 시각화 중...")
    visualize_results(df_qte, inequality_results, df_panel, spillover_results)

    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()
