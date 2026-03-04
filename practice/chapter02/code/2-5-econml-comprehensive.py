"""
제2장: EconML 패키지 활용 - 한국 기초연금 정책 효과 분석
LinearDML, CausalForestDML, DRLearner를 활용한 종합 분석

실제 연구 활용을 위한 코드:
- 실제 EconML 라이브러리 사용
- 신뢰구간 및 통계적 추론 제공
- 이질적 처치효과(CATE) 분석
- 정책 targeting 효율성 평가
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# EconML 라이브러리 임포트
from econml.dml import LinearDML, CausalForestDML
from econml.dr import DRLearner

# 시각화 설정 (한글 폰트 깨짐 방지)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def visualize_econml_results(X_df, treatment, satisfaction, true_cate, true_ate,
                             dml_effect, cf_effect, dr_effect, cf_ci_lower, cf_ci_upper):
    """
    EconML 분석 결과 시각화

    Parameters:
    -----------
    X_df : DataFrame
        공변량 데이터프레임
    treatment : array-like
        처치 지시자
    satisfaction : array-like
        결과 변수
    true_cate : array-like
        참값 CATE
    true_ate : float
        참값 ATE
    dml_effect : array-like
        DML 추정 효과
    cf_effect : array-like
        Causal Forest 추정 효과
    dr_effect : array-like
        DR 추정 효과
    cf_ci_lower, cf_ci_upper : array-like
        Causal Forest 95% 신뢰구간
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. 추정 방법 비교
    methods = ['True\nATE', 'LinearDML', 'CausalForest', 'DRLearner', 'Naive']
    estimates = [
        true_ate,
        np.mean(dml_effect),
        np.mean(cf_effect),
        np.mean(dr_effect),
        satisfaction[treatment==1].mean() - satisfaction[treatment==0].mean()
    ]
    colors = ['green', 'blue', 'orange', 'purple', 'red']

    bars = axes[0, 0].bar(methods, estimates, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=true_ate, color='green', linestyle='--',
                       linewidth=2, label='True ATE')
    axes[0, 0].set_ylabel('Average Treatment Effect')
    axes[0, 0].set_title('Method Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    for bar, est in zip(bars, estimates):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{est:.2f}', ha='center', va='bottom', fontweight='bold')

    # 2. 소득 분위별 CATE (신뢰구간 포함)
    income_quintiles = pd.qcut(X_df['income'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    cate_by_income = pd.DataFrame({
        'quintile': income_quintiles,
        'cf_cate': cf_effect,
        'cf_lower': cf_ci_lower,
        'cf_upper': cf_ci_upper,
        'true_cate': true_cate
    }).groupby('quintile').agg({
        'cf_cate': 'mean',
        'cf_lower': 'mean',
        'cf_upper': 'mean',
        'true_cate': 'mean'
    })

    x_pos = np.arange(5)
    width = 0.35

    axes[0, 1].bar(x_pos - width/2, cate_by_income['true_cate'], width,
                   label='True CATE', alpha=0.7, color='green')
    axes[0, 1].bar(x_pos + width/2, cate_by_income['cf_cate'], width,
                   label='CausalForest', alpha=0.7, color='orange')
    # 신뢰구간 표시
    axes[0, 1].errorbar(x_pos + width/2, cate_by_income['cf_cate'],
                        yerr=[cate_by_income['cf_cate'] - cate_by_income['cf_lower'],
                              cate_by_income['cf_upper'] - cate_by_income['cf_cate']],
                        fmt='none', color='black', capsize=3)

    axes[0, 1].set_xlabel('Income Quintile')
    axes[0, 1].set_ylabel('CATE')
    axes[0, 1].set_title('Heterogeneous Effects by Income (with 95% CI)')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(cate_by_income.index)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. CATE 분포
    axes[0, 2].hist(true_cate, bins=30, alpha=0.6, label='True CATE', density=True)
    axes[0, 2].hist(cf_effect, bins=30, alpha=0.6, label='CF CATE', density=True)
    axes[0, 2].axvline(x=true_ate, color='red', linestyle='--',
                       linewidth=2, label=f'True ATE: {true_ate:.2f}')
    axes[0, 2].set_xlabel('CATE')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('CATE Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # 4. CATE vs 소득 (신뢰구간 포함)
    # 소득 구간별 평균으로 시각화
    income_bins = pd.cut(X_df['income'], bins=20)
    scatter_df = pd.DataFrame({
        'income_bin': income_bins,
        'income': X_df['income'],
        'cf_cate': cf_effect,
        'cf_lower': cf_ci_lower,
        'cf_upper': cf_ci_upper,
        'true_cate': true_cate
    }).groupby('income_bin').agg({
        'income': 'mean',
        'cf_cate': 'mean',
        'cf_lower': 'mean',
        'cf_upper': 'mean',
        'true_cate': 'mean'
    }).dropna()

    axes[1, 0].plot(scatter_df['income'], scatter_df['true_cate'], 'g-',
                    linewidth=2, label='True CATE')
    axes[1, 0].plot(scatter_df['income'], scatter_df['cf_cate'], 'o-',
                    color='orange', linewidth=1.5, markersize=4, label='CausalForest')
    axes[1, 0].fill_between(scatter_df['income'], scatter_df['cf_lower'],
                            scatter_df['cf_upper'], alpha=0.2, color='orange')
    axes[1, 0].set_xlabel('Income (10,000 KRW)')
    axes[1, 0].set_ylabel('CATE')
    axes[1, 0].set_title('CATE vs Income (with 95% CI)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Targeting 효율성
    targeting_strategies = ['All', 'Top 50%', 'Top 30%', 'Q1-Q2', 'Q1 only']

    # All
    eff_all = true_ate

    # Top 50%
    top_50_pct = np.percentile(cf_effect, 50)
    eff_top50 = true_cate[cf_effect >= top_50_pct].mean()

    # Top 30%
    top_30_pct = np.percentile(cf_effect, 70)
    eff_top30 = true_cate[cf_effect >= top_30_pct].mean()

    # Q1-Q2
    q12_mask = income_quintiles.isin(['Q1', 'Q2'])
    eff_q12 = true_cate[q12_mask].mean()

    # Q1 only
    q1_mask = income_quintiles == 'Q1'
    eff_q1 = true_cate[q1_mask].mean()

    efficiencies = [eff_all, eff_top50, eff_top30, eff_q12, eff_q1]
    efficiency_indices = [e / eff_all for e in efficiencies]

    bars = axes[1, 1].bar(targeting_strategies, efficiency_indices, alpha=0.7)
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--',
                       linewidth=2, label='Baseline')
    axes[1, 1].set_ylabel('Efficiency Index')
    axes[1, 1].set_title('Targeting Strategy Efficiency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)

    for bar, idx in zip(bars, efficiency_indices):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{idx:.2f}', ha='center', va='bottom', fontweight='bold')

    # 6. 결과 요약표
    summary_text = "EconML Analysis Summary\n"
    summary_text += "=" * 50 + "\n\n"
    summary_text += f"{'Method':<15} {'ATE':>8} {'Bias':>8}\n"
    summary_text += "-" * 35 + "\n"
    summary_text += f"{'LinearDML':<15} {np.mean(dml_effect):8.2f} {np.mean(dml_effect) - true_ate:8.2f}\n"
    summary_text += f"{'CausalForest':<15} {np.mean(cf_effect):8.2f} {np.mean(cf_effect) - true_ate:8.2f}\n"
    summary_text += f"{'DRLearner':<15} {np.mean(dr_effect):8.2f} {np.mean(dr_effect) - true_ate:8.2f}\n"
    summary_text += f"{'True ATE':<15} {true_ate:8.2f} {'0.00':>8}\n\n"

    summary_text += "Income Quintile CATE:\n"
    summary_text += "-" * 35 + "\n"
    for q in cate_by_income.index:
        summary_text += f"{q:<5} True:{cate_by_income.loc[q, 'true_cate']:6.2f}  "
        summary_text += f"CF:{cate_by_income.loc[q, 'cf_cate']:6.2f}\n"

    axes[1, 2].text(0.05, 0.5, summary_text, fontsize=9,
                    verticalalignment='center', family='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')

    plt.suptitle('EconML Comprehensive Analysis: Korean Basic Pension',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('econml_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """메인 실행 함수"""

    print("=" * 70)
    print("제2장: EconML 패키지 활용 - 한국 기초연금 정책 효과 분석")
    print("=" * 70)

    # 데이터 로드
    print("\n1. 데이터 로드 중...")
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../data/2-5-econml.csv')
    df = pd.read_csv(data_path)

    X_df = df[['age', 'income', 'health', 'family_size', 'region']]
    X = X_df.values
    treatment = df['treatment'].values
    satisfaction = df['satisfaction'].values
    true_cate = df['CATE_true'].values
    true_ate = true_cate.mean()

    print(f"   - 샘플 수: {len(df)}")
    print(f"   - 수급자 수: {treatment.sum()}명 ({treatment.mean()*100:.1f}%)")
    print(f"   - 참값 ATE: {true_ate:.2f}")
    print(f"   - 참값 CATE 범위: [{true_cate.min():.2f}, {true_cate.max():.2f}]")

    # =========================================================================
    # EconML 방법론 적용 (실제 라이브러리 사용)
    # =========================================================================
    print("\n2. EconML 방법론 적용 중...")

    # 2.1 LinearDML
    print("   - LinearDML 추정 중...")
    dml = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=123),
        model_t=GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=123),
        discrete_treatment=True,
        random_state=123
    )
    dml.fit(Y=satisfaction, T=treatment, X=X)
    dml_effect = dml.effect(X)
    ate_dml = np.mean(dml_effect)

    # LinearDML 신뢰구간
    dml_inference = dml.effect_inference(X)
    dml_ci = dml_inference.conf_int(alpha=0.05)

    # 2.2 CausalForestDML
    print("   - CausalForestDML 추정 중...")
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=123),
        model_t=GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=123),
        n_estimators=500,
        min_samples_leaf=10,
        max_depth=8,
        discrete_treatment=True,
        random_state=123
    )
    cf.fit(Y=satisfaction, T=treatment, X=X)
    cf_effect = cf.effect(X)
    ate_cf = np.mean(cf_effect)

    # CausalForest 신뢰구간
    cf_inference = cf.effect_inference(X)
    cf_ci = cf_inference.conf_int(alpha=0.05)
    cf_ci_lower = cf_ci[0].flatten()
    cf_ci_upper = cf_ci[1].flatten()

    # 2.3 DRLearner
    print("   - DRLearner 추정 중...")
    dr = DRLearner(
        model_propensity=GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=123),
        model_regression=GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=123),
        model_final=GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=123),
        random_state=123
    )
    dr.fit(Y=satisfaction, T=treatment, X=X)
    dr_effect = dr.effect(X)
    ate_dr = np.mean(dr_effect)

    # Naive 추정
    naive_est = satisfaction[treatment==1].mean() - satisfaction[treatment==0].mean()

    # OLS
    X_with_T = np.column_stack([X, treatment])
    ols_model = LinearRegression()
    ols_model.fit(X_with_T, satisfaction)
    ols_est = ols_model.coef_[-1]

    # =========================================================================
    # 결과 출력
    # =========================================================================
    print("\n" + "=" * 70)
    print("EconML 기초연금 정책 효과 분석 결과")
    print("=" * 70)

    print(f"\n{'추정 방법':<20} | {'평균 처치효과':>12} | {'참값 ATE':>10} | {'편향':>8}")
    print("-" * 65)
    print(f"{'LinearDML':<20} | {ate_dml:12.2f} | {true_ate:10.2f} | {ate_dml - true_ate:8.2f}")
    print(f"{'CausalForestDML':<20} | {ate_cf:12.2f} | {true_ate:10.2f} | {ate_cf - true_ate:8.2f}")
    print(f"{'DRLearner':<20} | {ate_dr:12.2f} | {true_ate:10.2f} | {ate_dr - true_ate:8.2f}")
    print(f"{'Naive 차이':<20} | {naive_est:12.2f} | {true_ate:10.2f} | {naive_est - true_ate:8.2f}")
    print(f"{'OLS (공변량 조정)':<20} | {ols_est:12.2f} | {true_ate:10.2f} | {ols_est - true_ate:8.2f}")

    # =========================================================================
    # 소득 분위별 CATE
    # =========================================================================
    print("\n[소득 분위별 처치효과 (CATE) 분석]")
    income_quintiles = pd.qcut(X_df['income'], q=5, labels=['Q1 (최저)', 'Q2', 'Q3', 'Q4', 'Q5 (최고)'])

    cate_by_income = pd.DataFrame({
        'quintile': income_quintiles,
        'income': X_df['income'],
        'cf_cate': cf_effect,
        'cf_lower': cf_ci_lower,
        'cf_upper': cf_ci_upper,
        'dr_cate': dr_effect,
        'true_cate': true_cate
    }).groupby('quintile').agg({
        'income': 'mean',
        'cf_cate': 'mean',
        'cf_lower': 'mean',
        'cf_upper': 'mean',
        'dr_cate': 'mean',
        'true_cate': 'mean'
    })

    print(f"\n{'소득 분위':<12} | {'평균 소득':>10} | {'CF CATE':>9} | {'95% CI':>18} | {'참값 CATE':>10}")
    print("-" * 80)
    for idx, row in cate_by_income.iterrows():
        ci_str = f"[{row['cf_lower']:.2f}, {row['cf_upper']:.2f}]"
        print(f"{idx:<12} | {row['income']:10.1f} | {row['cf_cate']:9.2f} | "
              f"{ci_str:>18} | {row['true_cate']:10.2f}")

    # =========================================================================
    # CATE 이질성 통계적 검정
    # =========================================================================
    print("\n[CATE 이질성 통계적 검정]")

    # 분위별 CATE 차이 검정
    q1_cate = cf_effect[income_quintiles == 'Q1 (최저)']
    q5_cate = cf_effect[income_quintiles == 'Q5 (최고)']

    from scipy import stats
    t_stat, p_value = stats.ttest_ind(q1_cate, q5_cate)

    print(f"   Q1 평균 CATE: {q1_cate.mean():.2f} (n={len(q1_cate)})")
    print(f"   Q5 평균 CATE: {q5_cate.mean():.2f} (n={len(q5_cate)})")
    print(f"   t-통계량: {t_stat:.3f}")
    print(f"   p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"   → Q1과 Q5의 CATE 차이가 통계적으로 유의함 (p < 0.05)")
    else:
        print(f"   → Q1과 Q5의 CATE 차이가 통계적으로 유의하지 않음")

    # =========================================================================
    # Targeting 효율성
    # =========================================================================
    print("\n[정책 targeting 효율성 분석]")

    eff_all = true_ate
    top_50_pct = np.percentile(cf_effect, 50)
    eff_top50 = true_cate[cf_effect >= top_50_pct].mean()
    top_30_pct = np.percentile(cf_effect, 70)
    eff_top30 = true_cate[cf_effect >= top_30_pct].mean()
    q12_mask = income_quintiles.isin(['Q1 (최저)', 'Q2'])
    eff_q12 = true_cate[q12_mask].mean()
    q1_mask = income_quintiles == 'Q1 (최저)'
    eff_q1 = true_cate[q1_mask].mean()

    print(f"\n{'Targeting 전략':<20} | {'수급자 비율':>12} | {'평균 효과':>10} | {'효율성 지수':>12}")
    print("-" * 70)
    print(f"{'무차별 지급 (현행)':<20} | {'100%':>12} | {eff_all:10.2f} | {'1.00':>12}")
    print(f"{'CATE 상위 50%':<20} | {'50%':>12} | {eff_top50:10.2f} | {eff_top50/eff_all:12.2f}")
    print(f"{'CATE 상위 30%':<20} | {'30%':>12} | {eff_top30:10.2f} | {eff_top30/eff_all:12.2f}")
    print(f"{'소득 Q1-Q2만':<20} | {'40%':>12} | {eff_q12:10.2f} | {eff_q12/eff_all:12.2f}")
    print(f"{'소득 Q1만':<20} | {'20%':>12} | {eff_q1:10.2f} | {eff_q1/eff_all:12.2f}")

    # =========================================================================
    # CI 포함률 검증 (Coverage)
    # =========================================================================
    print("\n[신뢰구간 포함률 검증]")
    coverage = np.mean((cf_ci_lower <= true_cate) & (true_cate <= cf_ci_upper))
    print(f"   95% CI 포함률: {coverage*100:.1f}% (기대값: 95%)")
    if coverage >= 0.90:
        print(f"   → 신뢰구간이 적절한 포함률을 보임")
    else:
        print(f"   → 주의: 신뢰구간 포함률이 기대보다 낮음")

    # 시각화
    print("\n3. 시각화 생성 중...")
    visualize_econml_results(X_df, treatment, satisfaction, true_cate, true_ate,
                             dml_effect, cf_effect, dr_effect, cf_ci_lower, cf_ci_upper)

    # =========================================================================
    # 결과 해석
    # =========================================================================
    print("\n" + "=" * 70)
    print("분석 결과 해석")
    print("=" * 70)

    print("\n1. 방법론의 일관성과 신뢰성:")
    print(f"   - LinearDML ({ate_dml:.2f}), CausalForest ({ate_cf:.2f}), DRLearner ({ate_dr:.2f})")
    print(f"     모두 참값 ATE {true_ate:.2f}에 근접")
    print(f"   - 편향: {ate_dml - true_ate:.2f} ~ {ate_cf - true_ate:.2f}")
    print(f"   - Naive 추정 ({naive_est:.2f})은 {(naive_est - true_ate)/true_ate*100:.1f}% 과대추정")

    print("\n2. 이질적 효과와 targeting 기회:")
    print(f"   - 소득 최저 분위(Q1) CATE: {cate_by_income.loc['Q1 (최저)', 'true_cate']:.2f}")
    print(f"   - 소득 최고 분위(Q5) CATE: {cate_by_income.loc['Q5 (최고)', 'true_cate']:.2f}")
    ratio = cate_by_income.loc['Q1 (최저)', 'true_cate'] / cate_by_income.loc['Q5 (최고)', 'true_cate']
    print(f"   - {ratio:.1f}배 차이")
    print(f"   - CATE 상위 30% targeting 시 효율성 {eff_top30/eff_all:.2f}배 증가")

    print("\n3. CausalForest의 이질성 포착:")
    cf_q1 = cate_by_income.loc['Q1 (최저)', 'cf_cate']
    cf_q5 = cate_by_income.loc['Q5 (최고)', 'cf_cate']
    true_q1 = cate_by_income.loc['Q1 (최저)', 'true_cate']
    true_q5 = cate_by_income.loc['Q5 (최고)', 'true_cate']
    print(f"   - CausalForest Q1 추정: {cf_q1:.2f} (참값: {true_q1:.2f})")
    print(f"   - CausalForest Q5 추정: {cf_q5:.2f} (참값: {true_q5:.2f})")
    print(f"   - 추정된 이질성 비율: {cf_q1/cf_q5:.2f}배 (참값: {true_q1/true_q5:.2f}배)")

    print("\n※ 본 코드는 교육 목적의 가상 데이터 분석입니다.")
    print("   실제 정책 분석 시 공식 통계 데이터 사용이 필수적입니다.")

    print("\n분석 완료! 결과가 econml_comprehensive_analysis.png로 저장되었습니다.")

    return {
        'X_df': X_df,
        'treatment': treatment,
        'satisfaction': satisfaction,
        'true_cate': true_cate,
        'true_ate': true_ate,
        'dml_effect': dml_effect,
        'cf_effect': cf_effect,
        'dr_effect': dr_effect,
        'cf_ci_lower': cf_ci_lower,
        'cf_ci_upper': cf_ci_upper,
        'cate_by_income': cate_by_income
    }


if __name__ == "__main__":
    results = main()
