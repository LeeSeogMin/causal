"""
제5장: 이질적 처치효과 추정 (CATE in RDD)
T-learner, Gradient Boosting, BART를 이용한 조건부 평균처치효과 추정
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from matplotlib import font_manager as fm

# 한글 폰트 설정 (Windows - 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 랜덤 시드 설정
np.random.seed(42)

print("=" * 80)
print("RDD에서 이질적 처치효과 추정 (CATE)")
print("=" * 80)

# 1. 데이터 로드
print("\n[1단계] 데이터 로드")
print("-" * 80)

# CSV 파일에서 데이터 읽기
import os
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/5-6-cate-rdd-data.csv')
df = pd.read_csv(data_path)

cutoff = 50
base_effect = 3.6

print(f"데이터 파일: {data_path}")
print(f"총 관측치 수: {len(df)}")
print(f"평균 처치효과: {df['true_cate'].mean():.2f}")
print(f"처치효과 표준편차: {df['true_cate'].std():.2f}")
print(f"처치효과 범위: [{df['true_cate'].min():.2f}, {df['true_cate'].max():.2f}]")

# 2. 기준점 근처 데이터 추출
bandwidth = 0.75 * df['score'].std()
near_cutoff = df[np.abs(df['score'] - cutoff) <= bandwidth].copy()

print(f"\n유효 표본 크기: {len(near_cutoff)}")
print(f"처치군: {near_cutoff['treat'].sum()}")
print(f"대조군: {(1-near_cutoff['treat']).sum()}")

# 3. 평균 처치효과 (baseline)
print("\n[2단계] 평균 처치효과 추정 (baseline)")
print("-" * 80)

near_cutoff['score_centered'] = near_cutoff['score'] - cutoff
near_cutoff['treat_score'] = near_cutoff['treat'] * near_cutoff['score_centered']

baseline_model = smf.ols('outcome ~ treat + score_centered + treat_score',
                         data=near_cutoff).fit()

ate_estimate = baseline_model.params['treat']
print(f"평균 처치효과 (ATE): {ate_estimate:.2f}")
print(f"표준오차: {baseline_model.bse['treat']:.2f}")

# 4. 선형 상호작용 모형
print("\n[3단계] 선형 상호작용 모형으로 CATE 추정")
print("-" * 80)

# 주요 공변량과의 상호작용
formula_interaction = '''outcome ~ treat + score_centered + treat_score +
                         pre_outcome + education + income + age +
                         treat:pre_outcome + treat:education + treat:income'''

interaction_model = smf.ols(formula_interaction, data=near_cutoff).fit()

# CATE 예측
near_cutoff['cate_linear'] = (interaction_model.params['treat'] +
                              interaction_model.params['treat:pre_outcome'] * near_cutoff['pre_outcome'] +
                              interaction_model.params['treat:education'] * near_cutoff['education'] +
                              interaction_model.params['treat:income'] * near_cutoff['income'])

print(f"평균 CATE: {near_cutoff['cate_linear'].mean():.2f}")
print(f"CATE 범위: [{near_cutoff['cate_linear'].min():.2f}, {near_cutoff['cate_linear'].max():.2f}]")

print(f"\n상호작용 계수:")
print(f"  처치 × 사전성과: {interaction_model.params['treat:pre_outcome']:.3f} (p={interaction_model.pvalues['treat:pre_outcome']:.3f})")
print(f"  처치 × 교육: {interaction_model.params['treat:education']:.3f} (p={interaction_model.pvalues['treat:education']:.3f})")
print(f"  처치 × 소득: {interaction_model.params['treat:income']:.3f} (p={interaction_model.pvalues['treat:income']:.3f})")

# 5. T-learner로 CATE 추정
print("\n[4단계] T-learner (Gradient Boosting)로 CATE 추정")
print("-" * 80)

covariates = ['score', 'pre_outcome', 'education', 'income', 'age', 'experience']

# 처치군과 대조군 분리
treated = near_cutoff[near_cutoff['treat'] == 1]
control = near_cutoff[near_cutoff['treat'] == 0]

# 처치군 모델
model_1 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                    learning_rate=0.1, random_state=42)
model_1.fit(treated[covariates], treated['outcome'])

# 대조군 모델
model_0 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                    learning_rate=0.1, random_state=42)
model_0.fit(control[covariates], control['outcome'])

# CATE 예측: τ(x) = E[Y|X, D=1] - E[Y|X, D=0]
near_cutoff['cate_tlearner'] = (model_1.predict(near_cutoff[covariates]) -
                                model_0.predict(near_cutoff[covariates]))

print(f"평균 CATE (T-learner): {near_cutoff['cate_tlearner'].mean():.2f}")
print(f"CATE 범위: [{near_cutoff['cate_tlearner'].min():.2f}, {near_cutoff['cate_tlearner'].max():.2f}]")

# 교차 검증 RMSE
if len(treated) > 10 and len(control) > 10:
    cv_scores_1 = cross_val_score(model_1, treated[covariates], treated['outcome'],
                                   cv=min(5, len(treated)//10), scoring='neg_mean_squared_error')
    cv_scores_0 = cross_val_score(model_0, control[covariates], control['outcome'],
                                   cv=min(5, len(control)//10), scoring='neg_mean_squared_error')
    rmse_1 = np.sqrt(-cv_scores_1.mean())
    rmse_0 = np.sqrt(-cv_scores_0.mean())
    print(f"처치군 모델 CV-RMSE: {rmse_1:.2f}")
    print(f"대조군 모델 CV-RMSE: {rmse_0:.2f}")

# 6. Causal Forest 대체 (Random Forest로 근사)
print("\n[5단계] Causal Forest 근사 (Random Forest)")
print("-" * 80)

# T-learner with Random Forest
rf_1 = RandomForestRegressor(n_estimators=200, max_depth=5,
                             min_samples_leaf=10, random_state=42)
rf_1.fit(treated[covariates], treated['outcome'])

rf_0 = RandomForestRegressor(n_estimators=200, max_depth=5,
                             min_samples_leaf=10, random_state=42)
rf_0.fit(control[covariates], control['outcome'])

near_cutoff['cate_rf'] = (rf_1.predict(near_cutoff[covariates]) -
                          rf_0.predict(near_cutoff[covariates]))

print(f"평균 CATE (Random Forest): {near_cutoff['cate_rf'].mean():.2f}")
print(f"CATE 범위: [{near_cutoff['cate_rf'].min():.2f}, {near_cutoff['cate_rf'].max():.2f}]")

# 7. 하위 집단 분석
print("\n[6단계] 하위 집단별 CATE 분석")
print("-" * 80)

# 사전 성과 기준으로 3분위
near_cutoff['pre_outcome_tercile'] = pd.qcut(near_cutoff['pre_outcome'], 3,
                                              labels=['하위 33%', '중위 34%', '상위 33%'])

subgroup_analysis = near_cutoff.groupby('pre_outcome_tercile').agg({
    'cate_tlearner': ['mean', 'std', 'min', 'max'],
    'true_cate': 'mean',
    'score': 'count'
}).round(2)

subgroup_analysis.columns = ['_'.join(col).strip() for col in subgroup_analysis.columns]
subgroup_analysis = subgroup_analysis.rename(columns={
    'cate_tlearner_mean': 'CATE_mean',
    'cate_tlearner_std': 'CATE_std',
    'cate_tlearner_min': 'CATE_min',
    'cate_tlearner_max': 'CATE_max',
    'true_cate_mean': 'True_CATE',
    'score_count': 'N'
})

print("\n사전 성과 분위별 CATE:")
print(subgroup_analysis)

# 8. 정책 대상화 시뮬레이션
print("\n[7단계] 정책 대상화 시뮬레이션")
print("-" * 80)

# CATE 기반 대상화
near_cutoff_sorted = near_cutoff.sort_values('cate_tlearner', ascending=False)

targeting_results = []

for target_pct in [1.0, 0.75, 0.5, 0.25]:
    n_target = int(len(near_cutoff_sorted) * target_pct)
    targeted = near_cutoff_sorted.head(n_target)

    avg_effect = targeted['cate_tlearner'].mean()
    total_effect = targeted['cate_tlearner'].sum()
    efficiency = avg_effect / near_cutoff['cate_tlearner'].mean()

    targeting_results.append({
        'target_pct': f'{target_pct*100:.0f}%',
        'n_target': n_target,
        'avg_effect': avg_effect,
        'total_effect': total_effect,
        'efficiency': efficiency
    })

targeting_df = pd.DataFrame(targeting_results)
print("\n정책 대상화 결과:")
print(targeting_df.to_string(index=False))

# 9. 방법론 비교
print("\n[8단계] CATE 추정 방법론 비교")
print("-" * 80)

# RMSE 계산 (진정한 CATE와 비교)
near_cutoff_valid = near_cutoff.dropna(subset=['cate_linear', 'cate_tlearner', 'cate_rf'])

if len(near_cutoff_valid) > 0:
    rmse_linear = np.sqrt(mean_squared_error(near_cutoff_valid['true_cate'],
                                             near_cutoff_valid['cate_linear']))
    rmse_tlearner = np.sqrt(mean_squared_error(near_cutoff_valid['true_cate'],
                                               near_cutoff_valid['cate_tlearner']))
    rmse_rf = np.sqrt(mean_squared_error(near_cutoff_valid['true_cate'],
                                         near_cutoff_valid['cate_rf']))

    method_comparison = pd.DataFrame({
        'Method': ['Linear Interaction', 'T-learner (GB)', 'Causal Forest (RF)'],
        'Mean_CATE': [
            near_cutoff_valid['cate_linear'].mean(),
            near_cutoff_valid['cate_tlearner'].mean(),
            near_cutoff_valid['cate_rf'].mean()
        ],
        'RMSE': [rmse_linear, rmse_tlearner, rmse_rf],
        'Range': [
            f"[{near_cutoff_valid['cate_linear'].min():.2f}, {near_cutoff_valid['cate_linear'].max():.2f}]",
            f"[{near_cutoff_valid['cate_tlearner'].min():.2f}, {near_cutoff_valid['cate_tlearner'].max():.2f}]",
            f"[{near_cutoff_valid['cate_rf'].min():.2f}, {near_cutoff_valid['cate_rf'].max():.2f}]"
        ]
    })

    print("\n방법론별 CATE 추정 비교:")
    print(method_comparison.to_string(index=False))

# 10. 시각화
print("\n[9단계] 시각화 생성")
print("-" * 80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# (a) CATE 분포 (T-learner)
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(near_cutoff['cate_tlearner'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(near_cutoff['cate_tlearner'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'평균 CATE={near_cutoff["cate_tlearner"].mean():.2f}')
ax1.axvline(ate_estimate, color='green', linestyle='--', linewidth=2,
           label=f'ATE={ate_estimate:.2f}')
ax1.set_xlabel('CATE (조건부 평균처치효과)', fontsize=11)
ax1.set_ylabel('빈도', fontsize=11)
ax1.set_title('(a) CATE 분포 (T-learner)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3, axis='y')

# (b) 사전 성과 vs CATE
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(near_cutoff['pre_outcome'], near_cutoff['cate_tlearner'],
           alpha=0.5, s=30, color='coral')
z = np.polyfit(near_cutoff['pre_outcome'], near_cutoff['cate_tlearner'], 1)
p = np.poly1d(z)
x_trend = np.linspace(near_cutoff['pre_outcome'].min(), near_cutoff['pre_outcome'].max(), 100)
ax2.plot(x_trend, p(x_trend), 'r-', linewidth=2, label='추세선')
ax2.set_xlabel('사전 성과 (표준화)', fontsize=11)
ax2.set_ylabel('CATE', fontsize=11)
ax2.set_title('(b) 사전 성과 vs CATE', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# (c) 하위 집단별 CATE
ax3 = fig.add_subplot(gs[1, 0])
terciles = subgroup_analysis.index
means = subgroup_analysis['CATE_mean']
ax3.bar(range(len(terciles)), means, alpha=0.7, color=['steelblue', 'coral', 'green'])
ax3.set_xticks(range(len(terciles)))
ax3.set_xticklabels(terciles, rotation=15, ha='right')
ax3.set_ylabel('평균 CATE', fontsize=11)
ax3.set_title('(c) 사전 성과 분위별 평균 CATE', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

# (d) 정책 대상화 효율성
ax4 = fig.add_subplot(gs[1, 1])
target_pcts = [100, 75, 50, 25]
efficiencies = targeting_df['efficiency'].values
ax4.plot(target_pcts, efficiencies, 'o-', color='darkgreen', markersize=10, linewidth=2)
ax4.axhline(1, color='red', linestyle='--', linewidth=1, label='무작위 (기준)')
ax4.set_xlabel('대상화 비율 (%)', fontsize=11)
ax4.set_ylabel('효율성 (평균 효과 비율)', fontsize=11)
ax4.set_title('(d) CATE 기반 정책 대상화 효율성', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# (e) 방법론별 RMSE
if len(near_cutoff_valid) > 0:
    ax5 = fig.add_subplot(gs[1, 2])
    methods = method_comparison['Method']
    rmses = method_comparison['RMSE']
    x_pos = range(len(methods))
    ax5.bar(x_pos, rmses, alpha=0.7, color=['purple', 'orange', 'teal'])
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(['Linear', 'T-learner', 'RF'], rotation=15, ha='right')
    ax5.set_ylabel('RMSE', fontsize=11)
    ax5.set_title('(e) CATE 추정 방법론별 RMSE', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')

# (f) 진정한 CATE vs 추정 CATE (T-learner)
ax6 = fig.add_subplot(gs[2, 0])
ax6.scatter(near_cutoff_valid['true_cate'], near_cutoff_valid['cate_tlearner'],
           alpha=0.5, s=30, color='steelblue')
min_val = min(near_cutoff_valid['true_cate'].min(), near_cutoff_valid['cate_tlearner'].min())
max_val = max(near_cutoff_valid['true_cate'].max(), near_cutoff_valid['cate_tlearner'].max())
ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='완벽한 예측')
ax6.set_xlabel('진정한 CATE', fontsize=11)
ax6.set_ylabel('추정 CATE (T-learner)', fontsize=11)
ax6.set_title('(f) 진정한 CATE vs 추정값', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)

# (g) 교육 연수별 CATE
ax7 = fig.add_subplot(gs[2, 1])
education_bins = pd.cut(near_cutoff['education'], bins=5)
cate_by_edu = near_cutoff.groupby(education_bins)['cate_tlearner'].mean()
ax7.plot(range(len(cate_by_edu)), cate_by_edu.values, 'o-', color='coral',
        markersize=8, linewidth=2)
ax7.set_xticks(range(len(cate_by_edu)))
ax7.set_xticklabels([f'{int(i.left)}-{int(i.right)}' for i in cate_by_edu.index],
                   rotation=15, ha='right', fontsize=9)
ax7.set_xlabel('교육 연수 구간', fontsize=11)
ax7.set_ylabel('평균 CATE', fontsize=11)
ax7.set_title('(g) 교육 연수별 평균 CATE', fontsize=12, fontweight='bold')
ax7.grid(alpha=0.3)

# (h) 소득별 CATE
ax8 = fig.add_subplot(gs[2, 2])
income_quartiles = pd.qcut(near_cutoff['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
cate_by_income = near_cutoff.groupby(income_quartiles)['cate_tlearner'].mean()
ax8.bar(range(len(cate_by_income)), cate_by_income.values, alpha=0.7, color='green')
ax8.set_xticks(range(len(cate_by_income)))
ax8.set_xticklabels(cate_by_income.index)
ax8.set_xlabel('소득 분위', fontsize=11)
ax8.set_ylabel('평균 CATE', fontsize=11)
ax8.set_title('(h) 소득 분위별 평균 CATE', fontsize=12, fontweight='bold')
ax8.grid(alpha=0.3, axis='y')

plt.savefig('5-6-cate-rdd.png', dpi=300, bbox_inches='tight')
print("그래프 저장 완료: practice/chapter05/5-6-cate-rdd.png")

# 11. 최종 요약
print("\n" + "=" * 80)
print("CATE-RDD 분석 요약")
print("=" * 80)
print(f"✓ 평균 처치효과 (ATE): {ate_estimate:.2f}")
print(f"✓ CATE 범위: [{near_cutoff['cate_tlearner'].min():.2f}, {near_cutoff['cate_tlearner'].max():.2f}]")
print(f"✓ 사전 성과 상위 33% CATE: {subgroup_analysis.loc['상위 33%', 'CATE_mean']:.2f}")
print(f"✓ 사전 성과 하위 33% CATE: {subgroup_analysis.loc['하위 33%', 'CATE_mean']:.2f}")
print(f"✓ 상위 25% 대상화시 효율성: {targeting_df.iloc[-1]['efficiency']:.2f}x")
if len(near_cutoff_valid) > 0:
    print(f"✓ 최소 RMSE 방법: {method_comparison.loc[method_comparison['RMSE'].idxmin(), 'Method']} (RMSE={method_comparison['RMSE'].min():.2f})")
print(f"✓ 이질성 정도: CATE 표준편차 = {near_cutoff['cate_tlearner'].std():.2f}")
print("=" * 80)
