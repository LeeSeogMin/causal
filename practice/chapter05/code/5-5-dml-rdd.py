"""
제5장: Double Machine Learning for RDD
고차원 공변량 조정과 유연한 함수 형태를 위한 DML-RDD

본문 이론:
- DML은 고차원 공변량을 유연하게 조정하여 분산 감소
- Neyman 직교화로 정규화 편향 제거
- 비선형 관계 포착으로 선형 조정 대비 우수한 성능
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import font_manager as fm

# 한글 폰트 설정 (플랫폼 독립적)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 랜덤 시드 설정
np.random.seed(42)

print("=" * 80)
print("Double Machine Learning for RDD")
print("=" * 80)

# 1. 데이터 로드
print("\n[1단계] 데이터 로드")
print("-" * 80)

# CSV 파일에서 데이터 읽기
import os
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/5-5-dml-rdd-data.csv')
df = pd.read_csv(data_path)

cutoff = 50
TRUE_EFFECT = 3.5  # 진정한 처치효과

# 공변량 이름 추출
cov_names = [col for col in df.columns if col not in ['score', 'treat', 'outcome']]
n_covariates = len(cov_names)

print(f"데이터 파일: {data_path}")
print(f"총 관측치 수: {len(df)}")
print(f"공변량 개수: {n_covariates}")
print(f"처치군 수: {df['treat'].sum()}")
print(f"진정한 처치효과: {TRUE_EFFECT}")

# 2. 전통적 RDD (공변량 조정 없음)
print("\n[2단계] 전통적 RDD (공변량 조정 없음)")
print("-" * 80)

bandwidth = 10  # 고정 대역폭
near_cutoff = df[np.abs(df['score'] - cutoff) <= bandwidth].copy()
near_cutoff['score_centered'] = near_cutoff['score'] - cutoff
near_cutoff['treat_score'] = near_cutoff['treat'] * near_cutoff['score_centered']

traditional_model = smf.ols('outcome ~ treat + score_centered + treat_score',
                            data=near_cutoff).fit()

trad_estimate = traditional_model.params['treat']
trad_se = traditional_model.bse['treat']
trad_bias = trad_estimate - TRUE_EFFECT
trad_mse = trad_bias**2 + trad_se**2

print(f"유효 표본 크기: {len(near_cutoff)}")
print(f"처치효과 추정: {trad_estimate:.3f} (진정한 값: {TRUE_EFFECT})")
print(f"표준오차: {trad_se:.3f}")
print(f"편향: {trad_bias:.3f}")
print(f"MSE: {trad_mse:.3f}")

# 3. 선형 공변량 조정 RDD
print("\n[3단계] 선형 공변량 조정 RDD (상위 5개 공변량)")
print("-" * 80)

# 상위 5개 중요 공변량만 포함
top_covs = cov_names[:5]
formula_linear = 'outcome ~ treat + score_centered + treat_score + ' + ' + '.join(top_covs)

linear_adj_model = smf.ols(formula_linear, data=near_cutoff).fit()

linear_estimate = linear_adj_model.params['treat']
linear_se = linear_adj_model.bse['treat']
linear_bias = linear_estimate - TRUE_EFFECT
linear_mse = linear_bias**2 + linear_se**2

print(f"처치효과 추정: {linear_estimate:.3f}")
print(f"표준오차: {linear_se:.3f}")
print(f"편향: {linear_bias:.3f}")
print(f"MSE: {linear_mse:.3f}")
print(f"SE 감소율: {(1 - linear_se/trad_se)*100:.1f}%")

# 4. DML-RDD (Random Forest)
print("\n[4단계] DML-RDD (Random Forest)")
print("-" * 80)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
theta_list_rf = []
r2_outcome_list = []
r2_treat_list = []

covariates_cols = cov_names

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(near_cutoff)):
    train_data = near_cutoff.iloc[train_idx]
    test_data = near_cutoff.iloc[test_idx]

    # 1단계: 결과 모델 학습 (μ(X)) - 전체 훈련 데이터 사용
    rf_outcome = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=5, 
        random_state=42, n_jobs=-1
    )
    rf_outcome.fit(train_data[covariates_cols], train_data['outcome'])

    # 교차 검증 R²
    y_pred_outcome = rf_outcome.predict(train_data[covariates_cols])
    r2_outcome = r2_score(train_data['outcome'], y_pred_outcome)
    r2_outcome_list.append(r2_outcome)

    # 2단계: 처치 모델 학습 (π(X))
    rf_treat = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=5, 
        random_state=42, n_jobs=-1
    )
    rf_treat.fit(train_data[covariates_cols], train_data['treat'])

    y_pred_treat = rf_treat.predict(train_data[covariates_cols])
    r2_treat = r2_score(train_data['treat'], y_pred_treat)
    r2_treat_list.append(r2_treat)

    # 3단계: 테스트 셋에서 잔차 계산 (직교화)
    y_res = test_data['outcome'].values - rf_outcome.predict(test_data[covariates_cols])
    d_res = test_data['treat'].values - rf_treat.predict(test_data[covariates_cols])

    # 4단계: 직교화된 모멘트로 theta 추정
    if np.sum(d_res ** 2) > 1e-10:
        theta = np.sum(y_res * d_res) / np.sum(d_res ** 2)
        theta_list_rf.append(theta)

dml_rf_estimate = np.mean(theta_list_rf)
dml_rf_se = np.std(theta_list_rf) / np.sqrt(len(theta_list_rf))
dml_rf_bias = dml_rf_estimate - TRUE_EFFECT
dml_rf_mse = dml_rf_bias**2 + dml_rf_se**2

# 변수 중요도 (마지막 fold 모델 사용)
if 'rf_outcome' in locals():
    feature_importance = pd.DataFrame({
        'feature': covariates_cols,
        'importance': rf_outcome.feature_importances_
    }).sort_values('importance', ascending=False)

print(f"Fold 수: {len(theta_list_rf)}")
print(f"처치효과 추정: {dml_rf_estimate:.3f}")
print(f"표준오차: {dml_rf_se:.3f}")
print(f"편향: {dml_rf_bias:.3f}")
print(f"MSE: {dml_rf_mse:.3f}")
print(f"95% 신뢰구간: [{dml_rf_estimate - 1.96*dml_rf_se:.3f}, {dml_rf_estimate + 1.96*dml_rf_se:.3f}]")
print(f"결과 모델 평균 R²: {np.mean(r2_outcome_list):.3f}")
print(f"처치 모델 평균 R²: {np.mean(r2_treat_list):.3f}")

print(f"\n상위 5개 중요 변수:")
print(feature_importance.head().to_string(index=False))
print(f"표준오차: {dml_rf_se:.3f}")
print(f"95% 신뢰구간: [{dml_rf_estimate - 1.96*dml_rf_se:.3f}, {dml_rf_estimate + 1.96*dml_rf_se:.3f}]")
print(f"결과 모델 평균 R²: {np.mean(r2_outcome_list):.3f}")
print(f"처치 모델 평균 R²: {np.mean(r2_treat_list):.3f}")

print(f"\n상위 5개 중요 변수:")
print(feature_importance.head().to_string(index=False))

# 5. DML-RDD (Gradient Boosting)
print("\n[5단계] DML-RDD (Gradient Boosting)")
print("-" * 80)

theta_list_gb = []

for train_idx, test_idx in kf.split(near_cutoff):
    train_data = near_cutoff.iloc[train_idx]
    test_data = near_cutoff.iloc[test_idx]

    gb_outcome = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, 
        min_samples_leaf=5, random_state=42
    )
    gb_outcome.fit(train_data[covariates_cols], train_data['outcome'])

    gb_treat = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, 
        min_samples_leaf=5, random_state=42
    )
    gb_treat.fit(train_data[covariates_cols], train_data['treat'])

    y_res = test_data['outcome'].values - gb_outcome.predict(test_data[covariates_cols])
    d_res = test_data['treat'].values - gb_treat.predict(test_data[covariates_cols])

    if np.sum(d_res ** 2) > 1e-10:
        theta = np.sum(y_res * d_res) / np.sum(d_res ** 2)
        theta_list_gb.append(theta)

dml_gb_estimate = np.mean(theta_list_gb)
dml_gb_se = np.std(theta_list_gb) / np.sqrt(len(theta_list_gb))
dml_gb_bias = dml_gb_estimate - TRUE_EFFECT
dml_gb_mse = dml_gb_bias**2 + dml_gb_se**2

print(f"처치효과 추정: {dml_gb_estimate:.3f}")
print(f"표준오차: {dml_gb_se:.3f}")
print(f"편향: {dml_gb_bias:.3f}")
print(f"MSE: {dml_gb_mse:.3f}")

# 6. 방법론 비교
print("\n[6단계] 방법론 비교")
print("-" * 80)

# 비교 테이블 생성
methods_comparison = pd.DataFrame({
    'Method': [
        'Traditional RDD',
        'Linear Adjustment (5 covs)',
        'DML-RDD (Random Forest)',
        'DML-RDD (Gradient Boosting)'
    ],
    'Estimate': [trad_estimate, linear_estimate, dml_rf_estimate, dml_gb_estimate],
    'SE': [trad_se, linear_se, dml_rf_se, dml_gb_se],
    'Bias': [trad_bias, linear_bias, dml_rf_bias, dml_gb_bias],
    'MSE': [trad_mse, linear_mse, dml_rf_mse, dml_gb_mse],
    'CI_lower': [
        trad_estimate - 1.96*trad_se,
        linear_estimate - 1.96*linear_se,
        dml_rf_estimate - 1.96*dml_rf_se,
        dml_gb_estimate - 1.96*dml_gb_se
    ],
    'CI_upper': [
        trad_estimate + 1.96*trad_se,
        linear_estimate + 1.96*linear_se,
        dml_rf_estimate + 1.96*dml_rf_se,
        dml_gb_estimate + 1.96*dml_gb_se
    ],
    'N_covariates': [0, 5, n_covariates, n_covariates]
})

# 신뢰구간에 진정한 효과 포함 여부
methods_comparison['Covers_True'] = (
    (methods_comparison['CI_lower'] <= TRUE_EFFECT) & 
    (methods_comparison['CI_upper'] >= TRUE_EFFECT)
)

print(f"\n진정한 처치효과: {TRUE_EFFECT}")
print("\n방법론별 비교:")
print(methods_comparison[['Method', 'Estimate', 'SE', 'Bias', 'MSE', 'Covers_True']].to_string(index=False))

# 7. 시각화
print("\n[7단계] 시각화 생성")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) 방법론별 추정값 비교
ax = axes[0, 0]
x_pos = np.arange(len(methods_comparison))
estimates = methods_comparison['Estimate']
errors = 1.96 * methods_comparison['SE']

colors = ['steelblue', 'coral', 'green', 'purple']
bars = ax.bar(x_pos, estimates, yerr=errors, capsize=5, alpha=0.7, color=colors)
ax.axhline(TRUE_EFFECT, color='red', linestyle='--', linewidth=2, label=f'True Effect = {TRUE_EFFECT}')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Traditional', 'Linear Adj', 'DML (RF)', 'DML (GB)'],
                  rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Treatment Effect Estimate', fontsize=11)
ax.set_title('(a) Method Comparison (with 95% CI)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (b) MSE 비교
ax = axes[0, 1]
ax.bar(x_pos, methods_comparison['MSE'], alpha=0.7, color=colors)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Traditional', 'Linear Adj', 'DML (RF)', 'DML (GB)'],
                  rotation=15, ha='right', fontsize=9)
ax.set_ylabel('MSE (Bias² + Variance)', fontsize=11)
ax.set_title('(b) Mean Squared Error Comparison', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 최소 MSE 하이라이트
min_idx = methods_comparison['MSE'].idxmin()
ax.bar(min_idx, methods_comparison.loc[min_idx, 'MSE'], alpha=0.9, 
       color=colors[min_idx], edgecolor='black', linewidth=2)

# (c) 변수 중요도
ax = axes[1, 0]
top_10_features = feature_importance.head(10)
ax.barh(range(len(top_10_features)), top_10_features['importance'], color='darkgreen', alpha=0.7)
ax.set_yticks(range(len(top_10_features)))
ax.set_yticklabels(top_10_features['feature'])
ax.set_xlabel('Importance', fontsize=11)
ax.set_title('(c) Variable Importance (Random Forest)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# (d) 편향 vs 표준오차 분해
ax = axes[1, 1]
width = 0.35
bias_sq = methods_comparison['Bias']**2
variance = methods_comparison['SE']**2

ax.bar(x_pos - width/2, bias_sq, width, label='Bias²', alpha=0.7, color='salmon')
ax.bar(x_pos + width/2, variance, width, label='Variance', alpha=0.7, color='skyblue')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Traditional', 'Linear Adj', 'DML (RF)', 'DML (GB)'],
                  rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Component Value', fontsize=11)
ax.set_title('(d) MSE Decomposition: Bias² vs Variance', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(base_path, '5-5-dml-rdd.png'), dpi=300, bbox_inches='tight')
print("그래프 저장 완료: 5-5-dml-rdd.png")

# 8. 최종 요약
print("\n" + "=" * 80)
print("DML-RDD 분석 요약")
print("=" * 80)
print(f"✓ 진정한 처치효과: {TRUE_EFFECT}")
print(f"✓ 전통 RDD: 추정={trad_estimate:.3f}, SE={trad_se:.3f}, MSE={trad_mse:.3f}")
print(f"✓ 선형 조정: 추정={linear_estimate:.3f}, SE={linear_se:.3f}, MSE={linear_mse:.3f}")
print(f"✓ DML-RF: 추정={dml_rf_estimate:.3f}, SE={dml_rf_se:.3f}, MSE={dml_rf_mse:.3f}")
print(f"✓ DML-GB: 추정={dml_gb_estimate:.3f}, SE={dml_gb_se:.3f}, MSE={dml_gb_mse:.3f}")
print(f"\n✓ SE 감소율 (DML-RF vs Traditional): {(1-dml_rf_se/trad_se)*100:.1f}%")
print(f"✓ MSE 감소율 (DML-RF vs Traditional): {(1-dml_rf_mse/trad_mse)*100:.1f}%")
print(f"✓ 최소 MSE 방법: {methods_comparison.loc[methods_comparison['MSE'].idxmin(), 'Method']}")
print(f"✓ 가장 중요한 변수: {feature_importance.iloc[0]['feature']} (중요도 {feature_importance.iloc[0]['importance']:.3f})")
print("=" * 80)
