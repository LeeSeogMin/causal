"""
제2장 2.6절: Causal ML과 Predictive Analytics 통합
DML + XGBoost를 활용한 한국 탄소 배출 감축 정책 분석

교육 목적: 인과추론과 예측 분석을 통합하여 정책 시나리오를 시뮬레이션

수정 이력
---------
2026-08-17
1) 그래프 축 이름과 범례를 한글로 쓰면서 폰트는 'Arial'로 지정해,
   저장된 PNG의 한글이 전부 네모로 깨졌다.
   조치: Malgun Gothic → AppleGothic → NanumGothic 순으로 설치된 폰트를
   찾아 지정하고, 셋 다 없으면 축 이름을 영문으로 내린다.
2) 그래프를 practice/chapter02/output/에 저장해 다른 실습 결과와 위치가
   달랐다. 조치: 다른 스크립트와 같이 code/ 폴더에 저장한다.
3) 마지막 plt.show()를 plt.close()로 바꿨다. 배치 실행에서 창이 뜨지 않는다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from sklearn.model_selection import train_test_split
import xgboost as xgb
from econml.dml import CausalForestDML, LinearDML
from econml.dr import DRLearner
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows/Mac/Linux 호환)
from matplotlib import font_manager
_installed = {f.name for f in font_manager.fontManager.ttflist}
KOREAN_FONT = next((f for f in ['Malgun Gothic', 'AppleGothic', 'NanumGothic']
                    if f in _installed), None)
if KOREAN_FONT:
    plt.rcParams['font.family'] = KOREAN_FONT
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def L(ko, en):
    """한글 폰트가 없으면 영문 레이블로 내린다."""
    return ko if KOREAN_FONT else en

print("="*80)
print("제2장 2.6절: Causal ML과 Predictive Analytics 통합")
print("한국 탄소 배출 감축 정책 분석 (DML + XGBoost)")
print("="*80)
print()

# =============================================================================
# 1. 데이터 로드: 한국 제조업 탄소 배출
# =============================================================================
print("1. 한국 제조업 탄소 배출 데이터 로드 중...")
import os
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/2-6-carbon.csv')
df = pd.read_csv(data_path)

X = df[['firm_size', 'industry', 'energy_efficiency', 'rd_investment', 'region']].values
feature_names = ['firm_size', 'industry', 'energy_efficiency', 'rd_investment', 'region']
X_df = pd.DataFrame(X, columns=feature_names)

carbon_tax = df['carbon_tax'].values
emission = df['emission'].values
true_cate = df['CATE_true'].values

n = len(df)
print(f"  총 기업 수: {n:,}개")
print(f"  탄소세 적용 기업: {carbon_tax.sum():,}개 ({carbon_tax.mean()*100:.1f}%)")
print(f"  평균 배출량: {emission.mean():.1f} 톤")
print(f"  진짜 평균 처치효과 (ATE): {true_cate.mean():.1f} 톤")
print()

# =============================================================================
# 2. 학습/테스트 데이터 분할
# =============================================================================
print("2. 데이터 분할 (70% 학습, 30% 테스트)...")
X_train, X_test, emission_train, emission_test, tax_train, tax_test, cate_train, cate_test = train_test_split(
    X, emission, carbon_tax, true_cate, test_size=0.3, random_state=2025
)
print(f"  학습 데이터: {len(X_train):,}개")
print(f"  테스트 데이터: {len(X_test):,}개")
print()

# =============================================================================
# 3. Causal ML 모델 학습
# =============================================================================
print("3. Causal ML 모델 학습 중...")

# 3.1 Causal Forest DML with XGBoost
print("  3.1 CausalForestDML + XGBoost 학습...")
xgb_y_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=2025, verbosity=0)
xgb_t_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=2025, verbosity=0)

cf_xgb = CausalForestDML(
    model_y=xgb_y_model,
    model_t=xgb_t_model,
    n_estimators=2000,
    min_samples_leaf=10,
    max_depth=10,
    random_state=2025,
    verbose=0
)
cf_xgb.fit(emission_train, tax_train, X=X_train)
cate_cf = cf_xgb.effect(X_test)

# 3.2 LinearDML
print("  3.2 LinearDML 학습...")
dml = LinearDML(
    model_y=xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=2025, verbosity=0),
    model_t=xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=2025, verbosity=0),
    discrete_treatment=True,
    random_state=2025
)
dml.fit(Y=emission_train, T=tax_train, X=X_train)
ate_dml = dml.effect(X_test).mean()

# 3.3 Doubly Robust Learner
print("  3.3 DRLearner 학습...")
dr = DRLearner(
    model_propensity=xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=2025, verbosity=0),
    model_regression=xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=2025, verbosity=0),
    random_state=2025
)
dr.fit(Y=emission_train, T=tax_train, X=X_train)
cate_dr = dr.effect(X_test)

print("  모델 학습 완료!")
print()

# =============================================================================
# 4. 인과효과 추정 결과 분석
# =============================================================================
print("4. 인과효과 추정 결과 분석")
print("-"*80)

# ATE 추정
ate_cf = cate_cf.mean()
ate_dr = cate_dr.mean()
true_ate = true_cate.mean()
naive_ate = emission[carbon_tax==1].mean() - emission[carbon_tax==0].mean()

print(f"  진짜 ATE: {true_ate:.2f} 톤")
print(f"  CausalForestDML 추정: {ate_cf:.2f} 톤 (편향: {ate_cf - true_ate:.2f} 톤)")
print(f"  LinearDML 추정: {ate_dml:.2f} 톤 (편향: {ate_dml - true_ate:.2f} 톤)")
print(f"  DRLearner 추정: {ate_dr:.2f} 톤 (편향: {ate_dr - true_ate:.2f} 톤)")
print(f"  Naive 추정 (선택편향): {naive_ate:.2f} 톤 (편향: {naive_ate - true_ate:.2f} 톤)")
print()

# CATE 정확도
cate_corr_cf = np.corrcoef(cate_cf, cate_test)[0, 1]
cate_corr_dr = np.corrcoef(cate_dr, cate_test)[0, 1]
print(f"  CATE 상관계수 (진짜 vs. CausalForest): {cate_corr_cf:.3f}")
print(f"  CATE 상관계수 (진짜 vs. DRLearner): {cate_corr_dr:.3f}")
print()

# =============================================================================
# 5. 에너지 효율별 이질적 처치효과 분석
# =============================================================================
print("5. 에너지 효율별 이질적 처치효과 (CATE) 분석")
print("-"*80)

efficiency_test = X_test[:, 2]
efficiency_quintiles = pd.qcut(efficiency_test, q=5, labels=['하위 20%', '하위 40%', '중위', '상위 40%', '상위 20%'])

cate_by_efficiency = pd.DataFrame({
    'quintile': efficiency_quintiles,
    'efficiency': efficiency_test,
    'cate_cf': cate_cf,
    'cate_dr': cate_dr,
    'true_cate': cate_test
})

cate_summary = cate_by_efficiency.groupby('quintile').agg({
    'efficiency': 'mean',
    'cate_cf': 'mean',
    'cate_dr': 'mean',
    'true_cate': 'mean'
}).round(2)

print(cate_summary)
print()

# =============================================================================
# 6. 예측 분석: 정책 시나리오별 배출량 예측
# =============================================================================
print("6. 정책 시나리오별 배출량 예측")
print("-"*80)

# 무처치 결과 예측 모델 (XGBoost)
emission_model = xgb.XGBRegressor(n_estimators=300, max_depth=8, random_state=2025, verbosity=0)
emission_model.fit(X_train[tax_train==0], emission_train[tax_train==0])
emission_no_tax = emission_model.predict(X_test)

# 시나리오 정의
scenarios = {}

# S1: 현 상태 유지
scenarios['S1_현상유지'] = emission_test

# S2: 전면 탄소세 적용
scenarios['S2_전면적용'] = emission_no_tax + cate_cf

# S3: 에너지 효율 하위 50%만 적용
low_efficiency_mask = efficiency_test < np.median(efficiency_test)
emission_s3 = emission_no_tax.copy()
emission_s3[low_efficiency_mask] += cate_cf[low_efficiency_mask]
scenarios['S3_하위50%'] = emission_s3

# S4: 에너지 효율 하위 30%만 적용
bottom_30_mask = efficiency_test < np.percentile(efficiency_test, 30)
emission_s4 = emission_no_tax.copy()
emission_s4[bottom_30_mask] += cate_cf[bottom_30_mask]
scenarios['S4_하위30%'] = emission_s4

# S5: 대기업 (상위 30%) + 효율 하위 50%
firm_size_test = X_test[:, 0]
large_firm_mask = firm_size_test > np.percentile(firm_size_test, 70)
emission_s5 = emission_no_tax.copy()
target_mask = low_efficiency_mask | large_firm_mask
emission_s5[target_mask] += cate_cf[target_mask]
scenarios['S5_대기업+하위50%'] = emission_s5

# 시나리오 결과 요약
scenario_results = pd.DataFrame({
    name: {
        '평균배출량': values.mean(),
        '총배출량': values.sum(),
        '감축률': (scenarios['S1_현상유지'].mean() - values.mean()) / scenarios['S1_현상유지'].mean() * 100,
        '수급자비율': (np.sum(values < emission_no_tax) / len(values)) * 100
    }
    for name, values in scenarios.items()
}).T

print(scenario_results.round(1))
print()

# =============================================================================
# 7. Feature Importance 분석
# =============================================================================
print("7. Feature Importance 분석 (CATE 예측)")
print("-"*80)

# XGBoost로 CATE 예측 모델 학습
cate_predictor = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=2025, verbosity=0)
cate_predictor.fit(X_train, cf_xgb.effect(X_train))

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': cate_predictor.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.to_string(index=False))
print()

# =============================================================================
# 8. 시각화
# =============================================================================
print("8. 결과 시각화...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 8.1 에너지 효율 vs. CATE
ax = axes[0, 0]
scatter = ax.scatter(efficiency_test, cate_cf, c=firm_size_test, cmap='viridis', alpha=0.5, s=30)
ax.plot(efficiency_test, cate_test, 'r.', alpha=0.3, markersize=3, label=L('진짜 CATE','True CATE'))
ax.set_xlabel(L('에너지 효율','Energy efficiency'))
ax.set_ylabel(L('CATE (톤)','CATE (tons)'))
ax.set_title(L('에너지 효율과 처치효과의 관계','Efficiency vs treatment effect'))
ax.legend()
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label=L('기업 규모','Firm size'))

# 8.2 CATE 분포
ax = axes[0, 1]
ax.hist(cate_cf, bins=50, alpha=0.6, label=L('CausalForest 추정','CausalForest estimate'), edgecolor='black')
ax.hist(cate_test, bins=50, alpha=0.6, label='진짜 CATE', edgecolor='black')
ax.axvline(ate_cf, color='blue', linestyle='--', linewidth=2, label=L(f'평균 (추정): {ate_cf:.1f}', f'Mean (est): {ate_cf:.1f}'))
ax.axvline(true_ate, color='red', linestyle='--', linewidth=2, label=L(f'평균 (참값): {true_ate:.1f}', f'Mean (true): {true_ate:.1f}'))
ax.set_xlabel(L('CATE (톤)','CATE (tons)'))
ax.set_ylabel(L('빈도','Frequency'))
ax.set_title(L('처치효과 분포','CATE distribution'))
ax.legend()
ax.grid(True, alpha=0.3)

# 8.3 시나리오별 배출량 비교
ax = axes[0, 2]
scenario_means = [scenarios[s].mean() for s in scenarios.keys()]
scenario_names = [L('S1\n현상유지', 'S1\nbase'), L('S2\n전면적용', 'S2\nall'),
                  L('S3\n하위50%', 'S3\nbottom50'), L('S4\n하위30%', 'S4\nbottom30'),
                  L('S5\n대기업+\n하위50%', 'S5\nlarge+b50')]
colors = ['gray', 'red', 'orange', 'yellow', 'green']
bars = ax.bar(scenario_names, scenario_means, color=colors, edgecolor='black')
ax.set_ylabel(L('평균 배출량 (톤)','Mean emission (tons)'))
ax.set_title(L('정책 시나리오별 배출량 비교','Emission by policy scenario'))
ax.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, scenario_means)):
    ax.text(i, val + 5, f'{val:.1f}', ha='center', fontsize=9)

# 8.4 효율 분위별 CATE
ax = axes[1, 0]
quintile_order = ['하위 20%', '하위 40%', '중위', '상위 40%', '상위 20%']
x_pos = np.arange(len(quintile_order))
cate_cf_means = [cate_by_efficiency[cate_by_efficiency['quintile'] == q]['cate_cf'].mean() for q in quintile_order]
cate_true_means = [cate_by_efficiency[cate_by_efficiency['quintile'] == q]['true_cate'].mean() for q in quintile_order]

width = 0.35
ax.bar(x_pos - width/2, cate_cf_means, width, label='CausalForest 추정', alpha=0.8, edgecolor='black')
ax.bar(x_pos + width/2, cate_true_means, width, label='진짜 CATE', alpha=0.8, edgecolor='black')
ax.set_xlabel(L('에너지 효율 분위','Efficiency quintile'))
ax.set_ylabel(L('평균 CATE (톤)','Mean CATE (tons)'))
ax.set_title(L('효율 분위별 평균 처치효과','Mean CATE by efficiency quintile'))
ax.set_xticks(x_pos)
ax.set_xticklabels(quintile_order if KOREAN_FONT else ['B20','B40','Mid','T40','T20'], rotation=15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 8.5 Feature Importance
ax = axes[1, 1]
ax.barh(importance_df['feature'], importance_df['importance'], edgecolor='black')
ax.set_xlabel('Feature Importance')
ax.set_title(L('CATE 예측 Feature Importance','Feature importance for CATE'))
ax.grid(True, alpha=0.3, axis='x')

# 8.6 시나리오별 감축률과 비용 효율성
ax = axes[1, 2]
reduction_rates = scenario_results['감축률'].values
coverage_rates = scenario_results['수급자비율'].values
efficiency_indices = -reduction_rates / coverage_rates * 100  # 효율성 지수

x_pos = np.arange(len(scenario_results))
bars1 = ax.bar(x_pos - 0.2, reduction_rates, 0.4, label=L('감축률 (%)','Reduction (%)'), alpha=0.8, edgecolor='black')
ax2 = ax.twinx()
bars2 = ax2.bar(x_pos + 0.2, efficiency_indices, 0.4, color='orange', label=L('효율성 지수','Efficiency index'), alpha=0.8, edgecolor='black')

ax.set_xlabel(L('정책 시나리오','Policy scenario'))
ax.set_ylabel(L('감축률 (%)','Reduction (%)'), color='blue')
ax2.set_ylabel(L('효율성 지수','Efficiency index'), color='orange')
ax.set_title(L('시나리오별 감축 효과와 효율성','Reduction and efficiency by scenario'))
ax.set_xticks(x_pos)
ax.set_xticklabels(scenario_names, rotation=15)
ax.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='orange')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
import os
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '2-6-causal-predictive-integration.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  시각화 저장 완료: {output_path}")
print()

# =============================================================================
# 9. 결과 요약
# =============================================================================
print("="*80)
print("분석 요약")
print("="*80)
print()
print(f"1. 인과효과 추정:")
print(f"   - 진짜 ATE: {true_ate:.2f} 톤")
print(f"   - CausalForestDML: {ate_cf:.2f} 톤 (오차: {abs(ate_cf - true_ate)/abs(true_ate)*100:.1f}%)")
print(f"   - Naive 추정 편향: {abs(naive_ate - true_ate):.2f} 톤")
print()
print(f"2. 이질적 효과:")
print(f"   - CATE 범위: {cate_cf.min():.1f} ~ {cate_cf.max():.1f} 톤")
print(f"   - 효율 하위 20% 평균 CATE: {cate_cf_means[0]:.1f} 톤")
print(f"   - 효율 상위 20% 평균 CATE: {cate_cf_means[-1]:.1f} 톤")
print(f"   - 효과 비율: {abs(cate_cf_means[0]/cate_cf_means[-1]):.1f}배")
print()
print(f"3. 최적 정책 시나리오:")
print(f"   - S2 (전면적용): {-scenario_results.loc['S2_전면적용', '감축률']:.1f}% 감축")
print(f"   - S3 (하위50%): {-scenario_results.loc['S3_하위50%', '감축률']:.1f}% 감축, 비용 효율성 우수")
print(f"   - S5 (대기업+하위50%): {-scenario_results.loc['S5_대기업+하위50%', '감축률']:.1f}% 감축, 균형적")
print()
print("※ 본 분석은 교육 목적의 가상 데이터를 사용하였습니다.")
print("  실제 정책 분석 시 한국환경공단의 공식 배출량 데이터를 사용해야 합니다.")
print("="*80)

plt.close()
