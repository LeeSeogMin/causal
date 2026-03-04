"""
제5장: Fuzzy RDD 추정
2단계 최소제곱법(2SLS)을 이용한 불완전 순응 상황의 RDD 추정
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from matplotlib import font_manager as fm

# 한글 폰트 설정 (플랫폼 독립적)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 랜덤 시드 설정
np.random.seed(42)

print("=" * 80)
print("Fuzzy RDD 추정: 2단계 최소제곱법")
print("=" * 80)

# 1. 데이터 로드
print("\n[1단계] 데이터 로드")
print("-" * 80)

# CSV 파일에서 데이터 읽기
import os
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/5-4-fuzzy-rdd-data.csv')
df = pd.read_csv(data_path)

cutoff = 50
treatment_effect_true = 3.65  # 순응자에 대한 LATE

print(f"데이터 파일: {data_path}")
print(f"총 관측치 수: {len(df)}")
print(f"배정 지시자 Z=1: {df['above_cutoff'].sum()}")
print(f"실제 처치 D=1: {df['treatment'].sum()}")
print(f"순응률 (Z=1일 때 D=1): {(df[df['above_cutoff']==1]['treatment']==1).mean():.3f}")
print(f"Always-taker 비율 (Z=0일 때도 D=1): {(df[df['above_cutoff']==0]['treatment']==1).mean():.3f}")

# 2. Fuzzy RDD 추정 (2단계 최소제곱법)
print("\n[2단계] Fuzzy RDD 2SLS 추정")
print("-" * 80)

# 대역폭 선택
bandwidth = 0.75 * df['score'].std()
near_cutoff = df[np.abs(df['score'] - cutoff) <= bandwidth].copy()
near_cutoff['score_centered'] = near_cutoff['score'] - cutoff
near_cutoff['above_score'] = near_cutoff['above_cutoff'] * near_cutoff['score_centered']

print(f"선택된 대역폭: {bandwidth:.2f}")
print(f"유효 표본 크기: {len(near_cutoff)}")

# 1단계: 처치를 배정 지시자에 회귀
print("\n--- 1단계 (First Stage) ---")
first_stage = smf.ols('treatment ~ above_cutoff + score_centered + above_score',
                      data=near_cutoff).fit()

print(f"배정 지시자 계수: {first_stage.params['above_cutoff']:.3f}")
print(f"1단계 t-통계량: {first_stage.tvalues['above_cutoff']:.2f}")
print(f"1단계 p-value: {first_stage.pvalues['above_cutoff']:.4f}")
print(f"1단계 R²: {first_stage.rsquared:.3f}")

# 1단계 F-통계량 계산
f_stat = first_stage.tvalues['above_cutoff'] ** 2
print(f"1단계 F-통계량: {f_stat:.2f}")
print(f"약한 도구 판정: {'강한 도구변수' if f_stat > 10 else '약한 도구변수'}")

# 축약형 (Reduced Form): 결과를 배정 지시자에 직접 회귀
print("\n--- 축약형 (Reduced Form / ITT) ---")
reduced_form = smf.ols('outcome ~ above_cutoff + score_centered + above_score',
                       data=near_cutoff).fit()

itt_estimate = reduced_form.params['above_cutoff']
itt_se = reduced_form.bse['above_cutoff']

print(f"ITT 효과: {itt_estimate:.3f}")
print(f"ITT 표준오차: {itt_se:.3f}")
print(f"ITT p-value: {reduced_form.pvalues['above_cutoff']:.4f}")

# 2단계: 결과를 예측된 처치에 회귀
print("\n--- 2단계 (Second Stage) ---")
near_cutoff['treatment_hat'] = first_stage.fittedvalues

second_stage = smf.ols('outcome ~ treatment_hat + score_centered + above_score',
                       data=near_cutoff).fit()

# Fuzzy RDD 추정값 (LATE)
fuzzy_estimate = second_stage.params['treatment_hat']

# 강건 표준오차로 보정 (수동 계산)
# LATE = ITT / First Stage
late = itt_estimate / first_stage.params['above_cutoff']
late_se = itt_se / abs(first_stage.params['above_cutoff'])

print(f"Fuzzy RDD (LATE) 추정: {late:.3f}")
print(f"표준오차: {late_se:.3f}")
print(f"95% 신뢰구간: [{late - 1.96*late_se:.3f}, {late + 1.96*late_se:.3f}]")
print(f"t-통계량: {late / late_se:.2f}")
print(f"p-value: {2 * (1 - stats.t.cdf(abs(late / late_se), len(near_cutoff)-4)):.4f}")

# 3. Sharp vs Fuzzy 비교
print("\n[3단계] Sharp RDD와 Fuzzy RDD 비교")
print("-" * 80)

# Sharp RDD (실제 처치를 배정 지시자로 대체)
near_cutoff['treat_score_sharp'] = near_cutoff['above_cutoff'] * near_cutoff['score_centered']
sharp_model = smf.ols('outcome ~ above_cutoff + score_centered + treat_score_sharp',
                      data=near_cutoff).fit()

sharp_estimate = sharp_model.params['above_cutoff']
sharp_se = sharp_model.bse['above_cutoff']

comparison = pd.DataFrame({
    'Method': ['Sharp RDD', 'Fuzzy RDD (LATE)', 'Reduced Form (ITT)'],
    'Estimate': [sharp_estimate, late, itt_estimate],
    'SE': [sharp_se, late_se, itt_se],
    'CI_lower': [sharp_estimate - 1.96*sharp_se, late - 1.96*late_se, itt_estimate - 1.96*itt_se],
    'CI_upper': [sharp_estimate + 1.96*sharp_se, late + 1.96*late_se, itt_estimate + 1.96*itt_se],
    'Interpretation': ['전체 집단 ATE', '순응자 LATE', '배정 효과 ITT']
})

print("\n추정 방법 비교:")
print(comparison.to_string(index=False))

# 4. 순응 패턴 분석
print("\n[4단계] 불완전 순응 패턴 분석")
print("-" * 80)

# Complier, Always-taker, Never-taker 비율 추정
compliers = first_stage.params['above_cutoff']  # Z=1일 때 D 증가분
always_takers = (df[df['above_cutoff']==0]['treatment']==1).mean()
never_takers = 1 - compliers - always_takers

compliance_table = pd.DataFrame({
    'Type': ['순응자 (Complier)', 'Always-taker', 'Never-taker', '반항자 (Defier)'],
    'Description': ['Z=1일 때만 D=1', 'Z와 무관하게 D=1', 'Z와 무관하게 D=0', '단조성 가정으로 0%'],
    'Proportion': [compliers, always_takers, never_takers, 0.0]
})

print("\n순응 패턴:")
print(compliance_table.to_string(index=False))

# 5. 시각화
print("\n[5단계] 시각화 생성")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) 1단계: 처치 수혜율 점프
ax = axes[0, 0]
bins = np.linspace(df['score'].min(), df['score'].max(), 30)
bin_centers = (bins[:-1] + bins[1:]) / 2

treat_rate_by_bin = []
for i in range(len(bins)-1):
    mask = (df['score'] >= bins[i]) & (df['score'] < bins[i+1])
    if mask.sum() > 0:
        treat_rate_by_bin.append(df[mask]['treatment'].mean())
    else:
        treat_rate_by_bin.append(np.nan)

ax.scatter(bin_centers, treat_rate_by_bin, alpha=0.6, s=50, color='steelblue')

# 좌우 회귀선
left_data = near_cutoff[near_cutoff['score'] < cutoff]
right_data = near_cutoff[near_cutoff['score'] >= cutoff]

if len(left_data) > 0:
    x_left = np.linspace(left_data['score'].min(), cutoff, 100)
    y_left = (first_stage.params['Intercept'] +
              first_stage.params['score_centered'] * (x_left - cutoff))
    ax.plot(x_left, y_left, 'b-', linewidth=2, label='좌측 추세')

if len(right_data) > 0:
    x_right = np.linspace(cutoff, right_data['score'].max(), 100)
    y_right = (first_stage.params['Intercept'] + first_stage.params['above_cutoff'] +
               (first_stage.params['score_centered'] + first_stage.params['above_score']) * (x_right - cutoff))
    ax.plot(x_right, y_right, 'r-', linewidth=2, label='우측 추세')

ax.axvline(cutoff, color='black', linestyle='--', linewidth=1.5, label='기준점')
ax.set_xlabel('배정 변수 (점수)', fontsize=11)
ax.set_ylabel('처치 수혜율', fontsize=11)
ax.set_title(f'(a) 1단계: 처치 수혜율 점프 ({compliers:.2%})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (b) 축약형: 결과 변수 불연속
ax = axes[0, 1]
ax.scatter(df[df['above_cutoff']==0]['score'], df[df['above_cutoff']==0]['outcome'],
          alpha=0.3, s=20, label='Z=0', color='steelblue')
ax.scatter(df[df['above_cutoff']==1]['score'], df[df['above_cutoff']==1]['outcome'],
          alpha=0.3, s=20, label='Z=1', color='coral')

# 축약형 회귀선
if len(left_data) > 0:
    x_left = np.linspace(left_data['score'].min(), cutoff, 100)
    y_left = (reduced_form.params['Intercept'] +
              reduced_form.params['score_centered'] * (x_left - cutoff))
    ax.plot(x_left, y_left, 'b-', linewidth=2)

if len(right_data) > 0:
    x_right = np.linspace(cutoff, right_data['score'].max(), 100)
    y_right = (reduced_form.params['Intercept'] + reduced_form.params['above_cutoff'] +
               (reduced_form.params['score_centered'] + reduced_form.params['above_score']) * (x_right - cutoff))
    ax.plot(x_right, y_right, 'r-', linewidth=2)

ax.axvline(cutoff, color='black', linestyle='--', linewidth=1.5)
ax.set_xlabel('배정 변수 (점수)', fontsize=11)
ax.set_ylabel('결과 변수', fontsize=11)
ax.set_title(f'(b) 축약형 (ITT={itt_estimate:.2f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (c) Sharp vs Fuzzy vs ITT 비교
ax = axes[1, 0]
methods = comparison['Method']
estimates = comparison['Estimate']
errors = 1.96 * comparison['SE']

x_pos = np.arange(len(methods))
colors = ['steelblue', 'coral', 'green']
bars = ax.bar(x_pos, estimates, yerr=errors, capsize=5, color=colors, alpha=0.7)
ax.axhline(treatment_effect_true, color='red', linestyle='--', linewidth=2, label='진정한 LATE')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.set_ylabel('처치효과 추정값', fontsize=11)
ax.set_title('(c) 추정 방법 비교', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (d) 순응 패턴
ax = axes[1, 1]
types = compliance_table['Type']
proportions = compliance_table['Proportion']
colors_comp = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

wedges, texts, autotexts = ax.pie(proportions, labels=types, autopct='%1.1f%%',
                                    colors=colors_comp, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_weight('bold')
ax.set_title('(d) 순응 패턴 분포', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('5-4-fuzzy-rdd.png', dpi=300, bbox_inches='tight')
print("그래프 저장 완료: practice/chapter05/5-4-fuzzy-rdd.png")

# 6. 최종 요약
print("\n" + "=" * 80)
print("Fuzzy RDD 분석 요약")
print("=" * 80)
print(f"✓ 1단계 F-통계량: {f_stat:.2f} ({'강한 도구변수' if f_stat > 10 else '약한 도구변수'})")
print(f"✓ 처치 수혜율 점프: {compliers:.2%}")
print(f"✓ Fuzzy RDD (LATE): {late:.2f} (95% CI: [{late - 1.96*late_se:.2f}, {late + 1.96*late_se:.2f}])")
print(f"✓ 축약형 (ITT): {itt_estimate:.2f} (95% CI: [{itt_estimate - 1.96*itt_se:.2f}, {itt_estimate + 1.96*itt_se:.2f}])")
print(f"✓ Sharp RDD (비교): {sharp_estimate:.2f}")
print(f"✓ 순응자 비율: {compliers:.1%}, Always-taker: {always_takers:.1%}, Never-taker: {never_takers:.1%}")
print(f"✓ LATE vs ITT 관계: LATE ({late:.2f}) = ITT ({itt_estimate:.2f}) / First Stage ({compliers:.2f})")
print("=" * 80)
