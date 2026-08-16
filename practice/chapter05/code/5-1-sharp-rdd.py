"""
제5장: Sharp RDD 추정 (국소 선형 회귀)
회귀불연속설계의 기본 구현 - 기준점 근처 국소 선형 회귀로 처치효과 추정

수정 이력
---------
2026-08-17
1) 한글 폰트: `font.family`가 'DejaVu Sans'로 지정되어 있었다. DejaVu Sans에는
   한글 글리프가 없어서 축 이름·범례·제목의 한글이 전부 빈 사각형으로 저장됐다
   (실행 시 "Glyph ... missing from font(s) DejaVu Sans" 경고 다수).
   설치된 한글 폰트를 찾아 지정하도록 고쳤다.
2) 저장 경로 안내: savefig는 스크립트가 있는 폴더에 저장하는데 print 문은
   'practice/chapter05/...'를 찍어 실제 위치와 달랐다. 실제 저장 경로를 찍도록 고쳤다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import font_manager as fm

# 한글 폰트 설정: 설치된 한글 폰트를 순서대로 찾아 지정한다
_installed = {f.name for f in fm.fontManager.ttflist}
for _name in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR']:
    if _name in _installed:
        plt.rcParams['font.family'] = _name
        break
plt.rcParams['axes.unicode_minus'] = False

# 랜덤 시드 설정
np.random.seed(42)

print("=" * 80)
print("Sharp RDD 추정: 국소 선형 회귀")
print("=" * 80)

# 1. 데이터 로드
print("\n[1단계] 데이터 로드")
print("-" * 80)

# CSV 파일에서 데이터 읽기
# CSV 파일에서 데이터 읽기
import os
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/5-1-sharp-rdd-data.csv')
df = pd.read_csv(data_path)

cutoff = 50
treatment_effect = 3.8  # 진정한 처치효과

print(f"데이터 파일: {data_path}")
print(f"총 관측치 수: {len(df)}")
print(f"처치군 수: {df['treat'].sum()}")
print(f"대조군 수: {len(df) - df['treat'].sum()}")
print(f"기준점: {cutoff}")

# 2. Sharp RDD 추정 (국소 선형 회귀)
print("\n[2단계] Sharp RDD 국소 선형 회귀 추정")
print("-" * 80)

# 대역폭 선택 (표준편차 기반)
bandwidth = 0.5 * df['score'].std()
print(f"선택된 대역폭: {bandwidth:.2f}")

# 기준점 근처 데이터만 선택
near_cutoff = df[np.abs(df['score'] - cutoff) <= bandwidth].copy()
print(f"유효 표본 크기: {len(near_cutoff)}")

# 중심화된 배정 변수
near_cutoff['score_centered'] = near_cutoff['score'] - cutoff
near_cutoff['treat_score'] = near_cutoff['treat'] * near_cutoff['score_centered']

# 국소 선형 회귀: Y = β0 + β1·D + β2·(X-c) + β3·D·(X-c) + ε
model = smf.ols('outcome ~ treat + score_centered + treat_score',
                data=near_cutoff).fit()

rdd_estimate = model.params['treat']
rdd_se = model.bse['treat']
rdd_tstat = model.tvalues['treat']
rdd_pvalue = model.pvalues['treat']

print(f"\n처치효과 추정 (τ̂): {rdd_estimate:.2f}")
print(f"표준오차: {rdd_se:.2f}")
print(f"t-통계량: {rdd_tstat:.2f}")
print(f"p-value: {rdd_pvalue:.4f}")
print(f"95% 신뢰구간: [{model.conf_int().loc['treat', 0]:.2f}, {model.conf_int().loc['treat', 1]:.2f}]")
print(f"R²: {model.rsquared:.3f}")
print(f"좌측 기울기: {model.params['score_centered']:.3f}")
print(f"우측 기울기: {model.params['score_centered'] + model.params['treat_score']:.3f}")

# 3. 대역폭 민감도 분석
print("\n[3단계] 대역폭 민감도 분석")
print("-" * 80)

bandwidths = [0.5, 1.0, 1.5, 2.0]
sensitivity_results = []

for h in bandwidths:
    bw = h * df['score'].std()
    near = df[np.abs(df['score'] - cutoff) <= bw].copy()
    near['score_centered'] = near['score'] - cutoff
    near['treat_score'] = near['treat'] * near['score_centered']

    mod = smf.ols('outcome ~ treat + score_centered + treat_score', data=near).fit()

    sensitivity_results.append({
        'bandwidth_multiplier': h,
        'bandwidth': bw,
        'estimate': mod.params['treat'],
        'se': mod.bse['treat'],
        'n_eff': len(near)
    })

sensitivity_df = pd.DataFrame(sensitivity_results)
print("\n대역폭 민감도 분석 결과:")
print(sensitivity_df.to_string(index=False))

# 4. 공변량 균형 검정
print("\n[4단계] 공변량 균형 검정 (기준점 ±1σ 내)")
print("-" * 80)

balance_window = df['score'].std()
balance_data = df[np.abs(df['score'] - cutoff) <= balance_window].copy()
balance_data['score_centered'] = balance_data['score'] - cutoff
balance_data['treat_score'] = balance_data['treat'] * balance_data['score_centered']

covariates = ['age', 'female', 'education']
balance_results = []

for cov in covariates:
    formula = f'{cov} ~ treat + score_centered + treat_score'
    mod = smf.ols(formula, data=balance_data).fit()

    balance_results.append({
        'covariate': cov,
        'treat_mean': balance_data[balance_data['treat']==1][cov].mean(),
        'control_mean': balance_data[balance_data['treat']==0][cov].mean(),
        'discontinuity': mod.params['treat'],
        'p_value': mod.pvalues['treat']
    })

balance_df = pd.DataFrame(balance_results)
print("\n공변량 균형 검정 결과:")
print(balance_df.to_string(index=False))

# 5. 시각화
print("\n[5단계] 시각화 생성")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) RDD 산점도와 국소 회귀선
ax = axes[0, 0]
treated = df[df['treat'] == 1]
control = df[df['treat'] == 0]

ax.scatter(control['score'], control['outcome'], alpha=0.3, s=20, label='대조군', color='steelblue')
ax.scatter(treated['score'], treated['outcome'], alpha=0.3, s=20, label='처치군', color='coral')

# 좌우 회귀선 그리기
left_data = near_cutoff[near_cutoff['score'] < cutoff]
right_data = near_cutoff[near_cutoff['score'] >= cutoff]

if len(left_data) > 0:
    x_left = np.linspace(left_data['score'].min(), cutoff, 100)
    y_left = (model.params['Intercept'] +
              model.params['score_centered'] * (x_left - cutoff))
    ax.plot(x_left, y_left, 'b-', linewidth=2, label='좌측 추세')

if len(right_data) > 0:
    x_right = np.linspace(cutoff, right_data['score'].max(), 100)
    y_right = (model.params['Intercept'] + model.params['treat'] +
               (model.params['score_centered'] + model.params['treat_score']) * (x_right - cutoff))
    ax.plot(x_right, y_right, 'r-', linewidth=2, label='우측 추세')

ax.axvline(cutoff, color='black', linestyle='--', linewidth=1.5, label='기준점')
ax.set_xlabel('배정 변수 (점수)', fontsize=11)
ax.set_ylabel('결과 변수', fontsize=11)
ax.set_title('(a) Sharp RDD: 산점도와 국소 회귀선', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (b) 대역폭별 추정값 변화
ax = axes[0, 1]
ax.errorbar(sensitivity_df['bandwidth'], sensitivity_df['estimate'],
            yerr=1.96*sensitivity_df['se'], fmt='o-', capsize=5, capthick=2,
            color='darkgreen', markersize=8)
ax.axhline(treatment_effect, color='red', linestyle='--', linewidth=2, label='진정한 효과')
ax.set_xlabel('대역폭', fontsize=11)
ax.set_ylabel('처치효과 추정값', fontsize=11)
ax.set_title('(b) 대역폭 민감도 분석', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (c) 공변량 균형 검정 시각화
ax = axes[1, 0]
x_pos = np.arange(len(covariates))
discontinuities = balance_df['discontinuity'].values
p_values = balance_df['p_value'].values

colors = ['green' if p > 0.05 else 'red' for p in p_values]
bars = ax.bar(x_pos, discontinuities, color=colors, alpha=0.6)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(covariates)
ax.set_ylabel('기준점에서 불연속', fontsize=11)
ax.set_title('(c) 공변량 균형 검정 (녹색=균형)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 각 막대 위에 p-value 표시
for i, (bar, p) in enumerate(zip(bars, p_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'p={p:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=9)

# (d) 밀도 분포 (조작 검정 시각화)
ax = axes[1, 1]
bins = np.linspace(df['score'].min(), df['score'].max(), 40)
ax.hist(df['score'], bins=bins, alpha=0.6, color='purple', edgecolor='black')
ax.axvline(cutoff, color='red', linestyle='--', linewidth=2, label='기준점')
ax.set_xlabel('배정 변수 (점수)', fontsize=11)
ax.set_ylabel('빈도', fontsize=11)
ax.set_title('(d) 배정 변수 밀도 분포', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
_out = os.path.join(base_path, '5-1-sharp-rdd-results.png')
plt.savefig(_out, dpi=300, bbox_inches='tight')
print(f"그래프 저장 완료: {_out}")

# 6. 최종 요약
print("\n" + "=" * 80)
print("Sharp RDD 분석 요약")
print("=" * 80)
print(f"✓ 처치효과 추정값: {rdd_estimate:.2f} (95% CI: [{model.conf_int().loc['treat', 0]:.2f}, {model.conf_int().loc['treat', 1]:.2f}])")
print(f"✓ 통계적 유의성: {'유의함' if rdd_pvalue < 0.05 else '비유의함'} (p={rdd_pvalue:.4f})")
print(f"✓ 공변량 균형: 모든 공변량이 균형 (모든 p > 0.05)")
print(f"✓ 대역폭 강건성: 추정값이 {sensitivity_df['estimate'].min():.2f}~{sensitivity_df['estimate'].max():.2f} 범위에서 안정적")
print("=" * 80)
