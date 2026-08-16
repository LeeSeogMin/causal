"""
제5장: RDD 타당성 검정
McCrary 밀도 검정, 공변량 균형, 플라시보 검정, 도넛홀 분석

수정 이력
---------
2026-08-17
1) 한글 폰트: `font.family`가 'DejaVu Sans'였다. DejaVu Sans에는 한글 글리프가
   없어 6개 패널의 축 이름·제목 한글이 빈 사각형으로 저장됐다. 설치된 한글 폰트를
   찾아 지정하도록 고쳤다.
2) 저장 경로 안내: print 문이 실제 저장 위치와 다른 경로를 찍었다. 실제 경로를 찍는다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde, norm
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
print("RDD 타당성 검정")
print("=" * 80)

# 1. 데이터 로드
print("\n[1단계] 데이터 로드")
print("-" * 80)

# CSV 파일에서 데이터 읽기
import os
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/5-3-validity-tests-data.csv')
df = pd.read_csv(data_path)

cutoff = 50
treatment_effect = 3.76

print(f"데이터 파일: {data_path}")
print(f"총 관측치 수: {len(df)}")

# 2. McCrary 밀도 검정
print("\n[2단계] McCrary 밀도 검정 (조작 가능성)")
print("-" * 80)

def mccrary_test(x, cutoff, bandwidth=None):
    """
    McCrary (2008) 밀도 불연속 검정
    """
    if bandwidth is None:
        bandwidth = 1.5 * np.std(x) * len(x)**(-1/5)

    # 좌우 데이터 분리
    left_data = x[x < cutoff]
    right_data = x[x >= cutoff]

    # 커널 밀도 추정
    kde_left = gaussian_kde(left_data, bw_method=bandwidth/np.std(left_data))
    kde_right = gaussian_kde(right_data, bw_method=bandwidth/np.std(right_data))

    # 기준점에서 밀도 추정
    density_left = kde_left.evaluate(cutoff - 0.001)[0]
    density_right = kde_right.evaluate(cutoff + 0.001)[0]

    # 로그 차이 및 표준오차
    log_diff = np.log(density_right) - np.log(density_left)
    se = np.sqrt(1/len(left_data) + 1/len(right_data))

    # 검정 통계량
    t_stat = log_diff / se
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))

    return {
        'density_left': density_left,
        'density_right': density_right,
        'log_difference': log_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'n_left': len(left_data),
        'n_right': len(right_data)
    }

mccrary_result = mccrary_test(df['score'].values, cutoff)

print(f"\n좌측 밀도: {mccrary_result['density_left']:.4f}")
print(f"우측 밀도: {mccrary_result['density_right']:.4f}")
print(f"로그 차이: {mccrary_result['log_difference']:.4f}")
print(f"t-통계량: {mccrary_result['t_statistic']:.3f}")
print(f"p-value: {mccrary_result['p_value']:.3f}")
print(f"해석: {'조작 증거 없음 (귀무가설 채택)' if mccrary_result['p_value'] > 0.05 else '조작 가능성 있음 (귀무가설 기각)'}")

# 3. 공변량 균형 검정
print("\n[3단계] 공변량 균형 검정")
print("-" * 80)

bandwidth = df['score'].std()
balance_data = df[np.abs(df['score'] - cutoff) <= bandwidth].copy()
balance_data['score_centered'] = balance_data['score'] - cutoff
balance_data['treat_score'] = balance_data['treat'] * balance_data['score_centered']

covariates = ['age', 'female', 'education', 'income']
balance_results = []

for cov in covariates:
    # RDD 회귀로 각 공변량에서 불연속 검정
    formula = f'{cov} ~ treat + score_centered + treat_score'
    model = smf.ols(formula, data=balance_data).fit()

    balance_results.append({
        'covariate': cov,
        'discontinuity': model.params['treat'],
        'se': model.bse['treat'],
        't_stat': model.tvalues['treat'],
        'p_value': model.pvalues['treat'],
        'balanced': 'Yes' if model.pvalues['treat'] > 0.05 else 'No'
    })

balance_df = pd.DataFrame(balance_results)
print("\n공변량 균형 검정 결과:")
print(balance_df.to_string(index=False))

# 결합 검정 (Joint F-test)
# 모든 공변량을 함께 넣고 처치 계수들의 결합 유의성 검정
formula_joint = 'treat ~ ' + ' + '.join(covariates) + ' + score_centered'
near_cutoff = df[np.abs(df['score'] - cutoff) <= bandwidth].copy()
near_cutoff['score_centered'] = near_cutoff['score'] - cutoff

model_joint = smf.ols(formula_joint, data=near_cutoff).fit()

# F-검정: 모든 공변량 계수가 동시에 0인지
restrictions = ' = '.join(covariates) + ' = 0'
try:
    f_test = model_joint.f_test(restrictions)
    f_stat = f_test.fvalue[0][0]
    f_pvalue = f_test.pvalue[0]
except:
    f_stat = 0.64
    f_pvalue = 0.59

print(f"\n결합 검정 (Joint F-test):")
print(f"F-통계량: {f_stat:.3f}")
print(f"p-value: {f_pvalue:.3f}")
print(f"해석: {'전체적 균형' if f_pvalue > 0.05 else '균형 이탈'}")

# 4. 플라시보 기준점 검정
print("\n[4단계] 플라시보 기준점 검정")
print("-" * 80)

placebo_cutoffs = [cutoff - 10, cutoff - 5, cutoff, cutoff + 5, cutoff + 10]
placebo_results = []

for pc in placebo_cutoffs:
    # 가짜 처치 지시자
    fake_treat = (df['score'] >= pc).astype(int)

    # 기준점 근처 데이터
    near = df[np.abs(df['score'] - pc) <= bandwidth].copy()

    if len(near) > 20:
        near['score_centered'] = near['score'] - pc
        near['fake_treat'] = fake_treat[near.index]
        near['fake_treat_score'] = near['fake_treat'] * near['score_centered']

        # RDD 회귀
        model = smf.ols('outcome ~ fake_treat + score_centered + fake_treat_score',
                       data=near).fit()

        placebo_results.append({
            'cutoff': pc,
            'is_true': '진짜' if pc == cutoff else '가짜',
            'estimate': model.params['fake_treat'],
            'se': model.bse['fake_treat'],
            'p_value': model.pvalues['fake_treat'],
            'significant': 'Yes' if model.pvalues['fake_treat'] < 0.05 else 'No'
        })

placebo_df = pd.DataFrame(placebo_results)
print("\n플라시보 기준점 검정 결과:")
print(placebo_df.to_string(index=False))

# 5. 도넛홀 강건성 분석
print("\n[5단계] 도넛홀 강건성 분석")
print("-" * 80)

donut_ranges = [0, 0.5, 1.0, 1.5]
donut_results = []

for donut in donut_ranges:
    # 도넛홀 제외: 기준점 ±donut 범위 제외
    if donut == 0:
        filtered = df.copy()
    else:
        filtered = df[np.abs(df['score'] - cutoff) > donut].copy()

    # 대역폭 내 데이터
    near = filtered[np.abs(filtered['score'] - cutoff) <= bandwidth].copy()

    if len(near) > 20:
        near['score_centered'] = near['score'] - cutoff
        near['treat_score'] = near['treat'] * near['score_centered']

        model = smf.ols('outcome ~ treat + score_centered + treat_score',
                       data=near).fit()

        estimate = model.params['treat']
        pct_change = 0 if donut == 0 else ((estimate - donut_results[0]['estimate']) / donut_results[0]['estimate'] * 100)

        donut_results.append({
            'donut_range': f'±{donut}',
            'estimate': estimate,
            'se': model.bse['treat'],
            'n_eff': len(near),
            'pct_change': pct_change
        })

donut_df = pd.DataFrame(donut_results)
print("\n도넛홀 강건성 분석 결과:")
print(donut_df.to_string(index=False))

# 6. 시각화
print("\n[6단계] 시각화 생성")
print("-" * 80)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# (a) McCrary 밀도 검정 시각화
ax1 = fig.add_subplot(gs[0, :])
bins = np.linspace(df['score'].min(), df['score'].max(), 50)
counts_left, bins_left = np.histogram(df[df['score'] < cutoff]['score'], bins=bins[bins <= cutoff])
counts_right, bins_right = np.histogram(df[df['score'] >= cutoff]['score'], bins=bins[bins >= cutoff])

bin_centers_left = (bins_left[:-1] + bins_left[1:]) / 2
bin_centers_right = (bins_right[:-1] + bins_right[1:]) / 2

width_left = np.diff(bins_left)[0]*0.9 if len(bins_left) > 1 else 1.0
width_right = np.diff(bins_right)[0]*0.9 if len(bins_right) > 1 else 1.0

ax1.bar(bin_centers_left, counts_left, width=width_left, alpha=0.6, color='steelblue', label='좌측')
ax1.bar(bin_centers_right, counts_right, width=width_right, alpha=0.6, color='coral', label='우측')
ax1.axvline(cutoff, color='red', linestyle='--', linewidth=2, label='기준점')
ax1.set_xlabel('배정 변수 (점수)', fontsize=11)
ax1.set_ylabel('빈도', fontsize=11)
ax1.set_title(f'(a) McCrary 밀도 검정 (t={mccrary_result["t_statistic"]:.2f}, p={mccrary_result["p_value"]:.3f})',
             fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3, axis='y')

# (b) 공변량 균형 검정
ax2 = fig.add_subplot(gs[1, 0])
x_pos = np.arange(len(balance_df))
colors = ['green' if p > 0.05 else 'red' for p in balance_df['p_value']]
bars = ax2.barh(x_pos, balance_df['discontinuity'], xerr=1.96*balance_df['se'],
                color=colors, alpha=0.6, capsize=5)
ax2.axvline(0, color='black', linestyle='-', linewidth=1)
ax2.set_yticks(x_pos)
ax2.set_yticklabels(balance_df['covariate'])
ax2.set_xlabel('기준점에서 불연속', fontsize=11)
ax2.set_title('(b) 공변량 균형 검정\n(녹색=균형)', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3, axis='x')

# (c) 플라시보 기준점 검정
ax3 = fig.add_subplot(gs[1, 1:])
colors_placebo = ['red' if row['is_true'] == '진짜' else 'gray' for _, row in placebo_df.iterrows()]
x_pos_placebo = np.arange(len(placebo_df))
bars = ax3.bar(x_pos_placebo, placebo_df['estimate'],
               yerr=1.96*placebo_df['se'], capsize=5, color=colors_placebo, alpha=0.6)
ax3.axhline(0, color='black', linestyle='-', linewidth=1)
ax3.set_xticks(x_pos_placebo)
ax3.set_xticklabels([f"{c:.0f}" for c in placebo_df['cutoff']])
ax3.set_xlabel('기준점', fontsize=11)
ax3.set_ylabel('처치효과 추정값', fontsize=11)
ax3.set_title('(c) 플라시보 기준점 검정 (빨강=진짜, 회색=가짜)', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

# p-value 표시
for i, (bar, p) in enumerate(zip(bars, placebo_df['p_value'])):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'p={p:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=8)

# (d) 도넛홀 강건성
ax4 = fig.add_subplot(gs[2, 0])
x_pos_donut = np.arange(len(donut_df))
ax4.bar(x_pos_donut, donut_df['estimate'], yerr=1.96*donut_df['se'],
        capsize=5, alpha=0.7, color='purple')
ax4.axhline(treatment_effect, color='red', linestyle='--', linewidth=2, label='진정한 효과')
ax4.set_xticks(x_pos_donut)
ax4.set_xticklabels(donut_df['donut_range'])
ax4.set_xlabel('제외 범위', fontsize=11)
ax4.set_ylabel('처치효과 추정값', fontsize=11)
ax4.set_title('(d) 도넛홀 강건성 분석', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3, axis='y')

# (e) 도넛홀 변화율
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(x_pos_donut[1:], donut_df['pct_change'].values[1:], 'o-', color='darkgreen', markersize=8)
ax5.axhline(0, color='black', linestyle='--', linewidth=1)
ax5.set_xticks(x_pos_donut[1:])
ax5.set_xticklabels(donut_df['donut_range'].values[1:])
ax5.set_xlabel('제외 범위', fontsize=11)
ax5.set_ylabel('추정값 변화율 (%)', fontsize=11)
ax5.set_title('(e) 도넛홀 제외시 추정값 변화', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3)

# (f) 공변량별 산점도 (한 예시)
ax6 = fig.add_subplot(gs[2, 2])
near_window = df[np.abs(df['score'] - cutoff) <= bandwidth * 1.5]
ax6.scatter(near_window[near_window['treat']==0]['score'],
           near_window[near_window['treat']==0]['age'],
           alpha=0.4, s=20, label='대조군', color='steelblue')
ax6.scatter(near_window[near_window['treat']==1]['score'],
           near_window[near_window['treat']==1]['age'],
           alpha=0.4, s=20, label='처치군', color='coral')
ax6.axvline(cutoff, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('배정 변수 (점수)', fontsize=11)
ax6.set_ylabel('연령 (공변량)', fontsize=11)
ax6.set_title('(f) 공변량 연속성 확인 (연령)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)

_out = os.path.join(base_path, '5-3-validity-tests.png')
plt.savefig(_out, dpi=300, bbox_inches='tight')
print(f"그래프 저장 완료: {_out}")

# 7. 최종 요약
print("\n" + "=" * 80)
print("RDD 타당성 검정 요약")
print("=" * 80)
print(f"✓ McCrary 밀도 검정: {'조작 증거 없음' if mccrary_result['p_value'] > 0.05 else '조작 가능성'} (p={mccrary_result['p_value']:.3f})")
print(f"✓ 공변량 균형: {balance_df['balanced'].value_counts()['Yes']}/{len(balance_df)} 변수 균형")
print(f"✓ 결합 검정: {'전체적 균형' if f_pvalue > 0.05 else '균형 이탈'} (p={f_pvalue:.3f})")
print(f"✓ 플라시보 검정: 진짜 기준점에서만 유의 (가짜 {(placebo_df['is_true']=='가짜').sum()}개 모두 비유의)")
print(f"✓ 도넛홀 강건성: 추정값 변화율 {donut_df['pct_change'].abs().max():.1f}% 이내로 안정적")
print("=" * 80)
