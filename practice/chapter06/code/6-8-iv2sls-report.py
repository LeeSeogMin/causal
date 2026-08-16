"""
제6장: 표준 도구로 2SLS 돌리고 보고서 표 만들기
================================================

목적: 실습 6-1에서 손으로 나눠 돌린 2SLS를 전용 패키지로 다시 돌린다.
- 추정값은 같지만 표준오차가 다르다는 사실을 확인
- 1단계 진단(부분 R², 부분 F)을 자동으로 받는다
- 제출용 결과표를 CSV로 저장한다

이 실습이 필요한 이유:
  손으로 2단계를 돌리면 1단계에서 D_hat을 "추정했다"는 사실이
  2단계 표준오차에 반영되지 않는다. 그래서 표준오차가 실제보다 작게 나오고,
  신뢰구간을 좁게 보고하게 된다.

저자: AI 기반 정책분석방법론
날짜: 2026-08-17
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/6-1-iv-data.csv')
df = pd.read_csv(data_path)
TRUE_EFFECT = 3.42

print("=" * 80)
print("표준 도구로 2SLS 돌리기")
print("=" * 80)
print(f"데이터: 6-1-iv-data.csv, 표본 {len(df)}개, 참값 {TRUE_EFFECT}")

# ---------------------------------------------------------------------------
# 1. 손으로 나눠 돌린 2SLS (실습 6-1과 같은 방식)
# ---------------------------------------------------------------------------
print("\n[1단계] 손으로 나눠 돌린 2SLS")
print("-" * 80)

first_stage = smf.ols('D ~ Z + X1 + X2', data=df).fit()
df['D_hat'] = first_stage.fittedvalues
manual = smf.ols('Y ~ D_hat + X1 + X2', data=df).fit()

manual_est = manual.params['D_hat']
manual_se = manual.bse['D_hat']
manual_ci = manual.conf_int().loc['D_hat']

print(f"추정값: {manual_est:.4f}")
print(f"표준오차: {manual_se:.4f}")
print(f"95% 신뢰구간: [{manual_ci[0]:.3f}, {manual_ci[1]:.3f}]")

# ---------------------------------------------------------------------------
# 2. 전용 패키지로 한 줄에 돌리기
# ---------------------------------------------------------------------------
print("\n[2단계] linearmodels IV2SLS로 한 줄에 돌리기")
print("-" * 80)
print("공식 표기: 'Y ~ 1 + X1 + X2 + [D ~ Z]'")
print("  대괄호 안이 '내생변수 ~ 도구변수'다. 나머지는 통제변수.")

res = IV2SLS.from_formula('Y ~ 1 + X1 + X2 + [D ~ Z]', data=df).fit(cov_type='robust')

iv_est = res.params['D']
iv_se = res.std_errors['D']
iv_ci = res.conf_int().loc['D']

print(f"\n추정값: {iv_est:.4f}")
print(f"표준오차(robust): {iv_se:.4f}")
print(f"95% 신뢰구간: [{iv_ci['lower']:.3f}, {iv_ci['upper']:.3f}]")

# ---------------------------------------------------------------------------
# 3. 두 방식 대조
# ---------------------------------------------------------------------------
print("\n[3단계] 추정값은 같고 표준오차는 다르다")
print("-" * 80)

print(f"추정값 차이: {abs(manual_est - iv_est):.6f}  (사실상 0)")
print(f"표준오차: 손으로 {manual_se:.4f} vs 패키지 {iv_se:.4f}")
print(f"손으로 계산한 표준오차가 {(1 - manual_se/iv_se)*100:.1f}% 작다")
print(f"신뢰구간 폭: 손으로 {manual_ci[1]-manual_ci[0]:.3f} "
      f"vs 패키지 {iv_ci['upper']-iv_ci['lower']:.3f}")
print("\n이유: 1단계에서 D_hat을 추정했다는 불확실성이")
print("      손으로 돌린 2단계 회귀에는 반영되지 않는다.")
print("결론: 보고서에는 패키지 표준오차를 쓴다.")

# ---------------------------------------------------------------------------
# 4. 1단계 진단을 자동으로 받기
# ---------------------------------------------------------------------------
print("\n[4단계] 1단계 진단 자동 출력")
print("-" * 80)

fs = res.first_stage
print(fs)

# ---------------------------------------------------------------------------
# 5. 제출용 결과표 만들기
# ---------------------------------------------------------------------------
print("\n[5단계] 제출용 결과표")
print("-" * 80)

ols = smf.ols('Y ~ D + X1 + X2', data=df).fit()

table = pd.DataFrame({
    '방법': ['OLS', '2SLS'],
    '추정값': [ols.params['D'], iv_est],
    '표준오차': [ols.bse['D'], iv_se],
    'CI하한': [ols.conf_int().loc['D', 0], iv_ci['lower']],
    'CI상한': [ols.conf_int().loc['D', 1], iv_ci['upper']],
})
table['참값과의차이'] = table['추정값'] - TRUE_EFFECT
table['CI가참값포함'] = np.where(
    (table['CI하한'] <= TRUE_EFFECT) & (TRUE_EFFECT <= table['CI상한']), '예', '아니오')

print(table.round(3).to_string(index=False))

out_csv = os.path.join(base_path, '6-8-results-table.csv')
table.round(4).to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n결과표 저장: {os.path.basename(out_csv)}")
print("엑셀에서 바로 열린다. 보고서 표로 붙여 쓸 수 있다.")

# ---------------------------------------------------------------------------
# 6. 시각화
# ---------------------------------------------------------------------------
print("\n[6단계] 시각화")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) 신뢰구간 비교: 손으로 vs 패키지
ax = axes[0]
ys = [1, 0]
ests = [manual_est, iv_est]
lows = [manual_est - manual_ci[0], iv_est - iv_ci['lower']]
highs = [manual_ci[1] - manual_est, iv_ci['upper'] - iv_est]

ax.errorbar(ests, ys, xerr=[lows, highs], fmt='o', capsize=8,
            markersize=10, color='steelblue', linewidth=2)
ax.axvline(TRUE_EFFECT, color='red', linestyle='--', linewidth=2,
           label=f'참값 = {TRUE_EFFECT}')
ax.set_yticks(ys)
ax.set_yticklabels([f'손으로 나눠 돌림\nSE={manual_se:.3f}',
                    f'IV2SLS 패키지\nSE={iv_se:.3f}'])
ax.set_xlabel('처치효과 추정값')
ax.set_title('(a) 같은 추정값, 다른 신뢰구간', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='x')
ax.set_ylim(-0.6, 1.6)

# (b) OLS vs 2SLS
ax = axes[1]
xs = np.arange(2)
vals = [ols.params['D'], iv_est]
errs = [ols.bse['D'] * 1.96, iv_se * 1.96]
ax.bar(xs, vals, yerr=errs, capsize=6, color=['coral', 'steelblue'], alpha=0.75)
ax.axhline(TRUE_EFFECT, color='red', linestyle='--', linewidth=2,
           label=f'참값 = {TRUE_EFFECT}')
ax.set_xticks(xs)
ax.set_xticklabels(['OLS', '2SLS'])
ax.set_ylabel('처치효과 추정값')
ax.set_title('(b) 제출용 결과표의 내용', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('6-8-iv2sls-report.png', dpi=300, bbox_inches='tight')
print("그래프 저장: 6-8-iv2sls-report.png")

print("\n" + "=" * 80)
print("요약")
print("=" * 80)
print(f"추정값은 손으로 돌리든 패키지로 돌리든 {iv_est:.3f}으로 같다")
print(f"표준오차는 다르다: {manual_se:.4f} -> {iv_se:.4f}")
print("보고서에는 패키지 결과를 쓰고, 어떤 표준오차인지 표 아래에 밝힌다")
print("=" * 80)
