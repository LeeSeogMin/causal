"""
제6장: 교육 수익률 추정 실습
==============================

목적: 도구변수(2SLS)를 실제 정책 문제에 적용하는 완결 사례
- 대학 교육의 임금 프리미엄 추정
- 1단계 진단(F통계량)
- 부모 학력 집단별 효과 + 신뢰구간
- 1단계 계수에 근거한 정책 시뮬레이션

저자: AI 기반 정책분석방법론
날짜: 2026-08-16 (수정)

수정 이력
---------
2026-08-16: 아래 네 가지 문제를 수정.
  1) 도구변수가 너무 약해 추정치가 참값을 못 따라감
     -> 정원 확대의 진학 영향 계수를 0.05에서 0.35로 올리고 중심화
  2) 부모 학력별 효과 차이(0.01/단계)가 임금 잡음(sd 0.15)에 묻혀
     추정 방향이 참값과 반대로 나옴
     -> 효과 차이를 0.10/단계로 키우고 표본을 5,000에서 50,000으로 늘려
        집단별 표준오차(0.02~0.03)가 집단 간 차이(0.10)보다 작아지게 만듦
  3) 2단계 신경망을 예측값으로 학습하고 실제값으로 평가해 외삽 발생
     -> 표준 2SLS로 교체(linearmodels.IV2SLS)
  4) 정책 시뮬레이션의 진학률 반응이 근거 없는 상수(0.05)로 하드코딩
     -> 1단계 회귀에서 추정한 계수로 계산
  5) 요약에서 참값과 어긋난 결론을 단정
     -> 추정 순서와 참값 순서를 대조해 실제 결과만 출력
"""

import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt

np.random.seed(42)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("교육 수익률 추정: 도구변수(2SLS) 적용")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. 데이터 생성 (가상의 한국 대졸자 데이터)
# ---------------------------------------------------------------------------
print("\n[1단계] 데이터 생성")
print("-" * 80)

# 집단별 효과를 구분하려면 집단마다 표본이 충분해야 한다.
# 5,000명이면 집단별 표준오차가 0.09로 집단 간 차이보다 커서 순서를 못 가린다.
n = 50000

age = np.random.normal(28, 3, n)
gender = np.random.binomial(1, 0.5, n)
parent_edu = np.random.choice([1, 2, 3], n, p=[0.4, 0.4, 0.2])  # 1=중졸 2=고졸 3=대졸
region = np.random.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2])

# 도구변수: 지역별 대학 정원 증가율(%)
# 지역마다 평균이 다르고, 같은 지역 안에서도 학교별 사정으로 흩어진다.
college_expansion = (np.random.normal(5, 2, n) * (region == 1) +
                     np.random.normal(3, 1.5, n) * (region == 2) +
                     np.random.normal(1, 1, n) * (region == 3))
expansion_centered = college_expansion - college_expansion.mean()

# 관측 불가능한 능력 (내생성의 원인)
ability = np.random.randn(n)

# 처치: 대학 졸업 여부
# 정원 확대 계수 0.35 -> 도구변수가 진학을 실제로 움직인다
GAMMA_Z = 0.35
college_prob = expit(
    -0.4 + GAMMA_Z * expansion_centered + 0.3 * parent_edu +
    0.5 * ability + 0.01 * age - 0.2 * gender + np.random.randn(n) * 0.2
)
college = np.random.binomial(1, college_prob)

# 결과: 로그 임금
# 참값 효과: 고졸 가정 기준 0.15, 부모 학력 한 단계당 0.10씩 커짐
# 중졸 가정 0.05 / 고졸 가정 0.15 / 대졸 가정 0.25
true_premium = 0.15 + 0.10 * (parent_edu - 2)

log_wage = (2.5 + 0.05 * age - 0.15 * gender + 0.2 * parent_edu + 0.3 * ability +
            college * true_premium + np.random.randn(n) * 0.15)

df = pd.DataFrame({
    'log_wage': log_wage,
    'college': college,
    'expansion': expansion_centered,
    'age': age,
    'gender': gender,
    'parent_edu': parent_edu,
    'region': region,
})

print(f"샘플 크기: {n}")
print(f"대졸률: {college.mean():.1%}")
print(f"평균 대졸 프리미엄 (참값): {true_premium.mean():.3f} "
      f"(임금 {(np.exp(true_premium.mean())-1)*100:.1f}% 증가)")
print("부모 학력별 참값:")
for lv, name in [(1, '중졸 이하'), (2, '고졸'), (3, '대졸 이상')]:
    print(f"  {name}: {true_premium[parent_edu == lv].mean():.3f}")

# ---------------------------------------------------------------------------
# 2. OLS (편향된 기준선)
# ---------------------------------------------------------------------------
print("\n[2단계] OLS 추정 (능력 편향이 남아 있는 기준선)")
print("-" * 80)

ols = smf.ols('log_wage ~ college + age + gender + parent_edu + C(region)', data=df).fit()
ols_effect = ols.params['college']

print(f"OLS 대졸 프리미엄: {ols_effect:.3f} "
      f"(임금 {(np.exp(ols_effect)-1)*100:.1f}% 증가)")
print(f"참값 {true_premium.mean():.3f} 대비: "
      f"{(ols_effect - true_premium.mean())/true_premium.mean()*100:+.1f}%")

# ---------------------------------------------------------------------------
# 3. 1단계 진단
# ---------------------------------------------------------------------------
print("\n[3단계] 1단계 진단 (도구변수가 진학을 움직이는가)")
print("-" * 80)

first_stage = smf.ols('college ~ expansion + age + gender + parent_edu + C(region)',
                      data=df).fit()
pi_z = first_stage.params['expansion']
t_z = first_stage.tvalues['expansion']
f_z = t_z ** 2

print(f"정원 증가율 계수: {pi_z:.4f} (t = {t_z:.2f})")
print(f"1단계 F통계량: {f_z:.2f}")
print("해석: 정원 증가율이 1%p 높은 지역에서 대졸 확률이 "
      f"{pi_z*100:.2f}%p 높다")

if f_z > 10:
    print("판정: 강한 도구변수 (F > 10)")
else:
    print("판정: 약한 도구변수 위험 (F < 10) - 추정치를 신뢰하지 않는다")

# ---------------------------------------------------------------------------
# 4. 2SLS 추정
# ---------------------------------------------------------------------------
print("\n[4단계] 2SLS 추정")
print("-" * 80)

iv_res = IV2SLS.from_formula(
    'log_wage ~ 1 + age + gender + parent_edu + C(region) + [college ~ expansion]',
    data=df).fit(cov_type='robust')

iv_effect = iv_res.params['college']
iv_se = iv_res.std_errors['college']
iv_ci = iv_res.conf_int().loc['college']

print(f"2SLS 대졸 프리미엄: {iv_effect:.3f} (SE: {iv_se:.3f})")
print(f"95% 신뢰구간: [{iv_ci['lower']:.3f}, {iv_ci['upper']:.3f}]")
print(f"임금 증가율: {(np.exp(iv_effect)-1)*100:.1f}%")
print(f"참값 {true_premium.mean():.3f} 대비: "
      f"{(iv_effect - true_premium.mean())/true_premium.mean()*100:+.1f}%")
print(f"신뢰구간이 참값을 포함하는가: "
      f"{'예' if iv_ci['lower'] <= true_premium.mean() <= iv_ci['upper'] else '아니오'}")

# ---------------------------------------------------------------------------
# 5. 부모 학력 집단별 효과 (집단별 1단계 진단 포함)
# ---------------------------------------------------------------------------
print("\n[5단계] 부모 학력 집단별 효과")
print("-" * 80)
print(f"{'집단':<10} {'1단계 F':>9} {'추정값':>9} {'95% 신뢰구간':>20} {'참값':>8}")
print("-" * 80)

group_results = []
for lv, name in [(1, '중졸 이하'), (2, '고졸'), (3, '대졸 이상')]:
    sub = df[df.parent_edu == lv]

    fs_sub = smf.ols('college ~ expansion + age + gender + C(region)', data=sub).fit()
    f_sub = fs_sub.tvalues['expansion'] ** 2

    iv_sub = IV2SLS.from_formula(
        'log_wage ~ 1 + age + gender + C(region) + [college ~ expansion]',
        data=sub).fit(cov_type='robust')

    est = iv_sub.params['college']
    ci = iv_sub.conf_int().loc['college']
    truth = true_premium[parent_edu == lv].mean()

    group_results.append({
        'level': lv, 'name': name, 'f': f_sub, 'est': est,
        'lo': ci['lower'], 'hi': ci['upper'], 'truth': truth,
        'n': len(sub),
    })

    flag = '' if f_sub > 10 else '  <- 약한 도구변수'
    print(f"{name:<10} {f_sub:>9.1f} {est:>9.3f} "
          f"[{ci['lower']:>7.3f}, {ci['upper']:>6.3f}] {truth:>8.3f}{flag}")

print("-" * 80)
est_order = [g['est'] for g in group_results]
truth_order = [g['truth'] for g in group_results]
if np.argsort(est_order).tolist() == np.argsort(truth_order).tolist():
    print("추정 순서가 참값 순서와 일치한다")
else:
    print("추정 순서가 참값 순서와 어긋난다 - 신뢰구간 폭을 함께 볼 것")

# ---------------------------------------------------------------------------
# 6. 정책 시뮬레이션 (1단계 계수에 근거)
# ---------------------------------------------------------------------------
policy_shock = 1.0  # 정원 증가율을 1%p 인상

print(f"\n[6단계] 정책 시뮬레이션: 정원 증가율을 {policy_shock:.0f}%p 올리면")
print("-" * 80)

delta_rate = pi_z * policy_shock                 # 1단계 계수에서 유도한 진학률 변화
delta_log_wage = iv_effect * delta_rate          # 2SLS 추정치와 곱한 평균 임금 변화

print(f"현재 대졸률: {college.mean():.1%}")
print(f"정원 증가율 +{policy_shock:.0f}%p 적용 시:")

print(f"  - 대졸률 변화: +{delta_rate*100:.1f}%p "
      f"(1단계 계수 {pi_z:.4f} x {policy_shock:.0f})")
print(f"  - 예상 대졸률: {(college.mean()+delta_rate):.1%}")
print(f"  - 평균 임금 변화: +{(np.exp(delta_log_wage)-1)*100:.2f}%")
print("\n계산 구조: 평균 임금 변화 = (늘어난 대졸률) x (대졸 프리미엄)")
print(f"           = {delta_rate:.3f} x {iv_effect:.3f} = {delta_log_wage:.4f}")
print("정책 효과는 '효과 크기'와 '실제로 영향받는 사람 수'의 곱이다.")

# ---------------------------------------------------------------------------
# 7. 시각화
# ---------------------------------------------------------------------------
print("\n[7단계] 시각화")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) OLS vs 2SLS vs 참값
ax = axes[0]
labels = ['참값', '2SLS', 'OLS']
values = [true_premium.mean(), iv_effect, ols_effect]
errors = [0, iv_se * 1.96, ols.bse['college'] * 1.96]
ax.bar(np.arange(3), values, yerr=errors, capsize=5,
       color=['green', 'steelblue', 'coral'], alpha=0.75)
ax.axhline(true_premium.mean(), color='red', linestyle='--', linewidth=2)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(labels)
ax.set_ylabel('대졸 프리미엄 (로그 임금)')
ax.set_title('(a) OLS는 능력 편향으로 과대추정', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (b) 부모 학력별 효과: 추정값과 참값
ax = axes[1]
xs = np.arange(3)
ests = [g['est'] for g in group_results]
los = [g['est'] - g['lo'] for g in group_results]
his = [g['hi'] - g['est'] for g in group_results]
truths = [g['truth'] for g in group_results]

ax.errorbar(xs, ests, yerr=[los, his], fmt='o', capsize=6, markersize=9,
            color='steelblue', label='2SLS 추정값 (95% CI)')
ax.plot(xs, truths, 'r^--', markersize=10, label='참값')
ax.set_xticks(xs)
ax.set_xticklabels(['중졸\n이하', '고졸', '대졸\n이상'])
ax.set_ylabel('대졸 프리미엄 (로그 임금)')
ax.set_title('(b) 부모 학력별 효과', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (c) 집단별 1단계 F
ax = axes[2]
fs = [g['f'] for g in group_results]
bar_colors = ['steelblue' if f > 10 else 'coral' for f in fs]
ax.bar(xs, fs, color=bar_colors, alpha=0.75)
ax.axhline(10, color='red', linestyle='--', linewidth=2, label='기준선 F=10')
ax.set_xticks(xs)
ax.set_xticklabels(['중졸\n이하', '고졸', '대졸\n이상'])
ax.set_ylabel('1단계 F통계량')
ax.set_title('(c) 집단별 도구변수 강도', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('6-7-education-returns.png', dpi=300, bbox_inches='tight')
print("그래프 저장: 6-7-education-returns.png")

# ---------------------------------------------------------------------------
# 8. 요약
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("교육 수익률 분석 요약")
print("=" * 80)
print(f"1단계 F통계량: {f_z:.1f} (강한 도구변수)")
print(f"OLS: {ols_effect:.3f} / 2SLS: {iv_effect:.3f} / 참값: {true_premium.mean():.3f}")
print(f"OLS가 {(ols_effect - true_premium.mean())/true_premium.mean()*100:+.0f}% "
      f"과대추정한 부분이 능력 편향이다")
print("부모 학력별 추정값: " +
      " / ".join(f"{g['name']} {g['est']:.3f}" for g in group_results))
lo3, hi1 = group_results[2]['lo'], group_results[0]['hi']
if lo3 > hi1:
    print("중졸 가정과 대졸 가정의 신뢰구간이 겹치지 않는다 - 차이를 말할 수 있다")
else:
    print("중졸 가정과 대졸 가정의 신뢰구간이 겹친다 - 차이를 단정할 수 없다")
print("\n주의: 가상 데이터 기반 교육용 예제다. 실제 정책 판단에 쓰지 않는다.")
print("=" * 80)
