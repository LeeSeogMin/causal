"""
제1장 실습 3: 무작위 배정이 답을 바꾼다
========================================

목적: 같은 모집단, 같은 분석 방법인데 데이터를 어떻게 모았느냐에 따라
      결론이 달라지는 것을 확인하고, 그 차이가 얼마짜리인지 계산한다.

두 가지 데이터를 만든다.
  (가) 관찰 데이터: 고객이 스스로 쿠폰을 받아 갔다 (실습 2와 같은 구조)
  (나) 실험 데이터: 동전을 던져 무작위로 쿠폰을 배정했다

저자: AI 기반 정책분석방법론
날짜: 2026-08-17
"""

import os
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
base_path = os.path.dirname(os.path.abspath(__file__))

TRUE_EFFECT = 3000
COUPON_COST = 5000        # 쿠폰 1장을 발행하는 데 드는 비용
N_CUSTOMERS = 1_000_000   # 전체 고객 수

print("=" * 78)
print("무작위 배정이 답을 바꾼다")
print("=" * 78)


def make_customers(n, seed):
    """같은 모집단에서 고객을 뽑는다. 쿠폰 배정 방식만 나중에 달라진다."""
    rng = np.random.default_rng(seed)
    loyalty = rng.normal(0, 1, n)
    age = rng.normal(38, 10, n)
    months = rng.poisson(24, n)
    noise = rng.normal(0, 8000, n)
    return loyalty, age, months, noise, rng


def purchase_amount(coupon, loyalty, age, months, noise):
    return (40000 + TRUE_EFFECT * coupon + 12000 * loyalty
            + 200 * months + 150 * (age - 38) + noise)


# ---------------------------------------------------------------------------
# 1. 관찰 데이터
# ---------------------------------------------------------------------------
print("\n[1단계] 관찰 데이터: 고객이 스스로 쿠폰을 받아 갔다")
print("-" * 78)

n = 5000
loy_o, age_o, mon_o, noi_o, rng_o = make_customers(n, 1)
p = expit(-0.5 + 0.9 * loy_o + 0.01 * (age_o - 38))
coupon_o = rng_o.binomial(1, p)
y_o = purchase_amount(coupon_o, loy_o, age_o, mon_o, noi_o)

obs = pd.DataFrame({'purchase': y_o.round(0), 'coupon': coupon_o,
                    'loyalty': loy_o.round(3), 'age': age_o.round(1), 'months': mon_o})
obs.to_csv(os.path.join(base_path, '../data/1-3-observational.csv'),
           index=False, encoding='utf-8-sig')

obs_gap = y_o[coupon_o == 1].mean() - y_o[coupon_o == 0].mean()
obs_loy_gap = loy_o[coupon_o == 1].mean() - loy_o[coupon_o == 0].mean()

print(f"쿠폰 수령률: {coupon_o.mean():.1%}")
print(f"두 집단 충성도 차이: {obs_loy_gap:+.2f}  (0에서 멀수록 집단이 다르다)")
print(f"평균 구매액 차이: {obs_gap:,.0f}원")

# ---------------------------------------------------------------------------
# 2. 실험 데이터
# ---------------------------------------------------------------------------
print("\n[2단계] 실험 데이터: 동전을 던져 무작위로 배정했다")
print("-" * 78)

loy_e, age_e, mon_e, noi_e, rng_e = make_customers(n, 2)
coupon_e = rng_e.binomial(1, 0.5, n)          # 충성도와 무관하게 배정
y_e = purchase_amount(coupon_e, loy_e, age_e, mon_e, noi_e)

exp = pd.DataFrame({'purchase': y_e.round(0), 'coupon': coupon_e,
                    'loyalty': loy_e.round(3), 'age': age_e.round(1), 'months': mon_e})
exp.to_csv(os.path.join(base_path, '../data/1-3-experimental.csv'),
           index=False, encoding='utf-8-sig')

exp_gap = y_e[coupon_e == 1].mean() - y_e[coupon_e == 0].mean()
exp_loy_gap = loy_e[coupon_e == 1].mean() - loy_e[coupon_e == 0].mean()

t, pval = stats.ttest_ind(y_e[coupon_e == 1], y_e[coupon_e == 0])
se = np.sqrt(y_e[coupon_e == 1].var(ddof=1) / (coupon_e == 1).sum()
             + y_e[coupon_e == 0].var(ddof=1) / (coupon_e == 0).sum())
lo, hi = exp_gap - 1.96 * se, exp_gap + 1.96 * se

print(f"쿠폰 수령률: {coupon_e.mean():.1%}")
print(f"두 집단 충성도 차이: {exp_loy_gap:+.2f}  (무작위 배정이라 0에 가깝다)")
print(f"평균 구매액 차이: {exp_gap:,.0f}원")
print(f"95% 신뢰구간: [{lo:,.0f}원, {hi:,.0f}원]")
print(f"신뢰구간이 참값 {TRUE_EFFECT:,}원을 포함하는가: "
      f"{'예' if lo <= TRUE_EFFECT <= hi else '아니오'}")

# ---------------------------------------------------------------------------
# 3. 두 결과 대조
# ---------------------------------------------------------------------------
print("\n[3단계] 같은 계산, 다른 답")
print("-" * 78)

table = pd.DataFrame({
    '데이터': ['관찰', '실험'],
    '쿠폰배정': ['고객이 선택', '동전 던지기'],
    '충성도차이': [round(obs_loy_gap, 2), round(exp_loy_gap, 2)],
    '추정효과': [round(obs_gap), round(exp_gap)],
    '참값대비': [f'{obs_gap/TRUE_EFFECT:.1f}배', f'{exp_gap/TRUE_EFFECT:.1f}배'],
})
print(table.to_string(index=False))
print(f"\n참 효과: {TRUE_EFFECT:,}원")
print("두 데이터에 같은 계산(평균 차이)을 적용했는데 답이 다르다.")
print("차이를 만든 것은 계산 방법이 아니라 쿠폰을 누구에게 줬는가다.")

# ---------------------------------------------------------------------------
# 4. 이 차이는 얼마짜리인가
# ---------------------------------------------------------------------------
print("\n[4단계] 이 차이는 얼마짜리인가")
print("-" * 78)

print(f"쿠폰 1장 비용: {COUPON_COST:,}원")
print(f"전체 고객: {N_CUSTOMERS:,}명\n")

for label, est in [('관찰 데이터를 믿으면', obs_gap), ('실제로는', TRUE_EFFECT)]:
    margin = est - COUPON_COST
    total = margin * N_CUSTOMERS
    verdict = '발행한다' if margin > 0 else '발행하지 않는다'
    print(f"{label}")
    print(f"  쿠폰 1장당 순이익: {est:,.0f} - {COUPON_COST:,} = {margin:,.0f}원 → {verdict}")
    print(f"  전 고객 발행 시:   {total/1e8:,.1f}억원")

loss = (TRUE_EFFECT - COUPON_COST) * N_CUSTOMERS
print(f"\n관찰 데이터를 믿고 100만 장을 발행하면 실제로는 "
      f"{abs(loss)/1e8:,.1f}억원 손실이다.")
print("예측을 잘하는 것과 옳은 결정을 내리는 것은 다른 문제다.")

# ---------------------------------------------------------------------------
# 5. 시각화
# ---------------------------------------------------------------------------
print("\n[5단계] 시각화")
print("-" * 78)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
ax.bar([0, 1], [obs_loy_gap, exp_loy_gap], color=['coral', 'steelblue'], alpha=0.85)
ax.axhline(0, color='black', linewidth=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(['관찰 데이터', '실험 데이터'])
ax.set_ylabel('두 집단의 충성도 차이')
ax.set_title('(a) 배정 방식이 집단을 가른다', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

ax = axes[1]
ax.bar([0, 1, 2], [obs_gap, exp_gap, TRUE_EFFECT],
       color=['coral', 'steelblue', 'green'], alpha=0.85)
ax.axhline(TRUE_EFFECT, color='red', linestyle='--', linewidth=2,
           label=f'참값 {TRUE_EFFECT:,}원')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['관찰', '실험', '참값'])
ax.set_ylabel('쿠폰 효과 추정 (원)')
ax.set_title('(b) 같은 계산, 다른 답', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

ax = axes[2]
margins = [(obs_gap - COUPON_COST) * N_CUSTOMERS / 1e8,
           (TRUE_EFFECT - COUPON_COST) * N_CUSTOMERS / 1e8]
colors = ['coral' if m < 0 else 'steelblue' for m in margins]
ax.bar([0, 1], margins, color=['steelblue', 'coral'], alpha=0.85)
ax.axhline(0, color='black', linewidth=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(['관찰 데이터를\n믿었을 때 기대', '실제 결과'])
ax.set_ylabel('100만 장 발행 손익 (억원)')
ax.set_title('(c) 틀린 추정치의 값', fontsize=12, fontweight='bold')
for i, v in enumerate(margins):
    ax.text(i, v + (2 if v > 0 else -4), f'{v:,.0f}억', ha='center', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(base_path, '1-3-randomization.png'),
            dpi=150, bbox_inches='tight')
print("그래프 저장: 1-3-randomization.png")

print("\n" + "=" * 78)
print("요약")
print("=" * 78)
print(f"관찰 데이터: {obs_gap:,.0f}원  /  실험 데이터: {exp_gap:,.0f}원  /  "
      f"참값: {TRUE_EFFECT:,}원")
print("무작위 배정은 두 집단을 처치 말고는 같게 만든다")
print("무작위 배정이 불가능할 때 쓰는 방법이 2장부터 배울 내용이다")
print("=" * 78)
