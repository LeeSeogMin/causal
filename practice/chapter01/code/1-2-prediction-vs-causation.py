"""
제1장 실습 2: 예측과 인과는 다른 질문에 답한다
================================================

목적: 잘 맞히는 예측 모형이 "무엇을 해야 하는가"에는 답하지 못한다는 것을 숫자로 확인한다.

상황: 온라인 쇼핑몰이 일부 고객에게 할인 쿠폰을 보냈다.
      쿠폰을 받은 고객의 구매액이 훨씬 높다. 쿠폰 덕분인가?

데이터는 이 스크립트가 직접 만든다. 참 효과를 알고 있어야
추정값이 맞았는지 틀렸는지 확인할 수 있기 때문이다.

저자: AI 기반 정책분석방법론
날짜: 2026-08-17
"""

import os
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
base_path = os.path.dirname(os.path.abspath(__file__))

print("=" * 78)
print("예측과 인과는 다른 질문에 답한다")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1. 데이터 만들기
# ---------------------------------------------------------------------------
print("\n[1단계] 데이터 만들기")
print("-" * 78)

n = 5000

# 관측되지 않는 원인: 이 쇼핑몰에 대한 애착(충성도)
loyalty = np.random.normal(0, 1, n)

# 관측되는 변수
age = np.random.normal(38, 10, n)
months = np.random.poisson(24, n)                       # 가입 기간(개월)
app_open = np.round(8 + 6 * loyalty + np.random.normal(0, 2, n)).clip(0)  # 월 앱 접속 횟수

# 쿠폰 수령 여부: 앱을 자주 여는 고객이 쿠폰을 더 잘 받는다
coupon_prob = expit(-0.5 + 0.9 * loyalty + 0.01 * (age - 38))
coupon = np.random.binomial(1, coupon_prob)

# 구매액: 쿠폰의 참 효과는 3,000원이다
TRUE_EFFECT = 3000
purchase = (40000
            + TRUE_EFFECT * coupon
            + 12000 * loyalty
            + 200 * months
            + 150 * (age - 38)
            + np.random.normal(0, 8000, n))

df = pd.DataFrame({
    'purchase': purchase.round(0),
    'coupon': coupon,
    'age': age.round(1),
    'months': months,
    'app_open': app_open,
    'loyalty': loyalty.round(3),      # 실제로는 관측 불가. 확인용으로만 저장한다
})
df.to_csv(os.path.join(base_path, '../data/1-2-coupon-observational.csv'),
          index=False, encoding='utf-8-sig')

print(f"고객 수: {n:,}명")
print(f"쿠폰 수령률: {coupon.mean():.1%}")
print(f"쿠폰의 참 효과: {TRUE_EFFECT:,}원  (데이터를 만들 때 넣은 값)")
print("관측되는 변수: 구매액, 쿠폰 수령 여부, 나이, 가입기간, 월 앱 접속 횟수")
print("관측되지 않는 변수: 충성도")

# ---------------------------------------------------------------------------
# 2. 단순 비교
# ---------------------------------------------------------------------------
print("\n[2단계] 쿠폰 받은 고객과 안 받은 고객을 그냥 비교하면")
print("-" * 78)

mean_yes = df.loc[df.coupon == 1, 'purchase'].mean()
mean_no = df.loc[df.coupon == 0, 'purchase'].mean()
naive_gap = mean_yes - mean_no

print(f"쿠폰 받은 고객 평균 구매액:   {mean_yes:>10,.0f}원  (n={int(coupon.sum()):,})")
print(f"쿠폰 안 받은 고객 평균 구매액: {mean_no:>10,.0f}원  (n={int(n-coupon.sum()):,})")
print(f"차이:                        {naive_gap:>10,.0f}원")
print(f"\n참 효과 {TRUE_EFFECT:,}원의 {naive_gap/TRUE_EFFECT:.1f}배로 부풀려졌다.")

# ---------------------------------------------------------------------------
# 3. 예측 모형
# ---------------------------------------------------------------------------
print("\n[3단계] 예측 모형을 학습시키면")
print("-" * 78)

X = df[['coupon', 'age', 'months', 'app_open']]
y = df['purchase']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
r2 = r2_score(y_te, rf.predict(X_te))

print(f"테스트셋 결정계수(R²): {r2:.3f}")
print("구매액을 상당히 잘 맞힌다.")

print("\n변수 중요도:")
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
names = {'coupon': '쿠폰 수령', 'age': '나이', 'months': '가입기간', 'app_open': '앱 접속'}
for k, v in imp.items():
    print(f"  {names[k]:<8} {v:.3f}")

print("\n예측 모형이 답하는 질문: '이 고객이 얼마를 살 것인가'")
print("예측 모형이 답하지 못하는 질문: '쿠폰을 보내면 얼마가 늘어나는가'")

# ---------------------------------------------------------------------------
# 4. 왜 어긋났는가
# ---------------------------------------------------------------------------
print("\n[4단계] 왜 어긋났는가")
print("-" * 78)

loy_yes = df.loc[df.coupon == 1, 'loyalty'].mean()
loy_no = df.loc[df.coupon == 0, 'loyalty'].mean()
app_yes = df.loc[df.coupon == 1, 'app_open'].mean()
app_no = df.loc[df.coupon == 0, 'app_open'].mean()

print("두 집단은 쿠폰 말고도 다른 점이 있다.")
print(f"  충성도 평균     쿠폰 받음 {loy_yes:>6.2f}  vs  안 받음 {loy_no:>6.2f}")
print(f"  월 앱 접속 횟수  쿠폰 받음 {app_yes:>6.1f}  vs  안 받음 {app_no:>6.1f}")
print("\n쿠폰을 받은 고객은 원래부터 이 쇼핑몰을 자주 쓰던 고객이다.")
print(f"단순 비교값 {naive_gap:,.0f}원 안에는 쿠폰 효과와 충성도 차이가 섞여 있다.")

# ---------------------------------------------------------------------------
# 5. 시각화
# ---------------------------------------------------------------------------
print("\n[5단계] 시각화")
print("-" * 78)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
ax.bar([0, 1], [mean_no, mean_yes], color=['lightgray', 'steelblue'], alpha=0.85)
ax.set_xticks([0, 1])
ax.set_xticklabels(['쿠폰 안 받음', '쿠폰 받음'])
ax.set_ylabel('평균 구매액 (원)')
ax.set_title(f'(a) 단순 비교: 차이 {naive_gap:,.0f}원', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

ax = axes[1]
ax.hist(df.loc[df.coupon == 0, 'loyalty'], bins=40, alpha=0.6,
        label='쿠폰 안 받음', color='lightgray')
ax.hist(df.loc[df.coupon == 1, 'loyalty'], bins=40, alpha=0.6,
        label='쿠폰 받음', color='steelblue')
ax.set_xlabel('충성도 (관측 불가)')
ax.set_ylabel('고객 수')
ax.set_title('(b) 두 집단은 애초에 다르다', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

ax = axes[2]
ax.bar([0, 1], [naive_gap, TRUE_EFFECT], color=['coral', 'green'], alpha=0.85)
ax.set_xticks([0, 1])
ax.set_xticklabels(['단순 비교', '참 효과'])
ax.set_ylabel('쿠폰 효과 추정 (원)')
ax.set_title('(c) 단순 비교는 과대추정', fontsize=12, fontweight='bold')
for i, v in enumerate([naive_gap, TRUE_EFFECT]):
    ax.text(i, v + 300, f'{v:,.0f}', ha='center', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(base_path, '1-2-prediction-vs-causation.png'),
            dpi=150, bbox_inches='tight')
print("그래프 저장: 1-2-prediction-vs-causation.png")

print("\n" + "=" * 78)
print("요약")
print("=" * 78)
print(f"예측 성능 R² = {r2:.3f} 으로 구매액은 잘 맞힌다")
print(f"그런데 쿠폰 효과는 {naive_gap:,.0f}원으로 나와, 참값 {TRUE_EFFECT:,}원의 "
      f"{naive_gap/TRUE_EFFECT:.1f}배다")
print("잘 맞히는 모형과 옳은 개입 판단은 별개다")
print("=" * 78)
