"""
제8장 실습 1: 고정 배분 A/B 테스트와 표본 크기 계산
====================================================

목적
- 고정 1:1 배분 A/B 테스트를 4가지 시나리오에서 돌린다.
- 각 시나리오에 필요한 표본 크기를 직접 계산한다.
- n=100/팔에서 실제로 얻는 검정력을 계산한다.

데이터: practice/chapter08/data/8-1-fixed-rct.csv
출력 그림: 8-1-fixed-rct-simulation.png

수정 이력
- 2026-08-17
  (1) [근본 원인] 3단계의 "50명으로 0.90 검정력 달성 가능", "150명 과잉",
      "총 비효율 350명"이 계산 결과가 아니라 코드에 박아 넣은 문장이었다.
      검정력 함수가 코드에 없어서 검증할 방법도 없었다.
      → 두 비율 비교의 표본 크기 공식과 검정력 공식을 직접 구현하고,
        모든 숫자를 계산 결과로 바꿨다. statsmodels 값과 교차 확인한다.
  (2) [근본 원인] 그림 출력이 없어 배분·검정력·표본 크기의 관계를 볼 수 없었다.
      → 4개 패널 PNG를 저장한다.
  (3) stats.ttest_ind가 등분산을 가정하고 있었다. 두 팔의 성공률이 다르면
      분산도 다르므로 equal_var=False(Welch)로 바꿨다.

저자: AI 기반 정책분석방법론
날짜: 2025-11-20 (최종 수정 2026-08-17)
"""

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

ALPHA = 0.05
POWER_TARGET = 0.80

print("=" * 80)
print("실습 1. 고정 배분 A/B 테스트와 표본 크기 계산")
print("=" * 80)


# ---------------------------------------------------------------------------
# 표본 크기 / 검정력 공식 (두 비율 비교, 양측검정, 1:1 배분)
# ---------------------------------------------------------------------------
def required_n_per_arm(p1, p2, alpha=ALPHA, power=POWER_TARGET):
    """팔 하나에 필요한 표본 수. 정규근사 plug-in 공식."""
    delta = abs(p2 - p1)
    if delta == 0:
        return np.inf
    p_bar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    term = (z_a * np.sqrt(2 * p_bar * (1 - p_bar))
            + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
    return term ** 2 / delta ** 2


def achieved_power(p1, p2, n_per_arm, alpha=ALPHA):
    """팔당 n명일 때 실제로 얻는 검정력."""
    delta = abs(p2 - p1)
    p_bar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    sd_null = np.sqrt(2 * p_bar * (1 - p_bar))
    sd_alt = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return stats.norm.cdf((delta * np.sqrt(n_per_arm) - z_a * sd_null) / sd_alt)


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
base_dir = pathlib.Path(__file__).resolve().parents[1]
data_path = base_dir / 'data' / '8-1-fixed-rct.csv'
df = pd.read_csv(data_path)

print(f"\n[0단계] 데이터 로드")
print("-" * 80)
print(f"  파일: {data_path.name}")
print(f"  총 참가자 수: {len(df)}명")
print(f"  시나리오 수: {df['scenario'].nunique()}개")

scenario_names = {
    'scenario1': '시나리오 1 (소효과)',
    'scenario2': '시나리오 2 (중효과)',
    'scenario3': '시나리오 3 (대효과)',
    'scenario4': '시나리오 4 (무효과)',
}

# ---------------------------------------------------------------------------
# 1단계: 고정 1:1 배분 결과
# ---------------------------------------------------------------------------
print("\n[1단계] 고정 1:1 배분 결과")
print("-" * 80)

results = []
for sid, sname in scenario_names.items():
    d = df[df['scenario'] == sid]
    y_c = d[d['treatment'] == 0]['outcome'].values
    y_t = d[d['treatment'] == 1]['outcome'].values

    ate = y_t.mean() - y_c.mean()
    _, p_value = stats.ttest_ind(y_t, y_c, equal_var=False)

    p_c_true = d['p_control_true'].iloc[0]
    p_t_true = d['p_treatment_true'].iloc[0]

    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    ci_lo, ci_hi = ate - 1.96 * se, ate + 1.96 * se

    results.append({
        'scenario': sname,
        'p_control_true': p_c_true,
        'p_treatment_true': p_t_true,
        'n_per_arm': len(y_c),
        'obs_control': y_c.mean(),
        'obs_treatment': y_t.mean(),
        'ate': ate,
        'se': se,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'p_value': p_value,
        'significant': p_value < ALPHA,
    })

    print(f"\n{sname}")
    print(f"  참 성공률   : 대조군 {p_c_true:.2f}, 처치군 {p_t_true:.2f} "
          f"(참 효과 {p_t_true - p_c_true:+.2f})")
    print(f"  배분        : 각 팔 {len(y_c)}명")
    print(f"  관측 성공률 : 대조군 {y_c.mean():.2f}, 처치군 {y_t.mean():.2f}")
    print(f"  관측 효과   : {ate:+.3f}  95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"  p값         : {p_value:.4f}  →  {'유의함' if p_value < ALPHA else '유의하지 않음'}")

res = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 2단계: 표본 크기 계산
# ---------------------------------------------------------------------------
print("\n[2단계] 각 시나리오에 필요한 표본 크기 (검정력 0.80, 양측 α=0.05)")
print("-" * 80)

z_a = stats.norm.ppf(1 - ALPHA / 2)
z_b = stats.norm.ppf(POWER_TARGET)
print(f"  z(1-α/2) = {z_a:.4f},  z(power) = {z_b:.4f}")
print()
print(f"  {'시나리오':<20}{'참 효과':>8}{'필요 n/팔':>11}{'필요 총원':>10}"
      f"{'실제 n/팔':>11}{'과부족':>10}")

need_list = []
for r in results:
    p1, p2 = r['p_control_true'], r['p_treatment_true']
    n_req = required_n_per_arm(p1, p2)
    need_list.append(n_req)
    if np.isinf(n_req):
        print(f"  {r['scenario']:<20}{p2 - p1:>+8.2f}{'무한':>11}{'무한':>10}"
              f"{r['n_per_arm']:>11}{'-':>10}")
    else:
        n_req_c = int(np.ceil(n_req))
        gap = r['n_per_arm'] - n_req_c
        print(f"  {r['scenario']:<20}{p2 - p1:>+8.2f}{n_req_c:>11}{n_req_c * 2:>10}"
              f"{r['n_per_arm']:>11}{gap:>+10}")

res['n_required'] = need_list

print("\n  [손계산 확인] 시나리오 3 (0.40 → 0.70)")
p1, p2 = 0.40, 0.70
p_bar = (p1 + p2) / 2
a = z_a * np.sqrt(2 * p_bar * (1 - p_bar))
b = z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
print(f"    p_bar = (0.40+0.70)/2 = {p_bar:.3f}")
print(f"    {z_a:.3f} x sqrt(2 x {p_bar:.3f} x {1-p_bar:.3f}) = {a:.4f}")
print(f"    {z_b:.3f} x sqrt(0.40x0.60 + 0.70x0.30)  = {b:.4f}")
print(f"    n/팔 = ({a:.4f} + {b:.4f})^2 / {(p2-p1)**2:.4f} = {required_n_per_arm(p1, p2):.2f}")

try:
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    n_sm = NormalIndPower().solve_power(
        effect_size=proportion_effectsize(p2, p1),
        power=POWER_TARGET, alpha=ALPHA, ratio=1)
    print(f"    statsmodels 교차 확인: {n_sm:.2f}")
except Exception as exc:  # pragma: no cover
    print(f"    statsmodels 교차 확인 생략 ({exc})")

# ---------------------------------------------------------------------------
# 3단계: n=100/팔에서 실제로 얻는 검정력
# ---------------------------------------------------------------------------
print("\n[3단계] 각 팔 100명일 때 실제 검정력")
print("-" * 80)
print(f"  {'시나리오':<20}{'참 효과':>8}{'검정력':>9}{'판정':>26}")

pow_list = []
for r in results:
    p1, p2 = r['p_control_true'], r['p_treatment_true']
    if p1 == p2:
        pw = ALPHA
        verdict = '효과 없음 (이 값은 1종 오류율)'
    else:
        pw = achieved_power(p1, p2, r['n_per_arm'])
        verdict = '충분' if pw >= POWER_TARGET else '부족 (효과를 놓칠 확률이 큼)'
    pow_list.append(pw)
    print(f"  {r['scenario']:<20}{p2 - p1:>+8.2f}{pw:>9.3f}{verdict:>26}")

res['power_at_n'] = pow_list

# ---------------------------------------------------------------------------
# 4단계: 요약표
# ---------------------------------------------------------------------------
print("\n[4단계] 요약표")
print("-" * 80)
out = res[['scenario', 'p_control_true', 'p_treatment_true', 'n_per_arm',
           'ate', 'p_value', 'n_required', 'power_at_n']].copy()
out.columns = ['시나리오', '참p_A', '참p_B', 'n/팔', '관측효과', 'p값', '필요n/팔', '검정력']
out['필요n/팔'] = out['필요n/팔'].map(lambda v: '무한' if np.isinf(v) else f"{np.ceil(v):.0f}")
print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

n1, n2, n3 = (int(np.ceil(need_list[i])) for i in range(3))
print("\n  읽는 법")
print(f"   - 시나리오 1: 필요 {n1}명/팔인데 100명만 썼다. 효과가 있는데도 못 찾는다.")
print(f"   - 시나리오 2: 필요 {n2}명/팔인데 100명만 썼다. 역시 못 찾는다.")
print(f"   - 시나리오 3: 필요 {n3}명/팔인데 100명을 썼다. {100 - n3}명/팔이 남는다.")
print(f"   - 시나리오 4: 효과가 0인데 p={results[3]['p_value']:.3f}로 유의하게 나왔다. 1종 오류다.")

# ---------------------------------------------------------------------------
# 5단계: 그림
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
short = ['1 소효과', '2 중효과', '3 대효과', '4 무효과']
xs = np.arange(4)

# (a) 관측 효과와 신뢰구간
ax = axes[0, 0]
true_eff = res['p_treatment_true'] - res['p_control_true']
ax.errorbar(xs, res['ate'], yerr=1.96 * res['se'], fmt='o', color='#2c6fbb',
            capsize=6, markersize=8, label='관측 효과 (95% CI)')
ax.plot(xs, true_eff, '^', color='#2e8b57', markersize=11, label='참 효과')
ax.axhline(0, color='#888888', linewidth=1)
ax.set_xticks(xs); ax.set_xticklabels(short)
ax.set_ylabel('전환율 차이 (B - A)')
ax.set_title('(a) 관측 효과 vs 참 효과 (각 팔 100명)', fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (b) 효과 크기별 필요 표본 수
ax = axes[0, 1]
deltas = np.linspace(0.02, 0.35, 200)
needs = [required_n_per_arm(0.40, 0.40 + d) for d in deltas]
ax.plot(deltas, needs, color='#2c6fbb', linewidth=2.2)
ax.axhline(100, color='#c0392b', linestyle='--', linewidth=1.6,
           label='이 실습의 n = 100/팔')
for i, r in enumerate(results):
    d = r['p_treatment_true'] - r['p_control_true']
    if d > 0:
        ax.plot(d, required_n_per_arm(r['p_control_true'], r['p_treatment_true']),
                'o', color='#d6a300', markersize=9)
        ax.annotate(short[i], (d, required_n_per_arm(r['p_control_true'], r['p_treatment_true'])),
                    textcoords='offset points', xytext=(8, 8), fontsize=9)
ax.set_yscale('log')
ax.set_xlabel('참 효과 크기 (전환율 차이, 기준 0.40)')
ax.set_ylabel('필요 표본 수 (팔당, 로그축)')
ax.set_title('(b) 효과가 절반이면 표본은 네 배', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (c) 표본 수에 따른 검정력
ax = axes[1, 0]
ns = np.arange(10, 1200, 5)
colors = ['#c0392b', '#d6a300', '#2e8b57']
for (p1, p2), c, lab in zip([(0.40, 0.45), (0.40, 0.55), (0.40, 0.70)],
                            colors, ['소효과 +0.05', '중효과 +0.15', '대효과 +0.30']):
    ax.plot(ns, [achieved_power(p1, p2, n) for n in ns], color=c, linewidth=2, label=lab)
ax.axhline(0.80, color='#333333', linestyle='--', linewidth=1.4, label='검정력 0.80')
ax.axvline(100, color='#2c6fbb', linestyle=':', linewidth=1.8, label='n = 100/팔')
ax.set_xlabel('팔당 표본 수'); ax.set_ylabel('검정력')
ax.set_xlim(0, 1200); ax.set_ylim(0, 1.02)
ax.set_title('(c) 팔당 100명으로 잡히는 효과는 대효과뿐', fontweight='bold')
ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.3)

# (d) 필요 표본 vs 실제 표본
ax = axes[1, 1]
req = [min(required_n_per_arm(r['p_control_true'], r['p_treatment_true']), 2000)
       for r in results]
ax.bar(xs - 0.2, req, width=0.4, color='#d6a300', label='필요 n/팔 (검정력 0.80)')
ax.bar(xs + 0.2, res['n_per_arm'], width=0.4, color='#2c6fbb', label='실제 n/팔')
ax.set_yscale('log')
ax.set_xticks(xs); ax.set_xticklabels(short)
ax.set_ylabel('팔당 표본 수 (로그축)')
ax.set_ylim(10, 6000)
ax.set_title('(d) 필요한 만큼 모았는가', fontweight='bold')
ax.legend(fontsize=9, loc='upper left'); ax.grid(axis='y', alpha=0.3)
ax.text(3, 2600, '무효과:\n필요 표본\n정의되지 않음', ha='center', fontsize=8.5,
        color='#c0392b')

plt.tight_layout()
png = pathlib.Path(__file__).with_name('8-1-fixed-rct-simulation.png')
plt.savefig(png, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n그림 저장: {png.name}")

print("\n" + "=" * 80)
print("정리")
print("=" * 80)
print("1. 표본 크기는 실험을 시작하기 전에 정한다. 필요 n은 효과 크기로 정해진다.")
print("2. 효과가 절반이면 필요 표본은 약 네 배가 된다 (분모가 델타의 제곱).")
print(f"3. 팔당 100명은 +0.30 효과만 잡는다. +0.05는 {n1}명이 필요하다.")
print("4. 효과가 0이어도 20번에 1번은 유의하게 나온다 (시나리오 4).")
print("=" * 80)
