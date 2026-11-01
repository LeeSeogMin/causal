"""
8장 강의노트용 개념도 생성
==========================

실행: python diagrams/make_chapter08.py
출력: diagrams/8-1.png ~ 8-7.png

- 8-1.png: 스스로 고른 두 집단과 동전으로 나눈 두 집단
- 8-2.png: 효과 크기와 필요 인원 (효과가 절반이면 인원은 약 4배)
- 8-3.png: 중간에 여러 번 들여다보면 1종 오류가 커진다
- 8-4.png: Pocock 경계와 O'Brien-Fleming 경계
- 8-5.png: 톰슨 샘플링 — 시도가 적은 안은 분포가 넓다
- 8-6.png: 톰슨 샘플링의 배분 비율 변화와 기대 손실
- 8-7.png: 베이지안 적응형 설계 — Beta 갱신과 사후확률 궤적
- 8-8.png: 매뉴얼(ch08-easy.md)용 작업 고르기 순서도

그림 속 실행 결과 수치는 practice/chapter08/results/*.log 에서 옮긴 값이다.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

_available = {f.name for f in font_manager.fontManager.ttflist}
for _name in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans KR']:
    if _name in _available:
        plt.rcParams['font.family'] = _name
        break
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))

BLUE = '#d9e8ff'; BLUE_E = '#2c6fbb'
GREEN = '#d9f2d9'; GREEN_E = '#2e8b57'
RED = '#ffd9d9'; RED_E = '#c0392b'
YELLOW = '#fff2cc'; YELLOW_E = '#d6a300'
GRAY = '#eeeeee'; GRAY_E = '#888888'


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=200, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# 8-1. 스스로 고른 두 집단과 동전으로 나눈 두 집단
# ---------------------------------------------------------------------------
rng = np.random.default_rng(8)
n = 2000
motivated = rng.random(n) < 0.5          # 측정하지 못한 특성: 구매 의욕이 높은 사람
# 스스로 고르는 경우: 의욕이 높은 사람이 새 화면(B안)을 더 많이 고른다
self_b = np.where(motivated, rng.random(n) < 0.75, rng.random(n) < 0.25)
# 동전으로 나누는 경우
coin_b = rng.random(n) < 0.5

share = {
    'self': (motivated[~self_b].mean(), motivated[self_b].mean()),
    'coin': (motivated[~coin_b].mean(), motivated[coin_b].mean()),
}

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ax, key, title, note, col in [
        (axes[0], 'self', '사용자가 스스로 고른 경우',
         '두 집단의 구성이 이미 다르다\n→ 결과 차이에 의욕 차이가 섞인다', RED_E),
        (axes[1], 'coin', '동전으로 나눈 경우',
         '두 집단의 구성이 거의 같다\n→ 결과 차이는 화면 차이로 읽는다', GREEN_E)]:
    a, b = share[key]
    bars = ax.bar(['A안 집단', 'B안 집단'], [a * 100, b * 100],
                  color=[GRAY, BLUE], edgecolor=[GRAY_E, BLUE_E], linewidth=2,
                  width=0.55)
    for bar, v in zip(bars, [a, b]):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 100 + 2,
                f'{v * 100:.0f}%', ha='center', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.text(0.5, -0.22, note, transform=ax.transAxes, ha='center', va='top',
            fontsize=10.5, color=col)
    ax.grid(axis='y', alpha=0.3)
axes[0].set_ylabel('구매 의욕이 높은 사람의 비율 (%)', fontsize=10.5)
fig.suptitle('그림 8-1. 동전으로 나누면 측정하지 못한 특성까지 두 집단에 고르게 퍼진다',
             fontsize=13, fontweight='bold', y=1.03)
save('8-1.png')
print('8-1 의욕 비율:', {k: tuple(round(x, 3) for x in v) for k, v in share.items()})


# ---------------------------------------------------------------------------
# 8-2. 효과 크기와 필요 인원
# ---------------------------------------------------------------------------
def required_n_per_arm(p1, p2, alpha=0.05, power=0.80):
    p_bar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    term = (z_a * np.sqrt(2 * p_bar * (1 - p_bar))
            + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
    return term ** 2 / abs(p2 - p1) ** 2


deltas = np.linspace(0.04, 0.32, 200)
ns = [required_n_per_arm(0.40, 0.40 + d) for d in deltas]
marks = [(0.05, '시나리오 1', (0.085, 900)), (0.15, '시나리오 2', (0.175, 330)),
         (0.30, '시나리오 3', (0.235, 140))]

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(deltas, ns, color='black', linewidth=2.5)
for d, name, pos in marks:
    nv = required_n_per_arm(0.40, 0.40 + d)
    ax.plot(d, nv, 'o', color=RED_E, markersize=10)
    ax.annotate(f'{name}\n효과 +{d:.2f} → 팔당 {np.ceil(nv):.0f}명',
                xy=(d, nv), xytext=pos,
                fontsize=10.5, color=RED_E, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=RED_E, linewidth=1.4))
ax.axhline(100, color=BLUE_E, linestyle='--', linewidth=2)
ax.text(0.10, 88, '실습 데이터의 실제 인원: 팔당 100명', color=BLUE_E,
        fontsize=10, ha='left', va='top', fontweight='bold')
ax.set_yscale('log')
ax.set_yticks([20, 50, 100, 200, 500, 1000, 2000])
ax.set_yticklabels(['20', '50', '100', '200', '500', '1,000', '2,000'])
ax.set_xlabel('잡으려는 효과 크기 (전환율 차이, A안 0.40 기준)', fontsize=11)
ax.set_ylabel('팔 하나에 필요한 인원 (로그 눈금)', fontsize=11)
ax.grid(alpha=0.3)
ax.set_title('그림 8-2. 잡으려는 효과가 작아질수록 필요한 인원은 빠르게 늘어난다',
             fontsize=12.5, fontweight='bold', pad=10)
fig.text(0.5, -0.04,
         '효과 +0.30 → +0.15로 절반이 되면 42명 → 173명으로 약 4배. '
         '검정력 0.80, 유의수준 0.05 기준.',
         ha='center', fontsize=10.5, color='#333333')
save('8-2.png')


# ---------------------------------------------------------------------------
# 8-3. 중간에 여러 번 들여다보면 1종 오류가 커진다
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                         gridspec_kw={'width_ratios': [1.25, 1]})

# 왼쪽: 효과가 0인 실험 몇 개의 누적 검정 통계량 궤적
rng = np.random.default_rng(3)
k = 5
looks = np.arange(1, k + 1)
paths = []
while len(paths) < 6:
    z = np.cumsum(rng.standard_normal(k)) / np.sqrt(looks)
    crossed_mid = np.any(np.abs(z[:-1]) > 1.96) and abs(z[-1]) < 1.96
    if len(paths) == 0 and not crossed_mid:
        continue
    paths.append(z)
ax = axes[0]
for i, z in enumerate(paths):
    if i == 0:
        ax.plot(looks, z, 'o-', color=RED_E, linewidth=2.6, markersize=7,
                label='중간에 한 번 선을 넘은 실험')
        j = int(np.argmax(np.abs(z[:-1]) > 1.96))
        ax.annotate('여기서 멈추면\n"효과 있음"으로 잘못 선언',
                    xy=(looks[j], z[j]), xytext=(looks[j] + 0.6, 2.9),
                    fontsize=10, color=RED_E, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=RED_E))
    else:
        ax.plot(looks, z, 'o-', color=GRAY_E, linewidth=1.4, markersize=4,
                alpha=0.8)
for s in (1.96, -1.96):
    ax.axhline(s, color=BLUE_E, linestyle='--', linewidth=2)
ax.text(5.15, 2.45, '유의 기준선\n(p값 0.05)', color=BLUE_E, fontsize=9.5,
        va='center')
ax.set_xticks(looks)
ax.set_xlabel('중간 분석 차례', fontsize=10.5)
ax.set_ylabel('누적 검정 통계량', fontsize=10.5)
ax.set_ylim(-3.6, 3.6)
ax.set_xlim(0.7, 6.0)
ax.legend(fontsize=9.5, loc='lower left')
ax.grid(alpha=0.3)
ax.set_title('효과가 0인 실험 6개의 궤적', fontsize=11.5, fontweight='bold')

# 오른쪽: 8-2 로그 1단계 표
K = ['1', '2', '3', '5', '10', '20']
err = [0.0498, 0.0825, 0.1062, 0.1409, 0.1911, 0.2484]
ax = axes[1]
bars = ax.bar(K, err, color=RED, edgecolor=RED_E, linewidth=1.8)
for bar, v in zip(bars, err):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006, f'{v:.3f}',
            ha='center', fontsize=10, fontweight='bold')
ax.axhline(0.05, color=BLUE_E, linestyle='--', linewidth=2)
ax.text(5.4, 0.056, '설계한 5%', color=BLUE_E, fontsize=10, ha='right')
ax.set_xlabel('들여다본 횟수', fontsize=10.5)
ax.set_ylabel('효과가 없는데 "있다"고 판정할 확률', fontsize=10)
ax.set_ylim(0, 0.29)
ax.grid(axis='y', alpha=0.3)
ax.set_title('들여다본 횟수와 1종 오류 (20만 번 반복)', fontsize=11.5,
             fontweight='bold')

fig.suptitle('그림 8-3. 유의할 때까지 들여다보면 우연이 "효과"로 둔갑한다',
             fontsize=13, fontweight='bold', y=1.03)
save('8-3.png')


# ---------------------------------------------------------------------------
# 8-4. Pocock 경계와 O'Brien-Fleming 경계 (8-2 로그 2단계)
# ---------------------------------------------------------------------------
stage_n = [67, 133, 200]
pocock_z = [2.294, 2.294, 2.294]
pocock_p = [0.02177, 0.02177, 0.02177]
obf_z = [3.468, 2.452, 2.002]
obf_p = [0.00052, 0.01419, 0.04525]

fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.plot(stage_n, pocock_z, 's-', color=GREEN_E, linewidth=2.6, markersize=10,
        label='Pocock: 세 번 모두 같은 기준')
ax.plot(stage_n, obf_z, 'o-', color=RED_E, linewidth=2.6, markersize=10,
        label="O'Brien-Fleming: 처음엔 엄격, 끝에서 완화")
ax.axhline(1.96, color=BLUE_E, linestyle='--', linewidth=2,
           label='보정 없는 기준 1.96 (p값 0.05)')
for x, z, p in zip(stage_n, pocock_z, pocock_p):
    ax.text(x - 4, z + 0.10, f'p값 {p}', color=GREEN_E, fontsize=9.5,
            ha='right')
for i, (x, z, p) in enumerate(zip(stage_n, obf_z, obf_p)):
    dz, va = (0.06, 'bottom') if i < 2 else (-0.06, 'top')
    ax.text(x + 5, z + dz, f'p값 {p}', color=RED_E, fontsize=9.5,
            ha='left', va=va)
ax.set_xticks(stage_n)
ax.set_xticklabels(['1차 (67명)', '2차 (133명)', '3차·최종 (200명)'])
ax.set_xlim(40, 230)
ax.set_ylim(1.2, 3.9)
ax.set_ylabel('멈추려면 넘어야 하는 검정 통계량', fontsize=10.5)
ax.text(135, 1.45, '통계량이 경계선 아래에 머물면 다음 차례까지 계속 모집한다',
        ha='center', fontsize=10, color='#555555')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)
ax.set_title('그림 8-4. 세 번 들여다보면서 전체 오류를 5%로 지키는 두 가지 방법',
             fontsize=12.5, fontweight='bold', pad=10)
save('8-4.png')


# ---------------------------------------------------------------------------
# 8-5. 톰슨 샘플링 — 시도가 적은 안은 분포가 넓다
# ---------------------------------------------------------------------------
x = np.linspace(0, 1, 500)
a_s, a_n = 1, 3      # 안 A: 3명 중 1명 성공
b_s, b_n = 50, 90    # 안 B: 90명 중 50명 성공
pa = stats.beta(1 + a_s, 1 + a_n - a_s)
pb = stats.beta(1 + b_s, 1 + b_n - b_s)
rng = np.random.default_rng(5)
draws_a = pa.rvs(200_000, random_state=rng)
draws_b = pb.rvs(200_000, random_state=rng)
p_a_wins = float(np.mean(draws_a > draws_b))

fig, ax = plt.subplots(figsize=(9.8, 4.6))
ax.fill_between(x, pa.pdf(x), color=RED, alpha=0.8)
ax.plot(x, pa.pdf(x), color=RED_E, linewidth=2.4,
        label=f'안 A: {a_n}명 중 {a_s}명 성공 → Beta({1 + a_s}, {1 + a_n - a_s})')
ax.fill_between(x, pb.pdf(x), color=BLUE, alpha=0.7)
ax.plot(x, pb.pdf(x), color=BLUE_E, linewidth=2.4,
        label=f'안 B: {b_n}명 중 {b_s}명 성공 → Beta({1 + b_s}, {1 + b_n - b_s})')
ax.annotate('시도가 적어 넓게 퍼져 있다\n→ 가끔 큰 값이 뽑혀 기회를 얻는다',
            xy=(0.72, pa.pdf(0.72)), xytext=(0.70, 4.6), fontsize=10.5,
            color=RED_E, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED_E))
ax.annotate('시도가 많아 좁게 모여 있다\n→ 참값 근처 값만 뽑힌다',
            xy=(0.49, pb.pdf(0.49)), xytext=(0.02, 6.2), fontsize=10.5,
            color=BLUE_E, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BLUE_E))
ax.set_xlabel('전환율', fontsize=11)
ax.set_ylabel('그럴듯한 정도', fontsize=11)
ax.set_yticks([])
ax.set_xlim(0, 1)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)
ax.set_title('그림 8-5. 톰슨 샘플링은 두 분포에서 하나씩 뽑아 큰 쪽에 다음 사용자를 보낸다',
             fontsize=12, fontweight='bold', pad=10)
fig.text(0.5, -0.04,
         f'두 분포에서 20만 번 뽑아 보면 안 A가 이기는 비율은 {p_a_wins * 100:.0f}%다. '
         '안 A는 평균이 낮아도 이만큼 다시 시도된다.',
         ha='center', fontsize=10.5, color='#333333')
save('8-5.png')
print(f'8-5 안 A가 이기는 비율: {p_a_wins:.4f}')


# ---------------------------------------------------------------------------
# 8-6. 톰슨 샘플링의 배분 비율 변화와 기대 손실 (8-3 로그 3·4단계)
# ---------------------------------------------------------------------------
t = [50, 100, 200, 500, 1000]
arm0 = [8.0, 5.0, 3.0, 1.2, 0.6]
arm1 = [76.0, 84.0, 91.0, 96.0, 97.9]
arm2 = [16.0, 11.0, 6.0, 2.8, 1.5]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                         gridspec_kw={'width_ratios': [1.3, 1]})
ax = axes[0]
ax.plot(t, arm1, 'o-', color=GREEN_E, linewidth=2.6, markersize=8,
        label='안 1 (참 0.55, 최적)')
ax.plot(t, arm2, 's-', color=YELLOW_E, linewidth=2.2, markersize=7,
        label='안 2 (참 0.40)')
ax.plot(t, arm0, '^-', color=RED_E, linewidth=2.2, markersize=7,
        label='안 0 (참 0.30)')
ax.axhline(33.3, color=GRAY_E, linestyle='--', linewidth=2,
           label='고정 배분이면 계속 33.3%')
for xi, yi in zip(t, arm1):
    ax.text(xi, yi + 3, f'{yi:.0f}%' if yi < 97 else f'{yi}%',
            ha='center', fontsize=9.5, color=GREEN_E, fontweight='bold')
ax.set_xscale('log')
ax.set_xticks(t)
ax.set_xticklabels([str(v) for v in t])
ax.set_ylim(0, 108)
ax.set_xlabel('지금까지 배정한 인원 (로그 눈금)', fontsize=10.5)
ax.set_ylabel('누적 배분 비율 (%)', fontsize=10.5)
ax.legend(fontsize=9.5, loc='center right')
ax.grid(alpha=0.3)
ax.set_title('최적 안에 보내는 비율이 점점 오른다', fontsize=11.5,
             fontweight='bold')

ax = axes[1]
names = ['고정 1:1:1', 'Epsilon-\nGreedy', '톰슨\n샘플링']
regret = [133.33, 43.35, 3.75]
bars = ax.bar(names, regret, color=[GRAY, YELLOW, GREEN],
              edgecolor=[GRAY_E, YELLOW_E, GREEN_E], linewidth=2, width=0.6)
for bar, v in zip(bars, regret):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 3, f'{v:.2f}',
            ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('기대 손실 (열등한 안을 겪은 사용자 수 환산)', fontsize=9.5)
ax.set_ylim(0, 150)
ax.grid(axis='y', alpha=0.3)
ax.set_title('1,000명 실험의 기대 손실', fontsize=11.5, fontweight='bold')

fig.suptitle('그림 8-6. 톰슨 샘플링은 배분을 최적 안으로 옮겨 손실을 133.33에서 3.75로 줄였다',
             fontsize=12.5, fontweight='bold', y=1.03)
save('8-6.png')


# ---------------------------------------------------------------------------
# 8-7. 베이지안 적응형 설계 (8-4 로그 2단계)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                         gridspec_kw={'width_ratios': [1, 1.15]})

# 왼쪽: 대효과 시나리오 n=50 시점의 두 사후분포
ax = axes[0]
x = np.linspace(0, 1, 500)
post_a = stats.beta(1 + 11, 1 + 14)   # A안 25명 중 11명
post_b = stats.beta(1 + 19, 1 + 6)    # B안 25명 중 19명
ax.fill_between(x, post_a.pdf(x), color=GRAY, alpha=0.9)
ax.plot(x, post_a.pdf(x), color=GRAY_E, linewidth=2.4,
        label='A안: 25명 중 11명 → Beta(12, 15)')
ax.fill_between(x, post_b.pdf(x), color=BLUE, alpha=0.7)
ax.plot(x, post_b.pdf(x), color=BLUE_E, linewidth=2.4,
        label='B안: 25명 중 19명 → Beta(20, 7)')
ax.text(0.5, 0.97, 'B안이 더 클 확률 = 0.9887', transform=ax.transAxes,
        ha='center', va='top', fontsize=11, fontweight='bold', color=BLUE_E)
ax.set_xlabel('전환율', fontsize=10.5)
ax.set_yticks([])
ax.set_ylim(0, 6.2)
ax.legend(fontsize=9.5, loc='upper left', bbox_to_anchor=(0, 0.88))
ax.grid(alpha=0.3)
ax.set_title('대효과 실험, 첫 중간 분석(50명) 시점', fontsize=11.5,
             fontweight='bold')

# 오른쪽: 세 시나리오의 사후확률 궤적
ax = axes[1]
ax.axhspan(0.99, 1.02, color=GREEN, alpha=0.8)
ax.axhspan(-0.02, 0.05, color=RED, alpha=0.8)
ax.text(205, 1.045, 'B안 채택 구역 (0.99 이상)', fontsize=9.5,
        color=GREEN_E, va='center', ha='right', fontweight='bold')
ax.text(205, 0.015, 'B안 폐기 구역 (0.05 이하)', fontsize=9.5,
        color=RED_E, va='center', ha='right', fontweight='bold')
traj = [
    ('대효과 (0.40 vs 0.70)', [50, 100], [0.9887, 0.9993], BLUE_E, 'o'),
    ('중효과 (0.40 vs 0.55)', [50, 100, 150, 200],
     [0.8718, 0.9441, 0.9632, 0.9545], YELLOW_E, 's'),
    ('무효과 (0.40 vs 0.40)', [50, 100, 150], [0.2865, 0.0803, 0.0023],
     GRAY_E, '^'),
]
for name, xs_, ys_, col, mk in traj:
    ax.plot(xs_, ys_, marker=mk, color=col, linewidth=2.4, markersize=9,
            label=name)
ax.annotate('100명에서 채택·중단', xy=(100, 0.9993), xytext=(105, 0.80),
            fontsize=9.5, color=BLUE_E, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BLUE_E))
ax.annotate('200명 완주, 판정 유보', xy=(200, 0.9545), xytext=(150, 0.66),
            fontsize=9.5, color=YELLOW_E, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=YELLOW_E))
ax.annotate('150명에서 폐기·중단', xy=(150, 0.0023), xytext=(152, 0.30),
            fontsize=9.5, color='#555555', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#555555'))
ax.set_xticks([50, 100, 150, 200])
ax.set_xlim(40, 210)
ax.set_ylim(-0.02, 1.08)
ax.set_xlabel('누적 인원 (중간 분석 시점)', fontsize=10.5)
ax.set_ylabel('B안이 A안보다 나을 확률', fontsize=10.5)
ax.legend(fontsize=9.5, loc='center left')
ax.grid(alpha=0.3)
ax.set_title('세 실험의 사후확률 궤적', fontsize=11.5, fontweight='bold')

fig.suptitle('그림 8-7. 사후확률이 위·아래 문턱 중 하나를 넘으면 멈춘다',
             fontsize=13, fontweight='bold', y=1.03)
save('8-7.png')

# ---------------------------------------------------------------------------
# 8-8. 매뉴얼용 작업 고르기 순서도
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec, fs=10.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor=fc, edgecolor=ec, linewidth=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold')


def arrow(ax, p1, p2, label=None, dx=0, dy=0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=18,
                                 color='#333333', linewidth=1.8,
                                 shrinkA=4, shrinkB=4))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy, label,
                ha='center', va='center', fontsize=9.5, color='#333333')


fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.6)
ax.axis('off')

box(ax, 5.5, 5.0, 6.4, 0.75, '실험의 목적이 무엇인가', YELLOW, YELLOW_E, fs=12)

box(ax, 1.9, 3.5, 3.3, 0.95, '효과 크기를\n정확히 보고한다', BLUE, BLUE_E)
box(ax, 5.5, 3.5, 3.3, 0.95, '실험 중 손실을\n줄인다', GREEN, GREEN_E)
box(ax, 9.1, 3.5, 3.3, 0.95, '"B안이 나을 확률"로\n판정하고 일찍 멈춘다', RED, RED_E)
arrow(ax, (3.6, 4.62), (2.2, 4.0))
arrow(ax, (5.5, 4.62), (5.5, 4.0))
arrow(ax, (7.4, 4.62), (8.8, 4.0))

box(ax, 1.9, 2.05, 3.3, 0.75, '작업 A: 필요 인원 확인', GRAY, GRAY_E)
arrow(ax, (1.9, 3.02), (1.9, 2.44))
box(ax, 0.95, 0.55, 1.8, 0.85, '작업 B\n끝난 뒤 판정', GRAY, GRAY_E, fs=10)
box(ax, 2.95, 0.55, 1.8, 0.85, '작업 C\n중간에 멈추기', GRAY, GRAY_E, fs=10)
arrow(ax, (1.6, 1.66), (1.0, 1.0), '중간에\n안 봄', dx=-0.45)
arrow(ax, (2.2, 1.66), (2.9, 1.0), '중간에\n봄', dx=0.45)

box(ax, 5.5, 2.05, 3.3, 0.75, '작업 D: 좋은 안에\n더 많이 보내기', GRAY, GRAY_E)
arrow(ax, (5.5, 3.02), (5.5, 2.44))
box(ax, 9.1, 2.05, 3.3, 0.75, '작업 E: 사후확률로\n멈추기', GRAY, GRAY_E)
arrow(ax, (9.1, 3.02), (9.1, 2.44))

ax.text(7.3, 0.55, '작업 D를 고르면 나쁜 안의 효과는 정확히 알 수 없다.\n'
        '효과 크기 보고가 목적이면 작업 A→B 또는 A→C를 쓴다.',
        ha='center', va='center', fontsize=10, color=RED_E)
ax.set_title('그림 8-8. 목적에 따라 작업을 고른다', fontsize=13,
             fontweight='bold', pad=6)
save('8-8.png')

print("생성 완료:")
for f in [f'8-{i}.png' for i in range(1, 9)]:
    p = os.path.join(OUT, f)
    print(f"  {f}  ({os.path.getsize(p) / 1024:.0f} KB)")
