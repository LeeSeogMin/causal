"""
3장 2부 그림 생성 스크립트

3-4-covariate-balance.py와 3-5-comprehensive-comparison.py의 함수를 불러
같은 계산을 재수행하고, 강의자료용 그림 두 장을 diagrams/에 저장한다.

- 그림 3-3 (diagrams/3-3.png): λ에 따른 예측 손실-균형 맞교환 (2패널)
- 그림 3-4 (diagrams/3-4.png): 네 방법 종합 비교 — ATT와 균형 (2패널)

주의: 난수 스트림을 본편과 맞추기 위해 모듈 import와 계산 순서를
각 본편 스크립트의 main()과 동일하게 유지한다 (3-4 전체 → 3-5 전체).

실행: practice/chapter03/code 폴더에서  python 3-5-figures.py  (수 분 소요)
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

BLUE = "#0072B2"
ORANGE = "#E69F00"

_here = Path(__file__).parent
out_dir = (_here / "../../../diagrams").resolve()


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, _here / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set_korean_font():
    # 본편 모듈들이 font.family를 Arial로 덮어쓰므로 로드 후 재지정한다
    malgun = Path(r"C:\Windows\Fonts\malgun.ttf")
    if malgun.exists():
        fm.fontManager.addfont(str(malgun))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(malgun)).get_name()
    else:
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------
# 1) 3-4 재수행: λ별 BCE·가중 SMD·매칭 후 SMD
# ---------------------------------------------------------------
psm34 = _load("psm34", "3-4-covariate-balance.py")

X4, T4, data4, _ = psm34.load_data()
y4 = data4["outcome"].values
lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
rows = []
for lam in lambdas:
    _, bce, weighted_smd, ps = psm34.train_model(X4, T4, lambda_balance=lam)
    att, matched_smd, n_pairs = psm34.match_and_estimate(ps, T4, y4, X4)
    rows.append((lam, bce, weighted_smd, matched_smd, att, n_pairs))
    print(f"lambda={lam:.1f}  BCE={bce:.4f}  wSMD={weighted_smd:.4f}  "
          f"mSMD={matched_smd:.4f}  ATT={att:.3f}  pairs={n_pairs}")

# ---------------------------------------------------------------
# 2) 3-5 재수행: 네 방법 ATT·SE·SMD (모듈 import가 시드를 재설정한다)
# ---------------------------------------------------------------
psm35 = _load("psm35", "3-5-comprehensive-comparison.py")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

X5, T5, y5, data5, _ = psm35.load_and_prepare_data()
true_att = data5.loc[T5 == 1, "true_te"].mean()

methods = []
start = time.time()
ps_lr = LogisticRegression(max_iter=1000).fit(X5, T5).predict_proba(X5)[:, 1]
methods.append(("로지스틱", *psm35.match_and_evaluate(ps_lr, T5, y5, X5, data5), time.time() - start))
start = time.time()
ps_rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20,
                               random_state=42).fit(X5, T5).predict_proba(X5)[:, 1]
methods.append(("랜덤 포레스트", *psm35.match_and_evaluate(ps_rf, T5, y5, X5, data5), time.time() - start))
start = time.time()
ps_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                   random_state=42).fit(X5, T5).predict_proba(X5)[:, 1]
methods.append(("그래디언트\n부스팅", *psm35.match_and_evaluate(ps_gb, T5, y5, X5, data5), time.time() - start))
start = time.time()
ps_lbc = psm35.train_lbcnet(X5, T5)
methods.append(("LBC-Net", *psm35.match_and_evaluate(ps_lbc, T5, y5, X5, data5), time.time() - start))

for name, att, se, avg_smd, max_smd, dur in methods:
    print(f"{name!r}: ATT={att:.3f} SE={se:.3f} avgSMD={avg_smd:.3f} "
          f"maxSMD={max_smd:.3f} time={dur:.2f}s")

# ---------------------------------------------------------------
# 그림 3-3: λ 맞교환 (2패널 — 이중 축을 쓰지 않는다)
# ---------------------------------------------------------------
_set_korean_font()

lam_v = [r[0] for r in rows]
bce_v = [r[1] for r in rows]
wsmd_v = [r[2] for r in rows]
msmd_v = [r[3] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150)

ax1.plot(lam_v, bce_v, marker="o", color=BLUE, linewidth=2)
ax1.set_xlabel("λ (균형 가중치)")
ax1.set_ylabel("BCE 손실 (예측)")
ax1.set_title("(a) λ가 커지면 예측 손실은 상승", fontsize=10)

ax2.plot(lam_v, wsmd_v, marker="o", color=ORANGE, linewidth=2, label="가중 SMD")
ax2.plot(lam_v, msmd_v, marker="s", linestyle="--", color=BLUE, linewidth=2,
         label="매칭 후 SMD")
ax2.axhline(0.1, color="#c0392b", linestyle=":", linewidth=1, label="기준 0.1")
ax2.set_xlabel("λ (균형 가중치)")
ax2.set_ylabel("평균 |SMD| (균형)")
ax2.set_title("(b) 균형 지표의 변화", fontsize=10)
ax2.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(out_dir / "3-3.png")
plt.close(fig)

# ---------------------------------------------------------------
# 그림 3-4: 네 방법 종합 비교 (ATT ± 95% CI, 평균 SMD)
# ---------------------------------------------------------------
names = [m[0] for m in methods]
atts = np.array([m[1] for m in methods])
ses = np.array([m[2] for m in methods])
avg_smds = np.array([m[3] for m in methods])
xs = np.arange(len(names))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150)

ax1.axhline(true_att, color="#c0392b", linestyle="--", linewidth=1.2,
            label=f"참 ATT = {true_att:.1f}")
ax1.errorbar(xs, atts, yerr=1.96 * ses, fmt="o", color=BLUE,
             markersize=7, capsize=4, linewidth=1.5, label="추정 ATT (95% CI)")
for x, a in zip(xs, atts):
    ax1.annotate(f"{a:.2f}", (x, a), textcoords="offset points",
                 xytext=(8, 4), fontsize=8.5, color="#333333")
ax1.set_xticks(xs)
ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylabel("ATT 추정치")
ax1.set_title("(a) ATT와 95% 신뢰구간 vs 참값", fontsize=10)
ax1.legend(frameon=False, fontsize=8, loc="upper right")

ax2.bar(xs, avg_smds, width=0.55, color=BLUE)
ax2.axhline(0.1, color="#c0392b", linestyle="--", linewidth=1.2, label="기준 0.1")
for x, s in zip(xs, avg_smds):
    ax2.annotate(f"{s:.3f}", (x, s), textcoords="offset points",
                 xytext=(0, 4), ha="center", fontsize=8.5, color="#333333")
ax2.set_xticks(xs)
ax2.set_xticklabels(names, fontsize=9)
ax2.set_ylabel("매칭 후 평균 |SMD|")
ax2.set_title("(b) 공변량 균형", fontsize=10)
ax2.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(out_dir / "3-4.png")
plt.close(fig)

print("\n그림 저장 완료:")
print(f"  {out_dir / '3-3.png'}")
print(f"  {out_dir / '3-4.png'}")
