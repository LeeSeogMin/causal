"""
3.5절 세 방법(LBC-Net 제외) 종합 비교 그림 생성: diagrams/3-6.png

3-5-comprehensive-comparison.py의 데이터 로드·평가 함수를 재사용해
로지스틱·랜덤 포레스트·그래디언트 부스팅 세 방법만 같은 기준으로 평가한다.
(개조식판처럼 LBC-Net을 다루지 않는 자료에서 그림 3-4 대신 사용)

실행: practice/chapter03/code 폴더에서  python 3-5-figure-3methods.py
"""

# == RESULTS LOG (auto): 콘솔 출력을 results/<스크립트명>.log 에도 저장 ==
import sys as _sys
from pathlib import Path as _Path

_results_dir = _Path(__file__).resolve().parent.parent / "results"
_results_dir.mkdir(exist_ok=True)

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for _s in self._streams:
            _s.write(data)
            _s.flush()
    def flush(self):
        for _s in self._streams:
            _s.flush()

_sys.stdout = _Tee(_sys.stdout, open(_results_dir / (_Path(__file__).stem + ".log"), "w", encoding="utf-8"))
# == /RESULTS LOG ==


import importlib.util
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

BLUE = "#0072B2"

_here = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "psm35", _here / "3-5-comprehensive-comparison.py"
)
psm35 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psm35)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

_malgun = Path(r"C:\Windows\Fonts\malgun.ttf")
if _malgun.exists():
    fm.fontManager.addfont(str(_malgun))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(_malgun)).get_name()
else:
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

X, T, y, data, _ = psm35.load_and_prepare_data()
true_att = data.loc[T == 1, "true_te"].mean()

methods = []
ps = LogisticRegression(max_iter=1000).fit(X, T).predict_proba(X)[:, 1]
methods.append(("로지스틱", *psm35.match_and_evaluate(ps, T, y, X, data)))
ps = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20,
                            random_state=42).fit(X, T).predict_proba(X)[:, 1]
methods.append(("랜덤 포레스트", *psm35.match_and_evaluate(ps, T, y, X, data)))
ps = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                random_state=42).fit(X, T).predict_proba(X)[:, 1]
methods.append(("그래디언트\n부스팅", *psm35.match_and_evaluate(ps, T, y, X, data)))

for name, att, se, avg_smd, max_smd in methods:
    print(f"{name!r}: ATT={att:.3f} SE={se:.3f} avgSMD={avg_smd:.3f} maxSMD={max_smd:.3f}")

names = [m[0] for m in methods]
atts = np.array([m[1] for m in methods])
ses = np.array([m[2] for m in methods])
avg_smds = np.array([m[3] for m in methods])
xs = np.arange(len(names))

out_dir = (_here / "../../../diagrams").resolve()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150)

ax1.axhline(true_att, color="#c0392b", linestyle="--", linewidth=1.2,
            label=f"참 효과 = {true_att:.1f}")
ax1.errorbar(xs, atts, yerr=1.96 * ses, fmt="o", color=BLUE,
             markersize=7, capsize=4, linewidth=1.5, label="추정 ATT (95% 신뢰구간)")
for x, a in zip(xs, atts):
    ax1.annotate(f"{a:.2f}", (x, a), textcoords="offset points",
                 xytext=(8, 4), fontsize=8.5, color="#333333")
ax1.set_xticks(xs)
ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylabel("ATT 추정치")
ax1.set_title("(a) ATT와 95% 신뢰구간 vs 참값", fontsize=10)
ax1.legend(frameon=False, fontsize=8, loc="upper right")

ax2.bar(xs, avg_smds, width=0.5, color=BLUE)
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
fig.savefig(out_dir / "3-6.png")
plt.close(fig)

print(f"\n그림 저장 완료: {out_dir / '3-6.png'}")
