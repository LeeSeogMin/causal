"""
3.2절 개념 그림 생성: 나무 모형이 곡선을 배우는 방식 (diagrams/3-5.png)

개념 확인용 1차원 예시를 실제로 학습시켜 그린다.
- 나이 하나로 처치 확률이 곡선(단봉)을 그리는 작은 데이터를 만들고,
- 왼쪽 패널: 나무 한 그루(계단) vs 랜덤 포레스트 수백 그루의 평균 vs 참 곡선
- 오른쪽 패널: 그래디언트 부스팅이 나무 1 → 10 → 100개로 쌓이며 곡선에 접근

본편 실습 데이터와는 별개인 개념 예시이며, 시드 고정으로 재현된다.
실행: practice/chapter03/code 폴더에서  python 3-2-concept-figure.py
"""

# == RESULTS LOG (auto): 콘솔 출력과 그림을 results/ 에 저장 ==
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

# 그림 저장을 results/ 로 돌린다 (강의자료용 diagrams/ 경로는 그대로 둔다)
try:
    from matplotlib.figure import Figure as _Fig
    if not getattr(_Fig, "_results_patched", False):
        _orig_savefig = _Fig.savefig
        def _savefig_to_results(self, fname, *a, **k):
            try:
                _p = _Path(fname)
                if "diagrams" not in [s.lower() for s in _p.parts]:
                    fname = _results_dir / _p.name
                    print(f"[그림 저장 → {fname}]")
            except TypeError:
                pass
            return _orig_savefig(self, fname, *a, **k)
        _Fig.savefig = _savefig_to_results
        _Fig._results_patched = True
except ImportError:
    pass
# == /RESULTS LOG ==


from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#888888"

_malgun = Path(r"C:\Windows\Fonts\malgun.ttf")
if _malgun.exists():
    fm.fontManager.addfont(str(_malgun))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(_malgun)).get_name()
else:
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 개념 예시 데이터: 나이 하나로 처치 확률이 단봉 곡선을 그린다
rng = np.random.default_rng(42)
n = 2000
age = np.sort(rng.uniform(20, 65, n))
p_true = 0.15 + 0.6 * np.exp(-(((age - 45) / 8.0) ** 2))
treated = rng.binomial(1, p_true)
X = age.reshape(-1, 1)

grid = np.linspace(20, 65, 400).reshape(-1, 1)
p_grid = 0.15 + 0.6 * np.exp(-(((grid.ravel() - 45) / 8.0) ** 2))

# 모형 학습
tree1 = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, treated)
rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=40,
                            random_state=42).fit(X, treated)
gb_stages = {}
for n_est in (1, 10, 100):
    gb = GradientBoostingClassifier(n_estimators=n_est, learning_rate=0.1,
                                    max_depth=1, min_samples_leaf=100,
                                    random_state=42).fit(X, treated)
    gb_stages[n_est] = gb.predict_proba(grid)[:, 1]

out_dir = (Path(__file__).parent / "../../../diagrams").resolve()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.9), dpi=150)

# (a) 랜덤 포레스트: 계단 -> 평균 -> 곡선
ax1.plot(grid, p_grid, color=GRAY, linestyle="--", linewidth=1.6, label="참 확률 곡선")
ax1.plot(grid, tree1.predict_proba(grid)[:, 1], color=ORANGE, linewidth=2,
         drawstyle="steps-post", label="나무 한 그루 (계단)")
ax1.plot(grid, rf.predict_proba(grid)[:, 1], color=BLUE, linewidth=2.2,
         label="수백 그루의 평균 (랜덤 포레스트)")
ax1.set_title("(a) 랜덤 포레스트 — 계단을 평균해 곡선을 만든다", fontsize=10)

# (b) 그래디언트 부스팅: 나무를 쌓을수록 곡선에 접근
ax2.plot(grid, p_grid, color=GRAY, linestyle="--", linewidth=1.6, label="참 확률 곡선")
ax2.plot(grid, gb_stages[1], color="#b8d4ea", linewidth=2, label="나무 1개")
ax2.plot(grid, gb_stages[10], color="#5d9dc9", linewidth=2, label="나무 10개")
ax2.plot(grid, gb_stages[100], color=BLUE, linewidth=2.2, label="나무 100개")
ax2.set_title("(b) 그래디언트 부스팅 — 오답을 이어 고치며 접근한다", fontsize=10)

for ax in (ax1, ax2):
    ax.set_xlabel("나이")
    ax.set_ylabel("처치를 받을 확률")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(out_dir / "3-5.png")
plt.close(fig)

print(f"그림 저장 완료: {out_dir / '3-5.png'}")
