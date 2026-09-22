"""
3장 1부 그림 생성 스크립트

3-1-traditional-psm-limitations.py의 함수를 그대로 불러 같은 계산을 수행하고,
강의자료(lecture/chapter03.md)에 넣을 그림 두 장을 diagrams/에 저장한다.

- 그림 3-1 (diagrams/3-1.png): 성향점수 분포와 공통지지 영역, 극단값 기준선
- 그림 3-2 (diagrams/3-2.png): 매칭 전후 공변량별 SMD와 판정 기준선

실행: practice/chapter03/code 폴더에서  python 3-1-figures.py
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


import importlib.util
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 색: 처치군/매칭 후 = 파랑, 대조군/매칭 전 = 주황 (CVD 검증 통과 팔레트)
BLUE = "#0072B2"
ORANGE = "#E69F00"

# 본편 스크립트를 모듈로 로드해 동일한 파이프라인을 재사용한다
_here = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "psm31", _here / "3-1-traditional-psm-limitations.py"
)
psm31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psm31)

# 한글 글꼴은 본편 모듈 로드 "이후"에 지정한다
# (본편 스크립트가 font.family를 Arial로 덮어쓰기 때문)
_malgun = Path(r"C:\Windows\Fonts\malgun.ttf")
if _malgun.exists():
    fm.fontManager.addfont(str(_malgun))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(_malgun)).get_name()
else:
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

data, true_att = psm31.load_data()
feature_cols = ["age", "income", "education", "experience", "assets"]
feature_labels = ["나이", "소득", "교육년수", "경력년수", "자산"]

smd_before = psm31.analyze_before_matching(data, feature_cols)
ps_logit, ps_true, X_scaled, scaler = psm31.estimate_propensity_score(data, feature_cols)
T = data["treatment"].values
overlap_min, overlap_max, _, _ = psm31.analyze_common_support(ps_logit, T)
mt_idx, mc_idx, n_matched, n_unmatched = psm31.perform_matching(data, ps_logit, feature_cols)
smd_after = psm31.analyze_after_matching(data, feature_cols, mt_idx, mc_idx, smd_before)

out_dir = (_here / "../../../diagrams").resolve()

# ---------------------------------------------------------------
# 그림 3-1: 성향점수 분포와 공통지지 영역
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

bins = np.linspace(0, 1, 41)
# 공통지지 "밖" 구간을 음영으로 표시 (짝을 찾을 수 없는 구간)
ax.axvspan(0, overlap_min, color="#f0d9d9", zorder=0)
ax.axvspan(overlap_max, 1, color="#f0d9d9", zorder=0)
ax.hist(ps_logit[T == 0], bins=bins, color=ORANGE, alpha=0.65,
        label=f"대조군 ({int((T == 0).sum()):,}명)", zorder=2)
ax.hist(ps_logit[T == 1], bins=bins, color=BLUE, alpha=0.65,
        label=f"처치군 ({int((T == 1).sum()):,}명)", zorder=3)
for x in (0.05, 0.95):
    ax.axvline(x, color="#666666", linestyle="--", linewidth=1, zorder=4)

handles, labels = ax.get_legend_handles_labels()
handles += [
    plt.Rectangle((0, 0), 1, 1, color="#f0d9d9"),
    Line2D([0], [0], color="#666666", linestyle="--", linewidth=1),
]
labels += [
    f"공통지지 밖 [{overlap_min:.2f} 미만, {overlap_max:.2f} 초과]",
    "극단값 기준 (0.05 / 0.95)",
]
ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)

ax.set_xlabel("추정 성향점수")
ax.set_ylabel("인원수")
ax.set_xlim(0, 1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(out_dir / "3-1.png")
plt.close(fig)

# ---------------------------------------------------------------
# 그림 3-2: 매칭 전후 공변량별 SMD
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)

y = np.arange(len(feature_cols))[::-1]  # 표 3-2와 같은 순서로 위에서 아래로

ax.axvspan(-0.1, 0.1, color="#e8f2e6", zorder=0, label="균형 달성 구간 (|SMD| < 0.1)")
ax.axvline(0, color="#999999", linewidth=1, zorder=1)
for x in (-0.2, -0.1, 0.1, 0.2):
    ax.axvline(x, color="#bbbbbb", linestyle="--", linewidth=0.8, zorder=1)

for yi, sb, sa in zip(y, smd_before, smd_after):
    ax.plot([sb, sa], [yi, yi], color="#cccccc", linewidth=1.2, zorder=2)
ax.scatter(smd_before, y, s=70, facecolors="white", edgecolors=ORANGE,
           linewidths=2, zorder=3, label="매칭 전")
ax.scatter(smd_after, y, s=70, color=BLUE, zorder=4, label="매칭 후")

for yi, sa in zip(y, smd_after):
    ax.annotate(f"{sa:.3f}", (sa, yi), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=9, color="#333333")

ax.set_yticks(y)
ax.set_yticklabels(feature_labels)
ax.set_xlabel("SMD (표준화 평균차)")
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(out_dir / "3-2.png")
plt.close(fig)

print("\n그림 저장 완료:")
print(f"  {out_dir / '3-1.png'}")
print(f"  {out_dir / '3-2.png'}")
