"""
제2장: 인과 포레스트(Causal Forest)로 개인별 처치효과 추정
강의자료 2.4.2절 표 2-6에 대응한다.
EconML 패키지(권장) 또는 단순 구현체(Fallback)를 사용합니다.
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


import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# 시각화 설정
plt.rcParams['font.family'] = 'Arial'
# Windows 한글 폰트 설정 (필요시)
# plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_data():
    """외부 데이터 로드"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '../data/2-4-causal-ml.csv')

    if not os.path.exists(data_path):
        print("="*60)
        print("[오류] 데이터 파일을 찾을 수 없습니다.")
        print(f"경로: {data_path}")
        print("해결 방법: python 2-0-2-meta-data-gen.py 를 먼저 실행하여 데이터를 생성하세요.")
        print("="*60)
        sys.exit(1)

    print(f"데이터 로드: {data_path}")
    return pd.read_csv(data_path)

# =============================================================================
# 평가 및 시각화 유틸리티
# =============================================================================
def evaluate_cate_estimator(true_cate, estimated_cate, true_ate, name="Method"):
    ate_estimated = np.mean(estimated_cate)
    correlation = np.corrcoef(true_cate, estimated_cate)[0, 1]
    rmse = np.sqrt(np.mean((true_cate - estimated_cate)**2))
    ate_bias = ate_estimated - true_ate

    return {
        'method': name,
        'mean_cate': ate_estimated,
        'cate_std': np.std(estimated_cate),
        'cate_min': np.min(estimated_cate),
        'cate_max': np.max(estimated_cate),
        'correlation': correlation,
        'rmse': rmse,
        'ate_bias': ate_bias
    }

def visualize_cate(true_cate, estimated_cate, result):
    """추정치가 참값을 얼마나 따라가는지, 그리고 얼마나 퍼져 있는지를 본다."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 1. 참값 대비 추정치 산점도 - 점이 45도선에 붙을수록 정확하다
    axes[0].scatter(true_cate, estimated_cate, s=8, alpha=0.3, color='steelblue')
    lo = min(true_cate.min(), estimated_cate.min())
    hi = max(true_cate.max(), estimated_cate.max())
    axes[0].plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='perfect fit')
    axes[0].set_xlabel('True CATE')
    axes[0].set_ylabel('Estimated CATE')
    axes[0].set_title(f"True vs Estimated (corr={result['correlation']:.3f}, "
                      f"RMSE={result['rmse']:.3f})")
    axes[0].legend()

    # 2. 추정 CATE 분포 - 평균 하나에 가려진 이질성의 폭을 본다
    axes[1].hist(estimated_cate, bins=50, color='seagreen', alpha=0.75)
    axes[1].axvline(result['mean_cate'], color='black', linestyle='--',
                    label=f"mean = {result['mean_cate']:.2f}")
    axes[1].set_xlabel('Estimated CATE')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f"Distribution (range {result['cate_min']:.2f} "
                      f"~ {result['cate_max']:.2f})")
    axes[1].legend()

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'causal_ml_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"\n결과 그래프 저장됨: {output_path}")

# =============================================================================
# 단순 구현체 (EconML 없을 경우 사용)
# =============================================================================
class SimpleCausalForest: # T-learner logic using RF
    def __init__(self):
        self.rf0 = RandomForestRegressor(n_estimators=100)
        self.rf1 = RandomForestRegressor(n_estimators=100)
    def fit(self, X, D, Y):
        self.rf0.fit(X[D==0], Y[D==0])
        self.rf1.fit(X[D==1], Y[D==1])
        return self
    def effect(self, X):
        return self.rf1.predict(X) - self.rf0.predict(X)

# =============================================================================
# 메인 함수
# =============================================================================
def main():
    print(f"Python 실행 환경: {sys.executable}")

    # 1. 데이터 로드
    df = load_data()
    feature_cols = [c for c in df.columns if c.startswith('X')]
    X = df[feature_cols].values
    T = df['treatment'].values
    y = df['outcome'].values
    true_cate = df['true_cate'].values
    true_ate = true_cate.mean()

    print(f"데이터 로드 완료. 샘플 수: {len(df)}")
    print(f"True ATE: {true_ate:.4f}")

    # 2. EconML 활용 시도
    try:
        from econml.dml import CausalForestDML
        print("\n[EconML] 패키지 로드 성공. 인과 포레스트로 분석합니다.")

        # 보조 모형: 결과 회귀와 처치 모형에 같은 설정을 쓴다
        # random_state를 주지 않으면 실행할 때마다 추정값이 조금씩 달라진다
        est_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        prop_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

        print("Training Causal Forest...")
        cf = CausalForestDML(model_y=est_model, model_t=prop_model,
                             n_estimators=1000, discrete_treatment=True, random_state=42)
        cf.fit(y, T, X=X) # EconML signature: Y, T, X
        cate_cf = cf.effect(X)
        result = evaluate_cate_estimator(true_cate, cate_cf, true_ate, "Causal Forest")

    except ImportError:
        print("\n[경로/설치 주의] 'econml' 패키지가 설치되지 않았습니다.")
        print("  -> pip install econml")
        print("  -> 또는 practice/venv 가상환경이 활성화되었는지 확인하세요.")
        print("\n[Fallback] 단순 구현체(Simple Implementation)를 사용하여 교육용 예제를 실행합니다.")
        print("주의: 이 결과는 문서의 EconML 기반 결과와 다를 수 있으며, 이론적 성능을 완전히 대변하지 못합니다.")

        print("Training Simple Causal Forest (RF T-learner)...")
        cf = SimpleCausalForest().fit(X, T, y)
        cate_cf = cf.effect(X)
        result = evaluate_cate_estimator(true_cate, cate_cf, true_ate, "Causal Forest*")

    # 3. 결과 출력 (강의자료 표 2-6에 대응)
    print("\n" + "=" * 70)
    print("인과 포레스트의 개인별 효과 추정 성능")
    print("=" * 70)
    print(f"{'지표':<22} | {'값':>18} | 무엇을 재는가")
    print("-" * 70)
    print(f"{'평균 CATE':<22} | {result['mean_cate']:>18.3f} | 개인별 추정치의 평균")
    print(f"{'참값 평균 CATE':<22} | {true_ate:>18.3f} | 비교 기준")
    print(f"{'ATE 편향':<22} | {result['ate_bias']:>18.3f} | 평균 효과가 참값에서 벗어난 정도")
    print(f"{'참값과의 상관':<22} | {result['correlation']:>18.3f} | 효과 순위를 맞힌 정도")
    print(f"{'RMSE':<22} | {result['rmse']:>18.3f} | 개인별 값이 벗어난 평균 크기")
    print(f"{'추정 CATE의 범위':<22} | {result['cate_min']:>8.2f} ~ {result['cate_max']:<7.2f} | 포레스트가 잡아낸 이질성의 폭")
    print(f"{'참값 CATE의 범위':<22} | {true_cate.min():>8.2f} ~ {true_cate.max():<7.2f} | 자료에 실제로 있는 이질성의 폭")
    print("-" * 70)

    # 4. 타겟팅 효율 (강의자료 표 2-7에 대응)
    # 추정 CATE로 순위를 매겨 상위 p%를 고른 뒤, 그 집단의 '참값' 평균 효과를 본다
    order = np.argsort(-cate_cf)
    base = true_cate.mean()
    print("\n" + "=" * 70)
    print("대상을 좁혔을 때의 효율 (추정 CATE 상위 p%를 선정)")
    print("=" * 70)
    print(f"{'타겟팅 전략':<18} | {'비율':>5} | {'1인당 효과':>10} | {'총 효과':>10} | {'효율성 지수':>10}")
    print("-" * 70)
    for p, label in [(1.0, '무차별 지급'), (0.5, 'CATE 상위 50%'),
                     (0.3, 'CATE 상위 30%'), (0.1, 'CATE 상위 10%')]:
        idx = order[:int(len(true_cate) * p)]
        per_capita = true_cate[idx].mean()
        print(f"{label:<18} | {int(p*100):4d}% | {per_capita:10.2f} | "
              f"{true_cate[idx].sum():10.0f} | {per_capita/base:10.2f}")
    print("-" * 70)
    print("1인당 효과와 총 효과가 반대로 움직인다. 예산이 제약이면 1인당 효과를,")
    print("총량 목표가 제약이면 총 효과를 기준으로 고른다.")

    # 판정 기준 안내 (강의자료 2.4.2 '결과를 읽는 법')
    print("[안내] 상관과 RMSE는 참값 CATE를 아는 시뮬레이션이라서 계산된다.")
    print("       실제 자료에서는 성격이 다른 추정기를 함께 돌려 평균 효과가 모이는지 보고,")
    print("       상위 집단으로 분류된 개체의 특성이 해석되는지 확인한다(2.5절 참조).")

    # 시각화
    visualize_cate(true_cate, cate_cf, result)

if __name__ == "__main__":
    main()
