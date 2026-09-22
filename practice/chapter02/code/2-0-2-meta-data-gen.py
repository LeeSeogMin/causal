"""
제2장: 데이터 생성 (Meta-learners 및 Causal Forest 비교용)
이질적 처치효과(Heterogeneous Treatment Effects)가 포함된 합성 데이터 생성.
Causal Forest와 메타 학습기들의 성능 차이를 드러내기 위해 복잡한 구조를 가짐.
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


import numpy as np
import pandas as pd
from scipy.special import expit
import os

def generate_meta_learner_data(n=2000, seed=42):
    """
    메타 학습기 비교를 위한 이질적 처치효과 데이터 생성
    - X-learner가 유리하도록 약간의 불균형 처치 배정 (또는 복잡한 CATE)
    - Causal Forest가 잘 작동하는 비선형 CATE 구조
    """
    np.random.seed(seed)

    # 1. 공변량 생성 (5~10차원)
    # n=2000, p=10
    n_features = 10
    X = np.random.randn(n, n_features)
    
    # 2. 성향점수 (Propensity Score)
    # 약간의 선택 편향 및 불균형 (평균 처치율 0.2~0.3 정도로 낮춰서 불균형 상황 연출 가능)
    # 여기서는 적당한 선택편향만 부여
    # P(T=1|X) = sigmoid(X1 - 0.5*X2)
    logits = 0.5 * X[:, 0] - 0.5 * X[:, 1]
    true_propensity = expit(logits)
    
    # 처치 배정
    treatment = np.random.binomial(1, true_propensity)

    # 3. 이질적 처치효과 (CATE)
    # 복잡한 비선형 구조:
    # tau(x) = exp(X1) + 2.5 * sin(pi * X2) (Wager & Athey (2018) 유사 패턴)
    # X1, X2에 강하게 의존
    cate_true = np.exp(X[:, 0]) + 2.5 * np.sin(np.pi * X[:, 1])
    
    # 베이스라인 결과 (Y0)
    # Y0 = max(X3, 0) + X4 + 0.5 * X5 + noise
    # (처치효과와 교란요인이 겹치지 않게 하여 CATE 식별 난이도 조절)
    y0_true = np.maximum(X[:, 2], 0) + X[:, 3] + 0.5 * X[:, 4]
    
    # 노이즈
    epsilon = np.random.randn(n) * 0.5
    
    # 관측 결과 Y
    y1_true = y0_true + cate_true
    y_observed = treatment * y1_true + (1 - treatment) * y0_true + epsilon

    # 데이터프레임 생성
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['treatment'] = treatment
    df['outcome'] = y_observed
    
    # 메타데이터 (참값)
    df['true_cate'] = cate_true
    df['true_propensity'] = true_propensity
    
    true_ate = np.mean(cate_true)

    print(f"Data Generated: N={n}, Features={n_features}")
    print(f"Treatment Rate: {np.mean(treatment):.4f}")
    print(f"True ATE: {true_ate:.4f}")
    print(f"CATE Range: [{np.min(cate_true):.2f}, {np.max(cate_true):.2f}]")

    return df, true_ate

def main():
    df, true_ate = generate_meta_learner_data(n=2000, seed=123)
    
    # 저장 경로: practice/chapter02/data/2-4-causal-ml.csv
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '../data')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    save_path = os.path.join(data_dir, '2-4-causal-ml.csv')
    df.to_csv(save_path, index=False)
    print(f"데이터 저장 완료: {save_path}")
    
    # 메타 정보 저장
    meta_path = os.path.join(data_dir, '2-4-causal-ml-meta.txt')
    with open(meta_path, 'w') as f:
        f.write(f"True ATE: {true_ate}\n")

if __name__ == "__main__":
    main()
