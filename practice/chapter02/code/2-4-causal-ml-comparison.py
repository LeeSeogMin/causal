"""
제2장: Causal ML 핵심 방법론 비교
Causal Forest, X-learner, R-learner, S-learner, T-learner 성능 비교
문서의 이론을 뒷받침하기 위해 EconML 패키지(권장) 또는 단순 구현체(Fallback)를 사용합니다.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
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
        'correlation': correlation,
        'rmse': rmse,
        'ate_bias': ate_bias
    }

def visualize_comparison(true_cate, true_ate, results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    methods = [r['method'] for r in results]
    corrs = [r['correlation'] for r in results]
    rmses = [r['rmse'] for r in results]
    biases = [r['ate_bias'] for r in results]

    # 1. 상관계수
    sns.barplot(x=methods, y=corrs, ax=axes[0], palette='viridis')
    axes[0].set_title('Correlation with True CATE')
    axes[0].set_ylim(0, 1)

    # 2. RMSE
    sns.barplot(x=methods, y=rmses, ax=axes[1], palette='rocket')
    axes[1].set_title('RMSE (Lower is Better)')

    # 3. Bias
    sns.barplot(x=methods, y=biases, ax=axes[2], palette='coolwarm')
    axes[2].set_title('ATE Bias')
    axes[2].axhline(0, color='black', linestyle='--')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'causal_ml_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"\n결과 그래프 저장됨: {output_path}")

# =============================================================================
# 단순 구현체 (EconML 없을 경우 사용)
# =============================================================================
class SimpleSLearner:
    def __init__(self, model): self.model = model
    def fit(self, X, D, Y):
        self.model.fit(np.column_stack([X, D]), Y)
        return self
    def effect(self, X):
        return self.model.predict(np.column_stack([X, np.ones(len(X))])) - \
               self.model.predict(np.column_stack([X, np.zeros(len(X))]))

class SimpleTLearner:
    def __init__(self, model0, model1): 
        self.model0, self.model1 = model0, model1
    def fit(self, X, D, Y):
        self.model0.fit(X[D==0], Y[D==0])
        self.model1.fit(X[D==1], Y[D==1])
        return self
    def effect(self, X):
        return self.model1.predict(X) - self.model0.predict(X)

class SimpleXLearner:
    def __init__(self, model0, model1, prop_model):
        self.m0, self.m1, self.pm = model0, model1, prop_model
        self.tau0, self.tau1 = GradientBoostingRegressor(), GradientBoostingRegressor()
    def fit(self, X, D, Y):
        self.m0.fit(X[D==0], Y[D==0]); self.m1.fit(X[D==1], Y[D==1])
        imp1 = Y[D==1] - self.m0.predict(X[D==1])
        imp0 = self.m1.predict(X[D==0]) - Y[D==0]
        self.tau0.fit(X[D==0], imp0); self.tau1.fit(X[D==1], imp1)
        self.pm.fit(X, D)
        return self
    def effect(self, X):
        ps = self.pm.predict_proba(X)[:,1]
        return ps*self.tau0.predict(X) + (1-ps)*self.tau1.predict(X)

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

    results = []

    # 2. EconML 활용 시도
    try:
        from econml.dml import CausalForestDML, LinearDML
        from econml.metalearners import XLearner, SLearner, TLearner
        print("\n[EconML] 패키지 로드 성공. 고급 방법론을 사용하여 분석합니다.")

        # EconML Models
        # 공통 Base Models
        est_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        prop_model = GradientBoostingRegressor(n_estimators=100, max_depth=3) # or Classifier
        
        # 1. Causal Forest
        print("Training Causal Forest...")
        cf = CausalForestDML(model_y=est_model, model_t=prop_model, 
                             n_estimators=1000, discrete_treatment=True, random_state=42)
        cf.fit(y, T, X=X) # EconML signature: Y, T, X
        cate_cf = cf.effect(X)
        results.append(evaluate_cate_estimator(true_cate, cate_cf, true_ate, "Causal Forest"))

        # 2. X-learner
        print("Training X-learner...")
        xl = XLearner(models=est_model, propensity_model=LogisticRegression())
        xl.fit(y, T, X=X)
        cate_xl = xl.effect(X)
        results.append(evaluate_cate_estimator(true_cate, cate_xl, true_ate, "X-learner"))
        
        # 3. R-learner (LinearDML)
        print("Training R-learner (DML)...")
        rl = LinearDML(model_y=est_model, model_t=prop_model, discrete_treatment=True, random_state=42)
        rl.fit(y, T, X=X)
        cate_rl = rl.effect(X)
        results.append(evaluate_cate_estimator(true_cate, cate_rl, true_ate, "R-learner"))
        
        # 4. S-learner
        print("Training S-learner...")
        sl = SLearner(overall_model=est_model)
        sl.fit(y, T, X=X)
        cate_sl = sl.effect(X)
        results.append(evaluate_cate_estimator(true_cate, cate_sl, true_ate, "S-learner"))
        
        # 5. T-learner
        print("Training T-learner...")
        tl = TLearner(models=est_model)
        tl.fit(y, T, X=X)
        cate_tl = tl.effect(X)
        results.append(evaluate_cate_estimator(true_cate, cate_tl, true_ate, "T-learner"))

    except ImportError:
        print("\n[경로/설치 주의] 'econml' 패키지가 설치되지 않았습니다.")
        print("  -> pip install econml")
        print("  -> 또는 practice/venv 가상환경이 활성화되었는지 확인하세요.")
        print("\n[Fallback] 단순 구현체(Simple Implementations)를 사용하여 교육용 예제를 실행합니다.")
        print("주의: 이 결과는 문서의 EconML 기반 결과와 다를 수 있으며, 이론적 성능을 완전히 대변하지 못합니다.")
        
        # Simple Models
        print("Training Simple Causal Forest (RF T-learner)...")
        cf = SimpleCausalForest().fit(X, T, y)
        results.append(evaluate_cate_estimator(true_cate, cf.effect(X), true_ate, "Causal Forest*"))
        
        print("Training Simple X-learner...")
        xl = SimpleXLearner(GradientBoostingRegressor(), GradientBoostingRegressor(), LogisticRegression())
        xl.fit(X, T, y)
        results.append(evaluate_cate_estimator(true_cate, xl.effect(X), true_ate, "X-learner*"))
        
        # Simple others... (Skip for brevity or implement if needed)
        print("Training Simple S-learner...")
        sl = SimpleSLearner(GradientBoostingRegressor()).fit(X, T, y)
        results.append(evaluate_cate_estimator(true_cate, sl.effect(X), true_ate, "S-learner*"))

    # 3. 결과 출력 및 비교
    print("\n" + "=" * 80)
    print("Causal ML 방법론 성능 비교 결과")
    print("=" * 80)
    print(f"{'Method':<15} | {'RMSE':>8} | {'Corr':>8} | {'Bias':>8} | {'Mean CATE':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:<15} | {r['rmse']:8.3f} | {r['correlation']:8.3f} | {r['ate_bias']:8.3f} | {r['mean_cate']:10.3f}")
    print("-" * 80)
    print(f"{'True CATE':<15} | {'0.000':>8} | {'1.000':>8} | {'0.000':>8} | {true_ate:10.3f}")
    
    # 시각화
    visualize_comparison(true_cate, true_ate, results)

if __name__ == "__main__":
    main()
