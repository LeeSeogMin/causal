"""
간단한 DML 테스트 - 성향점수 모델 수정 검증
"""
import numpy as np
from scipy.special import expit
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# 간단한 데이터 생성
np.random.seed(42)
n = 100
X = np.random.randn(n, 3)
true_propensity = expit(0.5 * X[:, 0] - 0.3 * X[:, 1])
treatment = np.random.binomial(1, true_propensity)
Y = 2 + X[:, 0] - 0.5 * X[:, 1] + 3 * treatment + np.random.randn(n) * 0.5

print("=" * 60)
print("성향점수 모델 수정 검증")
print("=" * 60)

# 기존 방법 (잘못됨): Regressor
print("\n1. 기존 방법 (GradientBoostingRegressor):")
try:
    model_reg = GradientBoostingRegressor(n_estimators=50, max_depth=3)
    model_reg.fit(X, treatment)
    pred_reg = model_reg.predict(X)
    r2_reg = 1 - np.mean((treatment - pred_reg)**2) / np.var(treatment)
    print(f"   - 예측값 범위: [{pred_reg.min():.3f}, {pred_reg.max():.3f}]")
    print(f"   - R²: {r2_reg:.3f}")
    print(f"   - 문제: 예측값이 [0, 1] 범위를 벗어남!" if (pred_reg.min() < 0 or pred_reg.max() > 1) else "   - OK")
except Exception as e:
    print(f"   - 오류: {e}")

# 수정 방법 (올바름): Classifier
print("\n2. 수정 방법 (GradientBoostingClassifier):")
try:
    model_clf = GradientBoostingClassifier(n_estimators=50, max_depth=3)
    model_clf.fit(X, treatment)
    pred_clf = model_clf.predict_proba(X)[:, 1]
    r2_clf = 1 - np.mean((treatment - pred_clf)**2) / np.var(treatment)
    print(f"   - 예측값 범위: [{pred_clf.min():.3f}, {pred_clf.max():.3f}]")
    print(f"   - R²: {r2_clf:.3f}")
    print(f"   - 올바름: 예측값이 [0, 1] 범위 내!" if (0 <= pred_clf.min() and pred_clf.max() <= 1) else "   - 문제 있음")
except Exception as e:
    print(f"   - 오류: {e}")

print("\n" + "=" * 60)
print("결론: Classifier 사용 시 올바른 확률 예측 가능")
print("=" * 60)
