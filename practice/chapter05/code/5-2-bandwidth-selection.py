"""
제5장: 대역폭 선택 및 비모수적 추정
Imbens-Kalyanaraman 최적 대역폭 선택과 국소 다항 회귀
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from matplotlib import font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 랜덤 시드 설정
np.random.seed(42)

print("=" * 80)
print("대역폭 선택과 비모수적 RDD 추정")
print("=" * 80)

# 1. 데이터 로드
print("\n[1단계] 데이터 로드")
print("-" * 80)

# CSV 파일에서 데이터 읽기
import os
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/5-2-bandwidth-selection-data.csv')
df = pd.read_csv(data_path)

cutoff = 50
treatment_effect = 3.76

print(f"데이터 파일: {data_path}")
print(f"총 관측치 수: {len(df)}")
print(f"기준점: {cutoff}")
print(f"배정 변수 표준편차: {df['score'].std():.2f}")

# 2. Imbens-Kalyanaraman 최적 대역폭 선택
print("\n[2단계] Imbens-Kalyanaraman 최적 대역폭 계산")
print("-" * 80)

def imbens_kalyanaraman_bandwidth(y, x, cutoff):
    """
    Imbens-Kalyanaraman 최적 대역폭 계산
    """
    # Step 1: 3차 다항 회귀로 잔차 분산 추정
    x_poly3 = np.column_stack([x, x**2, x**3])
    model_global = sm.OLS(y, sm.add_constant(x_poly3)).fit()
    sigma_sq = np.var(model_global.resid)

    # Step 2: 좌우에서 2차 회귀로 2차 도함수 추정
    left_mask = x < cutoff
    right_mask = x >= cutoff

    x_left = x[left_mask] - cutoff
    x_right = x[right_mask] - cutoff
    y_left = y[left_mask]
    y_right = y[right_mask]

    # 좌측 2차 회귀
    if len(x_left) > 3:
        x_left_poly = np.column_stack([x_left, x_left**2])
        model_left = sm.OLS(y_left, sm.add_constant(x_left_poly)).fit()
        m2_left = 2 * model_left.params[2] if len(model_left.params) > 2 else 0
    else:
        m2_left = 0

    # 우측 2차 회귀
    if len(x_right) > 3:
        x_right_poly = np.column_stack([x_right, x_right**2])
        model_right = sm.OLS(y_right, sm.add_constant(x_right_poly)).fit()
        m2_right = 2 * model_right.params[2] if len(model_right.params) > 2 else 0
    else:
        m2_right = 0

    # Step 3: 최적 대역폭 계산
    N = len(y)
    C_K = 3.4375  # 삼각 커널 상수

    # 올바른 IK 방법: 좌우 2차 도함수의 절댓값 평균 사용
    # (Imbens & Kalyanaraman 2012 원 논문 기준)
    m2 = (abs(m2_left) + abs(m2_right)) / 2

    if m2 == 0 or N == 0:
        h_opt = df['score'].std()
    else:
        h_opt = C_K * ((sigma_sq / (N * m2**2)) ** 0.2)

    results = {
        'h_optimal': h_opt,
        'sigma_squared': sigma_sq,
        'm2_left': m2_left,
        'm2_right': m2_right,
        'm2_avg': m2,  # 평균값으로 변경
        'n_left': left_mask.sum(),
        'n_right': right_mask.sum()
    }

    return results

ik_results = imbens_kalyanaraman_bandwidth(
    df['outcome'].values,
    df['score'].values,
    cutoff
)

print(f"\nIK 최적 대역폭: {ik_results['h_optimal']:.2f}")
print(f"잔차 분산 (σ²): {ik_results['sigma_squared']:.2f}")
print(f"좌측 2차 도함수 (m₂): {ik_results['m2_left']:.4f}")
print(f"우측 2차 도함수 (m₂): {ik_results['m2_right']:.4f}")
print(f"좌측 표본 크기: {ik_results['n_left']}")
print(f"우측 표본 크기: {ik_results['n_right']}")

# 3. 다양한 대역폭 선택 방법 비교
print("\n[3단계] 대역폭 선택 방법 비교")
print("-" * 80)

# 여러 대역폭 선택 방법
bandwidth_methods = {
    'IK (2012)': ik_results['h_optimal'],
    'Rule-of-thumb (1σ)': df['score'].std(),
    'Half rule-of-thumb': 0.5 * df['score'].std(),
    'Double rule-of-thumb': 2.0 * df['score'].std()
}

comparison_results = []

for method_name, h in bandwidth_methods.items():
    near = df[np.abs(df['score'] - cutoff) <= h].copy()
    near['score_centered'] = near['score'] - cutoff
    near['treat_score'] = near['treat'] * near['score_centered']

    if len(near) > 10:
        model = smf.ols('outcome ~ treat + score_centered + treat_score', data=near).fit()

        # MSE 계산 (편향² + 분산)
        bias_sq = (model.params['treat'] - treatment_effect) ** 2
        variance = model.bse['treat'] ** 2
        mse = bias_sq + variance

        comparison_results.append({
            'method': method_name,
            'bandwidth': h,
            'estimate': model.params['treat'],
            'se': model.bse['treat'],
            'ci_lower': model.conf_int().loc['treat', 0],
            'ci_upper': model.conf_int().loc['treat', 1],
            'n_eff': len(near),
            'mse': mse
        })

comparison_df = pd.DataFrame(comparison_results)
print("\n대역폭 선택 방법 비교:")
print(comparison_df.to_string(index=False))

# 4. 커널 함수 선택의 영향
print("\n[4단계] 커널 함수 선택의 영향")
print("-" * 80)

h = ik_results['h_optimal']
near = df[np.abs(df['score'] - cutoff) <= h].copy()
near['score_centered'] = near['score'] - cutoff
near['treat_score'] = near['treat'] * near['score_centered']

# 삼각 커널 가중치
near['dist'] = np.abs(near['score_centered']) / h
near['weight_triangular'] = np.maximum(1 - near['dist'], 0)

# 엡스타인 커널 가중치
near['weight_epanechnikov'] = np.maximum(0.75 * (1 - near['dist']**2), 0)

# 균등 커널 (가중치 1)
near['weight_uniform'] = 1.0

kernel_results = []

for kernel_name, weight_col in [('Triangular', 'weight_triangular'),
                                 ('Epanechnikov', 'weight_epanechnikov'),
                                 ('Uniform', 'weight_uniform')]:
    model_wls = sm.WLS.from_formula('outcome ~ treat + score_centered + treat_score',
                                     data=near, weights=near[weight_col]).fit()

    kernel_results.append({
        'kernel': kernel_name,
        'estimate': model_wls.params['treat'],
        'se': model_wls.bse['treat']
    })

kernel_df = pd.DataFrame(kernel_results)
print("\n커널 함수별 추정 결과:")
print(kernel_df.to_string(index=False))

# 5. 교차 검증 기반 대역폭 선택
print("\n[5단계] 교차 검증 기반 대역폭 선택")
print("-" * 80)

def cv_bandwidth_selection(df, cutoff, h_candidates, n_folds=5):
    """
    교차 검증으로 최적 대역폭 선택
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []

    for h in h_candidates:
        fold_mse = []

        for train_idx, test_idx in kf.split(df):
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]

            # 훈련 데이터에서 모델 학습
            near_train = train_data[np.abs(train_data['score'] - cutoff) <= h].copy()
            if len(near_train) < 10:
                continue

            near_train['score_centered'] = near_train['score'] - cutoff
            near_train['treat_score'] = near_train['treat'] * near_train['score_centered']

            model = smf.ols('outcome ~ treat + score_centered + treat_score',
                           data=near_train).fit()

            # 테스트 데이터에서 예측
            near_test = test_data[np.abs(test_data['score'] - cutoff) <= h].copy()
            if len(near_test) == 0:
                continue

            near_test['score_centered'] = near_test['score'] - cutoff
            near_test['treat_score'] = near_test['treat'] * near_test['score_centered']

            predictions = model.predict(near_test)
            mse = np.mean((near_test['outcome'] - predictions) ** 2)
            fold_mse.append(mse)

        if len(fold_mse) > 0:
            cv_scores.append({
                'bandwidth': h,
                'cv_mse': np.mean(fold_mse)
            })

    return pd.DataFrame(cv_scores)

h_candidates = np.linspace(3, 15, 25)
cv_results = cv_bandwidth_selection(df, cutoff, h_candidates)

if len(cv_results) > 0:
    best_h_cv = cv_results.loc[cv_results['cv_mse'].idxmin(), 'bandwidth']
    print(f"\n교차 검증 최적 대역폭: {best_h_cv:.2f}")
    print(f"최소 CV MSE: {cv_results['cv_mse'].min():.2f}")

# 6. 시각화
print("\n[6단계] 시각화 생성")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) 대역폭 비교
ax = axes[0, 0]
x_pos = np.arange(len(comparison_df))
bars = ax.bar(x_pos, comparison_df['estimate'], yerr=1.96*comparison_df['se'],
              capsize=5, alpha=0.7, color='steelblue')
ax.axhline(treatment_effect, color='red', linestyle='--', linewidth=2, label='진정한 효과')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df['method'], rotation=15, ha='right', fontsize=9)
ax.set_ylabel('처치효과 추정값', fontsize=11)
ax.set_title('(a) 대역폭 선택 방법별 추정값', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# (b) MSE 비교
ax = axes[0, 1]
ax.bar(x_pos, comparison_df['mse'], alpha=0.7, color='coral')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df['method'], rotation=15, ha='right', fontsize=9)
ax.set_ylabel('MSE', fontsize=11)
ax.set_title('(b) 대역폭 선택 방법별 MSE', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# (c) 교차 검증 결과
ax = axes[1, 0]
if len(cv_results) > 0:
    ax.plot(cv_results['bandwidth'], cv_results['cv_mse'], 'o-', color='darkgreen', markersize=6)
    ax.axvline(best_h_cv, color='red', linestyle='--', linewidth=2, label=f'최적 h={best_h_cv:.2f}')
    ax.axvline(ik_results['h_optimal'], color='blue', linestyle='--', linewidth=2,
               label=f'IK h={ik_results["h_optimal"]:.2f}')
    ax.set_xlabel('대역폭', fontsize=11)
    ax.set_ylabel('교차 검증 MSE', fontsize=11)
    ax.set_title('(c) 교차 검증 기반 대역폭 선택', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

# (d) 커널 함수 비교
ax = axes[1, 1]
x_pos_kernel = np.arange(len(kernel_df))
ax.bar(x_pos_kernel, kernel_df['estimate'], yerr=1.96*kernel_df['se'],
       capsize=5, alpha=0.7, color='purple')
ax.axhline(treatment_effect, color='red', linestyle='--', linewidth=2, label='진정한 효과')
ax.set_xticks(x_pos_kernel)
ax.set_xticklabels(kernel_df['kernel'])
ax.set_ylabel('처치효과 추정값', fontsize=11)
ax.set_title('(d) 커널 함수별 추정값', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
_out = os.path.join(base_path, '5-2-bandwidth-selection.png')
plt.savefig(_out, dpi=300, bbox_inches='tight')
print(f"그래프 저장 완료: {_out}")

# 7. 최종 요약
print("\n" + "=" * 80)
print("대역폭 선택 분석 요약")
print("=" * 80)
print(f"✓ IK 최적 대역폭: {ik_results['h_optimal']:.2f}")
print(f"✓ IK 방법 추정값: {comparison_df[comparison_df['method']=='IK (2012)']['estimate'].values[0]:.2f}")
print(f"✓ 최소 MSE 방법: {comparison_df.loc[comparison_df['mse'].idxmin(), 'method']}")
print(f"✓ 커널 함수 영향: 미미함 (추정값 범위 {kernel_df['estimate'].min():.2f}~{kernel_df['estimate'].max():.2f})")
if len(cv_results) > 0:
    print(f"✓ 교차 검증 최적 대역폭: {best_h_cv:.2f}")
print("=" * 80)
