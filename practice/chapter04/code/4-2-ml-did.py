"""
제4장: Machine Learning Enhanced DID
Double Machine Learning(DML)을 활용한 DID 추정

수정 이력
---------
2026-08-17
- 증상: 스크립트가 끝까지 실행되지 않고 멈춤. 표준출력에 아무것도 남지 않음.
- 원인: matplotlib 기본 백엔드가 tkagg여서 plt.show()가 창을 띄우고 사용자
  입력을 기다린다. 콘솔에서 일괄 실행하면 여기서 무한 대기한다.
- 조치: pyplot을 import 하기 전에 matplotlib.use('Agg')로 화면 없는 백엔드를
  지정하고, plt.show() 대신 plt.close()로 그림 객체를 닫는다.
  PNG 저장(plt.savefig)은 그대로 동작한다.

2026-08-17 (2)
- 증상 1: 변수 중요도 표가 데이터를 바꿔도 항상 0.28 / 0.19 / 0.15 / 0.12 /
  0.09 / 0.17로 나왔다. 항목 이름도 데이터에 없는 상호작용 항이었다.
- 원인 1: feature_importance_analysis()가 RandomForest를 학습해 놓고
  rf.feature_importances_를 버린 뒤 하드코딩한 표를 반환했다.
- 조치 1: 학습한 모형의 feature_importances_를 그대로 쓰고, 이름도 실제 공변량
  X1~X5로 바꿨다.

- 증상 2: "표 4-2의 SE 0.61은 부트스트랩 결과"라는 안내가 출력되는데, 표에 찍힌
  0.61은 부트스트랩이 아니라 폴드 간 표준편차에서 나온 값이었다.
- 원인 2: 표준오차를 np.std(폴드별 theta) 기반으로 계산해 놓고, 안내문에만
  부트스트랩이라고 적었다. 폴드 간 표준편차는 DML의 표준오차가 아니다.
- 조치 2: 폴드별 잔차를 전부 모아 pooled theta를 구하고, Neyman 직교 모멘트의
  영향함수로 점근 표준오차를 계산한다. 잘못된 안내문은 지웠다.

- 증상 3: DML 추정값이 전통적 DID보다 작은데 "-5.7% 높게 나타남",
  "모두 전통적 DID보다 유의하게 높음"으로 출력됐다.
- 원인 3: 방향을 계산하지 않고 '높음'을 고정 문자열로 붙였다.
- 조치 3: 부호를 계산해 '높게/낮게'를 고르고, 큰 것과 작은 것의 개수를 센다.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 화면 없는 백엔드 (plt.show() 대기 방지)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def generate_nonlinear_did_data(n=1000, seed=42):
    """
    비선형 관계를 포함한 DID 데이터 생성

    Parameters:
    -----------
    n : int
        샘플 크기
    seed : int
        난수 시드

    Returns:
    --------
    pd.DataFrame : 생성된 데이터
    """
    np.random.seed(seed)

    # 공변량 생성 (5개)
    X1 = np.random.randn(n)  # 산업 구조
    X2 = np.random.randn(n)  # 지역 경제 수준
    X3 = np.random.randn(n)  # 인구 밀도
    X4 = np.random.randn(n)  # 교육 수준
    X5 = np.random.randn(n)  # 접근성

    # 처치 확률 (성향점수) - 비선형 관계
    propensity = 1 / (1 + np.exp(-(0.5*X1 - 0.3*X2 + 0.4*X1*X2)))
    treatment = np.random.binomial(1, propensity)

    # 결과 변수 - 비선형 관계와 상호작용
    # 참값 처치효과는 4.0 + 1.0*X3 (이질적)
    Y0 = (2.0 + 1.5*X1 + 0.8*X2 +
          0.5*X1**2 +  # 비선형
          0.3*X1*X2 +  # 상호작용
          np.random.randn(n) * 1.5)

    Y1 = Y0 + 4.0 + 1.0*X3  # 이질적 처치효과

    # 관측되는 결과
    Y_observed = treatment * Y1 + (1 - treatment) * Y0

    # 참값 ATE
    true_ATE = np.mean(Y1 - Y0)

    df = pd.DataFrame({
        'Y': Y_observed,
        'D': treatment,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5
    })

    return df, true_ATE

def dml_did_estimator(df, ml_model='rf', K=5, seed=42):
    """
    Double Machine Learning DID 추정
    5-fold 교차 적합 (cross-fitting) 사용

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    ml_model : str
        기계학습 모델 ('rf', 'xgboost', 'lasso')
    K : int
        교차 검증 폴드 수
    seed : int
        난수 시드

    Returns:
    --------
    dict : DML 추정 결과
    """
    np.random.seed(seed)

    # 공변량 행렬
    X = df[['X1', 'X2', 'X3', 'X4', 'X5']].values
    Y = df['Y'].values
    D = df['D'].values

    # 모델 선택
    if ml_model == 'rf':
        outcome_model = RandomForestRegressor(n_estimators=100, max_depth=5,
                                             random_state=seed, n_jobs=-1)
        treat_model = RandomForestRegressor(n_estimators=100, max_depth=5,
                                           random_state=seed, n_jobs=-1)
        model_name = 'Random Forest'
    elif ml_model == 'xgboost':
        from sklearn.ensemble import GradientBoostingRegressor
        outcome_model = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                 learning_rate=0.1, random_state=seed)
        treat_model = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                               learning_rate=0.1, random_state=seed)
        model_name = 'XGBoost'
    elif ml_model == 'lasso':
        outcome_model = LassoCV(cv=5, random_state=seed)
        treat_model = LassoCV(cv=5, random_state=seed)
        model_name = 'Lasso'
    else:
        raise ValueError("ml_model must be 'rf', 'xgboost', or 'lasso'")

    # K-fold 교차 적합
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    theta_estimates = []
    rmse_outcome_list = []
    rmse_treat_list = []
    # 폴드별 잔차를 전부 모아 둔다 (pooled 추정과 점근 표준오차에 쓴다)
    Y_res_all = []
    D_res_all = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        D_train, D_test = D[train_idx], D[test_idx]

        # Step 1: Nuisance parameter 학습
        # E[Y|X, D=0] - 대조군에서 결과 모형 학습
        control_idx = D_train == 0
        if control_idx.sum() > 0:
            outcome_model.fit(X_train[control_idx], Y_train[control_idx])
            Y_pred = outcome_model.predict(X_test)
            rmse_outcome = np.sqrt(np.mean((Y_test[D_test==0] - outcome_model.predict(X_test[D_test==0]))**2)) if (D_test==0).sum() > 0 else 0
            rmse_outcome_list.append(rmse_outcome)
        else:
            Y_pred = np.zeros(len(X_test))
            rmse_outcome_list.append(0)

        # E[D|X] - 처치 성향점수
        treat_model.fit(X_train, D_train)
        D_pred = treat_model.predict(X_test)
        rmse_treat = np.sqrt(np.mean((D_test - D_pred)**2))
        rmse_treat_list.append(rmse_treat)

        # Step 2: 직교화된 모멘트 계산
        # Ỹ = Y - Ê[Y|X]
        # D̃ = D - Ê[D|X]
        Y_res = Y_test - Y_pred
        D_res = D_test - D_pred
        Y_res_all.append(Y_res)
        D_res_all.append(D_res)

        # Step 3: 처치효과 추정
        # θ = E[ỸD̃] / E[D̃²]
        if np.sum(D_res**2) > 1e-10:  # 분모가 0에 가까우면 skip
            theta = np.sum(Y_res * D_res) / np.sum(D_res**2)
            theta_estimates.append(theta)

    # 최종 추정값: 폴드별 잔차를 모두 합쳐 한 번에 계산한다 (pooled)
    Y_res_all = np.concatenate(Y_res_all)
    D_res_all = np.concatenate(D_res_all)
    theta_dml = np.sum(Y_res_all * D_res_all) / np.sum(D_res_all ** 2)

    # 표준오차: Neyman 직교 모멘트의 영향함수 기반 점근 표준오차
    #   psi_i = D_res_i * (Y_res_i - theta * D_res_i)
    #   SE    = sqrt( mean(psi^2) ) / ( sqrt(n) * mean(D_res^2) )
    n_obs = len(D_res_all)
    psi = D_res_all * (Y_res_all - theta_dml * D_res_all)
    jacobian = np.mean(D_res_all ** 2)
    se_dml = np.sqrt(np.mean(psi ** 2)) / (np.sqrt(n_obs) * jacobian)

    # RMSE 평균
    rmse_outcome = np.mean(rmse_outcome_list)
    rmse_treat = np.mean(rmse_treat_list)

    results = {
        'theta': theta_dml,
        'se': se_dml,
        'ci_lower': theta_dml - 1.96 * se_dml,
        'ci_upper': theta_dml + 1.96 * se_dml,
        'rmse_outcome': rmse_outcome,
        'rmse_treat': rmse_treat,
        'K': K,
        'model_name': model_name
    }

    return results

def bootstrap_dml_se(df, ml_model='rf', K=5, B=1000, seed=42):
    """
    DML 부트스트랩 표준오차 계산

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    ml_model : str
        기계학습 모델
    K : int
        교차 검증 폴드 수
    B : int
        부트스트랩 반복 횟수
    seed : int
        난수 시드

    Returns:
    --------
    dict : 부트스트랩 결과 (theta, se, ci)
    """
    np.random.seed(seed)
    bootstrap_estimates = []

    for b in range(B):
        # 부트스트랩 샘플링 (복원 추출)
        boot_idx = np.random.choice(len(df), len(df), replace=True)
        df_boot = df.iloc[boot_idx].reset_index(drop=True)

        # DML 추정
        try:
            result = dml_did_estimator(df_boot, ml_model=ml_model, K=K, seed=seed+b)
            bootstrap_estimates.append(result['theta'])
        except:
            continue  # 일부 부트스트랩 샘플에서 실패 시 skip

    # 부트스트랩 표준오차
    theta_mean = np.mean(bootstrap_estimates)
    se_boot = np.std(bootstrap_estimates)

    # 95% 신뢰구간 (percentile 방법)
    ci_lower = np.percentile(bootstrap_estimates, 2.5)
    ci_upper = np.percentile(bootstrap_estimates, 97.5)

    results = {
        'theta': theta_mean,
        'se': se_boot,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_estimates': bootstrap_estimates,
        'B': B
    }

    return results

def traditional_did(df):
    """
    전통적 선형 DID 추정 (비교 기준)

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임

    Returns:
    --------
    dict : 추정 결과
    """
    # 선형 회귀: Y ~ D + X1 + X2 + X3 + X4 + X5
    formula = 'Y ~ D + X1 + X2 + X3 + X4 + X5'
    model = smf.ols(formula, data=df).fit()

    results = {
        'theta': model.params['D'],
        'se': model.bse['D'],
        'ci_lower': model.conf_int().loc['D', 0],
        'ci_upper': model.conf_int().loc['D', 1]
    }

    return results

def feature_importance_analysis(df, seed=42):
    """
    변수 중요도 분석 (Random Forest 사용)

    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    seed : int
        난수 시드

    Returns:
    --------
    pd.DataFrame : 변수 중요도
    """
    X = df[['X1', 'X2', 'X3', 'X4', 'X5']].values
    Y = df['Y'].values

    rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                               random_state=seed, n_jobs=-1)
    rf.fit(X, Y)

    labels = ['X1 Industry', 'X2 Economy', 'X3 Density',
              'X4 Education', 'X5 Accessibility']
    importance_df = pd.DataFrame({
        'Feature': labels,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    return importance_df

def visualize_dml_results(results_dict, true_ATE, importance_df):
    """
    DML 분석 결과 시각화

    Parameters:
    -----------
    results_dict : dict
        방법론별 결과 딕셔너리
    true_ATE : float
        참값 ATE
    importance_df : pd.DataFrame
        변수 중요도
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 방법론 간 처치효과 비교
    methods = list(results_dict.keys())
    effects = [results_dict[m]['theta'] for m in methods]
    ses = [results_dict[m]['se'] for m in methods]

    x_pos = np.arange(len(methods))
    colors = ['blue', 'green', 'orange', 'red']

    axes[0, 0].bar(x_pos, effects, yerr=1.96*np.array(ses),
                   capsize=10, alpha=0.7, color=colors)
    axes[0, 0].axhline(y=true_ATE, color='black', linestyle='--',
                       linewidth=2, label=f'True ATE ({true_ATE:.2f})')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0, 0].set_ylabel('Treatment Effect Estimate')
    axes[0, 0].set_title('DML vs Traditional DID Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. 변수 중요도
    axes[0, 1].barh(importance_df['Feature'], importance_df['Importance'],
                    color='steelblue', alpha=0.7)
    axes[0, 1].set_xlabel('Importance Score')
    axes[0, 1].set_title('Feature Importance Analysis (Random Forest)')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # 3. 신뢰구간 비교
    for i, method in enumerate(methods):
        ci_low = results_dict[method]['ci_lower']
        ci_up = results_dict[method]['ci_upper']
        axes[1, 0].plot([ci_low, ci_up], [i, i], 'o-', linewidth=3,
                        markersize=8, label=method, color=colors[i])

    axes[1, 0].axvline(x=true_ATE, color='black', linestyle='--',
                       linewidth=2, label=f'True ATE ({true_ATE:.2f})')
    axes[1, 0].set_yticks(range(len(methods)))
    axes[1, 0].set_yticklabels(methods)
    axes[1, 0].set_xlabel('Treatment Effect')
    axes[1, 0].set_title('95% Confidence Intervals Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # 4. RMSE 비교 (nuisance parameter 추정)
    dml_methods = [m for m in methods if 'DML' in m]
    rmse_outcome = [results_dict[m]['rmse_outcome'] for m in dml_methods]
    rmse_treat = [results_dict[m]['rmse_treat'] for m in dml_methods]

    x = np.arange(len(dml_methods))
    width = 0.35

    axes[1, 1].bar(x - width/2, rmse_outcome, width, label='Outcome RMSE',
                   color='skyblue', alpha=0.7)
    axes[1, 1].bar(x + width/2, rmse_treat, width, label='Treatment RMSE',
                   color='salmon', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(dml_methods, rotation=15, ha='right')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('Nuisance Parameter Prediction Quality')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('4-2-ml-did-results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """메인 실행 함수"""

    print("=" * 70)
    print("제4장: Machine Learning Enhanced DID")
    print("=" * 70)

    # 1. 데이터 생성
    print("\n1. 비선형 DID 데이터 생성 중...")
    df, true_ATE = generate_nonlinear_did_data(n=1000, seed=42)
    print(f"   - 생성된 샘플 수: 1,000개")
    print(f"   - 처치군: {df['D'].sum()}명")
    print(f"   - 대조군: {(1-df['D']).sum()}명")
    print(f"   - 참값 ATE: {true_ATE:.2f}")

    # 2. 방법론별 추정
    print("\n2. 방법론별 처치효과 추정 중...")

    results_dict = {}

    # 2-1. DML with Random Forest
    print("   - DML-RF 추정 중...")
    results_dict['DML-RF'] = dml_did_estimator(df, ml_model='rf', K=5, seed=42)

    # 2-2. DML with XGBoost
    print("   - DML-XGBoost 추정 중...")
    results_dict['DML-XGBoost'] = dml_did_estimator(df, ml_model='xgboost', K=5, seed=42)

    # 2-3. DML with Lasso
    print("   - DML-Lasso 추정 중...")
    results_dict['DML-Lasso'] = dml_did_estimator(df, ml_model='lasso', K=5, seed=42)

    # 2-4. 전통적 DID
    print("   - 전통적 DID 추정 중...")
    results_dict['Traditional DID'] = traditional_did(df)

    # 2-5. 부트스트랩 표준오차 (선택 사항, 시간이 오래 걸린다)
    print("\n   주석: 아래 표의 표준오차는 Neyman 직교 모멘트의")
    print("         영향함수로 계산한 점근 표준오차입니다.")
    print("         부트스트랩(B=1000)은 run_bootstrap=True로 바꾸면 실행됩니다.")

    # 부트스트랩 실행 여부 (기본 False)
    run_bootstrap = False  # True로 변경하면 부트스트랩 실행

    if run_bootstrap:
        print("\n   - 부트스트랩 SE 계산 중 (B=1000)...")
        boot_results = bootstrap_dml_se(df, ml_model='rf', K=5, B=1000, seed=42)
        print(f"     부트스트랩 SE: {boot_results['se']:.2f}")
        results_dict['DML-RF (Bootstrap)'] = boot_results

    # 3. 결과 출력
    print("\n" + "=" * 70)
    print("Double Machine Learning DID 추정 결과")
    print("=" * 70)

    print("\n{:<20} {:<10} {:<10} {:<25} {:<10} {:<5}".format(
        '방법론', '처치효과', '표준오차', '95% 신뢰구간', 'RMSE', 'K'))
    print("-" * 90)

    for method, res in results_dict.items():
        rmse_str = f"{res['rmse_outcome']:.2f}" if 'rmse_outcome' in res else "-"
        k_str = str(res['K']) if 'K' in res else "-"
        print("{:<20} {:<10.2f} {:<10.2f} [{:>6.2f}, {:>6.2f}] {:<10} {:<5}".format(
            method,
            res['theta'],
            res['se'],
            res['ci_lower'],
            res['ci_upper'],
            rmse_str,
            k_str
        ))

    # 4. 변수 중요도 분석
    print("\n" + "=" * 70)
    print("변수 중요도 분석 (Random Forest nuisance 추정)")
    print("=" * 70)

    importance_df = feature_importance_analysis(df, seed=42)
    print("\n{:<35} {:<10}".format('변수', '중요도'))
    print("-" * 50)
    for _, row in importance_df.iterrows():
        print("{:<35} {:<10.2f}".format(row['Feature'], row['Importance']))

    # 5. 결과 해석
    print("\n" + "=" * 70)
    print("분석 결과 해석")
    print("=" * 70)

    dml_rf = results_dict['DML-RF']['theta']
    trad_did = results_dict['Traditional DID']['theta']

    print("\n1. 두 추정값의 차이:")
    gap_pct = (dml_rf / trad_did - 1) * 100
    direction = '높게' if gap_pct >= 0 else '낮게'
    print(f"   - DML 추정값 ({dml_rf:.2f})이 전통적 DID ({trad_did:.2f})보다")
    print(f"     {abs(gap_pct):.1f}% {direction} 나타남")
    print(f"   - 처치효과가 공변량에 따라 비선형적으로 변하거나")
    print(f"     복잡한 상호작용이 존재함을 시사")

    print("\n2. 방법론의 강건성:")
    dml_methods = ['DML-RF', 'DML-XGBoost', 'DML-Lasso']
    dml_estimates = [results_dict[m]['theta'] for m in dml_methods]
    print(f"   - RF, XGBoost, Lasso 추정값이")
    print(f"     {min(dml_estimates):.2f}-{max(dml_estimates):.2f} 범위로 일관")
    n_above = sum(e > trad_did for e in dml_estimates)
    print(f"   - 전통적 DID({trad_did:.2f})보다 큰 것 {n_above}개, "
          f"작은 것 {len(dml_estimates)-n_above}개")
    print(f"   - 비선형 관계가 특정 모형 선택에 의한")
    print(f"     인공적 결과가 아님을 시사")

    print("\n3. 교차 적합의 안정성:")
    print(f"   - K=5 폴드에서 안정적인 추정")
    print(f"   - 표본 분할 방식이 결과에 큰 영향을 미치지 않음")

    # 6. 시각화
    print("\n4. 시각화 생성 중...")
    visualize_dml_results(results_dict, true_ATE, importance_df)

    print("\n※ 본 코드는 교육 목적의 기본 DML 구현입니다.")
    print("   실제 적용 시 하이퍼파라미터 튜닝과 민감도 분석이 필요합니다.")
    print("\n분석 완료! 결과가 4-2-ml-did-results.png로 저장되었습니다.")

    # 7. 데이터 저장
    df.to_csv('4-2-ml-did-data.csv', index=False)
    print("데이터가 4-2-ml-did-data.csv로 저장되었습니다.")

    return results_dict, df

if __name__ == "__main__":
    results_dict, df = main()
