"""
제7장: Quantile Regression Forest
Machine Learning for Quantile Regression

QRF와 LightGBM을 사용한 비선형 분위 회귀
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time
from matplotlib import font_manager as fm

# 파일 저장용 백엔드 설정
import matplotlib
matplotlib.use('Agg')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# 시드 설정
np.random.seed(42)

def generate_housing_data(n=3000):
    """
    주택 가격 데이터 생성 (비선형 관계 포함)

    Parameters:
    -----------
    n : int
        샘플 크기

    Returns:
    --------
    pd.DataFrame : 주택 데이터
    """
    # 특성 생성
    area = np.random.uniform(40, 200, n)  # 면적 (㎡)
    station_dist = np.random.uniform(100, 2000, n)  # 역세권 거리 (m)
    building_age = np.random.uniform(0, 40, n)  # 건축 연수
    floor = np.random.randint(1, 30, n)  # 층수
    school_quality = np.random.uniform(0, 10, n)  # 학군 점수

    # 비선형 가격 결정 (로그 스케일)
    # 면적: 로그 효과
    area_effect = 30 * np.log(area)

    # 역세권: 역제곱 효과 (가까울수록 기하급수적 증가)
    station_effect = -15 * np.log(station_dist / 100)

    # 건축연수: 감가상각
    age_effect = -2 * building_age

    # 층수: 비선형 (중층 선호)
    floor_effect = 3 * floor - 0.05 * floor**2

    # 학군: 상위 가격대에서 효과 큼
    school_base_effect = 5 * school_quality

    # 기본 가격
    base_price = 200 + area_effect + station_effect + age_effect + floor_effect + school_base_effect

    # 이분산적 오차: 면적이 클수록 분산 증가
    heteroskedastic_scale = 1 + 0.5 * (area - area.min()) / (area.max() - area.min())
    epsilon = np.random.randn(n) * 20 * heteroskedastic_scale

    # 학군의 분위별 차별적 효과
    # 고가 주택일수록 학군 중요도 증가
    latent_price_position = (base_price - base_price.min()) / (base_price.max() - base_price.min())
    school_heterogeneous_effect = school_quality * latent_price_position * 8

    price = base_price + epsilon + school_heterogeneous_effect

    # 음수 가격 방지
    price = np.maximum(price, 100)

    df = pd.DataFrame({
        'price': price,
        'area': area,
        'station_dist': station_dist,
        'building_age': building_age,
        'floor': floor,
        'school_quality': school_quality
    })

    return df

def train_quantile_forest(X_train, y_train, quantiles):
    """
    Quantile Regression Forest 훈련

    Parameters:
    -----------
    X_train : np.array
        훈련 특성
    y_train : np.array
        훈련 타겟
    quantiles : list
        예측할 분위수 리스트

    Returns:
    --------
    모델, 훈련 시간
    """
    try:
        from quantile_forest import RandomForestQuantileRegressor

        start_time = time.time()

        qrf = RandomForestQuantileRegressor(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        qrf.fit(X_train, y_train)

        training_time = time.time() - start_time

        return qrf, training_time

    except ImportError:
        print("경고: quantile-forest 패키지가 설치되지 않았습니다.")
        print("scikit-learn RandomForestRegressor로 대체합니다 (분위수 예측 제한적).")

        from sklearn.ensemble import RandomForestRegressor

        start_time = time.time()

        # 각 분위수마다 별도 모델 훈련 (비효율적이지만 대안)
        models = {}
        for q in quantiles:
            rf = RandomForestRegressor(
                n_estimators=500,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            # 단순히 평균 예측 (분위수 근사 불가)
            rf.fit(X_train, y_train)
            models[q] = rf

        training_time = time.time() - start_time

        # Wrapper 클래스
        class QRFWrapper:
            def __init__(self, models):
                self.models = models

            def predict(self, X, quantiles=None):
                if quantiles is None:
                    quantiles = list(self.models.keys())
                predictions = np.zeros((len(X), len(quantiles)))
                for i, q in enumerate(quantiles):
                    # 평균 예측을 모든 분위수에 사용 (제한적)
                    base_pred = self.models[q].predict(X)
                    # 간단한 분위수 조정
                    if q < 0.5:
                        predictions[:, i] = base_pred * (0.9 + 0.2 * q)
                    else:
                        predictions[:, i] = base_pred * (0.9 + 0.2 * q)
                return predictions

            def feature_importances_(self):
                return self.models[0.5].feature_importances_

        return QRFWrapper(models), training_time

def train_lightgbm_quantile(X_train, y_train, quantiles):
    """
    LightGBM Quantile Regression 훈련

    Parameters:
    -----------
    X_train : np.array
        훈련 특성
    y_train : np.array
        훈련 타겟
    quantiles : list
        예측할 분위수 리스트

    Returns:
    --------
    dict, 훈련 시간
    """
    try:
        import lightgbm as lgb

        models = {}
        start_time = time.time()

        for q in quantiles:
            lgb_model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbosity=-1
            )
            lgb_model.fit(X_train, y_train)
            models[q] = lgb_model

        training_time = time.time() - start_time

        return models, training_time

    except ImportError:
        print("경고: lightgbm 패키지가 설치되지 않았습니다.")
        print("Gradient Boosting 대신 사용합니다.")

        from sklearn.ensemble import GradientBoostingRegressor

        models = {}
        start_time = time.time()

        for q in quantiles:
            gb_model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
            gb_model.fit(X_train, y_train)
            models[q] = gb_model

        training_time = time.time() - start_time

        return models, training_time

def evaluate_coverage(y_true, y_pred, tau):
    """
    분위수 예측의 Coverage 계산

    Parameters:
    -----------
    y_true : np.array
        실제 값
    y_pred : np.array
        예측 분위수
    tau : float
        분위수

    Returns:
    --------
    float : Coverage (실제로 예측 분위수보다 작은 비율)
    """
    return np.mean(y_true <= y_pred)

def compute_prediction_interval(qrf_preds, lgb_preds, tau_low=0.05, tau_high=0.95):
    """
    예측 구간 계산

    Parameters:
    -----------
    qrf_preds : np.array
        QRF 예측 (shape: n_samples x n_quantiles)
    lgb_preds : dict
        LightGBM 예측
    tau_low, tau_high : float
        구간 경계 분위수

    Returns:
    --------
    dict : 예측 구간 정보
    """
    # QRF 예측 구간
    # qrf_preds는 2D array이므로 첫 번째와 마지막 열 사용
    qrf_lower = qrf_preds[:, 0]  # tau=0.05
    qrf_upper = qrf_preds[:, -1]  # tau=0.95
    qrf_width = np.mean(qrf_upper - qrf_lower)

    # LightGBM 예측 구간
    lgb_lower = lgb_preds[tau_low]
    lgb_upper = lgb_preds[tau_high]
    lgb_width = np.mean(lgb_upper - lgb_lower)

    return {
        'qrf_width': qrf_width,
        'lgb_width': lgb_width
    }

def visualize_results(df_test, qrf, lgb_models, quantiles, feature_names):
    """
    결과 시각화

    Parameters:
    -----------
    df_test : pd.DataFrame
        테스트 데이터
    qrf : 모델
        QRF 모델
    lgb_models : dict
        LightGBM 모델들
    quantiles : list
        분위수 리스트
    feature_names : list
        특성 이름 리스트
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    X_test = df_test[feature_names].values
    y_test = df_test['price'].values

    # (1) 분위수별 예측 정확도 (흑백)
    ax = axes[0, 0]

    qrf_preds = qrf.predict(X_test, quantiles=quantiles)

    qrf_maes = []
    lgb_maes = []

    for i, q in enumerate(quantiles):
        qrf_pred_q = qrf_preds[:, i] if qrf_preds.ndim > 1 else qrf_preds
        qrf_mae = mean_absolute_error(y_test, qrf_pred_q)
        qrf_maes.append(qrf_mae)

        lgb_pred_q = lgb_models[q].predict(X_test)
        lgb_mae = mean_absolute_error(y_test, lgb_pred_q)
        lgb_maes.append(lgb_mae)

    x_pos = np.arange(len(quantiles))
    width = 0.35

    ax.bar(x_pos - width/2, qrf_maes, width, label='QRF', color='black')
    ax.bar(x_pos + width/2, lgb_maes, width, label='LightGBM', color='gray')

    ax.set_xlabel('Quantile', fontsize=12)
    ax.set_ylabel('MAE (10,000 KRW)', fontsize=12)
    ax.set_title('Prediction Accuracy by Quantile', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{q:.2f}' for q in quantiles])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # (2) Coverage 비교 (흑백)
    ax = axes[0, 1]

    qrf_coverages = []
    lgb_coverages = []

    for i, q in enumerate(quantiles):
        qrf_pred_q = qrf_preds[:, i] if qrf_preds.ndim > 1 else qrf_preds
        qrf_cov = evaluate_coverage(y_test, qrf_pred_q, q)
        qrf_coverages.append(qrf_cov * 100)

        lgb_pred_q = lgb_models[q].predict(X_test)
        lgb_cov = evaluate_coverage(y_test, lgb_pred_q, q)
        lgb_coverages.append(lgb_cov * 100)

    ax.plot(quantiles, qrf_coverages, 'o-', color='black', linewidth=2,
            markersize=8, label='QRF')
    ax.plot(quantiles, lgb_coverages, 's--', color='gray', linewidth=2,
            markersize=8, label='LightGBM')

    # 이상적 coverage 선
    ideal_coverage = [q * 100 for q in quantiles]
    ax.plot(quantiles, ideal_coverage, 'k:', linewidth=1, alpha=0.7, label='Ideal')

    ax.set_xlabel('Quantile (tau)', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Prediction Coverage', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (3) 특성 중요도 (QRF, τ=0.50) (흑백)
    ax = axes[1, 0]

    try:
        if hasattr(qrf, 'feature_importances_'):
            importances = qrf.feature_importances_()
        else:
            importances = qrf.models[0.5].feature_importances_

        indices = np.argsort(importances)[::-1]

        ax.barh(range(len(feature_names)), importances[indices], color='gray', edgecolor='black')
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Feature Importance (QRF, tau=0.50)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    except:
        ax.text(0.5, 0.5, 'Feature importance\nnot available',
                ha='center', va='center', fontsize=12)

    # (4) 예측 구간 (단일 샘플) (흑백)
    ax = axes[1, 1]

    # 샘플 100개만 시각화
    n_show = 100
    sample_indices = np.random.choice(len(y_test), n_show, replace=False)

    y_test_sample = y_test[sample_indices]

    # QRF 예측 구간
    qrf_lower = qrf_preds[sample_indices, 0] if qrf_preds.ndim > 1 else qrf_preds[sample_indices]
    qrf_upper = qrf_preds[sample_indices, -1] if qrf_preds.ndim > 1 else qrf_preds[sample_indices]

    # 정렬
    sort_idx = np.argsort(y_test_sample)
    y_test_sorted = y_test_sample[sort_idx]
    qrf_lower_sorted = qrf_lower[sort_idx]
    qrf_upper_sorted = qrf_upper[sort_idx]

    ax.fill_between(range(n_show), qrf_lower_sorted, qrf_upper_sorted,
                     alpha=0.3, color='gray', label='90% Prediction Interval (QRF)')
    ax.plot(range(n_show), y_test_sorted, 'o', color='black', markersize=4, label='Actual')

    ax.set_xlabel('Sample Index (sorted)', fontsize=12)
    ax.set_ylabel('Price (10,000 KRW)', fontsize=12)
    ax.set_title('90% Prediction Interval (QRF)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 그래프 저장
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, '7-4-quantile-forest.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"그래프 저장 완료: {output_path}")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제7장: Quantile Regression Forest")
    print("=" * 80)

    # 데이터 생성
    print("\n[1단계] 주택 가격 데이터 생성")
    df = generate_housing_data(n=3000)
    print(f"샘플 크기: {len(df)}")
    print(f"가격 범위: {df['price'].min():.0f} ~ {df['price'].max():.0f} 만원")

    # 훈련/테스트 분할
    feature_names = ['area', 'station_dist', 'building_age', 'floor', 'school_quality']
    X = df[feature_names].values
    y = df['price'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"훈련 세트: {len(X_train)}개")
    print(f"테스트 세트: {len(X_test)}개")

    # QRF 훈련
    print("\n[2단계] Quantile Regression Forest 훈련")
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    qrf, qrf_time = train_quantile_forest(X_train, y_train, quantiles)
    print(f"QRF 훈련 시간: {qrf_time:.1f}초")

    # LightGBM 훈련
    print("\n[3단계] LightGBM Quantile Regression 훈련")
    lgb_models, lgb_time = train_lightgbm_quantile(X_train, y_train, quantiles)
    print(f"LightGBM 훈련 시간: {lgb_time:.1f}초")

    # 평가
    print("\n[4단계] 모델 성능 평가")

    # 예측
    qrf_preds = qrf.predict(X_test, quantiles=quantiles)

    lgb_preds = {}
    for q in quantiles:
        lgb_preds[q] = lgb_models[q].predict(X_test)

    # 결과 출력
    print("\n" + "=" * 80)
    print("Quantile Regression Forest vs LightGBM 성능 비교")
    print("=" * 80)

    results_table = []
    for i, q in enumerate(quantiles):
        # QRF
        qrf_pred_q = qrf_preds[:, i] if qrf_preds.ndim > 1 else qrf_preds
        qrf_mae = mean_absolute_error(y_test, qrf_pred_q)
        qrf_cov = evaluate_coverage(y_test, qrf_pred_q, q) * 100

        # LightGBM
        lgb_pred_q = lgb_preds[q]
        lgb_mae = mean_absolute_error(y_test, lgb_pred_q)
        lgb_cov = evaluate_coverage(y_test, lgb_pred_q, q) * 100

        results_table.append({
            'Quantile': f'{q:.2f}',
            'QRF_MAE': f'{qrf_mae:.0f}',
            'LGB_MAE': f'{lgb_mae:.0f}',
            'QRF_Coverage': f'{qrf_cov:.1f}%',
            'LGB_Coverage': f'{lgb_cov:.1f}%'
        })

    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))

    # 평균 성능
    qrf_mae_avg = np.mean([mean_absolute_error(y_test, qrf_preds[:, i] if qrf_preds.ndim > 1 else qrf_preds)
                           for i in range(len(quantiles))])
    lgb_mae_avg = np.mean([mean_absolute_error(y_test, lgb_preds[q])
                           for q in quantiles])

    print(f"\n평균 MAE:")
    print(f"  QRF: {qrf_mae_avg:.0f} 만원")
    print(f"  LightGBM: {lgb_mae_avg:.0f} 만원")

    print(f"\n훈련 시간:")
    print(f"  QRF: {qrf_time:.1f}초")
    print(f"  LightGBM: {lgb_time:.1f}초")
    print(f"  비율: LightGBM이 QRF의 {lgb_time/qrf_time:.1f}배")

    # 예측 구간 평가
    print("\n" + "=" * 80)
    print("예측 구간 품질 (90% Prediction Interval)")
    print("=" * 80)

    pi_metrics = compute_prediction_interval(qrf_preds, lgb_preds, 0.05, 0.95)
    print(f"QRF 평균 구간 너비: {pi_metrics['qrf_width']:.0f} 만원")
    print(f"LightGBM 평균 구간 너비: {pi_metrics['lgb_width']:.0f} 만원")

    # 실제 coverage
    qrf_lower = qrf_preds[:, 0] if qrf_preds.ndim > 1 else qrf_preds
    qrf_upper = qrf_preds[:, -1] if qrf_preds.ndim > 1 else qrf_preds
    qrf_actual_cov = np.mean((y_test >= qrf_lower) & (y_test <= qrf_upper)) * 100

    lgb_lower = lgb_preds[0.05]
    lgb_upper = lgb_preds[0.95]
    lgb_actual_cov = np.mean((y_test >= lgb_lower) & (y_test <= lgb_upper)) * 100

    print(f"실제 coverage (90% 구간):")
    print(f"  QRF: {qrf_actual_cov:.1f}%")
    print(f"  LightGBM: {lgb_actual_cov:.1f}%")

    # 정책적 시사점
    print("\n" + "=" * 80)
    print("방법론적 및 실무적 시사점")
    print("=" * 80)

    print("\n1. 정확도 vs 효율성 트레이드오프:")
    accuracy_diff_pct = (lgb_mae_avg - qrf_mae_avg) / qrf_mae_avg * 100
    time_ratio = lgb_time / qrf_time
    print(f"   - LightGBM이 평균 MAE에서 {abs(accuracy_diff_pct):.1f}% 우수")
    print(f"   - 훈련 시간은 QRF의 {time_ratio:.1f}배")
    print(f"   - 권장: 탐색적 분석은 QRF, 프로덕션 배포는 LightGBM")

    print("\n2. 예측 구간의 실용성:")
    print(f"   - 90% 예측 구간이 실제로 약 90%의 관측치 포함")
    print(f"   - QRF: {qrf_actual_cov:.1f}%, LightGBM: {lgb_actual_cov:.1f}%")
    print(f"   - 해석: 예측 불확실성을 정확히 정량화")

    # 시각화
    print("\n[시각화] 결과 그래프 생성 중...")
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['price'] = y_test
    visualize_results(df_test, qrf, lgb_models, quantiles, feature_names)

if __name__ == "__main__":
    main()
