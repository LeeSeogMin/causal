"""
제7장: Quantile Regression Forest
Machine Learning for Quantile Regression

QRF와 LightGBM을 사용한 비선형 분위 회귀

수정 이력 (2026-08-17)
---------------------
1. quantile-forest 패키지가 없어 sklearn RandomForestRegressor로 대체되고 있었다.
   대체 코드는 평균 예측에 (0.9 + 0.2*tau)를 곱하는 임시 보정이라 분위수 예측이
   아니었고, 그 결과 tau=0.05의 coverage가 14%로 나왔다.
   -> pip install quantile-forest로 실제 QRF를 설치해 쓴다. 대체 경로는 안내
      문구만 남기고 결과를 쓰지 못하도록 예외를 던진다.
2. 특성 중요도 그림에서 qrf.feature_importances_()를 함수처럼 호출했다.
   실제 QRF에서는 배열 속성이라 TypeError가 나고, bare except가 그 오류를 삼켜
   그림이 "not available"로 비어 있었다.
   -> 속성/함수를 구분해 읽도록 고쳤다.
3. 분위수 예측의 정확도를 MAE로 쟀다. MAE는 중위수를 최적으로 하는 손실이라
   극단 분위수 예측을 평가하는 척도가 아니다.
   -> 분위 회귀의 고유 손실인 pinball loss를 함께 계산해 출력한다.
4. "90% 예측 구간이 실제로 약 90%의 관측치 포함"이라는 문장이 실제 값(73%)과
   무관하게 박혀 있었다.
   -> 실제 값을 읽어 판정하도록 고쳤다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time

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
    except ImportError:
        raise SystemExit(
            "quantile-forest 패키지가 필요하다. 다음을 실행한 뒤 다시 돌린다.\n"
            "    pip install quantile-forest\n"
            "평균을 예측하는 RandomForestRegressor로 대신하면 분위수 예측이 되지 않는다."
        )

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


def pinball_loss(y_true, y_pred, tau):
    """
    분위 회귀의 고유 손실 (check function). 값이 작을수록 좋다.

    L = mean( tau * max(y - yhat, 0) + (1 - tau) * max(yhat - y, 0) )

    MAE는 tau=0.5에서만 올바른 척도다. tau=0.05 예측을 MAE로 재면
    "값을 중앙으로 끌어올린 예측"이 이겨 버린다.
    """
    resid = y_true - y_pred
    return np.mean(np.maximum(tau * resid, (tau - 1) * resid))

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
        qrf_maes.append(pinball_loss(y_test, qrf_pred_q, q))

        lgb_pred_q = lgb_models[q].predict(X_test)
        lgb_maes.append(pinball_loss(y_test, lgb_pred_q, q))

    x_pos = np.arange(len(quantiles))
    width = 0.35

    ax.bar(x_pos - width/2, qrf_maes, width, label='QRF', color='black')
    ax.bar(x_pos + width/2, lgb_maes, width, label='LightGBM', color='gray')

    ax.set_xlabel('Quantile', fontsize=12)
    ax.set_ylabel('Pinball loss (10,000 KRW)', fontsize=12)
    ax.set_title('Pinball Loss by Quantile (lower is better)', fontsize=13, fontweight='bold')
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

    importances = np.asarray(qrf.feature_importances_)
    indices = np.argsort(importances)

    ax.barh(range(len(feature_names)), importances[indices], color='gray', edgecolor='black')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (QRF)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

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

        qrf_pin = pinball_loss(y_test, qrf_pred_q, q)
        lgb_pin = pinball_loss(y_test, lgb_pred_q, q)

        results_table.append({
            'Quantile': f'{q:.2f}',
            'QRF_Pinball': f'{qrf_pin:.2f}',
            'LGB_Pinball': f'{lgb_pin:.2f}',
            'QRF_Coverage': f'{qrf_cov:.1f}%',
            'LGB_Coverage': f'{lgb_cov:.1f}%',
            'Ideal_Coverage': f'{q*100:.0f}%'
        })

    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))

    # 평균 성능
    qrf_pin_avg = np.mean([pinball_loss(y_test, qrf_preds[:, i], q)
                           for i, q in enumerate(quantiles)])
    lgb_pin_avg = np.mean([pinball_loss(y_test, lgb_preds[q], q)
                           for q in quantiles])

    print(f"\n평균 pinball loss (작을수록 좋다):")
    print(f"  QRF: {qrf_pin_avg:.2f}")
    print(f"  LightGBM: {lgb_pin_avg:.2f}")

    # coverage 오차
    qrf_cov_err = np.mean([abs(evaluate_coverage(y_test, qrf_preds[:, i], q) - q)
                           for i, q in enumerate(quantiles)]) * 100
    lgb_cov_err = np.mean([abs(evaluate_coverage(y_test, lgb_preds[q], q) - q)
                           for q in quantiles]) * 100
    print(f"\n평균 coverage 오차 (|실제 - 이상|, 작을수록 좋다):")
    print(f"  QRF: {qrf_cov_err:.1f}%p")
    print(f"  LightGBM: {lgb_cov_err:.1f}%p")

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

    print("\n1. 정확도와 훈련 시간:")
    pin_diff_pct = (lgb_pin_avg - qrf_pin_avg) / qrf_pin_avg * 100
    time_ratio = lgb_time / qrf_time
    better = 'LightGBM' if lgb_pin_avg < qrf_pin_avg else 'QRF'
    print(f"   - 평균 pinball loss는 {better}가 {abs(pin_diff_pct):.1f}% 낮다")
    print(f"   - 훈련 시간: QRF {qrf_time:.1f}초(1회), LightGBM {lgb_time:.1f}초"
          f"({len(quantiles)}회) = QRF의 {time_ratio:.1f}배")
    print(f"   - QRF는 한 번 훈련하면 어떤 분위수든 바로 예측한다")

    print("\n2. 90% 예측 구간이 실제로 담은 비율:")
    print(f"   - QRF: {qrf_actual_cov:.1f}%, LightGBM: {lgb_actual_cov:.1f}% (목표 90%)")
    for name, cov in [('QRF', qrf_actual_cov), ('LightGBM', lgb_actual_cov)]:
        if cov < 85:
            print(f"   - {name}의 구간은 목표에 {90-cov:.1f}%p 모자란다. "
                  f"그대로 쓰면 위험을 낮잡는다")
    print(f"   - 구간 너비: QRF {pi_metrics['qrf_width']:.0f}만원, "
          f"LightGBM {pi_metrics['lgb_width']:.0f}만원")

    # 시각화
    print("\n[시각화] 결과 그래프 생성 중...")
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['price'] = y_test
    visualize_results(df_test, qrf, lgb_models, quantiles, feature_names)

if __name__ == "__main__":
    main()
