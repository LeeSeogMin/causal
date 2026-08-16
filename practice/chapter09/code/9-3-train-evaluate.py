"""
Chapter 9 - Deep Learning Basics and Time Series Forecasting
9.5 기준선과 딥러닝 비교

단순 기준선 세 개(직전 값, 24시간 전 값, 이동평균)를 먼저 세우고,
LSTM·GRU가 그 기준선을 이기는지 같은 테스트 구간에서 확인한다.

수정 이력
---------
2026-08-17
1. 경로 오류 수정
   증상: `OUTPUT_DIR`이 저장소 루트 기준 상대경로여서 code 폴더에서 실행하면
         데이터 로드가 실패했다.
   수정: `__file__` 기준 절대경로로 바꿨다.

2. 그림 저장 (`plt.show()` → `savefig`)
   PNG 세 장을 code 폴더에 남긴다.

3. 단순 기준선 추가 (이 스크립트의 핵심 수정)
   증상: 딥러닝 세 모델끼리만 비교해, "이 정확도가 좋은가"를 판단할 기준이 없었다.
   수정: 직전 값 반복(persistence), 24시간 전 값 반복(seasonal naive),
         최근 24시간 이동평균 세 가지를 같은 테스트 구간에서 계산해 함께 보고한다.

4. 지어낸 결론 출력 삭제
   증상: 실행 결과와 무관하게 "Test MAPE 3~5% 수준", "산업 표준 5% 이내의 우수한
         정확도" 같은 문장이 항상 출력됐다. 실제 값이 그와 달라도 그대로 찍혔다.
   수정: 실제 계산값으로 기준선 대비 판정을 출력하도록 바꿨다.

5. `analyze_policy_phases` 이름·설명 수정
   증상: 테스트 구간을 3등분해 놓고 '정책 도입기/평가기/정착기'라고 불렀다.
         재생에너지 정책은 전체 시계열의 50% 지점에서 시작하므로 테스트 구간
         전체가 이미 정책 시행 후다. 세 구간은 정책 단계가 아니라 시간 3등분이다.
   수정: `analyze_test_thirds`로 바꾸고 '테스트 구간 전반부/중반부/후반부'로 적었다.

6. RMSE 추가
   MAE만 보고하던 것을 RMSE와 함께 보고한다. 큰 오차에 얼마나 벌점을 주는지
   두 지표를 나란히 놓고 읽을 수 있게 했다.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

import importlib.util

warnings.filterwarnings('ignore')

matplotlib.rc('font', family='Arial')
plt.rcParams['axes.unicode_minus'] = False

# 경로 (스크립트 위치 기준)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, os.pardir, 'data') + os.sep
FIG_DIR = BASE_DIR + os.sep

tf.random.set_seed(42)
np.random.seed(42)

def load_lstm_model_module():
    module_path = os.path.join(os.path.dirname(__file__), '9-2-lstm-model.py')
    spec = importlib.util.spec_from_file_location('chapter09_lstm_model', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_data():
    """
    시퀀스 데이터 로드 (9-1에서 생성)

    Returns:
    --------
    X_train, X_val, X_test : ndarray
        훈련/검증/테스트 입력 시퀀스
    y_train, y_val, y_test : ndarray
        훈련/검증/테스트 타겟 시퀀스
    """
    print("\n[1단계] 시퀀스 데이터 로드 중...")
    try:
        X = np.load(OUTPUT_DIR + '9-1-sequences-X.npy')
        y = np.load(OUTPUT_DIR + '9-1-sequences-y.npy')
        print(f"  - 입력 형태: {X.shape}")
        print(f"  - 타겟 형태: {y.shape}")
    except FileNotFoundError:
        print("  ⚠ 시퀀스 데이터 파일이 없습니다.")
        print("  먼저 9-1-data-preparation.py를 실행하세요.")
        raise

    # Train/Val/Test 분할
    n_samples = X.shape[0]
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print(f"\n  - 훈련 세트: {len(X_train):,} 샘플")
    print(f"  - 검증 세트: {len(X_val):,} 샘플")
    print(f"  - 테스트 세트: {len(X_test):,} 샘플")

    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_metrics(y_true, y_pred):
    """
    원 단위(MW)에서 세 지표를 계산한다.

    RMSE : 오차를 제곱해 평균 낸 뒤 제곱근. 큰 오차 하나가 값을 크게 올린다.
    MAE  : 오차 절댓값의 평균. 모든 오차를 같은 비중으로 센다.
    MAPE : 오차를 실제값으로 나눈 백분율의 평균. 단위가 다른 시계열끼리 비교할 때 쓴다.
    """
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / y_true)) * 100)
    return {'rmse': rmse, 'mae': mae, 'mape': mape}


def compute_baselines(X_test, scaler_params):
    """
    학습 없이 만드는 단순 기준선 세 개.

    persistence  : 마지막 관측값을 24시간 내내 그대로 쓴다.
    seasonal_24h : 어제 같은 시각의 값을 그대로 쓴다.
    moving_avg   : 최근 24시간 평균을 24시간 내내 그대로 쓴다.

    셋 다 파라미터가 0개다. 딥러닝 모델은 이 셋을 이겨야 쓸 값어치가 생긴다.
    """
    center = scaler_params['center']
    scale = scaler_params['scale']

    # X_test[:, :, 0]이 정규화된 demand다.
    last_value = X_test[:, -1, 0]                       # (n,)
    prev_day = X_test[:, -24:, 0]                       # (n, 24)
    mean_24h = X_test[:, -24:, 0].mean(axis=1)          # (n,)

    preds = {
        'persistence': np.repeat(last_value[:, None], 24, axis=1),
        'seasonal_24h': prev_day,
        'moving_avg_24h': np.repeat(mean_24h[:, None], 24, axis=1),
    }
    return {k: v * scale + center for k, v in preds.items()}


def train_model(model, X_train, y_train, X_val, y_val, model_name):
    """
    모델 학습 with callbacks

    Parameters:
    -----------
    model : keras.Model
        학습할 모델
    X_train, y_train : ndarray
        훈련 데이터
    X_val, y_val : ndarray
        검증 데이터
    model_name : str
        모델 이름

    Returns:
    --------
    history : History
        학습 기록
    """
    print(f"\n[{model_name}] 학습 시작...")

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    # 학습 (빠른 검증을 위해 20 에포크로 제한)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    print(f"  - 학습 완료 (총 {len(history.history['loss'])} 에포크)")
    print(f"  - 최종 훈련 손실: {history.history['loss'][-1]:.4f}")
    print(f"  - 최종 검증 손실: {history.history['val_loss'][-1]:.4f}")

    return history

def evaluate_model(model, X_test, y_test, scaler_params, model_name):
    """
    모델 평가 및 역정규화

    Parameters:
    -----------
    model : keras.Model
        평가할 모델
    X_test, y_test : ndarray
        테스트 데이터
    scaler_params : dict
        정규화 파라미터 (center, scale)
    model_name : str
        모델 이름

    Returns:
    --------
    metrics : dict
        평가 지표 (mse, mae, mape)
    y_pred : ndarray
        예측값 (역정규화)
    """
    print(f"\n[{model_name}] 평가 중...")

    # 예측
    y_pred_normalized = model.predict(X_test, verbose=0)

    # 역정규화
    center = scaler_params['center']
    scale = scaler_params['scale']

    y_pred = y_pred_normalized * scale + center
    y_true = y_test * scale + center

    metrics = compute_metrics(y_true, y_pred)

    print(f"  - Test RMSE: {metrics['rmse']:.1f} MW")
    print(f"  - Test MAE : {metrics['mae']:.1f} MW")
    print(f"  - Test MAPE: {metrics['mape']:.2f}%")

    return metrics, y_pred

def analyze_test_thirds(y_true, y_pred):
    """
    테스트 구간을 시간 순서로 3등분해 구간별 MAE를 본다.

    주의: 이 세 구간은 '정책 단계'가 아니다. 재생에너지 정책 효과는 전체 시계열의
    50% 지점에서 시작하므로 테스트 구간(뒤 15%)은 전부 정책 시행 후다.
    시간이 지날수록 오차가 커지는지만 확인하는 용도다.
    """
    print("\n테스트 구간 3등분 MAE...")

    n_samples = y_true.shape[0]
    a = int(n_samples * 0.3)
    b = int(n_samples * 0.7)

    mae_1 = float(np.mean(np.abs(y_true[:a] - y_pred[:a])))
    mae_2 = float(np.mean(np.abs(y_true[a:b] - y_pred[a:b])))
    mae_3 = float(np.mean(np.abs(y_true[b:] - y_pred[b:])))

    print(f"  - 전반부 MAE: {mae_1:.1f} MW")
    print(f"  - 중반부 MAE: {mae_2:.1f} MW")
    print(f"  - 후반부 MAE: {mae_3:.1f} MW")

    return {'first': mae_1, 'middle': mae_2, 'last': mae_3}

def visualize_training_results(histories, model_names):
    """학습 곡선 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # (1) 손실 곡선
    ax1 = axes[0]
    for i, (history, name) in enumerate(zip(histories, model_names)):
        ax1.plot(history.history['loss'],
                label=f'{name} Train',
                color=colors[i],
                alpha=0.7,
                linewidth=2)
        ax1.plot(history.history['val_loss'],
                label=f'{name} Val',
                color=colors[i],
                linestyle='--',
                linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('MSE Loss', fontsize=11)
    ax1.set_title('(a) Training Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (2) MAE 곡선
    ax2 = axes[1]
    for i, (history, name) in enumerate(zip(histories, model_names)):
        ax2.plot(history.history['mae'],
                label=f'{name} Train',
                color=colors[i],
                alpha=0.7,
                linewidth=2)
        ax2.plot(history.history['val_mae'],
                label=f'{name} Val',
                color=colors[i],
                linestyle='--',
                linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('MAE', fontsize=11)
    ax2.set_title('(b) Training MAE Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR + '9-3-learning-curves.png', dpi=130, bbox_inches='tight')
    plt.close()

def visualize_predictions(y_true, y_pred, baselines, all_metrics, model_name):
    """예측 결과와 기준선 비교 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    horizon = y_true.shape[1]          # 24시간
    time_index = np.arange(1, horizon + 1)
    w = 0                              # 테스트 구간 첫 창

    # (a) 24시간 예측 하나를 펼쳐 보기
    ax1 = axes[0, 0]
    ax1.plot(time_index, y_true[w], label='Actual', color='black', linewidth=2.5)
    ax1.plot(time_index, y_pred[w], label=model_name, color='#c0392b',
             linewidth=2, marker='o', markersize=3)
    ax1.plot(time_index, baselines['persistence'][w], label='Persistence',
             color='#2c6fbb', linewidth=1.5, linestyle='--')
    ax1.plot(time_index, baselines['seasonal_24h'][w], label='Yesterday same hour',
             color='#2e8b57', linewidth=1.5, linestyle=':')
    ax1.set_xlabel('Forecast horizon (hours ahead)', fontsize=11)
    ax1.set_ylabel('Demand (MW)', fontsize=11)
    ax1.set_title('(a) One 24-Hour Forecast Window', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (b) 오차 분포
    ax2 = axes[0, 1]
    errors = (y_true - y_pred).flatten()
    ax2.hist(errors, bins=50, color='steelblue', alpha=0.75, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax2.axvline(errors.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {errors.mean():.0f} MW')
    ax2.set_xlabel('Prediction error (MW)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'(b) Error Distribution - {model_name}', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # (c) 방법별 RMSE / MAE
    ax3 = axes[1, 0]
    names = list(all_metrics.keys())
    rmses = [all_metrics[n]['rmse'] for n in names]
    maes = [all_metrics[n]['mae'] for n in names]
    xpos = np.arange(len(names))
    ax3.bar(xpos - 0.2, rmses, 0.4, label='RMSE', color='#c0392b', alpha=0.8)
    ax3.bar(xpos + 0.2, maes, 0.4, label='MAE', color='#2c6fbb', alpha=0.8)
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax3.set_ylabel('Error (MW)', fontsize=11)
    ax3.set_title('(c) RMSE and MAE by Method', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # (d) 예측 시계별 MAE
    ax4 = axes[1, 1]
    ax4.plot(time_index, np.mean(np.abs(y_true - y_pred), axis=0),
             label=model_name, color='#c0392b', linewidth=2, marker='o', markersize=3)
    ax4.plot(time_index, np.mean(np.abs(y_true - baselines['persistence']), axis=0),
             label='Persistence', color='#2c6fbb', linewidth=1.5, linestyle='--')
    ax4.plot(time_index, np.mean(np.abs(y_true - baselines['seasonal_24h']), axis=0),
             label='Yesterday same hour', color='#2e8b57', linewidth=1.5, linestyle=':')
    ax4.set_xlabel('Forecast horizon (hours ahead)', fontsize=11)
    ax4.set_ylabel('MAE (MW)', fontsize=11)
    ax4.set_title('(d) MAE by Forecast Horizon', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR + '9-3-baseline-vs-deep.png', dpi=130, bbox_inches='tight')
    plt.close()

def visualize_attention_weights(model, X_test):
    """어텐션 가중치 시각화"""
    print("\n어텐션 가중치 분석 중...")

    # PolicyAwareLSTM인 경우만 어텐션 가중치 추출
    if not hasattr(model, 'get_attention_weights'):
        print("  ⚠ 이 모델은 어텐션 메커니즘이 없습니다.")
        return

    # 샘플 하나 선택
    sample_input = X_test[:1]
    attention_weights = model.get_attention_weights(sample_input)

    # 평균 어텐션 가중치 (헤드들의 평균)
    avg_attention = tf.reduce_mean(attention_weights[0], axis=0)  # (seq_length, seq_length)
    avg_attention_scores = tf.reduce_mean(avg_attention, axis=0).numpy()  # (seq_length,)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 어텐션 히트맵
    ax1 = axes[0]
    im = ax1.imshow(avg_attention.numpy(), cmap='viridis', aspect='auto')
    ax1.set_xlabel('Key Position (Hours)', fontsize=11)
    ax1.set_ylabel('Query Position (Hours)', fontsize=11)
    ax1.set_title('(a) Attention Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Attention Weight')

    # (2) 평균 어텐션 점수
    ax2 = axes[1]
    hours_back = np.arange(len(avg_attention_scores))
    ax2.bar(hours_back, avg_attention_scores, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Hours Back from Current', fontsize=11)
    ax2.set_ylabel('Average Attention Score', fontsize=11)
    ax2.set_title('(b) Average Attention Across Sequence', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIG_DIR + '9-3-attention.png', dpi=130, bbox_inches='tight')
    plt.close()

    # 주요 시점 분석
    top_5_indices = np.argsort(avg_attention_scores)[-5:][::-1]
    print(f"  - 균등 배분값 1/168 = {1/168:.4f}")
    print(f"  - 가중치가 높은 시점 5개 (창의 시작을 0으로 센 위치):")
    for i, idx in enumerate(top_5_indices, 1):
        print(f"    {i}. {idx}번째 시점 (점수: {avg_attention_scores[idx]:.4f})")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제9장: 기준선과 딥러닝 비교")
    print("=" * 80)

    # 1. 데이터 로드
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Scaler 파라미터 로드 (역정규화용)
    scaler_params = np.load(OUTPUT_DIR + '9-1-scaler-params.npy', allow_pickle=True).item()

    # 역정규화된 실제값
    y_true = y_test * scaler_params['scale'] + scaler_params['center']

    # 1-2. 학습하기 전에 기준선부터 세운다
    print("\n[2단계] 단순 기준선 계산 (학습 없음)...")
    baselines = compute_baselines(X_test, scaler_params)
    baseline_metrics = {}
    for name, pred in baselines.items():
        m = compute_metrics(y_true, pred)
        baseline_metrics[name] = m
        print(f"  - {name:15s} RMSE {m['rmse']:8.1f} MW | MAE {m['mae']:8.1f} MW | MAPE {m['mape']:5.2f}%")

    best_base = min(baseline_metrics, key=lambda k: baseline_metrics[k]['mae'])
    print(f"  → 가장 나은 기준선: {best_base} (MAE {baseline_metrics[best_base]['mae']:.1f} MW)")
    print("  → 딥러닝 모델은 이 값을 이겨야 한다")

    seq_length = X_train.shape[1]
    n_features = X_train.shape[2]
    pred_length = y_train.shape[1]

    # 2. 모델 생성
    print("\n[3단계] 딥러닝 모델 생성 중...")

    # Import model creation functions
    lstm_model_module = load_lstm_model_module()
    PolicyAwareLSTM = lstm_model_module.PolicyAwareLSTM
    create_baseline_lstm = lstm_model_module.create_baseline_lstm
    create_gru_model = lstm_model_module.create_gru_model

    # (1) Policy-Aware LSTM
    policy_lstm = PolicyAwareLSTM(
        lstm_units=[128, 64, 32],
        pred_length=pred_length,
        n_heads=4
    )
    _ = policy_lstm(X_train[:1])  # Build
    policy_lstm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mape']
    )

    # (2) Baseline LSTM
    baseline_lstm = create_baseline_lstm(seq_length, n_features, pred_length)
    baseline_lstm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mape']
    )

    # (3) GRU Model
    gru_model = create_gru_model(seq_length, n_features, pred_length)
    gru_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mape']
    )

    print("  - 모델 3개 생성 완료")

    # 3. 모델 학습
    print("\n[4단계] 모델 학습 중...")

    history_policy = train_model(policy_lstm, X_train, y_train, X_val, y_val,
                                 "LSTM+Attention")
    history_baseline = train_model(baseline_lstm, X_train, y_train, X_val, y_val,
                                   "LSTM")
    history_gru = train_model(gru_model, X_train, y_train, X_val, y_val,
                              "GRU")

    # 4. 모델 평가
    print("\n[5단계] 딥러닝 모델 평가...")

    metrics_policy, y_pred_policy = evaluate_model(
        policy_lstm, X_test, y_test, scaler_params, "LSTM+Attention"
    )
    metrics_baseline, y_pred_baseline = evaluate_model(
        baseline_lstm, X_test, y_test, scaler_params, "LSTM"
    )
    metrics_gru, y_pred_gru = evaluate_model(
        gru_model, X_test, y_test, scaler_params, "GRU"
    )

    # 5. 전체 비교표
    all_metrics = dict(baseline_metrics)
    all_metrics['LSTM+Attention'] = metrics_policy
    all_metrics['LSTM'] = metrics_baseline
    all_metrics['GRU'] = metrics_gru

    base_mae = baseline_metrics[best_base]['mae']

    print("\n" + "=" * 80)
    print("전체 비교 (같은 테스트 구간, 같은 지표)")
    print("=" * 80)
    print(f"{'방법':<16}{'RMSE(MW)':>11}{'MAE(MW)':>11}{'MAPE(%)':>10}"
          f"{'기준선 대비':>13}  판정")
    print("-" * 80)
    for name, m in all_metrics.items():
        diff = (m['mae'] / base_mae - 1) * 100
        verdict = "기준선을 이김" if m['mae'] < base_mae else "기준선을 못 이김"
        if name == best_base:
            verdict = "기준선(최고)"
        print(f"{name:<16}{m['rmse']:>11.1f}{m['mae']:>11.1f}{m['mape']:>10.2f}"
              f"{diff:>12.1f}%  {verdict}")

    winners = [n for n in ['LSTM+Attention', 'LSTM', 'GRU']
               if all_metrics[n]['mae'] < base_mae]
    print("-" * 80)
    print(f"기준선({best_base}) MAE {base_mae:.1f} MW를 이긴 딥러닝 모델: "
          f"{len(winners)}개 / 3개")
    if winners:
        print(f"  이긴 모델: {', '.join(winners)}")

    # 6. 테스트 구간 3등분
    print("\n[6단계] 구간별 오차 확인...")
    best_dl = min(['LSTM+Attention', 'LSTM', 'GRU'], key=lambda n: all_metrics[n]['mae'])
    y_pred_best = {'LSTM+Attention': y_pred_policy,
                   'LSTM': y_pred_baseline,
                   'GRU': y_pred_gru}[best_dl]
    analyze_test_thirds(y_true, y_pred_best)

    # 7. 시각화
    print("\n[7단계] 결과 시각화 중...")

    visualize_training_results(
        [history_policy, history_baseline, history_gru],
        ['LSTM+Attention', 'LSTM', 'GRU']
    )
    visualize_predictions(y_true, y_pred_best, baselines, all_metrics, best_dl)
    visualize_attention_weights(policy_lstm, X_test)

    print("\n출력 파일:")
    print("  - 9-3-learning-curves.png")
    print("  - 9-3-baseline-vs-deep.png")
    print("  - 9-3-attention.png")

    print("\n" + "=" * 80)
    print("실행 완료")
    print("=" * 80)

if __name__ == "__main__":
    main()
