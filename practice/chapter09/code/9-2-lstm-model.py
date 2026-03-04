"""
Chapter 9 - Deep Learning Basics and Policy Time Series Forecasting
9.5.2 LSTM Model Implementation

정책 효과를 반영한 LSTM 모델 구현
어텐션 메커니즘을 결합한 Policy-Aware LSTM
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
matplotlib.rc('font', family='Arial')
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = 'practice/chapter09/data/'

class PolicyAwareLSTM(keras.Model):
    """
    정책 인식 LSTM 모델
    다층 LSTM + 멀티헤드 어텐션 결합

    Parameters:
    -----------
    lstm_units : list
        각 LSTM 층의 유닛 수
    pred_length : int
        예측 길이 (24시간)
    n_heads : int
        어텐션 헤드 수
    """
    def __init__(self, lstm_units=[128, 64, 32], pred_length=24, n_heads=4):
        super(PolicyAwareLSTM, self).__init__()

        self.lstm_units = lstm_units
        self.pred_length = pred_length
        self.n_heads = n_heads

        # LSTM 층 생성
        self.lstm_layers = []
        for i, units in enumerate(lstm_units[:-1]):
            self.lstm_layers.append(
                keras.layers.LSTM(
                    units,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.1,
                    name=f'lstm_{i+1}'
                )
            )

        # 마지막 LSTM 층 (시퀀스 반환)
        self.lstm_layers.append(
            keras.layers.LSTM(
                lstm_units[-1],
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
                name=f'lstm_{len(lstm_units)}'
            )
        )

        # 멀티헤드 어텐션
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=lstm_units[-1] // n_heads,
            name='multi_head_attention'
        )

        # Global Average Pooling
        self.global_pool = keras.layers.GlobalAveragePooling1D(name='global_pool')

        # Fully connected layers
        self.dense1 = keras.layers.Dense(64, activation='relu', name='dense_1')
        self.dropout = keras.layers.Dropout(0.2, name='dropout')
        self.output_layer = keras.layers.Dense(pred_length, name='output')

    def call(self, inputs, training=False):
        """
        Forward pass

        Parameters:
        -----------
        inputs : tensor
            입력 시퀀스 (batch_size, seq_length, n_features)
        training : bool
            훈련 모드 여부

        Returns:
        --------
        output : tensor
            예측 값 (batch_size, pred_length)
        """
        x = inputs

        # LSTM 층들을 순차적으로 통과
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)

        # 어텐션 메커니즘 적용
        # Query, Key, Value 모두 LSTM 출력 사용
        attended = self.attention(x, x, x, training=training)

        # Global pooling
        pooled = self.global_pool(attended)

        # Dense layers
        x = self.dense1(pooled)
        x = self.dropout(x, training=training)
        output = self.output_layer(x)

        return output

    def get_attention_weights(self, inputs):
        """
        어텐션 가중치 추출 (시각화용)

        Parameters:
        -----------
        inputs : tensor
            입력 시퀀스

        Returns:
        --------
        attention_weights : tensor
            어텐션 가중치
        """
        x = inputs
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=False)

        # 어텐션 출력과 가중치 추출
        attended, attention_weights = self.attention(
            x, x, x,
            return_attention_scores=True,
            training=False
        )

        return attention_weights

def create_baseline_lstm(seq_length, n_features, pred_length=24):
    """
    비교를 위한 기본 LSTM 모델

    Parameters:
    -----------
    seq_length : int
        입력 시퀀스 길이
    n_features : int
        특징 수
    pred_length : int
        예측 길이

    Returns:
    --------
    model : keras.Sequential
        기본 LSTM 모델
    """
    model = keras.Sequential([
        keras.layers.LSTM(
            128,
            return_sequences=True,
            input_shape=(seq_length, n_features),
            dropout=0.2,
            name='lstm_1'
        ),
        keras.layers.LSTM(64, dropout=0.2, name='lstm_2'),
        keras.layers.Dense(64, activation='relu', name='dense'),
        keras.layers.Dropout(0.2, name='dropout'),
        keras.layers.Dense(pred_length, name='output')
    ], name='Baseline_LSTM')

    return model

def create_gru_model(seq_length, n_features, pred_length=24):
    """
    비교를 위한 GRU 모델

    Parameters:
    -----------
    seq_length : int
        입력 시퀀스 길이
    n_features : int
        특징 수
    pred_length : int
        예측 길이

    Returns:
    --------
    model : keras.Sequential
        GRU 모델
    """
    model = keras.Sequential([
        keras.layers.GRU(
            128,
            return_sequences=True,
            input_shape=(seq_length, n_features),
            dropout=0.2,
            name='gru_1'
        ),
        keras.layers.GRU(64, dropout=0.2, name='gru_2'),
        keras.layers.Dense(64, activation='relu', name='dense'),
        keras.layers.Dropout(0.2, name='dropout'),
        keras.layers.Dense(pred_length, name='output')
    ], name='GRU_Model')

    return model

def visualize_model_comparison(models_info):
    """모델 아키텍처 비교 시각화"""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = [info['name'] for info in models_info]
    param_counts = [info['params'] for info in models_info]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.barh(model_names, param_counts, color=colors, alpha=0.7)

    # 값 표시
    for i, (bar, params) in enumerate(zip(bars, param_counts)):
        ax.text(params + max(param_counts) * 0.01, i,
               f'{params:,}',
               va='center', fontsize=10)

    ax.set_xlabel('Number of Parameters', fontsize=11)
    ax.set_title('Model Complexity Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()
    # Visualization displayed (not saved)

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제9장: LSTM 모델 구현")
    print("=" * 80)

    # 데이터 로드 (9-1에서 생성한 시퀀스)
    print("\n[1단계] 시퀀스 데이터 로드 중...")
    try:
        X = np.load(OUTPUT_DIR + '9-1-sequences-X.npy')
        y = np.load(OUTPUT_DIR + '9-1-sequences-y.npy')
        print(f"  - 입력 형태: {X.shape}")
        print(f"  - 타겟 형태: {y.shape}")
    except FileNotFoundError:
        print("  ⚠ 시퀀스 데이터 파일이 없습니다.")
        print("  먼저 9-1-data-preparation.py를 실행하세요.")
        return

    seq_length = X.shape[1]
    n_features = X.shape[2]
    pred_length = y.shape[1]

    print(f"  - 시퀀스 길이: {seq_length} 시간")
    print(f"  - 특징 수: {n_features}")
    print(f"  - 예측 길이: {pred_length} 시간")

    # 2. 모델 생성
    print("\n[2단계] 모델 생성 중...")

    # (1) Policy-Aware LSTM
    print("\n  (1) Policy-Aware LSTM with Attention...")
    policy_lstm = PolicyAwareLSTM(
        lstm_units=[128, 64, 32],
        pred_length=pred_length,
        n_heads=4
    )
    # Build model by calling it once
    _ = policy_lstm(X[:1])
    policy_lstm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mape']
    )
    print("     ✓ Policy-Aware LSTM 생성 완료")
    print(f"     파라미터 수: {policy_lstm.count_params():,}")

    # (2) Baseline LSTM
    print("\n  (2) Baseline LSTM...")
    baseline_lstm = create_baseline_lstm(seq_length, n_features, pred_length)
    baseline_lstm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mape']
    )
    print("     ✓ Baseline LSTM 생성 완료")
    print(f"     파라미터 수: {baseline_lstm.count_params():,}")

    # (3) GRU Model
    print("\n  (3) GRU Model...")
    gru_model = create_gru_model(seq_length, n_features, pred_length)
    gru_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mape']
    )
    print("     ✓ GRU Model 생성 완료")
    print(f"     파라미터 수: {gru_model.count_params():,}")

    # 3. 모델 아키텍처 출력
    print("\n[3단계] 모델 아키텍처 상세...")
    print("\n" + "=" * 80)
    print("Policy-Aware LSTM Architecture")
    print("=" * 80)
    policy_lstm.summary()

    print("\n" + "=" * 80)
    print("Baseline LSTM Architecture")
    print("=" * 80)
    baseline_lstm.summary()

    print("\n" + "=" * 80)
    print("GRU Model Architecture")
    print("=" * 80)
    gru_model.summary()

    # 4. 모델 비교 시각화
    print("\n[4단계] 모델 복잡도 비교 시각화 중...")
    models_info = [
        {'name': 'Policy-Aware LSTM', 'params': policy_lstm.count_params()},
        {'name': 'Baseline LSTM', 'params': baseline_lstm.count_params()},
        {'name': 'GRU Model', 'params': gru_model.count_params()}
    ]
    visualize_model_comparison(models_info)

    # 5. 어텐션 메커니즘 설명
    print("\n[5단계] 어텐션 메커니즘 분석...")
    sample_input = X[:1]  # 하나의 샘플
    attention_weights = policy_lstm.get_attention_weights(sample_input)
    print(f"  - 어텐션 가중치 형태: {attention_weights.shape}")
    print(f"  - 해석: (batch, num_heads, seq_length, seq_length)")
    print(f"  - 각 헤드가 시퀀스의 어느 부분에 주목하는지 학습")

    # 어텐션 가중치 평균 계산
    avg_attention = tf.reduce_mean(attention_weights[0], axis=0)  # (seq_length, seq_length)
    avg_attention_scores = tf.reduce_mean(avg_attention, axis=0).numpy()  # (seq_length,)

    print(f"\n  - 평균 어텐션 점수 범위: [{avg_attention_scores.min():.4f}, {avg_attention_scores.max():.4f}]")
    print(f"  - 최고 점수 시점: {avg_attention_scores.argmax()} 시간 전")

    # 6. 분석 요약
    print("\n" + "=" * 80)
    print("모델 생성 완료")
    print("=" * 80)

    print("\n모델별 특징:")
    print("  1. Policy-Aware LSTM:")
    print("     - 3층 LSTM (128 → 64 → 32 units)")
    print("     - 멀티헤드 어텐션 (4 heads)")
    print("     - 정책 변화에 민감한 시점 학습")
    print(f"     - 총 파라미터: {policy_lstm.count_params():,}")

    print("\n  2. Baseline LSTM:")
    print("     - 2층 LSTM (128 → 64 units)")
    print("     - 어텐션 없음")
    print("     - 전통적 시계열 예측")
    print(f"     - 총 파라미터: {baseline_lstm.count_params():,}")

    print("\n  3. GRU Model:")
    print("     - 2층 GRU (128 → 64 units)")
    print("     - LSTM 대비 파라미터 약 25% 감소")
    print("     - 빠른 학습 속도")
    print(f"     - 총 파라미터: {gru_model.count_params():,}")

    print("\n설계 원칙:")
    print("  - Dropout (0.2)으로 과적합 방지")
    print("  - Recurrent Dropout (0.1)으로 시계열 정규화")
    print("  - 어텐션으로 중요 시점 자동 학습")
    print("  - Global Pooling으로 시퀀스 정보 요약")

    print("\n출력 파일:")
    print(f"  - {OUTPUT_DIR}9-2-policy-lstm-initial.weights.h5")

    # 7. 모델 저장 (선택적)
    print("\n[6단계] 모델 아키텍처 저장...")
    policy_lstm.save_weights(OUTPUT_DIR + '9-2-policy-lstm-initial.weights.h5')
    print(f"  - Policy-Aware LSTM 초기 가중치 저장 완료")

if __name__ == "__main__":
    main()
