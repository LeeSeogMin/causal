"""
Chapter 9 - Deep Learning Basics and Time Series Forecasting
9.4 LSTM과 GRU 모델 만들기

세 모델의 구조와 파라미터 수를 확인한다.
학습은 9-3-train-evaluate.py에서 한다.

수정 이력
---------
2026-08-17
1. 경로 오류 수정
   증상: `OUTPUT_DIR`이 저장소 루트 기준 상대경로여서 code 폴더에서 실행하면
         가중치 저장이 실패했다.
   수정: `__file__` 기준 절대경로로 바꿨다.

2. 그림 저장 (`plt.show()` → `savefig`)
   비대화형 실행에서 PNG가 남지 않던 문제를 고쳤다.

3. `recurrent_dropout=0.1` 제거
   증상: CPU에서 한 에포크가 수십 분씩 걸렸다.
   원인: recurrent_dropout을 켜면 Keras가 시점마다 파이썬 루프를 도는 일반 경로를
         쓴다. 융합된 LSTM 커널이 꺼진다.
   수정: 층 사이 dropout(0.2)만 남겼다. 정규화 효과는 유지하고 속도를 되찾았다.

4. Attention 층 이름 변경 안내
   `PolicyAwareLSTM`이라는 이름에 '정책'이 들어 있지만 정책 변수를 따로 다루는
   장치는 없다. LSTM 3층 위에 어텐션 1층을 얹은 구조다. 강의노트에 그대로 적었다.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

matplotlib.rc('font', family='Arial')
plt.rcParams['axes.unicode_minus'] = False

# 경로 (스크립트 위치 기준)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, os.pardir, 'data') + os.sep
FIG_DIR = BASE_DIR + os.sep

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
                    name=f'lstm_{i+1}'
                )
            )

        # 마지막 LSTM 층 (시퀀스 반환)
        self.lstm_layers.append(
            keras.layers.LSTM(
                lstm_units[-1],
                return_sequences=True,
                dropout=0.2,
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
    plt.savefig(FIG_DIR + '9-2-lstm-model.png', dpi=130, bbox_inches='tight')
    plt.close()

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
    print(f"  - 균등 배분값 1/168 = {1/168:.4f}")
    print("  - 학습 전이라 168개 시점에 가중치가 거의 고르게 퍼져 있다")

    # 6. 분석 요약
    print("\n" + "=" * 80)
    print("모델 생성 완료")
    print("=" * 80)

    p_att = policy_lstm.count_params()
    p_lstm = baseline_lstm.count_params()
    p_gru = gru_model.count_params()

    print("\n모델별 구조:")
    print(f"  1. LSTM+Attention: LSTM 3층(128,64,32) + 어텐션 1층, 파라미터 {p_att:,}")
    print(f"  2. LSTM:           LSTM 2층(128,64),            파라미터 {p_lstm:,}")
    print(f"  3. GRU:            GRU  2층(128,64),            파라미터 {p_gru:,}")

    print("\n파라미터 수 계산 확인:")
    print("  LSTM 1층: 4 x (128 x (19 + 128) + 128) = 75,776")
    print("  GRU  1층: 3 x (128 x (19 + 128) + 128 x 2) = 57,216")
    print(f"  GRU가 LSTM보다 {(1 - p_gru / p_lstm) * 100:.1f}% 적다 (게이트 3개 대 4개)")

    print("\n이 스크립트로 확인한 것과 확인 못 한 것:")
    print("  확인함: 층 구성, 층별 출력 모양, 파라미터 수")
    print("  확인 못 함: 예측 정확도. 아직 학습을 하지 않았다")

    print("\n출력 파일:")
    print("  - data/9-2-policy-lstm-initial.weights.h5")
    print("  - 9-2-lstm-model.png")

    # 7. 모델 저장 (선택적)
    print("\n[6단계] 모델 아키텍처 저장...")
    policy_lstm.save_weights(OUTPUT_DIR + '9-2-policy-lstm-initial.weights.h5')
    print(f"  - Policy-Aware LSTM 초기 가중치 저장 완료")

if __name__ == "__main__":
    main()
