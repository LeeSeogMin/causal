# -*- coding: utf-8 -*-
"""
9-3-mamba-ssm.py
Mamba 상태공간모델(State Space Model) 구현 예제

이 코드는 Gu & Dao(2023)의 Mamba 아키텍처를 단순화하여 구현한 교육용 예제이다.
실제 Mamba 라이브러리는 CUDA 커널 최적화를 포함하지만, 여기서는 순수 Python/TensorFlow로
핵심 개념을 이해할 수 있도록 구현하였다.

※ 이 코드는 교육 목적의 예제입니다. 실제 정책 분석에서는 공식 Mamba 라이브러리나
검증된 LSTM/GRU 모델을 사용하는 것이 권장됩니다.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


class MambaBlock(keras.Model):
    """
    Mamba 상태공간모델 블록

    Mamba의 핵심 혁신:
    1. 선택적 상태공간(Selective State Space): 입력에 따라 A, B, C 파라미터가 동적으로 결정
    2. 선택적 스캔 알고리즘: 효율적인 순환 연산
    3. 하드웨어 인식 구현: GPU 메모리 계층 최적화 (이 예제에서는 간소화)
    """

    def __init__(self, d_model, d_state=16, expand=2):
        """
        Args:
            d_model: 모델 차원 (입력/출력 차원)
            d_state: 상태 차원 (SSM 잠재 상태 크기)
            expand: 내부 차원 확장 비율
        """
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = int(self.expand * self.d_model)
        self.d_inner = d_inner

        # 입력 투영: 2배 크기로 확장 (x와 z 분기용)
        self.in_proj = keras.layers.Dense(d_inner * 2, use_bias=False)

        # 출력 투영
        self.out_proj = keras.layers.Dense(d_model, use_bias=False)

        # SSM 파라미터 A: 상태 전이 행렬 (학습되지 않음, 고정)
        # Mamba에서는 A를 대각 행렬로 파라미터화하여 안정성 확보
        self.A = self.add_weight(
            name='A',
            shape=(d_inner, d_state),
            initializer=keras.initializers.TruncatedNormal(stddev=0.1),
            trainable=False  # A는 고정
        )

        # B, C: 입력 의존적 투영 (선택적 SSM의 핵심)
        self.B_proj = keras.layers.Dense(d_state, use_bias=False, name='B_proj')
        self.C_proj = keras.layers.Dense(d_inner, use_bias=False, name='C_proj')

        # D: 직접 연결 (skip connection)
        self.D = self.add_weight(
            name='D',
            shape=(d_inner,),
            initializer='ones'
        )

        # Delta: 이산화 스텝 크기 (입력 의존적)
        self.delta_proj = keras.layers.Dense(d_inner, use_bias=False, name='delta')

        # 정규화 레이어
        self.norm = keras.layers.LayerNormalization()

    def selective_scan(self, x, delta, A, B, C, D):
        """
        선택적 스캔 알고리즘

        연속 시간 SSM을 이산화하고 순환적으로 상태를 갱신한다.
        실제 Mamba는 이 연산을 병렬화된 CUDA 커널로 구현하여 O(L) 복잡도를 달성한다.

        연속 시간 SSM:
            h'(t) = Ah(t) + Bx(t)
            y(t) = Ch(t) + Dx(t)

        이산화 (zero-order hold):
            Ā = exp(ΔA)
            B̄ = (ΔA)⁻¹(exp(ΔA) - I) · ΔB ≈ ΔB (근사)

        Args:
            x: 입력 시퀀스 [batch, length, d_inner]
            delta: 이산화 스텝 크기 [batch, length, d_inner]
            A: 상태 전이 행렬 [d_inner, d_state]
            B: 입력 투영 [batch, length, d_state]
            C: 출력 투영 [batch, length, d_inner]
            D: 직접 연결 [d_inner]

        Returns:
            출력 시퀀스 [batch, length, d_inner]
        """
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        # 이산화: Ā = exp(Δ * A)
        # delta: [batch, length, d_inner], A: [d_inner, d_state]
        # deltaA: [batch, length, d_inner, d_state]
        delta_expanded = tf.expand_dims(delta, -1)  # [batch, length, d_inner, 1]
        A_expanded = tf.reshape(A, [1, 1, self.d_inner, self.d_state])  # [1, 1, d_inner, d_state]
        deltaA = tf.exp(delta_expanded * A_expanded)  # [batch, length, d_inner, d_state]

        # deltaB ≈ delta * B
        delta_B = tf.expand_dims(delta, -1) * tf.expand_dims(B, 2)  # [batch, length, d_inner, d_state]

        # 순환 스캔 (간소화된 버전)
        # 실제 Mamba는 이를 병렬 스캔으로 구현
        outputs = []
        state = tf.zeros((batch_size, self.d_inner, self.d_state))

        for t in range(length):
            # 상태 갱신: h_t = Ā * h_{t-1} + B̄ * x_t
            deltaA_t = deltaA[:, t, :, :]  # [batch, d_inner, d_state]
            deltaB_t = delta_B[:, t, :, :]  # [batch, d_inner, d_state]
            x_t = tf.expand_dims(x[:, t, :], -1)  # [batch, d_inner, 1]

            state = deltaA_t * state + deltaB_t * x_t

            # 출력 계산: y_t = C * h_t
            C_t = C[:, t, :]  # [batch, d_inner]
            y_t = tf.reduce_sum(state * tf.expand_dims(C_t, -1), axis=-1)  # [batch, d_inner]

            outputs.append(y_t)

        y = tf.stack(outputs, axis=1)  # [batch, length, d_inner]

        # 직접 연결 추가: y = y + D * x
        y = y + D * x

        return y

    def call(self, x, training=None):
        """
        순전파

        Args:
            x: 입력 [batch, length, d_model]
            training: 학습 모드 여부

        Returns:
            출력 [batch, length, d_model]
        """
        # 잔차 연결을 위해 입력 저장
        residual = x

        # 정규화
        x = self.norm(x)

        # 입력 투영: x와 z 분기로 분할
        x_proj = self.in_proj(x)  # [batch, length, d_inner * 2]
        x_branch, z = tf.split(x_proj, 2, axis=-1)  # 각각 [batch, length, d_inner]

        # 선택적 SSM 파라미터 계산 (입력 의존적)
        delta = tf.nn.softplus(self.delta_proj(x_branch))  # [batch, length, d_inner]
        B = self.B_proj(x_branch)  # [batch, length, d_state]
        C = self.C_proj(x_branch)  # [batch, length, d_inner]

        # 선택적 스캔 적용
        y = self.selective_scan(x_branch, delta, self.A, B, C, self.D)

        # 게이트 곱셈: y = y * SiLU(z)
        y = y * tf.nn.silu(z)

        # 출력 투영
        output = self.out_proj(y)

        # 잔차 연결
        output = output + residual

        return output


class MambaModel(keras.Model):
    """
    정책 시계열 예측을 위한 Mamba 기반 모델
    """

    def __init__(self, d_model=64, d_state=16, n_layers=4, output_size=24):
        """
        Args:
            d_model: 모델 차원
            d_state: SSM 상태 차원
            n_layers: Mamba 블록 수
            output_size: 출력 차원 (예측 시점 수)
        """
        super(MambaModel, self).__init__()

        self.embedding = keras.layers.Dense(d_model)
        self.mamba_layers = [MambaBlock(d_model, d_state) for _ in range(n_layers)]
        self.output_norm = keras.layers.LayerNormalization()
        self.output_proj = keras.layers.Dense(output_size)

    def call(self, x, training=None):
        """
        Args:
            x: 입력 시퀀스 [batch, length, features]

        Returns:
            예측값 [batch, output_size]
        """
        # 임베딩
        x = self.embedding(x)

        # Mamba 레이어 적용
        for layer in self.mamba_layers:
            x = layer(x, training=training)

        # 마지막 시점의 출력 사용
        x = x[:, -1, :]  # [batch, d_model]

        # 정규화 및 출력 투영
        x = self.output_norm(x)
        output = self.output_proj(x)

        return output


def create_synthetic_data(n_samples=1000, seq_length=168, n_features=10):
    """
    시뮬레이션 정책 시계열 데이터 생성

    Args:
        n_samples: 샘플 수
        seq_length: 시퀀스 길이 (168 = 1주일 시간별 데이터)
        n_features: 특성 수

    Returns:
        X: 입력 데이터 [n_samples, seq_length, n_features]
        y: 타겟 데이터 [n_samples, 24] (24시간 전방 예측)
    """
    np.random.seed(42)

    # 시간 인덱스
    t = np.arange(seq_length + 24)

    X_list = []
    y_list = []

    for _ in range(n_samples):
        # 기본 패턴: 일간 주기 + 주간 주기 + 트렌드
        daily_pattern = 10 * np.sin(2 * np.pi * t / 24)
        weekly_pattern = 5 * np.sin(2 * np.pi * t / (24 * 7))
        trend = 0.01 * t

        # 정책 효과 시뮬레이션 (랜덤 시점에서 구조 변화)
        policy_start = np.random.randint(50, 100)
        policy_effect = np.where(t >= policy_start, 3 * (1 - np.exp(-0.1 * (t - policy_start))), 0)

        # 노이즈
        noise = np.random.normal(0, 1, len(t))

        # 기본 시계열
        base_series = daily_pattern + weekly_pattern + trend + policy_effect + noise

        # 다변량 특성 생성
        features = np.zeros((seq_length, n_features))
        features[:, 0] = base_series[:seq_length]  # 주요 지표

        for j in range(1, n_features):
            # 관련 특성들 (지연, 이동평균 등)
            lag = np.random.randint(1, 12)
            if j < n_features // 2:
                features[:, j] = np.roll(base_series[:seq_length], lag) + np.random.normal(0, 0.5, seq_length)
            else:
                # 이동평균
                window = np.random.randint(3, 12)
                features[:, j] = np.convolve(base_series[:seq_length], np.ones(window)/window, mode='same')

        # 타겟: 다음 24시간 예측
        target = base_series[seq_length:seq_length + 24]

        X_list.append(features)
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def compare_architectures():
    """
    Mamba, LSTM, GRU 아키텍처 비교 실험
    """
    print("=" * 60)
    print("시퀀스 모델 아키텍처 비교 실험")
    print("=" * 60)

    # 데이터 생성
    print("\n1. 시뮬레이션 데이터 생성...")
    X, y = create_synthetic_data(n_samples=800, seq_length=168, n_features=10)

    # 데이터 분할
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"   학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

    # 모델 정의
    models = {}

    # Mamba 모델
    print("\n2. 모델 구축...")
    print("   - Mamba SSM 모델 생성")
    models['Mamba'] = MambaModel(d_model=64, d_state=16, n_layers=2, output_size=24)

    # LSTM 모델
    print("   - LSTM 모델 생성")
    lstm_model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(168, 10)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(24)
    ])
    models['LSTM'] = lstm_model

    # GRU 모델
    print("   - GRU 모델 생성")
    gru_model = keras.Sequential([
        keras.layers.GRU(64, return_sequences=True, input_shape=(168, 10)),
        keras.layers.Dropout(0.2),
        keras.layers.GRU(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(24)
    ])
    models['GRU'] = gru_model

    # 모델 학습 및 평가
    results = {}

    for name, model in models.items():
        print(f"\n3. {name} 모델 학습...")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        # 학습
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=10,
            batch_size=32,
            verbose=0
        )

        # 평가
        loss, mae = model.evaluate(X_test, y_test, verbose=0)

        # 파라미터 수 계산
        params = model.count_params()

        results[name] = {
            'loss': loss,
            'mae': mae,
            'params': params,
            'history': history.history
        }

        print(f"   테스트 MSE: {loss:.4f}, MAE: {mae:.4f}")
        print(f"   파라미터 수: {params:,}")

    # 결과 비교 테이블
    print("\n" + "=" * 60)
    print("아키텍처 비교 결과")
    print("=" * 60)
    print(f"{'모델':<10} {'테스트 MSE':<15} {'테스트 MAE':<15} {'파라미터 수':<15}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<10} {res['loss']:<15.4f} {res['mae']:<15.4f} {res['params']:<15,}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 학습 곡선
    ax1 = axes[0]
    for name, res in results.items():
        ax1.plot(res['history']['loss'], label=f'{name} (Train)')
        ax1.plot(res['history']['val_loss'], '--', label=f'{name} (Val)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 성능 비교 막대 그래프
    ax2 = axes[1]
    model_names = list(results.keys())
    mae_values = [results[name]['mae'] for name in model_names]
    params_values = [results[name]['params'] / 1000 for name in model_names]  # K 단위

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax2.bar(x - width/2, mae_values, width, label='MAE', color='steelblue')
    ax2.set_ylabel('MAE')
    ax2.set_xlabel('Model')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)

    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, params_values, width, label='Parameters (K)', color='coral')
    ax2_twin.set_ylabel('Parameters (K)')

    ax2.set_title('Model Performance and Complexity')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('../../../diagrams/9-architecture-comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n시각화 저장: diagrams/9-architecture-comparison.png")

    return results


def demonstrate_selective_mechanism():
    """
    Mamba의 선택적 메커니즘 시연

    입력에 따라 A, B, C 파라미터가 어떻게 동적으로 변화하는지 보여준다.
    """
    print("\n" + "=" * 60)
    print("선택적 상태공간 메커니즘 시연")
    print("=" * 60)

    # 간단한 Mamba 블록 생성
    mamba = MambaBlock(d_model=32, d_state=8)

    # 두 가지 다른 입력 패턴
    np.random.seed(42)

    # 패턴 1: 부드러운 신호
    t = np.linspace(0, 4*np.pi, 50)
    smooth_signal = np.sin(t) + 0.1 * np.random.randn(50)

    # 패턴 2: 급격한 변화가 있는 신호
    abrupt_signal = np.zeros(50)
    abrupt_signal[:25] = 1.0
    abrupt_signal[25:] = -1.0
    abrupt_signal += 0.1 * np.random.randn(50)

    # 입력 준비
    x_smooth = tf.constant(smooth_signal.reshape(1, 50, 1), dtype=tf.float32)
    x_abrupt = tf.constant(abrupt_signal.reshape(1, 50, 1), dtype=tf.float32)

    # 입력을 d_model 차원으로 확장
    expand_layer = keras.layers.Dense(32)
    x_smooth_expanded = expand_layer(x_smooth)
    x_abrupt_expanded = expand_layer(x_abrupt)

    # 출력 계산
    y_smooth = mamba(x_smooth_expanded)
    y_abrupt = mamba(x_abrupt_expanded)

    print("\n입력 신호에 따른 출력 변화:")
    print(f"  - 부드러운 신호 출력 범위: [{y_smooth.numpy().min():.4f}, {y_smooth.numpy().max():.4f}]")
    print(f"  - 급격한 신호 출력 범위: [{y_abrupt.numpy().min():.4f}, {y_abrupt.numpy().max():.4f}]")

    print("\n선택적 메커니즘의 핵심:")
    print("  1. 부드러운 변화 → delta 값이 작음 → 상태 천천히 갱신")
    print("  2. 급격한 변화 → delta 값이 큼 → 상태 빠르게 갱신")
    print("  3. 이를 통해 모델이 중요한 변화점을 자동으로 감지")


if __name__ == "__main__":
    # 아키텍처 비교 실험 수행
    results = compare_architectures()

    # 선택적 메커니즘 시연
    demonstrate_selective_mechanism()

    print("\n" + "=" * 60)
    print("실험 완료")
    print("=" * 60)
    print("\n※ 이 코드는 교육 목적의 예제입니다.")
    print("   실제 정책 분석에서는 검증된 라이브러리 사용을 권장합니다.")
