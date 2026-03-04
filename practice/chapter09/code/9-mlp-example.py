"""
Chapter 9 - Deep Learning Basics and Policy Time Series Forecasting
9.1 Deep Learning Basics and Neural Networks - MLP Example

정책 효과 분류를 위한 다층 퍼셉트론 구현
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
matplotlib.rc('font', family='Arial')
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = 'practice/chapter09/data/'

def generate_policy_data(n_samples=5000):
    """
    정책 효과 분류를 위한 합성 데이터 생성

    Returns:
    --------
    X : ndarray
        특징 행렬 (n_samples, n_features)
    y : ndarray
        정책 성공 여부 레이블 (0 or 1)
    """
    np.random.seed(42)

    # 특징 생성
    # Feature 1: 정책 예산 (억원)
    budget = np.random.uniform(10, 500, n_samples)

    # Feature 2: 집행률 (%)
    execution_rate = np.random.uniform(30, 100, n_samples)

    # Feature 3: 대상 인구 (만명)
    target_population = np.random.uniform(1, 100, n_samples)

    # Feature 4: 정책 기간 (개월)
    duration = np.random.randint(3, 36, n_samples)

    # Feature 5: 이전 정책 성공 횟수
    previous_success = np.random.randint(0, 5, n_samples)

    # Feature 6: 지역 경제 수준 (GDP per capita, 천만원)
    gdp_per_capita = np.random.uniform(2.5, 5.0, n_samples)

    # 특징 행렬 생성
    X = np.column_stack([
        budget, execution_rate, target_population,
        duration, previous_success, gdp_per_capita
    ])

    # 레이블 생성 (복잡한 비선형 관계)
    # 정책 성공 확률 = f(예산, 집행률, 경험, 경제수준)
    success_prob = (
        0.3 +
        0.15 * (budget / 500) +  # 예산 효과
        0.25 * (execution_rate / 100) +  # 집행률 효과
        0.10 * (previous_success / 5) +  # 경험 효과
        0.15 * (gdp_per_capita / 5.0) +  # 경제 수준 효과
        0.05 * (duration / 36)  # 기간 효과
    )

    # 비선형 관계 추가 (상호작용 효과)
    success_prob += 0.10 * (execution_rate / 100) * (previous_success / 5)

    # 확률을 0-1 범위로 클리핑
    success_prob = np.clip(success_prob, 0, 1)

    # 이진 레이블 생성
    y = (np.random.random(n_samples) < success_prob).astype(int)

    return X, y

def create_mlp_model(input_dim, hidden_units=[64, 32]):
    """
    정책 효과 분류를 위한 다층 퍼셉트론

    Parameters:
    -----------
    input_dim : int
        입력 특징 수
    hidden_units : list
        은닉층 유닛 수 리스트

    Returns:
    --------
    model : keras.Model
        컴파일된 MLP 모델
    """
    model = keras.Sequential(name='PolicyEffectMLP')
    model.add(keras.layers.Input(shape=(input_dim,), name='input'))

    # 은닉층 추가
    for i, units in enumerate(hidden_units):
        model.add(keras.layers.Dense(
            units,
            activation='relu',
            name=f'hidden_{i+1}'
        ))
        model.add(keras.layers.BatchNormalization(name=f'bn_{i+1}'))
        model.add(keras.layers.Dropout(0.2, name=f'dropout_{i+1}'))

    # 출력층
    model.add(keras.layers.Dense(1, activation='sigmoid', name='output'))

    # 모델 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    return model

def visualize_results(history, y_test, y_pred, y_pred_prob):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 학습 곡선 - 손실
    ax1 = axes[0, 0]
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Binary Crossentropy Loss', fontsize=11)
    ax1.set_title('(a) Learning Curve - Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # (2) 학습 곡선 - 정확도
    ax2 = axes[0, 1]
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('(b) Learning Curve - Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # (3) 예측 확률 분포
    ax3 = axes[1, 0]
    ax3.hist(y_pred_prob[y_test == 0], bins=50, alpha=0.7, label='Failed (y=0)', color='red')
    ax3.hist(y_pred_prob[y_test == 1], bins=50, alpha=0.7, label='Success (y=1)', color='green')
    ax3.set_xlabel('Predicted Probability', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('(c) Prediction Probability Distribution', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # (4) Confusion Matrix
    ax4 = axes[1, 1]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    im = ax4.imshow(cm, cmap='Blues', aspect='auto')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Failed', 'Success'])
    ax4.set_yticklabels(['Failed', 'Success'])
    ax4.set_xlabel('Predicted Label', fontsize=11)
    ax4.set_ylabel('True Label', fontsize=11)
    ax4.set_title('(d) Confusion Matrix', fontsize=12, fontweight='bold')

    # Confusion matrix 값 표시
    for i in range(2):
        for j in range(2):
            text = ax4.text(j, i, f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                          ha='center', va='center', color='black', fontsize=11)

    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    plt.show()
    # Visualization displayed (not saved)

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제9장: 딥러닝 기초와 신경망 - MLP 예제")
    print("=" * 80)

    # 1. 데이터 생성
    print("\n[1단계] 정책 데이터 생성 중...")
    X, y = generate_policy_data(n_samples=5000)
    print(f"  - 샘플 수: {len(X):,}")
    print(f"  - 특징 수: {X.shape[1]}")
    print(f"  - 성공 비율: {y.mean():.2%}")

    # 데이터 분할
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"  - 훈련 세트: {len(X_train):,}")
    print(f"  - 검증 세트: {len(X_val):,}")
    print(f"  - 테스트 세트: {len(X_test):,}")

    # 데이터 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 데이터 저장
    feature_names = ['budget', 'execution_rate', 'target_population',
                    'duration', 'previous_success', 'gdp_per_capita']
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['success'] = y_train
    train_df.to_csv(OUTPUT_DIR + '9-mlp-training-data.csv', index=False, encoding='utf-8-sig')
    print(f"  - 훈련 데이터 저장: {OUTPUT_DIR}9-mlp-training-data.csv")

    # 2. 모델 생성
    print("\n[2단계] MLP 모델 생성 중...")
    model = create_mlp_model(input_dim=X.shape[1], hidden_units=[64, 32])
    print(f"  - 모델 아키텍처:")
    model.summary()

    # 3. 모델 학습
    print("\n[3단계] 모델 학습 중...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    print(f"  - 학습 완료 (총 {len(history.history['loss'])} 에포크)")
    print(f"  - 최종 훈련 정확도: {history.history['accuracy'][-1]:.4f}")
    print(f"  - 최종 검증 정확도: {history.history['val_accuracy'][-1]:.4f}")

    # 4. 모델 평가
    print("\n[4단계] 모델 평가 중...")
    y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    test_loss, test_acc, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)

    print("\n" + "=" * 80)
    print("MLP 모델 성능 평가")
    print("=" * 80)
    print(f"Test Loss (Binary Crossentropy): {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC-ROC: {test_auc:.4f}")

    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred,
                               target_names=['Failed', 'Success'],
                               digits=4))

    # 특징 중요도 분석 (간접적)
    print("\n특징별 평균 가중치 크기 (첫 번째 은닉층):")
    first_layer_weights = model.layers[0].get_weights()[0]  # Shape: (6, 64)
    feature_importance = np.abs(first_layer_weights).mean(axis=1)

    for i, name in enumerate(feature_names):
        print(f"  {name:20s}: {feature_importance[i]:.4f}")

    # 5. 시각화
    print("\n[5단계] 결과 시각화 중...")
    visualize_results(history, y_test, y_pred, y_pred_prob)

    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)
    print("\n출력 파일:")
    print(f"  - {OUTPUT_DIR}9-mlp-training-data.csv")

if __name__ == "__main__":
    main()
