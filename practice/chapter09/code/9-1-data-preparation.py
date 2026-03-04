"""
Chapter 9 - Deep Learning Basics and Policy Time Series Forecasting
9.5.1 Data Preparation and Preprocessing

한국 전력 수요 예측을 위한 데이터 준비 및 전처리
시간적 특징의 순환 인코딩과 시퀀스 생성
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
matplotlib.rc('font', family='Arial')
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = 'practice/chapter09/data/'

def generate_electricity_demand_data(n_days=365*2):
    """
    한국 전력 수요 시뮬레이션 데이터 생성
    2024년 KPX 데이터 패턴 기반

    Parameters:
    -----------
    n_days : int
        생성할 일수 (기본: 2년)

    Returns:
    --------
    df : DataFrame
        시간별 전력 수요 데이터
    """
    np.random.seed(42)

    # 시간 인덱스 생성 (시간별 데이터)
    hours = n_days * 24
    dates = pd.date_range(start='2023-01-01', periods=hours, freq='H')

    # 기본 특징 추출
    hour = dates.hour
    day_of_week = dates.dayofweek
    month = dates.month
    day_of_year = dates.dayofyear

    # 기저 수요 (약 60,000 MW)
    base_demand = 60000

    # 계절성 (여름, 겨울 높음)
    seasonal = 8000 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # 주간 패턴 (평일 > 주말)
    weekly = 3000 * (1 - 0.3 * ((day_of_week == 5) | (day_of_week == 6)).astype(int))

    # 시간대별 패턴 (오전 10시, 오후 2시, 저녁 7-9시 피크)
    hourly = (
        5000 * np.sin(2 * np.pi * (hour - 6) / 24) +  # 일반적 패턴
        2000 * np.exp(-0.5 * ((hour - 14) / 2)**2) +  # 오후 피크
        3000 * np.exp(-0.5 * ((hour - 20) / 1.5)**2)  # 저녁 피크
    )

    # 기온 효과 (여름: 냉방, 겨울: 난방)
    temperature = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 3, hours)
    temp_effect = 500 * (np.abs(temperature - 20))  # 20도에서 멀어질수록 수요 증가

    # 태양광 발전량 (낮 시간대, 여름에 많음)
    solar = np.maximum(0, 2000 * np.sin(np.pi * hour / 24) *
                      (1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 172) / 365)))

    # 풍력 발전량 (랜덤 성분 높음)
    wind = np.abs(np.random.normal(1000, 400, hours))

    # 재생에너지 정책 효과 (2024년 중반부터 증가)
    policy_effect = np.zeros(hours)
    policy_start = int(hours * 0.5)  # 중간 지점부터
    policy_effect[policy_start:] = 500 * (1 - np.exp(-0.01 * np.arange(hours - policy_start)))

    # 총 전력 수요 계산
    demand = (
        base_demand +
        seasonal +
        weekly +
        hourly +
        temp_effect -
        solar * 0.5 -  # 태양광이 수요를 일부 상쇄
        wind * 0.3 +   # 풍력이 수요를 일부 상쇄
        policy_effect +
        np.random.normal(0, 1000, hours)  # 노이즈
    )

    # SMP (System Marginal Price) 생성 (수요와 양의 상관관계)
    smp = 80 + 0.001 * (demand - 60000) + np.random.normal(0, 5, hours)
    smp = np.clip(smp, 40, 200)  # 40-200 원/kWh

    # 데이터프레임 생성
    df = pd.DataFrame({
        'datetime': dates,
        'demand': demand,
        'temperature': temperature,
        'solar_generation': solar,
        'wind_generation': wind,
        'smp': smp,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': ((day_of_week == 5) | (day_of_week == 6)).astype(int),
        'policy_active': (np.arange(hours) >= policy_start).astype(int)
    })

    return df

def create_temporal_features(df):
    """
    시간 특징을 순환 인코딩으로 변환

    Parameters:
    -----------
    df : DataFrame
        원본 데이터프레임

    Returns:
    --------
    df : DataFrame
        순환 인코딩이 추가된 데이터프레임
    """
    # 시간(hour) 순환 인코딩
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # 월(month) 순환 인코딩
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # 요일(day_of_week) 순환 인코딩
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df

def create_lag_features(df, target_col='demand', lags=[24, 48, 168]):
    """
    지연 특징 생성 (24시간, 48시간, 168시간 전)

    Parameters:
    -----------
    df : DataFrame
        원본 데이터프레임
    target_col : str
        타겟 컬럼 이름
    lags : list
        지연 시간 리스트

    Returns:
    --------
    df : DataFrame
        지연 특징이 추가된 데이터프레임
    """
    for lag in lags:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)

    return df

def create_rolling_features(df, target_col='demand', windows=[24, 168]):
    """
    이동 통계 특징 생성

    Parameters:
    -----------
    df : DataFrame
        원본 데이터프레임
    target_col : str
        타겟 컬럼 이름
    windows : list
        윈도우 크기 리스트

    Returns:
    --------
    df : DataFrame
        이동 통계 특징이 추가된 데이터프레임
    """
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}h'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_rolling_std_{window}h'] = df[target_col].rolling(window).std()

    return df

def create_sequences(data, seq_length=168, pred_length=24):
    """
    LSTM 학습을 위한 시퀀스 생성
    1주일(168시간) 입력 → 24시간 예측

    Parameters:
    -----------
    data : ndarray
        입력 데이터
    seq_length : int
        입력 시퀀스 길이
    pred_length : int
        예측 길이

    Returns:
    --------
    X : ndarray
        입력 시퀀스 (n_samples, seq_length, n_features)
    y : ndarray
        타겟 시퀀스 (n_samples, pred_length)
    """
    X, y = [], []

    for i in range(len(data) - seq_length - pred_length + 1):
        # 입력: 과거 168시간
        X.append(data[i:i+seq_length])
        # 출력: 미래 24시간 (demand 컬럼만)
        y.append(data[i+seq_length:i+seq_length+pred_length, 0])  # demand는 첫 번째 컬럼

    return np.array(X), np.array(y)

def visualize_data(df):
    """데이터 시각화"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # (1) 전력 수요 시계열
    ax1 = axes[0, 0]
    ax1.plot(df['datetime'][:24*30], df['demand'][:24*30], linewidth=1)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Demand (MW)', fontsize=11)
    ax1.set_title('(a) Electricity Demand (30 days)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # (2) 시간별 평균 수요
    ax2 = axes[0, 1]
    hourly_avg = df.groupby('hour')['demand'].mean()
    ax2.plot(hourly_avg.index, hourly_avg.values, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('Hour of Day', fontsize=11)
    ax2.set_ylabel('Average Demand (MW)', fontsize=11)
    ax2.set_title('(b) Average Demand by Hour', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # (3) 월별 평균 수요
    ax3 = axes[1, 0]
    monthly_avg = df.groupby('month')['demand'].mean()
    ax3.bar(monthly_avg.index, monthly_avg.values, alpha=0.7)
    ax3.set_xlabel('Month', fontsize=11)
    ax3.set_ylabel('Average Demand (MW)', fontsize=11)
    ax3.set_title('(c) Average Demand by Month', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # (4) 기온 vs 수요
    ax4 = axes[1, 1]
    sample_idx = np.random.choice(len(df), 5000, replace=False)
    ax4.scatter(df.iloc[sample_idx]['temperature'],
               df.iloc[sample_idx]['demand'],
               alpha=0.3, s=10)
    ax4.set_xlabel('Temperature (C)', fontsize=11)
    ax4.set_ylabel('Demand (MW)', fontsize=11)
    ax4.set_title('(d) Temperature vs Demand', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # (5) 재생에너지 발전량
    ax5 = axes[2, 0]
    ax5.plot(df['datetime'][:24*30], df['solar_generation'][:24*30],
            label='Solar', alpha=0.7, linewidth=1)
    ax5.plot(df['datetime'][:24*30], df['wind_generation'][:24*30],
            label='Wind', alpha=0.7, linewidth=1)
    ax5.set_xlabel('Date', fontsize=11)
    ax5.set_ylabel('Generation (MW)', fontsize=11)
    ax5.set_title('(e) Renewable Energy Generation', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # (6) SMP (System Marginal Price)
    ax6 = axes[2, 1]
    ax6.plot(df['datetime'][:24*30], df['smp'][:24*30], linewidth=1, color='orange')
    ax6.set_xlabel('Date', fontsize=11)
    ax6.set_ylabel('SMP (KRW/kWh)', fontsize=11)
    ax6.set_title('(f) System Marginal Price', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    # Visualization displayed (not saved)

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("제9장: 데이터 준비 및 전처리")
    print("=" * 80)

    # 1. 데이터 생성
    print("\n[1단계] 전력 수요 데이터 생성 중...")
    df = generate_electricity_demand_data(n_days=365*2)
    print(f"  - 총 시간: {len(df):,} 시간")
    print(f"  - 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"  - 평균 수요: {df['demand'].mean():.0f} MW")
    print(f"  - 최대 수요: {df['demand'].max():.0f} MW")
    print(f"  - 최소 수요: {df['demand'].min():.0f} MW")

    # 2. 순환 인코딩
    print("\n[2단계] 시간 특징 순환 인코딩 중...")
    df = create_temporal_features(df)
    print(f"  - 추가된 특징: hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos")

    # 3. 지연 특징 생성
    print("\n[3단계] 지연 특징 생성 중...")
    df = create_lag_features(df, target_col='demand', lags=[24, 48, 168])
    print(f"  - 추가된 특징: demand_lag_24h, demand_lag_48h, demand_lag_168h")

    # 4. 이동 통계 특징 생성
    print("\n[4단계] 이동 통계 특징 생성 중...")
    df = create_rolling_features(df, target_col='demand', windows=[24, 168])
    print(f"  - 추가된 특징: rolling_mean, rolling_std (24h, 168h)")

    # 결측치 제거 (지연 및 이동 통계로 인한)
    df_clean = df.dropna().reset_index(drop=True)
    print(f"\n  - 결측치 제거 후 샘플 수: {len(df_clean):,}")

    # 5. 데이터 저장
    print("\n[5단계] 데이터 저장 중...")
    df_clean.to_csv(OUTPUT_DIR + '9-1-preprocessed-data.csv', index=False, encoding='utf-8-sig')
    print(f"  - 전처리 데이터 저장: {OUTPUT_DIR}9-1-preprocessed-data.csv")

    # 6. 시퀀스 생성 예시
    print("\n[6단계] 시퀀스 생성 예시...")
    # 특징 선택 (순환 인코딩 + 지연 특징 + 이동 통계 + 외생 변수)
    feature_cols = [
        'demand',  # 타겟 (첫 번째 위치 중요)
        'temperature', 'solar_generation', 'wind_generation',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'dow_sin', 'dow_cos', 'is_weekend', 'policy_active',
        'demand_lag_24h', 'demand_lag_48h', 'demand_lag_168h',
        'demand_rolling_mean_24h', 'demand_rolling_std_24h',
        'demand_rolling_mean_168h', 'demand_rolling_std_168h'
    ]

    # RobustScaler로 정규화
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(df_clean[feature_cols])

    # 시퀀스 생성
    X, y = create_sequences(data_scaled, seq_length=168, pred_length=24)
    print(f"  - 입력 시퀀스 형태: {X.shape}")
    print(f"  - 타겟 시퀀스 형태: {y.shape}")
    print(f"  - 해석: {X.shape[0]:,}개 샘플, 각 168시간 입력 → 24시간 예측")

    # 시퀀스 데이터 저장 (numpy 형식)
    np.save(OUTPUT_DIR + '9-1-sequences-X.npy', X)
    np.save(OUTPUT_DIR + '9-1-sequences-y.npy', y)
    print(f"  - 시퀀스 데이터 저장: {OUTPUT_DIR}9-1-sequences-X.npy, 9-1-sequences-y.npy")

    # Scaler 파라미터 저장 (역정규화용)
    scaler_params = {
        'center': scaler.center_[0],  # demand의 중앙값
        'scale': scaler.scale_[0]     # demand의 IQR
    }
    np.save(OUTPUT_DIR + '9-1-scaler-params.npy', scaler_params)
    print(f"  - Scaler 파라미터 저장: {OUTPUT_DIR}9-1-scaler-params.npy")

    # 7. 시각화
    print("\n[7단계] 데이터 탐색 시각화 중...")
    visualize_data(df_clean)

    # 분석 결과 요약
    print("\n" + "=" * 80)
    print("데이터 전처리 완료")
    print("=" * 80)
    print("\n생성된 특징:")
    print("  1. 순환 인코딩: 시간, 월, 요일의 연속성 보존")
    print("  2. 지연 특징: 24시간, 48시간, 168시간 전 수요")
    print("  3. 이동 통계: 24시간, 168시간 이동평균 및 표준편차")
    print("  4. 외생 변수: 기온, 태양광/풍력 발전량, 정책 활성화 여부")
    print("\n시사점:")
    print("  - 순환 인코딩을 통해 23시와 0시의 연속성 학습 가능")
    print("  - 지연 특징으로 일간 및 주간 패턴 포착")
    print("  - 이동 통계로 최근 추세와 변동성 정보 제공")
    print("  - RobustScaler로 이상치에 강건한 정규화 수행")

    print("\n출력 파일:")
    print(f"  - {OUTPUT_DIR}9-1-preprocessed-data.csv")
    print(f"  - {OUTPUT_DIR}9-1-sequences-X.npy")
    print(f"  - {OUTPUT_DIR}9-1-sequences-y.npy")

if __name__ == "__main__":
    main()
