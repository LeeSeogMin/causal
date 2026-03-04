"""
CATE-RDD 데이터 생성
이론 증명을 위한 데이터:
- 처치효과가 공변량에 따라 비선형적으로 이질적
- T-learner/ML이 선형 상호작용 모형보다 우수하도록 설계
- 사전 성과가 높을수록 효과가 큼 (Matthew effect)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# 설정
n = 2000  # 표본 크기 증가
cutoff = 50
BASE_EFFECT = 3.0  # 기본 처치효과

print("=" * 80)
print("CATE-RDD 데이터 생성")
print(f"기본 처치효과: {BASE_EFFECT}")
print("=" * 80)

# 1. 배정 변수 생성
print("\n[1] 배정 변수 생성")
score = np.random.normal(50, 12, n)

# 2. 처치 배정 (Sharp RDD)
print("[2] 처치 배정")
treat = (score >= cutoff).astype(int)

# 3. 공변량 생성
print("[3] 공변량 생성")

# 사전 성과 (표준화)
pre_outcome = np.random.normal(0, 1, n)

# 교육 연수
education = np.random.normal(12, 3, n)

# 소득 (로그 스케일로 사용)
income = np.random.lognormal(10.5, 0.5, n)

# 연령
age = np.random.normal(35, 10, n)

# 경력
experience = np.random.normal(10, 5, n)

# 추가 공변량
cov_06 = np.random.normal(0, 1, n)
cov_07 = np.random.normal(0, 1, n)

# 4. 이질적 처치효과 (CATE) 생성 - 강한 비선형 패턴
print("[4] 이질적 처치효과 생성 (강한 비선형)")

# 핵심: CATE가 공변량의 비선형 함수
# 선형 모형은 이를 제대로 포착 못함, ML은 포착 가능

# 사전 성과의 비선형 효과 - 더 강하게
# 사전 성과가 높을수록 효과가 비선형적으로 급격히 증가
pre_effect = 3.0 * (1 / (1 + np.exp(-2.0 * pre_outcome)))  # S자 곡선: 0~3 범위

# 교육의 임계점 효과 (12년 이상에서 추가 효과) - 더 강하게
edu_effect = 1.0 * np.maximum(education - 12, 0)  # ReLU 형태

# 사전성과-교육 복잡한 상호작용 (선형이 못 잡는 패턴)
# 둘 다 높을 때만 추가 효과 발생
interaction_effect = 1.5 * np.maximum(pre_outcome - 0.5, 0) * np.maximum(education - 14, 0) / 3

# 연령의 역U자 효과 (35세 근처에서 최대) - 더 강하게
age_effect = 1.0 * np.exp(-0.005 * (age - 35)**2)

# 진정한 CATE - 노이즈 최소화
true_cate = (
    BASE_EFFECT +           # 기본 효과
    pre_effect +            # 사전 성과 비선형 효과 (0~3)
    edu_effect +            # 교육 임계점 효과
    interaction_effect +    # 비선형 상호작용
    age_effect +            # 연령 역U자 효과
    np.random.normal(0, 0.2, n)  # 개인별 랜덤 변동 (매우 작게)
)

print(f"진정한 CATE 평균: {true_cate.mean():.2f}")
print(f"진정한 CATE 범위: [{true_cate.min():.2f}, {true_cate.max():.2f}]")
print(f"진정한 CATE 표준편차: {true_cate.std():.2f}")

# 5. 결과 변수 생성
print("[5] 결과 변수 생성")

# 기본 결과 (공변량 효과) - 단순하게
base_outcome = (
    50 +
    2.0 * pre_outcome +           # 사전 성과 효과
    0.3 * education +             # 교육 효과
    0.05 * (score - cutoff)       # 배정 변수 효과 (작음)
)

# 결과 변수: 기본 + 처치효과 - 노이즈 줄임
noise = np.random.normal(0, 1.5, n)  # 노이즈 줄임
outcome = base_outcome + true_cate * treat + noise

# 6. DataFrame 생성
print("[6] DataFrame 생성")

df = pd.DataFrame({
    'score': score,
    'treat': treat,
    'outcome': outcome,
    'pre_outcome': pre_outcome,
    'education': education,
    'income': income,
    'age': age,
    'experience': experience,
    'cov_06': cov_06,
    'cov_07': cov_07,
    'true_cate': true_cate
})

# 7. 데이터 검증
print("\n[7] 데이터 검증")
print(f"총 관측치: {len(df)}")
print(f"처치군: {df['treat'].sum()} ({df['treat'].mean()*100:.1f}%)")

# 기준점 근처에서 CATE 분포 확인
bandwidth = 0.75 * df['score'].std()
near_cutoff = df[np.abs(df['score'] - cutoff) <= bandwidth]
print(f"\n기준점 ±{bandwidth:.1f} 내 관측치: {len(near_cutoff)}")
print(f"  처치군: {near_cutoff['treat'].sum()}")
print(f"  대조군: {(1-near_cutoff['treat']).sum()}")

# CATE 분위별 확인
print("\n사전 성과 분위별 진정한 CATE:")
near_cutoff_copy = near_cutoff.copy()
near_cutoff_copy['pre_tercile'] = pd.qcut(near_cutoff_copy['pre_outcome'], 3, labels=['하위', '중위', '상위'])
for tercile in ['하위', '중위', '상위']:
    subset = near_cutoff_copy[near_cutoff_copy['pre_tercile'] == tercile]
    print(f"  {tercile} 33%: 평균 CATE = {subset['true_cate'].mean():.2f}, N = {len(subset)}")

# 8. 저장
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../data/5-6-cate-rdd-data.csv')
df.to_csv(data_path, index=False)
print(f"\n데이터 저장: {data_path}")

# 9. 선형 vs 비선형 예측 가능성 확인
print("\n[8] 선형 vs 비선형 예측 가능성")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

X = near_cutoff_copy[['pre_outcome', 'education', 'income', 'age', 'experience']].values
y = near_cutoff_copy['true_cate'].values

# 선형 모형
lr = LinearRegression()
lr_scores = cross_val_score(lr, X, y, cv=5, scoring='neg_mean_squared_error')
lr_rmse = np.sqrt(-lr_scores.mean())

# GB 모형
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gb_scores = cross_val_score(gb, X, y, cv=5, scoring='neg_mean_squared_error')
gb_rmse = np.sqrt(-gb_scores.mean())

print(f"선형 모형 CV-RMSE (CATE 예측): {lr_rmse:.3f}")
print(f"GB 모형 CV-RMSE (CATE 예측): {gb_rmse:.3f}")
print(f"ML 개선율: {(1 - gb_rmse/lr_rmse)*100:.1f}%")

print("\n" + "=" * 80)
print("데이터 생성 완료")
print("=" * 80)
