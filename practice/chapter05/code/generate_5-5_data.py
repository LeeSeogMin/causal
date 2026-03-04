"""
DML-RDD 데이터 생성
본문 이론 증명을 위한 데이터: 
- 고차원 공변량이 결과 변수에 강하게 영향 (비선형)
- 기준점 근처에서 공변량 불균형 (교란) - 강화
- DML이 이 교란을 조정하여 선형 조정보다 우수
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# 설정
n = 3000  # 표본 크기
cutoff = 50  # 기준점
n_covariates = 25  # 공변량 개수
TRUE_EFFECT = 3.5  # 진정한 처치효과

print("=" * 80)
print("DML-RDD 데이터 생성")
print(f"진정한 처치효과: {TRUE_EFFECT}")
print("=" * 80)

# 1. 배정 변수 먼저 생성 (RDD의 핵심)
print("\n[1] 배정 변수 생성")
score = np.random.normal(50, 12, n)

# 2. 처치 배정 (Sharp RDD)
print("[2] 처치 배정")
treat = (score >= cutoff).astype(int)

# 3. 공변량 생성 - 기준점 근처에서 강한 불균형 발생
print("[3] 공변량 생성 (기준점 근처 강한 불균형)")

distance_from_cutoff = score - cutoff

# 핵심: 기준점 바로 위/아래에서 공변량이 강하게 다름
# 이것이 "교란"을 발생시켜 DML 조정이 필요하게 만듦

# 사전 결과 변수: 기준점 통과자가 체계적으로 더 높음 (선택 효과)
# 더 강한 효과로 수정
pre_outcome_base = np.random.normal(50, 10, n)
selection_effect = 3 * (score >= cutoff).astype(float)  # 처치군이 3점 더 높음
pre_outcome = pre_outcome_base + selection_effect

# 교육 연수: 기준점 통과자가 더 높음
education_base = np.random.normal(12, 3, n)
education = education_base + 0.8 * (score >= cutoff).astype(float)

# 경력: 기준점 통과자가 더 높음
experience_base = np.random.normal(10, 5, n)
experience = experience_base + 0.5 * (score >= cutoff).astype(float)

# 기타 공변량 (독립)
income = np.random.normal(40000, 15000, n)
age = np.random.normal(35, 10, n)

# 추가 공변량 20개 (일부는 처치와 상관)
additional_covs = np.random.normal(0, 1, (n, 20))
# 첫 5개는 처치와 상관되게
for i in range(5):
    additional_covs[:, i] += 0.3 * (score >= cutoff).astype(float)

# 공변량 이름
cov_names = ['pre_outcome', 'education', 'income', 'age', 'experience']
cov_names += [f'cov_{i:02d}' for i in range(6, 26)]

# 소득 값 보정 (음수 방지)
income_safe = np.maximum(income, 1000)

X = np.column_stack([pre_outcome, education, income_safe/10000, age, experience, additional_covs])

# 4. 결과 변수 생성 (비선형 효과 + 공변량이 강하게 영향)
print("[4] 결과 변수 생성 (비선형, 공변량 효과 강함)")

# 공변량의 비선형 효과 (DML/ML이 포착해야 함)
cov_effect = (
    # pre_outcome의 강한 비선형 효과
    0.6 * (pre_outcome - 50) + 
    0.01 * (pre_outcome - 50)**2 +
    # 교육-경력 상호작용 (선형 모형이 일부 놓침)
    0.15 * education * experience / 10 +
    # 소득의 로그 효과
    0.5 * np.log(income_safe/10000 + 1) +
    # 연령 효과
    0.03 * age +
    # 추가 공변량 효과
    0.2 * additional_covs[:, 0] +
    0.15 * additional_covs[:, 1] +
    0.1 * additional_covs[:, 2] +
    0.05 * additional_covs[:, 0] * additional_covs[:, 1]  # 상호작용
)

# 배정 변수의 연속적 효과 (작게)
score_effect = 0.02 * (score - cutoff)

# 결과 변수: 공변량 효과 + 처치효과만 (교란 효과는 공변량을 통해)
noise = np.random.normal(0, 2, n)
outcome = (
    50 +                          # 기본값
    cov_effect +                  # 공변량 효과
    score_effect +                # 배정 변수 효과 (작음)
    TRUE_EFFECT * treat +         # 처치효과 (진정한 값)
    noise                         # 노이즈
)

# 5. DataFrame 생성
print("[5] DataFrame 생성")
df = pd.DataFrame({
    'score': score,
    'treat': treat,
    'outcome': outcome
})

for i, name in enumerate(cov_names):
    df[name] = X[:, i]

# 6. 데이터 검증
print("\n[6] 데이터 검증")
print(f"총 관측치: {len(df)}")
print(f"처치군: {df['treat'].sum()} ({df['treat'].mean()*100:.1f}%)")
print(f"결과 변수 평균: {df['outcome'].mean():.2f}")
print(f"결과 변수 표준편차: {df['outcome'].std():.2f}")

# 기준점 근처 데이터 확인
near_cutoff = df[np.abs(df['score'] - cutoff) <= 10]
print(f"\n기준점 ±10 내 관측치: {len(near_cutoff)}")
print(f"  처치군 평균 결과: {near_cutoff[near_cutoff['treat']==1]['outcome'].mean():.2f}")
print(f"  대조군 평균 결과: {near_cutoff[near_cutoff['treat']==0]['outcome'].mean():.2f}")
print(f"  단순 차이: {near_cutoff[near_cutoff['treat']==1]['outcome'].mean() - near_cutoff[near_cutoff['treat']==0]['outcome'].mean():.2f}")

# 기준점 근처 공변량 불균형 확인
near_treat = near_cutoff[near_cutoff['treat']==1]['pre_outcome'].mean()
near_control = near_cutoff[near_cutoff['treat']==0]['pre_outcome'].mean()
print(f"\n기준점 근처 사전결과 불균형:")
print(f"  처치군 평균: {near_treat:.2f}")
print(f"  대조군 평균: {near_control:.2f}")
print(f"  차이: {near_treat - near_control:.2f}")

# 7. 저장
save_path = os.path.join(os.path.dirname(__file__), '../data/5-5-dml-rdd-data.csv')
df.to_csv(save_path, index=False)
print(f"\n데이터 저장: {save_path}")

# 8. 공변량-결과 상관관계
print("\n[7] 공변량-결과 상관관계 (상위 5개)")
correlations = df[cov_names + ['outcome']].corr()['outcome'].drop('outcome').abs().sort_values(ascending=False)
print(correlations.head().to_string())

print("\n" + "=" * 80)
print(f"데이터 생성 완료. 진정한 처치효과 = {TRUE_EFFECT}")
print("=" * 80)
