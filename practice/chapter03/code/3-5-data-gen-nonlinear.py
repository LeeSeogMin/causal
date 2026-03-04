"""
Chapter 3.5: 비선형 데이터 생성 (기계학습/딥러닝 우위를 보여주기 위함)
====================================================================

목적: 로지스틱 회귀가 실패하고 ML/DL이 성공하는 데이터 생성
- 강한 비선형 성향점수 함수
- 고차 상호작용
- 기계학습/딥러닝이 로지스틱 회귀보다 우수한 성능을 보여야 함

저자: AI 기반 정책분석방법론
날짜: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(2025)

def generate_nonlinear_psm_data(n=2000):
    """
    비선형 성향점수 함수를 가진 데이터 생성
    - 로지스틱 회귀는 실패하고 ML/DL이 성공해야 함
    """
    
    print("=" * 70)
    print("3.5절용 비선형 PSM 데이터 생성")
    print("=" * 70)
    
    # === 공변량 생성 (5개 연속형) ===
    # 나이: 20-65세
    age = np.random.uniform(20, 65, n)
    
    # 소득: 로그정규분포 (만원)
    income = np.random.lognormal(4.0, 0.8, n)
    income = np.clip(income, 10, 500)
    
    # 교육년수: 8-20년
    education = np.random.normal(14, 3, n)
    education = np.clip(education, 8, 20)
    
    # 경력년수: 나이와 상관
    experience = 0.6 * (age - 22) + np.random.normal(0, 3, n)
    experience = np.clip(experience, 0, 40)
    
    # 자산: 소득과 나이에 비선형적으로 의존
    assets = 50 + 2 * income + 0.5 * age + 0.01 * income * age + np.random.normal(0, 50, n)
    assets = np.clip(assets, 0, 1000)
    
    # === 표준화 (성향점수 모형에서 사용) ===
    age_std = (age - 42.5) / 12
    income_std = (income - 80) / 50
    edu_std = (education - 14) / 3
    exp_std = (experience - 15) / 10
    assets_std = (assets - 200) / 150
    
    # === 성향점수 모형 (강한 비선형 + 상호작용) ===
    # 로지스틱 회귀로는 절대 포착 불가능한 패턴
    logit_ps = (
        -0.5                                    # 기본값
        + 1.5 * age_std                         # 나이 선형
        - 2.5 * age_std**2                      # 나이 제곱 (역U자)
        + 1.0 * income_std                      # 소득 선형
        + 1.5 * np.exp(-income_std**2)          # 소득 가우시안
        - 0.8 * edu_std                         # 교육 선형
        + 2.0 * age_std * income_std            # 나이×소득 상호작용
        + 1.0 * np.sin(2 * np.pi * exp_std)     # 경력 주기함수
        - 0.5 * assets_std                      # 자산 선형
        + 1.2 * edu_std * exp_std               # 교육×경력 상호작용
        + 0.8 * age_std**3                      # 3차항
    )
    
    # 성향점수 (시그모이드)
    true_ps = 1 / (1 + np.exp(-logit_ps))
    
    # 처치 배정
    treatment = np.random.binomial(1, true_ps)
    
    # === 결과 변수 (이질적 처치효과) ===
    # 기본 결과
    Y0 = (
        50 
        + 2.0 * age_std 
        + 5.0 * income_std 
        + 3.0 * edu_std
        + 1.5 * exp_std
        + 2.0 * assets_std
        + 1.0 * age_std * edu_std
        + np.random.normal(0, 5, n)
    )
    
    # 처치효과: 평균 5.0
    true_te = 5.0 + 2.0 * (1 - income_std)  # 저소득층에서 효과 큼
    
    # 관측 결과
    Y1 = Y0 + true_te
    outcome = treatment * Y1 + (1 - treatment) * Y0
    
    # 진정한 ATT
    true_att = true_te[treatment == 1].mean()
    true_ate = true_te.mean()
    
    # === 데이터프레임 생성 ===
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'experience': experience,
        'assets': assets,
        'treatment': treatment,
        'outcome': outcome,
        'true_ps': true_ps,
        'true_te': true_te
    })
    
    # === 통계 출력 ===
    n_treated = treatment.sum()
    n_control = n - n_treated
    
    print(f"\n[데이터 요약]")
    print(f"총 샘플 수: {n}")
    print(f"처치군: {n_treated} ({100*n_treated/n:.1f}%)")
    print(f"대조군: {n_control} ({100*n_control/n:.1f}%)")
    print(f"\n[진정한 값]")
    print(f"진정한 ATT: {true_att:.3f}")
    print(f"진정한 ATE: {true_ate:.3f}")
    
    # 공변량 분포 차이 확인
    print(f"\n[공변량 분포 (처치군 vs 대조군)]")
    for var in ['age', 'income', 'education', 'experience', 'assets']:
        t_mean = data.loc[data['treatment']==1, var].mean()
        c_mean = data.loc[data['treatment']==0, var].mean()
        pooled_std = np.sqrt((data.loc[data['treatment']==1, var].var() + 
                             data.loc[data['treatment']==0, var].var()) / 2)
        smd = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
        print(f"  {var}: 처치군={t_mean:.2f}, 대조군={c_mean:.2f}, SMD={smd:.3f}")
    
    return data, true_att


def main():
    # 데이터 생성
    data, true_att = generate_nonlinear_psm_data(n=2000)
    
    # 저장
    output_path = Path(__file__).parent / '../data/nonlinear_psm_data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"\n데이터 저장 완료: {output_path}")
    
    # 메타 정보 저장
    meta_path = Path(__file__).parent / '../data/nonlinear_psm_meta.txt'
    with open(meta_path, 'w') as f:
        f.write(f"true_att={true_att:.6f}\n")
        f.write(f"n_samples={len(data)}\n")
        f.write(f"n_treated={data['treatment'].sum()}\n")
        f.write(f"n_control={(1-data['treatment']).sum()}\n")
    print(f"메타 정보 저장: {meta_path}")
    
    return data, true_att


if __name__ == "__main__":
    data, true_att = main()
