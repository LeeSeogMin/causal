"""
Chapter 3: 모델 오설정 및 공통지지 위반이 발생하는 데이터 생성
==============================================================

목적: 전통적 로지스틱 회귀 PSM의 한계를 보여주는 시뮬레이션 데이터 생성
- 비선형 성향점수 모형 (로지스틱 회귀로는 포착 불가)
- 공통지지 영역 위반 (처치군/대조군 분포 불일치)
- 강한 상호작용 효과

저자: AI 기반 정책분석방법론
날짜: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(2025)

def generate_misspecified_data(n=2000):
    """
    모델 오설정과 공통지지 위반이 발생하는 데이터 생성
    
    특징:
    1. 성향점수가 강한 비선형 (제곱, 상호작용, 지수함수 포함)
    2. 공변량 분포가 처치/대조군에서 크게 다름 (공통지지 문제)
    3. 처치효과가 이질적
    """
    
    print("=" * 70)
    print("모델 오설정 + 공통지지 위반 데이터 생성")
    print("=" * 70)
    
    # === 공변량 생성 ===
    # X1: 나이 (20-65)
    age = np.random.uniform(20, 65, n)
    
    # X2: 소득 (로그정규분포 - 우측 꼬리 긴 분포)
    income = np.random.lognormal(4.0, 0.8, n)
    income = np.clip(income, 10, 500)  # 10만원 ~ 500만원
    
    # X3: 교육년수 (8-20년)
    education = np.random.normal(14, 3, n)
    education = np.clip(education, 8, 20)
    
    # X4: 경력년수 (0-40년, 나이와 상관)
    experience = 0.6 * (age - 22) + np.random.normal(0, 3, n)
    experience = np.clip(experience, 0, 40)
    
    # X5: 자산 (소득과 나이에 의존, 비선형)
    assets = 50 + 2 * income + 0.5 * age + 0.01 * income * age + np.random.normal(0, 50, n)
    assets = np.clip(assets, 0, 1000)
    
    # === 성향점수 모형 (강한 비선형 + 상호작용) ===
    # 로지스틱 회귀로는 포착 불가능한 복잡한 패턴
    
    # 표준화
    age_std = (age - 42.5) / 12
    income_std = (income - 80) / 50
    edu_std = (education - 14) / 3
    exp_std = (experience - 15) / 10
    assets_std = (assets - 200) / 150
    
    # 비선형 성향점수 로짓 (더 극단적으로)
    logit_ps = (
        -1.5                                    # 기본값 (더 낮게)
        + 2.5 * age_std                         # 나이 선형 (강화)
        - 3.0 * age_std**2                      # 나이 제곱 (강한 역U자)
        + 2.0 * income_std                      # 소득 선형 (강화)
        + 1.5 * np.exp(-income_std**2)          # 소득 가우시안
        - 1.0 * edu_std                         # 교육 선형
        + 2.5 * age_std * income_std            # 나이×소득 강한 상호작용
        + 1.2 * np.sin(2 * np.pi * exp_std)     # 경력 주기함수 (2배 주기)
        - 0.8 * assets_std                      # 자산 선형
        + 1.5 * edu_std * exp_std               # 교육×경력 상호작용
        + 1.0 * age_std**3                      # 3차항 추가
    )
    
    # 성향점수 (시그모이드)
    true_ps = 1 / (1 + np.exp(-logit_ps))
    
    # 처치 배정 (공통지지 문제를 더 심하게)
    # 고성향점수군에서 처치율 매우 높임 → 공통지지 영역 심각하게 축소
    ps_modified = true_ps ** 0.5  # 더 극단적으로
    treatment = np.random.binomial(1, ps_modified)
    
    # === 결과 변수 (이질적 처치효과) ===
    # 기본 결과 (비선형)
    Y0 = (
        50 
        + 2.0 * age_std 
        + 5.0 * income_std 
        + 3.0 * edu_std
        + 1.5 * exp_std
        + 2.0 * assets_std
        + 1.0 * age_std * edu_std  # 상호작용
        + np.random.normal(0, 5, n)
    )
    
    # 이질적 처치효과 (평균 5.0, 소득이 낮을수록 효과 큼)
    true_te = 5.0 + 3.0 * (1 - income_std)  # 저소득층에서 효과 큼
    
    # 관측 결과
    Y1 = Y0 + true_te
    outcome = treatment * Y1 + (1 - treatment) * Y0
    
    # 진정한 ATT
    true_att = true_te[treatment == 1].mean()
    
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
    print(f"진정한 ATE: {true_te.mean():.3f}")
    
    # 공통지지 영역 분석
    ps_treated = true_ps[treatment == 1]
    ps_control = true_ps[treatment == 0]
    
    overlap_min = max(ps_treated.min(), ps_control.min())
    overlap_max = min(ps_treated.max(), ps_control.max())
    
    outside_treated = ((ps_treated < overlap_min) | (ps_treated > overlap_max)).sum()
    outside_control = ((ps_control < overlap_min) | (ps_control > overlap_max)).sum()
    
    print(f"\n[공통지지 영역 분석 (진정한 성향점수 기준)]")
    print(f"처치군 PS 범위: [{ps_treated.min():.3f}, {ps_treated.max():.3f}]")
    print(f"대조군 PS 범위: [{ps_control.min():.3f}, {ps_control.max():.3f}]")
    print(f"공통지지 영역: [{overlap_min:.3f}, {overlap_max:.3f}]")
    print(f"공통지지 밖 처치군: {outside_treated} ({100*outside_treated/n_treated:.1f}%)")
    print(f"공통지지 밖 대조군: {outside_control} ({100*outside_control/n_control:.1f}%)")
    
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
    data, true_att = generate_misspecified_data(n=2000)
    
    # 저장
    output_path = Path(__file__).parent / '../data/misspecified_data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"\n데이터 저장 완료: {output_path}")
    
    # 메타 정보 저장
    meta_path = Path(__file__).parent / '../data/misspecified_meta.txt'
    with open(meta_path, 'w') as f:
        f.write(f"true_att={true_att:.6f}\n")
        f.write(f"n_samples={len(data)}\n")
        f.write(f"n_treated={data['treatment'].sum()}\n")
        f.write(f"n_control={(1-data['treatment']).sum()}\n")
    print(f"메타 정보 저장: {meta_path}")
    
    return data, true_att


if __name__ == "__main__":
    data, true_att = main()
