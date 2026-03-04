"""
제2장: 데이터 생성 (DML 분석용)
비선형성과 교호작용이 강한 합성 데이터를 생성하여 DML의 성능을 검증하기 위함.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
import os

def generate_dml_data(n=2000, seed=42):
    """
    DML 분석을 위한 고품질 합성 데이터 생성
    - 강한 비선형성 및 교호작용 포함 (OLS가 실패하도록 설계)
    - 이진 처치 변수
    - DML의 우수성을 입증하기 위한 복잡한 데이터 구조
    """
    np.random.seed(seed)

    # 1. 공변량 생성 (5개 변수)
    X = np.random.randn(n, 5)

    # 2. 처치 배정 메커니즘 (Propensity Score) - 강한 비선형성
    # OLS가 포착하기 어려운 복잡한 성향점수 함수
    logits = (
        0.8 * X[:, 0]
        - 0.6 * X[:, 1]
        + 0.7 * X[:, 0] * X[:, 1]           # 강한 교호작용
        + 0.5 * (X[:, 0] ** 2)              # 제곱항
        - 0.4 * (X[:, 1] ** 2)              # 제곱항
        + 0.6 * np.sin(2 * X[:, 2])         # 주기적 비선형성
    )
    true_propensity = expit(logits)
    treatment = np.random.binomial(1, true_propensity)

    # 3. 잠재 결과 (Potential Outcomes) - 매우 강한 비선형성
    # OLS가 절대 포착할 수 없는 복잡한 패턴
    epsilon = np.random.randn(n) * 0.5

    # Base outcome (Control group) - 강한 비선형 confounding
    Y0_true = (
        2.0
        + 1.5 * X[:, 0]
        - 1.0 * X[:, 1]
        + 1.2 * (X[:, 0] ** 2)              # 강한 제곱항
        + 0.8 * (X[:, 1] ** 2)              # 강한 제곱항
        + 1.0 * X[:, 0] * X[:, 1]           # 강한 교호작용
        + 0.6 * X[:, 1] * X[:, 2]           # 추가 교호작용
        + 0.8 * np.sin(2 * X[:, 2])         # 주기적 비선형성
        + 0.5 * np.exp(-X[:, 3] ** 2)       # 지수적 비선형성
        + epsilon
    )

    # Treatment effect (Heterogeneous)
    # ATE ≈ 3.0 (X2의 평균이 0이므로)
    tau = 3.0 + 0.8 * X[:, 2]
    Y1_true = Y0_true + tau
    
    # Observed Outcome
    Y_observed = treatment * Y1_true + (1 - treatment) * Y0_true
    
    # True Parameters
    true_ate = np.mean(tau)
    
    print(f"Data Generated: N={n}")
    print(f"True ATE: {true_ate:.4f}")
    print(f"Treatment Rate: {np.mean(treatment):.4f}")
    
    # 데이터프레임 생성
    df = pd.DataFrame(X, columns=['X1', 'X2', 'X3', 'X4', 'X5'])
    df['treatment'] = treatment
    df['outcome'] = Y_observed
    
    # 검증용 메타데이터 (실제 분석에선 사용 불가하지만 학습용으로 저장)
    df['true_propensity'] = true_propensity
    df['true_cate'] = tau
    df['Y0_true'] = Y0_true
    df['Y1_true'] = Y1_true
    
    return df, true_ate

def main():
    # 데이터 생성
    df, true_ate = generate_dml_data(n=2000, seed=42)
    
    # 저장 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '../data')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    save_path = os.path.join(data_dir, '2-3-dml-data.csv')
    
    # CSV 저장
    df.to_csv(save_path, index=False)
    print(f"\n데이터가 저장되었습니다: {save_path}")
    
    # 메타데이터 별도 저장 (True ATE 등) - 파일명에 기록하거나 별도 txt로? 
    # 여기선 간단히 출력만 하고, 분석 코드에서 참값을 알 수 있도록 파일 헤더나 별도 파일 고려.
    # 편의를 위해 'meta_dml.txt'에 ATE 정보만 적어둠.
    with open(os.path.join(data_dir, '2-3-dml-meta.txt'), 'w') as f:
        f.write(f"True ATE: {true_ate}\n")

if __name__ == "__main__":
    main()
