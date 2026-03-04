"""
Chapter 3.5: 비선형 성향점수 데이터 생성 (v2)
=============================================

목적: ML/DL이 로지스틱 회귀보다 확실히 우수한 성능을 보이는 데이터
핵심: 비선형 교란이 강한 상황에서 정확한 PS 추정이 ATT 편향 감소로 직결

저자: AI 기반 정책분석방법론
날짜: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(2025)

def generate_nonlinear_confounded_data(n=3000):
    """
    강한 비선형 교란이 있는 데이터 생성
    - 처치 배정과 결과 모두에 비선형으로 영향을 미치는 교란변수
    - 정확한 PS 추정이 ATT 추정의 핵심
    """
    
    print("=" * 70)
    print("3.5절용 비선형 교란 데이터 생성 (v2)")
    print("=" * 70)
    
    # === 공변량 생성 ===
    # Z1: 핵심 비선형 교란변수
    Z1 = np.random.normal(0, 1, n)
    
    # Z2: 또 다른 비선형 교란변수  
    Z2 = np.random.normal(0, 1, n)
    
    # X1-X3: 추가 공변량
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)
    
    # === 성향점수 (강한 비선형 함수) ===
    # 로지스틱 회귀가 절대 맞출 수 없는 형태
    logit_ps = (
        - 0.5                           # 기준
        + 2.0 * Z1                      # Z1 선형
        - 3.0 * Z1**2                   # Z1 제곱 (역U자)
        + 1.5 * np.exp(-Z2**2)          # Z2 가우시안 범프
        + 2.0 * Z1 * Z2                 # Z1×Z2 상호작용
        + 1.0 * np.sin(2 * np.pi * Z1)  # Z1 주기함수
        + 0.5 * X1                      # X1 선형
        - 0.3 * X2                      # X2 선형
        + 1.2 * Z1 * X1                 # Z1×X1 상호작용
    )
    
    true_ps = 1 / (1 + np.exp(-logit_ps))
    
    # === 처치 배정 ===
    treatment = np.random.binomial(1, true_ps)
    
    # === 결과 변수 ===
    # 핵심: 교란변수가 결과에도 비선형으로 영향
    # 같은 비선형 패턴이 결과에도 존재해야 교란이 발생
    
    Y0 = (
        50                              # 기준
        + 3.0 * Z1                      # Z1 선형
        - 4.0 * Z1**2                   # Z1 제곱 (PS와 같은 패턴!)
        + 2.0 * np.exp(-Z2**2)          # Z2 가우시안 (PS와 같은 패턴!)
        + 3.0 * Z1 * Z2                 # 상호작용
        + 0.5 * X1                      
        + 0.3 * X2
        + 0.2 * X3
        + np.random.normal(0, 2, n)     # 노이즈
    )
    
    # 처치효과: 상수 6.0
    true_te = 6.0 * np.ones(n)
    
    Y1 = Y0 + true_te
    outcome = treatment * Y1 + (1 - treatment) * Y0
    
    # === 참값 계산 ===
    true_att = true_te[treatment == 1].mean()  # = 6.0
    true_ate = true_te.mean()  # = 6.0
    
    # === 선택편향 계산 (교란 효과) ===
    # E[Y0|T=1] - E[Y0|T=0] = 선택편향
    selection_bias = Y0[treatment == 1].mean() - Y0[treatment == 0].mean()
    naive_diff = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    
    print(f"\n처치군 비율: {treatment.mean()*100:.1f}%")
    print(f"참 ATT: {true_att:.3f}")
    print(f"참 ATE: {true_ate:.3f}")
    print(f"\n선택편향 (교란 효과): {selection_bias:.3f}")
    print(f"단순 평균차이: {naive_diff:.3f}")
    print(f"단순 추정의 편향: {naive_diff - true_att:.3f} ({(naive_diff - true_att)/true_att*100:.1f}%)")
    
    # === 실제 사용할 공변량으로 재구성 ===
    # 정책분석 맥락에 맞는 변수명으로 변환
    age = 40 + 10 * Z1  # 30-50세 범위
    income = 50 + 20 * Z2  # 30-70 범위
    education = 14 + 2 * X1  # 12-16년
    experience = 15 + 5 * X2  # 10-20년
    assets = 200 + 50 * X3  # 150-250 범위
    
    # 데이터프레임 구성
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'experience': experience,
        'assets': assets,
        'Z1': Z1,  # 원래 변수도 저장 (검증용)
        'Z2': Z2,
        'treatment': treatment,
        'outcome': outcome,
        'true_ps': true_ps,
        'true_te': true_te,
        'Y0': Y0,
        'Y1': Y1
    })
    
    # === 로지스틱 회귀 vs 실제 PS 비교 ===
    from sklearn.linear_model import LogisticRegression
    
    X_mat = df[['age', 'income', 'education', 'experience', 'assets']].values
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_mat, treatment)
    lr_ps = lr.predict_proba(X_mat)[:, 1]
    
    # PS 추정 오차
    ps_mae_lr = np.abs(lr_ps - true_ps).mean()
    ps_corr_lr = np.corrcoef(lr_ps, true_ps)[0, 1]
    
    print(f"\n로지스틱 회귀 PS 오차:")
    print(f"  MAE: {ps_mae_lr:.4f}")
    print(f"  상관계수: {ps_corr_lr:.4f}")
    
    # 공변량 균형 확인 (매칭 전)
    print(f"\n매칭 전 공변량 불균형 (SMD):")
    for col in ['age', 'income', 'education', 'experience', 'assets']:
        t_mean = df.loc[treatment == 1, col].mean()
        c_mean = df.loc[treatment == 0, col].mean()
        pooled_std = np.sqrt((df.loc[treatment == 1, col].var() + df.loc[treatment == 0, col].var()) / 2)
        smd = (t_mean - c_mean) / pooled_std
        print(f"  {col}: {abs(smd):.3f}")
    
    # === 저장 ===
    data_dir = Path(__file__).parent / '../data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(data_dir / 'nonlinear_confounded_data.csv', index=False)
    print(f"\n데이터 저장: {data_dir / 'nonlinear_confounded_data.csv'}")
    
    # 메타 정보 저장
    with open(data_dir / 'nonlinear_confounded_meta.txt', 'w', encoding='utf-8') as f:
        f.write("3.5절 비선형 교란 데이터 (v2)\n")
        f.write("=" * 50 + "\n")
        f.write(f"샘플 수: {n}\n")
        f.write(f"처치군: {treatment.sum()} ({treatment.mean()*100:.1f}%)\n")
        f.write(f"대조군: {n - treatment.sum()} ({(1-treatment.mean())*100:.1f}%)\n")
        f.write(f"참 ATT: {true_att:.3f}\n")
        f.write(f"선택편향: {selection_bias:.3f}\n")
        f.write(f"로지스틱 회귀 PS 상관계수: {ps_corr_lr:.4f}\n")
    
    return df, true_att

if __name__ == "__main__":
    df, true_att = generate_nonlinear_confounded_data(n=3000)
    print("\n데이터 생성 완료!")
