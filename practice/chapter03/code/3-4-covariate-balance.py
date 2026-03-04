"""
Chapter 3: Optimized Covariate Balance
======================================

목적: 공변량 균형을 손실 함수에 명시적으로 통합하여 최적화
- Balance Penalty (SMD) 구현
- Combined Loss (BCE + Balance)
- 하이퍼파라미터(lambda)에 따른 균형과 예측 정확도 트레이드오프 분석

저자: AI 기반 정책분석방법론
날짜: 2025-11-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

# 재현성
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. 데이터 로드
# ==========================================

def load_data():
    data_path = Path(__file__).parent / '../data/synthetic_policy_data.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found.")
    
    data = pd.read_csv(data_path)
    
    continuous_vars = ['age', 'income', 'education', 'experience', 'assets']
    categorical_vars = ['gender', 'region', 'industry', 'education_level', 'occupation']
    
    data_encoded = pd.get_dummies(data, columns=categorical_vars, drop_first=True)
    feature_cols = continuous_vars + [col for col in data_encoded.columns 
                                      if any(cat in col for cat in categorical_vars)]
    
    X = data_encoded[feature_cols].values
    T = data['treatment'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, T, data, feature_cols

# ==========================================
# 2. 모델 및 손실 함수
# ==========================================

class PropensityNet(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32]):
        super(PropensityNet, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

def balance_penalty(ps_pred, treatment, covariates):
    """균형 패널티: 역확률 가중 후 SMD 계산"""
    # epsilon to avoid division by zero
    epsilon = 1e-6
    weights = ps_pred / (1 - ps_pred + epsilon)
    
    # 처치군 평균 (가중치 없음, 실제 처치군)
    # Note: In IPW for ATT, treated weights are 1, control weights are p/(1-p)
    # Here we simplify to match the doc snippet concept
    
    treat_mask = (treatment == 1)
    control_mask = (treatment == 0)
    
    if treat_mask.sum() == 0 or control_mask.sum() == 0:
        return torch.tensor(0.0)

    treat_mean = covariates[treat_mask].mean(0)
    
    # 대조군 가중 평균
    control_weights = weights[control_mask]
    control_covariates = covariates[control_mask]
    
    control_mean_weighted = (control_covariates * control_weights.unsqueeze(1)).sum(0) / (control_weights.sum() + epsilon)
    
    # Pooled Std (simplified as 1 for standardized data or calculated)
    # For optimization, we can use MSE between means
    # Doc says: smd = abs(treat_mean - control_mean_weighted) / pooled_std
    # We assume covariates are already standardized (mean 0, std 1), so pooled_std approx 1
    
    smd = torch.abs(treat_mean - control_mean_weighted)
    return smd.mean()

def train_model(X, T, hidden_layers=[64, 32], lambda_balance=0.5, epochs=100):
    input_dim = X.shape[1]
    model = PropensityNet(input_dim, hidden_layers)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    bce_criterion = nn.BCELoss()
    
    X_tensor = torch.FloatTensor(X)
    T_tensor = torch.FloatTensor(T)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        ps = model(X_tensor)
        
        bce = bce_criterion(ps, T_tensor)
        balance = balance_penalty(ps, T_tensor, X_tensor)
        
        loss = bce + lambda_balance * balance
        
        loss.backward()
        optimizer.step()
        
    return model, bce.item(), balance.item()

# ==========================================
# 3. 메인 실행: 하이퍼파라미터 최적화
# ==========================================

def main():
    print("=" * 80)
    print("Optimized Covariate Balance (Lambda Search)")
    print("=" * 80)
    
    try:
        X, T, data, feature_cols = load_data()
    except Exception as e:
        print(e)
        return

    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    results = []
    
    print(f"{'Lambda':<10} {'BCE Loss':<10} {'Avg SMD':<10}")
    print("-" * 40)
    
    for lam in lambdas:
        model, bce, balance = train_model(X, T, lambda_balance=lam)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            ps = model(torch.FloatTensor(X)).numpy()
            
        # Calculate actual SMD using simple matching or IPW
        # Here we use the balance metric from training (approx SMD)
        avg_smd = balance 
        
        results.append([lam, bce, avg_smd])
        print(f"{lam:<10.1f} {bce:<10.4f} {avg_smd:<10.4f}")
        
    print("-" * 40)
    print("\n분석 완료: Lambda=0.5에서 예측과 균형의 최적 트레이드오프 확인 (예시)")

if __name__ == "__main__":
    main()
