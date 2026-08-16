"""
Chapter 3: Optimized Covariate Balance
======================================

목적: 공변량 균형을 손실 함수에 명시적으로 통합하여 최적화
- Balance Penalty (SMD) 구현
- Combined Loss (BCE + Balance)
- 하이퍼파라미터(lambda)에 따른 균형과 예측 정확도 트레이드오프 분석

저자: AI 기반 정책분석방법론
날짜: 2025-11-18

수정 이력
---------
2026-08-17
1) 무엇이 틀렸나 (A): 마지막 줄이 "Lambda=0.5에서 최적 트레이드오프 확인 (예시)"로
   고정 출력되어 있었다. 표를 보면 평균 SMD가 가장 낮은 지점은 λ=1.0이었고
   λ=0.5는 여섯 값 중 네 번째였다. 결과와 무관한 문장을 찍고 있었다.
   고침: 실행 결과에서 SMD 최소인 λ를 계산해 출력한다.
2) 무엇이 틀렸나 (B): `avg_smd = balance`가 학습 루프 마지막 배치의 값이었다.
   그 시점의 모델은 `model.train()` 상태여서 Dropout이 켜져 있고 BatchNorm이
   배치 통계를 쓴다. 보고한 SMD가 실제 추정 성향점수의 SMD가 아니었다.
   고침: 학습이 끝난 뒤 `model.eval()`로 전체 표본에 대해 성향점수를 다시 뽑고,
   그 값으로 IPW 가중 SMD를 계산한다. BCE도 같은 방식으로 다시 잰다.
3) 무엇이 틀렸나 (C): λ를 골라도 그 λ가 ATT 추정을 얼마나 바꾸는지 보여 주지
   않았다. 균형만 보고 λ를 고르면 판단 근거가 반쪽이다.
   고침: 각 λ에서 매칭을 수행해 ATT와 매칭 후 실제 SMD를 함께 출력한다.
4) 함께 고친 것: 결과 그림(3-4-lambda-tradeoff.png)을 저장한다.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

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

    # 학습이 끝난 모델을 평가 모드로 두고 전체 표본에서 다시 잰다.
    # (학습 루프 안의 값은 Dropout이 켜진 상태의 값이라 보고에 쓸 수 없다)
    model.eval()
    with torch.no_grad():
        ps_eval = model(X_tensor)
        bce_eval = bce_criterion(ps_eval, T_tensor).item()
        balance_eval = balance_penalty(ps_eval, T_tensor, X_tensor).item()

    return model, bce_eval, balance_eval, ps_eval.numpy()


def match_and_estimate(ps, T, y, X, caliper=0.01):
    """성향점수 최근접 이웃 매칭 후 ATT와 매칭 후 SMD 계산"""

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(ps[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(ps[treated_idx].reshape(-1, 1))

    keep = distances.flatten() <= caliper
    if keep.sum() == 0:
        return np.nan, np.nan, 0

    mt = treated_idx[keep]
    mc = control_idx[indices.flatten()[keep]]

    att = float((y[mt] - y[mc]).mean())

    smds = []
    for j in range(5):          # 연속형 공변량 5개
        pooled = np.sqrt((X[mt, j].var() + X[mc, j].var()) / 2)
        if pooled > 0:
            smds.append(abs(X[mt, j].mean() - X[mc, j].mean()) / pooled)

    return att, float(np.mean(smds)), int(keep.sum())

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

    y = data['outcome'].values
    true_att = 5.0

    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    results = []

    print(f"진정한 ATT: {true_att:.3f}\n")
    print(f"{'Lambda':<8}{'BCE':>9}{'가중 SMD':>11}{'매칭후 SMD':>12}"
          f"{'ATT':>9}{'편향':>9}{'매칭쌍':>8}")
    print("-" * 70)

    for lam in lambdas:
        model, bce, weighted_smd, ps = train_model(X, T, lambda_balance=lam)
        att, matched_smd, n_pairs = match_and_estimate(ps, T, y, X)

        results.append({
            'lambda': lam, 'bce': bce, 'weighted_smd': weighted_smd,
            'matched_smd': matched_smd, 'att': att,
            'bias': att - true_att, 'n_pairs': n_pairs,
        })
        print(f"{lam:<8.1f}{bce:>9.4f}{weighted_smd:>11.4f}{matched_smd:>12.4f}"
              f"{att:>9.3f}{att - true_att:>9.3f}{n_pairs:>8}")

    print("-" * 70)

    df = pd.DataFrame(results)

    best_balance_row = df.loc[df['weighted_smd'].idxmin()]
    best_bias_row = df.loc[df['bias'].abs().idxmin()]

    print(f"\n가중 SMD 최소: λ = {best_balance_row['lambda']:.1f} "
          f"(SMD {best_balance_row['weighted_smd']:.4f}, "
          f"ATT {best_balance_row['att']:.3f})")
    print(f"ATT 편향 최소: λ = {best_bias_row['lambda']:.1f} "
          f"(편향 {best_bias_row['bias']:+.3f}, "
          f"SMD {best_bias_row['weighted_smd']:.4f})")

    smd0 = df.loc[df['lambda'] == 0.0, 'weighted_smd'].values[0]
    bce0 = df.loc[df['lambda'] == 0.0, 'bce'].values[0]
    print(f"\nλ=0 대비 (균형 항을 빼고 예측만 최적화한 경우)")
    print(f"  가중 SMD: {smd0:.4f} -> {best_balance_row['weighted_smd']:.4f}")
    print(f"  BCE:      {bce0:.4f} -> {best_balance_row['bce']:.4f}")

    plot_tradeoff(df, Path(__file__).parent / '../data/3-4-lambda-tradeoff.png')


def plot_tradeoff(df, out_path):
    """λ - 예측손실 - 균형 - 편향 관계 그림"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(df['lambda'], df['bce'], 'o-', color='#2c6fbb')
    ax.set_xlabel('lambda (balance weight)'); ax.set_ylabel('BCE loss')
    ax.set_title('(a) Prediction loss vs lambda'); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df['lambda'], df['weighted_smd'], 'o-', color='#c0392b',
            label='IPW-weighted SMD')
    ax.plot(df['lambda'], df['matched_smd'], 's--', color='#2e8b57',
            label='SMD after matching')
    ax.axhline(0.1, color='gray', linestyle=':', label='SMD = 0.1')
    ax.set_xlabel('lambda (balance weight)'); ax.set_ylabel('Average |SMD|')
    ax.set_title('(b) Covariate balance vs lambda')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(df['lambda'], df['bias'], 'o-', color='#d6a300')
    ax.axhline(0, color='red', linestyle='--', label='True ATT = 5.0')
    ax.set_xlabel('lambda (balance weight)'); ax.set_ylabel('ATT bias')
    ax.set_title('(c) ATT bias vs lambda')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n결과 그림 저장: {out_path}")


if __name__ == "__main__":
    main()
