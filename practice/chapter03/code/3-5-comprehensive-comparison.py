"""
Chapter 3: PSM 방법론 종합 비교 (Comprehensive Comparison)
==========================================================

목적: 모든 PSM 방법론의 성능을 실제로 실행하고 비교
- Logistic Regression PSM
- Random Forest PSM
- Gradient Boosting PSM
- LBC-Net (Deep PSM)

저자: AI 기반 정책분석방법론
날짜: 2025-11-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

# 재현성
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================

def load_and_prepare_data():
    """데이터 로드 및 전처리 (비선형 교란 데이터 v2)"""
    data_path = Path(__file__).parent / '../data/nonlinear_confounded_data.csv'
    if not data_path.exists():
        print("데이터 파일이 없습니다. 3-5-data-gen-nonlinear-v2.py를 먼저 실행합니다.")
        raise FileNotFoundError(f"{data_path} not found.")

    data = pd.read_csv(data_path)

    # 원래 비선형 변수 사용 (더 정확한 테스트를 위해)
    feature_cols = ['Z1', 'Z2', 'age', 'income', 'education', 'experience', 'assets']

    X = data[feature_cols].values
    T = data['treatment'].values
    y = data['outcome'].values

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 참 ATT 출력
    true_att = data.loc[T == 1, 'true_te'].mean()
    print(f"참 ATT (True ATT): {true_att:.3f}")

    return X_scaled, T, y, data, feature_cols

# ==========================================
# 2. 모델 정의 (LBC-Net 포함)
# ==========================================

class LBCNet(nn.Module):
    """LBC-Net: 개선된 아키텍처 (Peng et al. 2024)"""
    def __init__(self, input_dim, hidden_dim=128):
        super(LBCNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()

def local_balance_loss(ps, X, T, n_bins=10):
    """Local Balance 패널티"""
    ps_bins = torch.linspace(0, 1, n_bins + 1)
    balance_loss = 0.0
    for i in range(n_bins):
        bin_mask = (ps >= ps_bins[i]) & (ps < ps_bins[i + 1])
        if bin_mask.sum() > 1:
            treated_mask = bin_mask & (T == 1)
            control_mask = bin_mask & (T == 0)
            if treated_mask.sum() > 0 and control_mask.sum() > 0:
                treated_mean = X[treated_mask].mean(0)
                control_mean = X[control_mask].mean(0)
                balance_loss += ((treated_mean - control_mean) ** 2).sum()
    return balance_loss / n_bins

def local_calibration_loss(ps, T, n_bins=10):
    """Local Calibration 패널티"""
    ps_bins = torch.linspace(0, 1, n_bins + 1)
    calibration_loss = 0.0
    for i in range(n_bins):
        bin_mask = (ps >= ps_bins[i]) & (ps < ps_bins[i + 1])
        if bin_mask.sum() > 1:
            ps_mean = ps[bin_mask].mean()
            actual_rate = T[bin_mask].mean()
            calibration_loss += (ps_mean - actual_rate) ** 2
    return calibration_loss / n_bins

def train_lbcnet(X, T, epochs=300, lr=0.0003, lambda_balance=5.0, lambda_calib=2.0):
    """LBC-Net 학습 (전체 균형 손실 적용)"""
    input_dim = X.shape[1]
    model = LBCNet(input_dim, hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    X_tensor = torch.FloatTensor(X)
    T_tensor = torch.FloatTensor(T)

    dataset = torch.utils.data.TensorDataset(X_tensor, T_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    best_balance = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_balance = 0.0

        for batch_X, batch_T in dataloader:
            optimizer.zero_grad()
            ps = model(batch_X)

            bce_loss = criterion(ps, batch_T)
            balance_loss_val = local_balance_loss(ps, batch_X, batch_T)
            calib_loss_val = local_calibration_loss(ps, batch_T)

            total_loss = bce_loss + lambda_balance * balance_loss_val + lambda_calib * calib_loss_val
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_balance += balance_loss_val.item()

        avg_balance = epoch_balance / len(dataloader)
        scheduler.step(avg_balance)

        if avg_balance < best_balance:
            best_balance = avg_balance
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 50:
            break

    model.eval()
    with torch.no_grad():
        ps_final = model(X_tensor).numpy()
    return ps_final

# ==========================================
# 3. 평가 함수
# ==========================================

def match_and_evaluate(ps, T, y, X, data, caliper=0.01):
    """매칭 및 평가"""
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    
    treated_ps = ps[treated_idx].reshape(-1, 1)
    control_ps = ps[control_idx].reshape(-1, 1)
    
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control_ps)
    distances, indices = nn.kneighbors(treated_ps)
    
    matched_pairs = []
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= caliper:
            matched_pairs.append((treated_idx[i], control_idx[idx]))
            
    if not matched_pairs:
        return 0, 0, 0, 0

    matched_treated = [p[0] for p in matched_pairs]
    matched_control = [p[1] for p in matched_pairs]
    
    # ATT
    att = (y[matched_treated] - y[matched_control]).mean()
    se = (y[matched_treated] - y[matched_control]).std() / np.sqrt(len(matched_pairs))
    
    # SMD
    smds = []
    for i in range(X.shape[1]):
        t_mean = X[matched_treated, i].mean()
        c_mean = X[matched_control, i].mean()
        pooled_std = np.sqrt((X[matched_treated, i].var() + X[matched_control, i].var()) / 2)
        if pooled_std == 0: pooled_std = 1e-6
        smds.append(abs(t_mean - c_mean) / pooled_std)
        
    avg_smd = np.mean(smds)
    max_smd = np.max(smds)
    
    return att, se, avg_smd, max_smd

# ==========================================
# 4. 메인 실행
# ==========================================

def main():
    print("=" * 80)
    print("PSM 방법론 종합 비교 (비선형 성향점수 데이터)")
    print("=" * 80)
    
    try:
        X, T, y, data, feature_cols = load_and_prepare_data()
        true_att = data.loc[T == 1, 'true_te'].mean()
    except Exception as e:
        print(e)
        return

    results = []
    
    # 1. Logistic Regression
    start = time.time()
    ps_lr = LogisticRegression(max_iter=1000).fit(X, T).predict_proba(X)[:, 1]
    att, se, avg_smd, max_smd = match_and_evaluate(ps_lr, T, y, X, data)
    dur = time.time() - start
    results.append(['Logistic', avg_smd, max_smd, att, se, dur])
    print(f"Logistic: ATT={att:.3f}, SMD={avg_smd:.3f}, Time={dur:.2f}s")

    # 2. Random Forest (비선형 관계 포착)
    start = time.time()
    ps_rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42).fit(X, T).predict_proba(X)[:, 1]
    att, se, avg_smd, max_smd = match_and_evaluate(ps_rf, T, y, X, data)
    dur = time.time() - start
    results.append(['Random Forest', avg_smd, max_smd, att, se, dur])
    print(f"Random Forest: ATT={att:.3f}, SMD={avg_smd:.3f}, Time={dur:.2f}s")

    # 3. Gradient Boosting (비선형 관계 포착)
    start = time.time()
    ps_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42).fit(X, T).predict_proba(X)[:, 1]
    att, se, avg_smd, max_smd = match_and_evaluate(ps_gb, T, y, X, data)
    dur = time.time() - start
    results.append(['Gradient Boosting', avg_smd, max_smd, att, se, dur])
    print(f"Gradient Boosting: ATT={att:.3f}, SMD={avg_smd:.3f}, Time={dur:.2f}s")

    # 4. LBC-Net
    start = time.time()
    ps_lbc = train_lbcnet(X, T)
    att, se, avg_smd, max_smd = match_and_evaluate(ps_lbc, T, y, X, data)
    dur = time.time() - start
    results.append(['LBC-Net', avg_smd, max_smd, att, se, dur])
    print(f"LBC-Net: ATT={att:.3f}, SMD={avg_smd:.3f}, Time={dur:.2f}s")

    # 결과 출력
    df = pd.DataFrame(results, columns=['Method', 'Avg SMD', 'Max SMD', 'ATT', 'SE', 'Time'])
    df['Bias (%)'] = ((df['ATT'] - true_att) / true_att * 100).round(1)
    print("\n" + "=" * 80)
    print("최종 비교 결과")
    print(f"참 ATT: {true_att:.3f}")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # 가장 좋은 방법 출력
    best_idx = df['Bias (%)'].abs().idxmin()
    print(f"\n최소 편향 방법: {df.loc[best_idx, 'Method']} (편향: {df.loc[best_idx, 'Bias (%)']:.1f}%)")

if __name__ == "__main__":
    main()
