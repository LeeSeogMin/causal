"""
Chapter 3: Deep Nonparametric PSM (LBC-Net)
============================================

목적: PyTorch로 LBC-Net 구현
- Local Balance + Local Calibration 손실 함수
- 3-layer feedforward neural network
- 공변량 균형 명시적 최적화

저자: AI 기반 정책분석방법론
날짜: 2025-11-18
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 재현성
torch.manual_seed(42)
np.random.seed(42)


class PSMDataset(Dataset):
    """PSM 데이터셋"""

    def __init__(self, X, T):
        self.X = torch.FloatTensor(X)
        self.T = torch.FloatTensor(T)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]


class LBCNet(nn.Module):
    """LBC-Net: 3-layer feedforward network (Peng et al. 2024)"""

    def __init__(self, input_dim, hidden_dim=128):
        super(LBCNet, self).__init__()

        # Peng et al. (2024) 권장: 더 깊은 네트워크와 Batch Normalization
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
    """Local Balance 패널티 계산"""

    # 성향점수 구간 분할
    ps_bins = torch.linspace(0, 1, n_bins + 1)

    balance_loss = 0.0
    for i in range(n_bins):
        bin_mask = (ps >= ps_bins[i]) & (ps < ps_bins[i + 1])

        if bin_mask.sum() > 1:
            treated_mask = bin_mask & (T == 1)
            control_mask = bin_mask & (T == 0)

            if treated_mask.sum() > 0 and control_mask.sum() > 0:
                # 각 구간 내 공변량 평균 차이
                treated_mean = X[treated_mask].mean(0)
                control_mean = X[control_mask].mean(0)
                balance_loss += ((treated_mean - control_mean) ** 2).sum()

    return balance_loss / n_bins


def local_calibration_loss(ps, T, n_bins=10):
    """Local Calibration 패널티 계산"""

    ps_bins = torch.linspace(0, 1, n_bins + 1)

    calibration_loss = 0.0
    for i in range(n_bins):
        bin_mask = (ps >= ps_bins[i]) & (ps < ps_bins[i + 1])

        if bin_mask.sum() > 1:
            ps_mean = ps[bin_mask].mean()
            actual_rate = T[bin_mask].mean()
            calibration_loss += (ps_mean - actual_rate) ** 2

    return calibration_loss / n_bins


def train_lbcnet(X, T, epochs=500, lr=0.0003, lambda_balance=5.0, lambda_calib=2.0):
    """LBC-Net 학습 (Peng et al. 2024 설정)"""

    print("=== LBC-Net 학습 시작 ===")
    print(f"하이퍼파라미터: epochs={epochs}, lr={lr}, λ_balance={lambda_balance}, λ_calib={lambda_calib}\n")

    input_dim = X.shape[1]
    model = LBCNet(input_dim, hidden_dim=128)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    dataset = PSMDataset(X, T)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    X_tensor = torch.FloatTensor(X)
    T_tensor = torch.FloatTensor(T)

    # 학습률 스케줄러 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    best_balance = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_balance = 0.0
        epoch_calib = 0.0

        for batch_X, batch_T in dataloader:
            optimizer.zero_grad()

            # 성향점수 예측
            ps = model(batch_X)

            # 손실 함수 계산
            bce_loss = criterion(ps, batch_T)
            balance_loss_val = local_balance_loss(ps, batch_X, batch_T)
            calib_loss_val = local_calibration_loss(ps, batch_T)

            # 총 손실 (균형 손실에 더 높은 가중치)
            total_loss = bce_loss + lambda_balance * balance_loss_val + lambda_calib * calib_loss_val

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_bce += bce_loss.item()
            epoch_balance += balance_loss_val.item()
            epoch_calib += calib_loss_val.item()

        avg_loss = epoch_loss / len(dataloader)
        avg_balance = epoch_balance / len(dataloader)

        # 학습률 조정
        scheduler.step(avg_balance)

        # Early stopping (균형 기준)
        if avg_balance < best_balance:
            best_balance = avg_balance
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  BCE: {epoch_bce / len(dataloader):.4f}, "
                  f"Balance: {avg_balance:.4f}, "
                  f"Calib: {epoch_calib / len(dataloader):.4f}")

        # Early stopping 실행 (더 긴 patience)
        if patience_counter >= 50:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"학습 완료 (Best balance loss: {best_balance:.4f})\n")

    # 최종 성향점수 예측
    model.eval()
    with torch.no_grad():
        ps_final = model(X_tensor).numpy()

    return ps_final, model


def load_and_prepare_data():
    """데이터 로드 및 전처리"""

    data_path = Path(__file__).parent / '../data/synthetic_policy_data.csv'
    data = pd.read_csv(data_path)

    continuous_vars = ['age', 'income', 'education', 'experience', 'assets']
    categorical_vars = ['gender', 'region', 'industry', 'education_level', 'occupation']

    data_encoded = pd.get_dummies(data, columns=categorical_vars, drop_first=True)
    feature_cols = continuous_vars + [col for col in data_encoded.columns
                                      if any(cat in col for cat in categorical_vars)]

    X = data_encoded[feature_cols].values
    T = data['treatment'].values
    y = data['outcome'].values

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"데이터 로드 완료: {len(data)}개 샘플, {X.shape[1]}개 특성\n")

    return X_scaled, T, y, data


def nearest_neighbor_matching(propensity_scores, T, caliper=0.01):
    """매칭 수행"""

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    treated_ps = propensity_scores[treated_idx].reshape(-1, 1)
    control_ps = propensity_scores[control_idx].reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control_ps)

    distances, indices = nn.kneighbors(treated_ps)

    matched_pairs = []
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= caliper:
            matched_pairs.append({
                'treated_idx': treated_idx[i],
                'control_idx': control_idx[idx]
            })

    matched_df = pd.DataFrame(matched_pairs)
    print(f"매칭 성공: {len(matched_df)}쌍 (매칭률: {len(matched_df)/len(treated_idx):.1%})\n")

    return matched_df


def calculate_smd(data, matched_df):
    """SMD 계산"""

    continuous_vars = ['age', 'income', 'education', 'experience', 'assets']
    smd_after = {}

    for var in continuous_vars:
        treated_matched = data.loc[matched_df['treated_idx'], var]
        control_matched = data.loc[matched_df['control_idx'].values, var]
        pooled_std = np.sqrt((treated_matched.var() + control_matched.var()) / 2)
        smd_after[var] = (treated_matched.mean() - control_matched.mean()) / pooled_std

    return smd_after


def estimate_att(data, matched_df):
    """ATT 추정"""

    treated_outcomes = data.loc[matched_df['treated_idx'], 'outcome'].values
    control_outcomes = data.loc[matched_df['control_idx'].values, 'outcome'].values

    att = (treated_outcomes - control_outcomes).mean()
    se = (treated_outcomes - control_outcomes).std() / np.sqrt(len(matched_df))

    return att, se


def main():
    """메인 실행"""

    print("=" * 70)
    print("Deep Nonparametric PSM (LBC-Net)")
    print("=" * 70)
    print()

    # 데이터 준비
    X, T, y, data = load_and_prepare_data()

    # LBC-Net 학습 (Peng et al. 2024 최적 설정)
    ps_lbcnet, model = train_lbcnet(
        X, T,
        epochs=500,
        lr=0.0003,
        lambda_balance=5.0,
        lambda_calib=2.0
    )

    print(f"성향점수 범위: [{ps_lbcnet.min():.4f}, {ps_lbcnet.max():.4f}]\n")

    # 매칭
    matched_df = nearest_neighbor_matching(ps_lbcnet, T)

    # 공변량 균형
    smd_after = calculate_smd(data, matched_df)
    avg_smd = np.mean([abs(v) for v in smd_after.values()])

    print("=== 공변량 균형 (SMD) ===")
    for var, smd in smd_after.items():
        print(f"{var}: {smd:.4f}")
    print(f"\n평균 SMD: {avg_smd:.4f}\n")

    # ATT 추정
    att, se = estimate_att(data, matched_df)

    print("=== ATT 추정 결과 ===")
    print(f"ATT: {att:.3f} (SE: {se:.3f})")
    print(f"95% CI: [{att - 1.96 * se:.3f}, {att + 1.96 * se:.3f}]")
    print(f"진정한 ATT: 5.000")
    print(f"편향: {att - 5.0:.3f}\n")

    print("=" * 70)
    print("LBC-Net의 장점:")
    print("  - 공변량 균형 명시적 최적화")
    print("  - 비선형 관계 자동 학습")
    print("  - Local Balance & Calibration 동시 달성")
    print("=" * 70)


if __name__ == "__main__":
    main()
