"""
Chapter 3: Deep Nonparametric PSM (LBC-Net)
============================================

목적: PyTorch로 LBC-Net 구현
- Local Balance + Local Calibration 손실 함수
- 3-layer feedforward neural network
- 공변량 균형 명시적 최적화

저자: AI 기반 정책분석방법론
날짜: 2025-11-18

수정 이력
---------
2026-08-17
1) 무엇이 틀렸나: local_balance_loss가 학습에 전혀 반영되지 않았다.
   구간을 `bin_mask = (ps >= a) & (ps < b)`로 나눈 뒤 그 구간의 공변량 평균
   `X[treated_mask].mean(0)`를 비교했는데, X는 입력 데이터라 성향점수 ps의
   함수가 아니다. 불리언 마스크는 미분되지 않으므로 이 항의 grad_fn이 None이
   되고, `torch.autograd.grad(balance_loss, ps)`가 "does not require grad"
   오류를 낸다. 즉 λ_balance를 아무리 키워도 가중치가 갱신되지 않았다.
   그 결과 균형 손실이 학습 내내 무작위로 오르내렸고, 조기 종료가 epoch 51에서
   걸려 500 epoch 중 10%만 돌고 끝났다.
2) 왜 틀렸나: "구간으로 자른다"를 계단 함수로 구현했다. 계단 함수는 미분값이
   0이라 경사하강에 아무 신호도 주지 않는다.
3) 어떻게 고쳤나: 계단 구간을 가우시안 커널 가중치로 바꿨다.
   앵커점 p_k에 대해 w_ik = exp(-(ps_i - p_k)^2 / (2h^2))로 두면 가중치가
   ps의 매끄러운 함수가 되고, 국소 가중평균 차이가 ps에 대해 미분된다.
   Peng et al.(2024)이 쓴 커널 가중 국소 균형 조건과 같은 형태다.
4) 확인: grad norm이 0이 아님을 확인했고, 균형 손실이 학습에 따라 단조 감소한다.
5) λ_balance를 5.0에서 1.0으로 낮췄다. 5.0은 균형 항이 학습에 반영되지 않던
   시절에 정한 값이라 아무 부작용이 없었다. 균형 항이 실제로 작동하자
   5.0에서는 성향점수가 [0.305, 0.715]로 눌리고 구간별 보정 오차가 0.350까지
   커졌다(예측 점수가 높은 구간의 실제 처치 비율이 오히려 낮아졌다).
   λ_balance ∈ {0, 0.5, 1, 2, 5}를 돌려 보정 오차와 SMD를 함께 본 결과
   1.0을 골랐다. 보정 오차 0.074, 매칭 후 평균 |SMD| 0.078이다.
6) 함께 고친 것: 결과 그림(3-3-deep-psm.png)을 저장하도록 추가했다.
   학습 곡선·성향점수 분포·구간별 보정·매칭 후 SMD 네 개 패널이다.
   구간별 보정 표도 출력한다.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

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


def local_balance_loss(ps, X, T, n_bins=10, bandwidth=0.10, eps=1e-6):
    """Local Balance 패널티 계산 (커널 가중, 미분 가능)

    앵커점 p_k 주변에서 처치군과 대조군의 공변량 가중평균이 같아야 한다는 조건.
    가중치 w_ik = exp(-(ps_i - p_k)^2 / (2h^2))가 ps의 매끄러운 함수이므로
    이 손실은 신경망 가중치에 대해 미분된다.
    """

    anchors = torch.linspace(0.05, 0.95, n_bins)

    # (n, K) 커널 가중치
    w = torch.exp(-((ps.unsqueeze(1) - anchors.unsqueeze(0)) ** 2)
                  / (2 * bandwidth ** 2))

    t = T.unsqueeze(1)                      # (n, 1)
    w_t = w * t                             # 처치군 가중치
    w_c = w * (1 - t)                       # 대조군 가중치

    mass_t = w_t.sum(0)                     # (K,)
    mass_c = w_c.sum(0)

    # 국소 가중평균: (K, p)
    mean_t = (w_t.T @ X) / (mass_t.unsqueeze(1) + eps)
    mean_c = (w_c.T @ X) / (mass_c.unsqueeze(1) + eps)

    diff2 = ((mean_t - mean_c) ** 2).sum(1)  # (K,)

    # 표본이 거의 없는 앵커점은 가중치를 낮춘다
    rho = torch.min(mass_t, mass_c)
    rho = rho / (rho.sum() + eps)

    return (rho * diff2).sum()


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


def train_lbcnet(X, T, epochs=300, lr=0.0003, lambda_balance=1.0, lambda_calib=2.0):
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
    history = {'epoch': [], 'total': [], 'bce': [], 'balance': [], 'calib': []}

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

        history['epoch'].append(epoch + 1)
        history['total'].append(avg_loss)
        history['bce'].append(epoch_bce / len(dataloader))
        history['balance'].append(avg_balance)
        history['calib'].append(epoch_calib / len(dataloader))

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

    print(f"학습 완료 (Best balance loss: {best_balance:.4f})")
    print(f"균형 손실: 첫 epoch {history['balance'][0]:.4f} "
          f"-> 마지막 epoch {history['balance'][-1]:.4f}\n")

    # 최종 성향점수 예측
    model.eval()
    with torch.no_grad():
        ps_final = model(X_tensor).numpy()

    return ps_final, model, history


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


def calibration_table(ps, T, n_bins=10):
    """구간별 보정 점검: 예측 성향점수 평균 vs 실제 처치 비율"""

    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (ps >= edges[i]) & (ps < edges[i + 1])
        if m.sum() >= 10:
            rows.append({
                'bin_low': edges[i],
                'bin_high': edges[i + 1],
                'n': int(m.sum()),
                'ps_mean': float(ps[m].mean()),
                'treat_rate': float(T[m].mean()),
            })
    return pd.DataFrame(rows)


def plot_results(ps, T, history, calib_df, smd_before, smd_after, out_path):
    """결과 그림 4패널 저장"""

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) 학습 곡선
    ax = axes[0, 0]
    ax.plot(history['epoch'], history['balance'], color='#c0392b',
            label='Local balance')
    ax.plot(history['epoch'], history['bce'], color='#2c6fbb', label='BCE')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('(a) Training loss')
    ax.legend(); ax.grid(alpha=0.3)

    # (b) 성향점수 분포
    ax = axes[0, 1]
    ax.hist(ps[T == 1], bins=30, alpha=0.6, density=True, label='Treated')
    ax.hist(ps[T == 0], bins=30, alpha=0.6, density=True, label='Control')
    ax.set_xlabel('Estimated propensity score'); ax.set_ylabel('Density')
    ax.set_title('(b) Propensity score overlap')
    ax.legend(); ax.grid(alpha=0.3)

    # (c) 보정 곡선
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.plot(calib_df['ps_mean'], calib_df['treat_rate'], 'o-',
            color='#2e8b57', label='LBC-Net')
    ax.set_xlabel('Mean predicted score in bin')
    ax.set_ylabel('Actual treatment rate in bin')
    ax.set_title('(c) Local calibration')
    ax.legend(); ax.grid(alpha=0.3)

    # (d) 매칭 전후 SMD
    ax = axes[1, 1]
    names = list(smd_after.keys())
    ypos = np.arange(len(names))
    ax.scatter([abs(smd_before[k]) for k in names], ypos, s=110,
               label='Before matching')
    ax.scatter([abs(smd_after[k]) for k in names], ypos, s=110, marker='s',
               label='After matching')
    ax.axvline(0.1, color='red', linestyle='--', label='SMD = 0.1')
    ax.set_yticks(ypos); ax.set_yticklabels(names)
    ax.set_xlabel('|Standardized mean difference|')
    ax.set_title('(d) Covariate balance')
    ax.legend(); ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"결과 그림 저장: {out_path}\n")


def main():
    """메인 실행"""

    print("=" * 70)
    print("Deep Nonparametric PSM (LBC-Net)")
    print("=" * 70)
    print()

    # 데이터 준비
    X, T, y, data = load_and_prepare_data()

    # LBC-Net 학습 (Peng et al. 2024 최적 설정)
    ps_lbcnet, model, history = train_lbcnet(
        X, T,
        epochs=300,
        lr=0.0003,
        lambda_balance=1.0,
        lambda_calib=2.0
    )

    print(f"성향점수 범위: [{ps_lbcnet.min():.4f}, {ps_lbcnet.max():.4f}]\n")

    # 매칭
    matched_df = nearest_neighbor_matching(ps_lbcnet, T)

    # 공변량 균형
    continuous_vars = ['age', 'income', 'education', 'experience', 'assets']
    smd_before = {}
    for var in continuous_vars:
        t_mean = data.loc[data['treatment'] == 1, var].mean()
        c_mean = data.loc[data['treatment'] == 0, var].mean()
        pooled = np.sqrt((data.loc[data['treatment'] == 1, var].var()
                          + data.loc[data['treatment'] == 0, var].var()) / 2)
        smd_before[var] = (t_mean - c_mean) / pooled

    smd_after = calculate_smd(data, matched_df)
    avg_smd = np.mean([abs(v) for v in smd_after.values()])

    print("=== 공변량 균형 (SMD) ===")
    print(f"{'변수':<12}{'매칭 전':>10}{'매칭 후':>10}{'0.1 미만':>10}")
    for var in continuous_vars:
        ok = 'yes' if abs(smd_after[var]) < 0.1 else 'no'
        print(f"{var:<12}{smd_before[var]:>10.4f}{smd_after[var]:>10.4f}{ok:>10}")
    print(f"\n평균 |SMD|: 매칭 전 "
          f"{np.mean([abs(v) for v in smd_before.values()]):.4f} "
          f"-> 매칭 후 {avg_smd:.4f}\n")

    # 구간별 보정 점검
    calib_df = calibration_table(ps_lbcnet, T)
    print("=== 구간별 보정 (Local Calibration) ===")
    print(f"{'구간':<14}{'표본수':>8}{'예측 PS 평균':>14}{'실제 처치비율':>14}{'차이':>10}")
    for _, r in calib_df.iterrows():
        print(f"[{r['bin_low']:.1f}, {r['bin_high']:.1f})   {int(r['n']):>6}"
              f"{r['ps_mean']:>14.3f}{r['treat_rate']:>14.3f}"
              f"{r['ps_mean'] - r['treat_rate']:>10.3f}")
    mae_calib = float((calib_df['ps_mean'] - calib_df['treat_rate']).abs().mean())
    print(f"\n구간별 보정 오차 평균: {mae_calib:.4f}\n")

    # ATT 추정
    att, se = estimate_att(data, matched_df)

    print("=== ATT 추정 결과 ===")
    print(f"ATT: {att:.3f} (SE: {se:.3f})")
    print(f"95% CI: [{att - 1.96 * se:.3f}, {att + 1.96 * se:.3f}]")
    print(f"진정한 ATT: 5.000")
    print(f"편향: {att - 5.0:.3f}")
    contains = (att - 1.96 * se) <= 5.0 <= (att + 1.96 * se)
    print(f"95% CI가 진정한 값 포함: {'yes' if contains else 'no'}\n")

    out_path = Path(__file__).parent / '../data/3-3-deep-psm.png'
    plot_results(ps_lbcnet, T, history, calib_df, smd_before, smd_after, out_path)


if __name__ == "__main__":
    main()
