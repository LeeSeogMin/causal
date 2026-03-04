"""
Chapter 3: 전통적 Propensity Score Matching (로지스틱 회귀)
==========================================================

목적: 로지스틱 회귀를 사용한 성향점수 추정 및 매칭
- 최근접 이웃 매칭 (1:1, 복원 추출)
- 공변량 균형 진단
- ATT 추정

저자: AI 기반 정책분석방법론
날짜: 2025-11-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False


def load_data():
    """시뮬레이션 데이터 로드"""

    data_path = Path(__file__).parent / '../data/synthetic_policy_data.csv'

    if not data_path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {data_path}\n"
            "먼저 3-data-generator.py를 실행하세요."
        )

    data = pd.read_csv(data_path)
    print(f"데이터 로드 완료: {len(data)}개 샘플")
    print(f"처치군: {data['treatment'].sum()}명, 대조군: {(1-data['treatment']).sum()}명\n")

    return data


def prepare_features(data):
    """특성 준비 및 전처리"""

    print("=== 특성 전처리 ===")

    # 연속형 변수
    continuous_vars = ['age', 'income', 'education', 'experience', 'assets']

    # 범주형 변수 원-핫 인코딩
    categorical_vars = ['gender', 'region', 'industry', 'education_level', 'occupation']

    # 원-핫 인코딩
    data_encoded = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

    # 특성 컬럼 추출
    feature_cols = continuous_vars + [col for col in data_encoded.columns
                                      if any(cat in col for cat in categorical_vars)]

    X = data_encoded[feature_cols].values
    T = data['treatment'].values
    y = data['outcome'].values

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"특성 개수: {X.shape[1]}")
    print(f"연속형: {len(continuous_vars)}, 범주형 (인코딩 후): {len(feature_cols) - len(continuous_vars)}\n")

    return X_scaled, T, y, feature_cols, data


def estimate_propensity_scores(X, T):
    """로지스틱 회귀로 성향점수 추정"""

    print("=== 성향점수 추정 (로지스틱 회귀) ===")

    # 로지스틱 회귀 모델
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, T)

    # 성향점수 예측
    propensity_scores = model.predict_proba(X)[:, 1]

    print(f"성향점수 범위: [{propensity_scores.min():.4f}, {propensity_scores.max():.4f}]")
    print(f"성향점수 평균: {propensity_scores.mean():.4f}\n")

    return propensity_scores, model


def nearest_neighbor_matching(propensity_scores, T, caliper=0.01):
    """최근접 이웃 매칭 (1:1, 복원 추출)"""

    print("=== 최근접 이웃 매칭 ===")
    print(f"매칭 방법: 1:1, 복원 추출")
    print(f"캘리퍼: {caliper}\n")

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    treated_ps = propensity_scores[treated_idx].reshape(-1, 1)
    control_ps = propensity_scores[control_idx].reshape(-1, 1)

    # Nearest Neighbors 모델
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control_ps)

    # 매칭 수행
    distances, indices = nn.kneighbors(treated_ps)

    # 캘리퍼 적용
    matched_pairs = []
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= caliper:
            matched_pairs.append({
                'treated_idx': treated_idx[i],
                'control_idx': control_idx[idx],
                'distance': dist
            })

    matched_df = pd.DataFrame(matched_pairs)

    print(f"매칭 성공: {len(matched_df)}쌍")
    print(f"매칭 실패: {len(treated_idx) - len(matched_df)}명 (처치군)")
    print(f"매칭률: {len(matched_df) / len(treated_idx):.1%}\n")

    return matched_df


def calculate_smd(data, feature_cols, matched_df):
    """표준화 평균 차이(SMD) 계산"""

    continuous_vars = ['age', 'income', 'education', 'experience', 'assets']

    smd_before = {}
    smd_after = {}

    for var in continuous_vars:
        # 매칭 전
        treated_mean = data[data['treatment'] == 1][var].mean()
        control_mean = data[data['treatment'] == 0][var].mean()
        pooled_std = data[var].std()
        smd_before[var] = (treated_mean - control_mean) / pooled_std

        # 매칭 후
        treated_matched = data.loc[matched_df['treated_idx'], var]
        control_matched = data.loc[matched_df['control_idx'].values, var]
        pooled_std_matched = np.sqrt((treated_matched.var() + control_matched.var()) / 2)
        smd_after[var] = (treated_matched.mean() - control_matched.mean()) / pooled_std_matched

    return smd_before, smd_after


def estimate_att(data, matched_df):
    """ATT 추정"""

    print("=== ATT 추정 ===")

    # 매칭된 샘플의 결과 변수
    treated_outcomes = data.loc[matched_df['treated_idx'], 'outcome'].values
    control_outcomes = data.loc[matched_df['control_idx'].values, 'outcome'].values

    # ATT 계산
    att = (treated_outcomes - control_outcomes).mean()
    se = (treated_outcomes - control_outcomes).std() / np.sqrt(len(matched_df))
    ci_lower = att - 1.96 * se
    ci_upper = att + 1.96 * se

    print(f"ATT 추정값: {att:.3f}")
    print(f"표준 오차: {se:.3f}")
    print(f"95% 신뢰구간: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"진정한 ATT: 5.000")
    print(f"편향: {att - 5.0:.3f}\n")

    return att, se, ci_lower, ci_upper


def plot_balance(smd_before, smd_after, output_dir='../data'):
    """공변량 균형 시각화 (Love Plot)"""

    fig, ax = plt.subplots(figsize=(10, 6))

    variables = list(smd_before.keys())
    y_pos = np.arange(len(variables))

    # 매칭 전/후 SMD
    before = [smd_before[var] for var in variables]
    after = [smd_after[var] for var in variables]

    # 플롯
    ax.scatter(before, y_pos, label='Matching Before', alpha=0.7, s=100)
    ax.scatter(after, y_pos, label='Matching After', alpha=0.7, s=100, marker='s')

    # 기준선
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='SMD = 0.1')
    ax.axvline(x=-0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Standardized Mean Difference (SMD)')
    ax.set_title('Covariate Balance: Before and After Matching (Logistic Regression PSM)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_path = Path(__file__).parent / output_dir / 'balance_logistic_psm.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"균형 플롯 저장: {output_path}\n")

    plt.close()


def plot_propensity_distribution(propensity_scores, T, output_dir='../data'):
    """성향점수 분포 시각화"""

    fig, ax = plt.subplots(figsize=(10, 6))

    treated_ps = propensity_scores[T == 1]
    control_ps = propensity_scores[T == 0]

    ax.hist(treated_ps, bins=30, alpha=0.5, label='Treated', density=True)
    ax.hist(control_ps, bins=30, alpha=0.5, label='Control', density=True)

    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.set_title('Propensity Score Distribution (Logistic Regression)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = Path(__file__).parent / output_dir / 'ps_dist_logistic.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"성향점수 분포 저장: {output_path}\n")

    plt.close()


def print_summary(smd_before, smd_after, att, se):
    """결과 요약 출력"""

    print("=" * 60)
    print("전통적 로지스틱 회귀 PSM 결과 요약")
    print("=" * 60)

    print("\n1. 공변량 균형 (SMD)")
    print("-" * 40)
    print(f"{'Variable':<15} {'Before':<10} {'After':<10} {'개선':<10}")
    print("-" * 40)

    for var in smd_before.keys():
        before = smd_before[var]
        after = smd_after[var]
        improvement = abs(before) - abs(after)
        print(f"{var:<15} {before:>9.3f} {after:>9.3f} {improvement:>9.3f}")

    avg_smd_before = np.mean([abs(v) for v in smd_before.values()])
    avg_smd_after = np.mean([abs(v) for v in smd_after.values()])

    print("-" * 40)
    print(f"{'평균 (절댓값)':<15} {avg_smd_before:>9.3f} {avg_smd_after:>9.3f}")
    print()

    print("2. 처치효과 추정")
    print("-" * 40)
    print(f"ATT 추정값: {att:.3f} (SE: {se:.3f})")
    print(f"진정한 ATT: 5.000")
    print(f"편향: {att - 5.0:.3f}")
    print()

    print("3. 방법론 특징")
    print("-" * 40)
    print("장점:")
    print("  - 구현 간단, 해석 용이")
    print("  - 계산 속도 빠름")
    print("\n단점:")
    print("  - 선형성 가정 (비선형 관계 포착 어려움)")
    print("  - 상호작용 항 수동 지정 필요")
    print("  - 모델 오설정 위험")
    print()


def main():
    """메인 실행 함수"""

    print("=" * 60)
    print("전통적 Propensity Score Matching (로지스틱 회귀)")
    print("=" * 60)
    print()

    # 1. 데이터 로드
    data = load_data()

    # 2. 특성 준비
    X, T, y, feature_cols, data_original = prepare_features(data)

    # 3. 성향점수 추정
    propensity_scores, model = estimate_propensity_scores(X, T)

    # 4. 성향점수 분포 시각화
    plot_propensity_distribution(propensity_scores, T)

    # 5. 매칭
    matched_df = nearest_neighbor_matching(propensity_scores, T, caliper=0.01)

    # 6. 공변량 균형 평가
    smd_before, smd_after = calculate_smd(data_original, feature_cols, matched_df)

    # 7. 균형 시각화
    plot_balance(smd_before, smd_after)

    # 8. ATT 추정
    att, se, ci_lower, ci_upper = estimate_att(data_original, matched_df)

    # 9. 결과 요약
    print_summary(smd_before, smd_after, att, se)

    print("=" * 60)
    print("분석 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
