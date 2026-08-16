"""
제2장: 전통적 Propensity Score Matching (PSM) 분석
로지스틱 회귀 기반 PSM의 한계와 공변량 균형 개선 분석

수정 이력
---------
2026-08-17
1) 시각화 단계에서 KeyError('true_propensity')로 스크립트가 중단됐다.
   원인: main()을 CSV 로드 방식으로 바꾸면서 data 딕셔너리에
   'true_propensity'와 'true_ATE'를 담지 않았는데,
   visualize_psm_results()는 두 키를 읽는다.
   조치: CSV에 이미 들어 있는 두 열을 data 딕셔너리에 넣었다.
2) 결과 해석 문구가 하드코딩되어 있어 실제 SMD와 어긋났다.
   조치: 매칭 후 SMD가 0.1을 넘는 공변량 개수를 계산해서 출력한다.
3) plt.show()가 배치 실행에서 불필요한 창을 띄우므로 plt.close()로 바꿨다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정 (한글 폰트 깨짐 방지)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def generate_psm_data(n=1000, seed=42):
    """
    PSM 분석을 위한 데이터 생성

    Parameters:
    -----------
    n : int
        샘플 크기
    seed : int
        난수 시드

    Returns:
    --------
    dict : 생성된 데이터
    """
    np.random.seed(seed)

    # 공변량 생성
    X = np.random.randn(n, 5)

    # 진짜 성향점수 (비선형, 상호작용 포함)
    true_propensity = expit(0.5 * X[:, 0] - 0.3 * X[:, 1])
    treatment = np.random.binomial(1, true_propensity)

    # 잠재결과
    Y0_true = 2 + X[:, 0] - 0.5 * X[:, 1] + np.random.randn(n) * 0.5
    Y1_true = Y0_true + 3 + 0.8 * X[:, 2]  # 이질적 처치효과

    # 관측 결과
    Y_observed = treatment * Y1_true + (1 - treatment) * Y0_true

    # 참값
    true_ATE = np.mean(Y1_true - Y0_true)
    true_ATT = np.mean((Y1_true - Y0_true)[treatment == 1])

    return {
        'X': X,
        'treatment': treatment,
        'Y_observed': Y_observed,
        'true_propensity': true_propensity,
        'Y0_true': Y0_true,
        'Y1_true': Y1_true,
        'true_ATE': true_ATE,
        'true_ATT': true_ATT
    }

def compute_smd(X, treatment, weights=None):
    """
    표준화 평균 차이(Standardized Mean Difference) 계산

    Parameters:
    -----------
    X : array-like
        공변량 행렬
    treatment : array-like
        처치 지시자
    weights : array-like, optional
        가중치

    Returns:
    --------
    array : 각 공변량의 SMD
    """
    if weights is None:
        weights = np.ones(len(X))

    mean_t = np.average(X[treatment==1], weights=weights[treatment==1], axis=0)
    mean_c = np.average(X[treatment==0], weights=weights[treatment==0], axis=0)
    std_pooled = np.sqrt((X[treatment==1].var(axis=0) + X[treatment==0].var(axis=0)) / 2)

    return np.abs(mean_t - mean_c) / std_pooled

def perform_psm(X, treatment, Y_observed, caliper=0.25):
    """
    Propensity Score Matching 수행

    Parameters:
    -----------
    X : array-like
        공변량 행렬
    treatment : array-like
        처치 지시자
    Y_observed : array-like
        관측된 결과
    caliper : float
        매칭 허용 거리 (성향점수 표준편차 단위)

    Returns:
    --------
    dict : PSM 결과
    """
    # 성향점수 추정 (로지스틱 회귀)
    ps_model = LogisticRegression(max_iter=500, random_state=42)
    ps_model.fit(X, treatment)
    propensity_scores = ps_model.predict_proba(X)[:, 1]

    # Caliper 계산
    ps_std = propensity_scores.std()
    caliper_distance = caliper * ps_std

    # 1:1 최근접 이웃 매칭
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(propensity_scores[treatment==0].reshape(-1, 1))

    treated_indices = np.where(treatment == 1)[0]
    matched_pairs = []
    distances_list = []

    for t_idx in treated_indices:
        ps_t = propensity_scores[t_idx]
        distances, indices = nn.kneighbors([[ps_t]])

        if distances[0][0] <= caliper_distance:
            c_idx = np.where(treatment == 0)[0][indices[0][0]]
            matched_pairs.append((t_idx, c_idx))
            distances_list.append(distances[0][0])

    # ATT 추정
    if len(matched_pairs) > 0:
        treated_outcomes = np.array([Y_observed[t] for t, c in matched_pairs])
        control_outcomes = np.array([Y_observed[c] for t, c in matched_pairs])
        att_psm = np.mean(treated_outcomes - control_outcomes)
    else:
        att_psm = np.nan

    # 매칭된 인덱스 추출
    matched_treated = np.array([t for t, c in matched_pairs])
    matched_control = np.array([c for t, c in matched_pairs])
    matched_indices = np.concatenate([matched_treated, matched_control])

    return {
        'propensity_scores': propensity_scores,
        'matched_pairs': matched_pairs,
        'distances': distances_list,
        'att_psm': att_psm,
        'matched_indices': matched_indices,
        'matched_treated': matched_treated,
        'matched_control': matched_control
    }

def visualize_psm_results(data, psm_results):
    """
    PSM 분석 결과 시각화

    Parameters:
    -----------
    data : dict
        generate_psm_data 함수의 반환값
    psm_results : dict
        perform_psm 함수의 반환값
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    X = data['X']
    treatment = data['treatment']
    Y_observed = data['Y_observed']
    ps_estimated = psm_results['propensity_scores']
    ps_true = data['true_propensity']

    # 1. 추정 성향점수 vs 진짜 성향점수
    axes[0, 0].scatter(ps_true, ps_estimated, alpha=0.5, s=10)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect')
    axes[0, 0].set_xlabel('True Propensity Score')
    axes[0, 0].set_ylabel('Estimated Propensity Score')
    axes[0, 0].set_title('Propensity Score Estimation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 매칭 전 성향점수 분포
    axes[0, 1].hist(ps_estimated[treatment == 1], bins=25, alpha=0.6,
                    label='Treated', color='red', density=True)
    axes[0, 1].hist(ps_estimated[treatment == 0], bins=25, alpha=0.6,
                    label='Control', color='blue', density=True)
    axes[0, 1].set_xlabel('Propensity Score')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Before Matching')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 매칭 후 성향점수 분포
    matched_treated = psm_results['matched_treated']
    matched_control = psm_results['matched_control']

    axes[0, 2].hist(ps_estimated[matched_treated], bins=25, alpha=0.6,
                    label='Treated', color='red', density=True)
    axes[0, 2].hist(ps_estimated[matched_control], bins=25, alpha=0.6,
                    label='Control', color='blue', density=True)
    axes[0, 2].set_xlabel('Propensity Score')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('After Matching')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 공변량 균형 (SMD)
    smd_before = compute_smd(X, treatment)
    matched_treatment = np.concatenate([np.ones(len(matched_treated)),
                                        np.zeros(len(matched_control))])
    smd_after = compute_smd(X[psm_results['matched_indices']],
                            matched_treatment.astype(int))

    x_pos = np.arange(5)
    width = 0.35

    axes[1, 0].bar(x_pos - width/2, smd_before, width, label='Before', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, smd_after, width, label='After', alpha=0.7)
    axes[1, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.5,
                       label='Threshold (0.1)')
    axes[1, 0].set_xlabel('Covariate')
    axes[1, 0].set_ylabel('Standardized Mean Difference')
    axes[1, 0].set_title('Covariate Balance Improvement')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f'X{i+1}' for i in range(5)])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 5. 매칭 거리 분포
    axes[1, 1].hist(psm_results['distances'], bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=np.median(psm_results['distances']), color='red',
                       linestyle='--', linewidth=2, label=f"Median: {np.median(psm_results['distances']):.3f}")
    axes[1, 1].set_xlabel('Matching Distance')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Matching Quality')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 6. 추정값 비교
    naive_estimate = Y_observed[treatment==1].mean() - Y_observed[treatment==0].mean()
    estimates = ['True ATT', 'PSM', 'Naive']
    values = [data['true_ATT'], psm_results['att_psm'], naive_estimate]
    colors = ['green', 'blue', 'red']

    bars = axes[1, 2].bar(estimates, values, color=colors, alpha=0.7)
    axes[1, 2].axhline(y=data['true_ATT'], color='green', linestyle='--',
                       alpha=0.5, label='True ATT')
    axes[1, 2].set_ylabel('Treatment Effect')
    axes[1, 2].set_title('Estimation Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    # 값 표시
    for bar, val in zip(bars, values):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Traditional Propensity Score Matching Analysis',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'traditional_psm_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   그래프 저장: {out_path}")

def main():
    """메인 실행 함수"""

    print("=" * 70)
    print("제2장: 전통적 Propensity Score Matching (PSM) 분석")
    print("=" * 70)

    # 데이터 로드
    print("\n1. 데이터 로드 중...")
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../data/2-2-psm-dml.csv')
    df = pd.read_csv(data_path)

    X = df[['X1', 'X2', 'X3', 'X4', 'X5']].values
    treatment = df['treatment'].values
    Y_observed = df['Y'].values
    Y0_true = df['Y0_true'].values
    Y1_true = df['Y1_true'].values

    true_ATT = np.mean((Y1_true - Y0_true)[treatment == 1])
    true_ATE = np.mean(Y1_true - Y0_true)

    data = {
        'X': X,
        'treatment': treatment,
        'Y_observed': Y_observed,
        'true_ATT': true_ATT,
        'true_ATE': true_ATE,
        'true_propensity': df['true_propensity'].values,
        'Y0_true': Y0_true,
        'Y1_true': Y1_true
    }

    # Naive 추정
    naive_estimate = Y_observed[treatment==1].mean() - Y_observed[treatment==0].mean()

    print(f"   - 생성된 샘플 수: {len(df)}")
    print(f"   - 처치군: {treatment.sum()}명")
    print(f"   - 대조군: {(1 - treatment).sum()}명")

    # PSM 수행
    print("\n2. Propensity Score Matching 수행 중...")
    psm_results = perform_psm(data['X'], data['treatment'], data['Y_observed'])

    # 결과 출력
    print("\n" + "=" * 70)
    print("전통적 PSM 추정 결과")
    print("=" * 70)

    print("\n[추정값 비교]")
    print(f"PSM 추정 ATT:    {psm_results['att_psm']:.2f}")
    print(f"참값 ATT:        {data['true_ATT']:.2f}")
    print(f"추정 오차:       {psm_results['att_psm'] - data['true_ATT']:.2f}")
    print(f"오차 비율:       {abs(psm_results['att_psm'] - data['true_ATT']) / abs(data['true_ATT']) * 100:.1f}%")
    print(f"\nNaive 추정 오차: {abs(naive_estimate - data['true_ATT']) / abs(data['true_ATT']) * 100:.1f}%")
    print(f"편향 감소율:     {(abs(naive_estimate - data['true_ATT']) - abs(psm_results['att_psm'] - data['true_ATT'])) / abs(naive_estimate - data['true_ATT']) * 100:.1f}%")

    # 공변량 균형
    print("\n[공변량 균형 개선]")
    smd_before = compute_smd(data['X'], data['treatment'])
    matched_treatment = np.concatenate([np.ones(len(psm_results['matched_treated'])),
                                        np.zeros(len(psm_results['matched_control']))])
    smd_after = compute_smd(data['X'][psm_results['matched_indices']],
                            matched_treatment.astype(int))

    print(f"\n공변량 | 매칭 전 SMD | 매칭 후 SMD | 개선율")
    print("-" * 55)
    for i in range(5):
        improvement = (smd_before[i] - smd_after[i]) / smd_before[i] * 100
        print(f"X{i+1}     | {smd_before[i]:11.2f} | {smd_after[i]:11.2f} | {improvement:6.1f}%")

    avg_smd_before = np.mean(smd_before)
    avg_smd_after = np.mean(smd_after)
    print(f"평균   | {avg_smd_before:11.2f} | {avg_smd_after:11.2f} | {(avg_smd_before - avg_smd_after) / avg_smd_before * 100:6.1f}%")

    # 매칭 품질
    print("\n[매칭 품질 지표]")
    print(f"매칭된 처치군 개체 수:    {len(psm_results['matched_treated'])}개")
    print(f"매칭 거리 (중위수):       {np.median(psm_results['distances']):.3f}")
    print(f"매칭 거리 (90th %ile):    {np.percentile(psm_results['distances'], 90):.3f}")

    # 공통지지 밖 개체
    outside_support = data['treatment'].sum() - len(psm_results['matched_treated'])
    print(f"공통지지 밖 제외:         {outside_support}개 ({outside_support / data['treatment'].sum() * 100:.1f}%)")

    # 시각화
    print("\n3. 시각화 생성 중...")
    visualize_psm_results(data, psm_results)

    # 결과 해석
    print("\n" + "=" * 70)
    print("분석 결과 해석")
    print("=" * 70)

    print("\n1. 부분적 편향 감소:")
    print(f"   - PSM은 선택편향을 {abs(naive_estimate - data['true_ATT']) / abs(data['true_ATT']) * 100:.1f}%에서")
    print(f"     {abs(psm_results['att_psm'] - data['true_ATT']) / abs(data['true_ATT']) * 100:.1f}%로 감소시켜 상당한 개선을 보임")
    print(f"   - 여전히 {abs(psm_results['att_psm'] - data['true_ATT']) / abs(data['true_ATT']) * 100:.1f}%의 잔여 편향 존재")
    print(f"   - 로지스틱 회귀의 모형 오설정이 주요 원인")

    n_over_after = int((smd_after > 0.1).sum())
    over_names = ', '.join(f'X{i+1}' for i in range(5) if smd_after[i] > 0.1)
    print("\n2. 불완전한 균형 달성:")
    print(f"   - 평균 SMD가 {avg_smd_before:.2f}에서 {avg_smd_after:.2f}로 개선됨")
    print(f"   - 매칭 후 SMD > 0.1인 공변량: {n_over_after}개"
          + (f" ({over_names})" if n_over_after else " (없음)"))
    print(f"   - 1차원 성향점수로 다차원 공변량 균형의 한계")

    print("\n3. 표본 손실과 외적 타당도:")
    print(f"   - 공통지지 영역 밖의 {outside_support}개 개체 제외")
    print(f"   - 추정 대상이 원래 모집단에서 변화")
    print(f"   - 내적 타당도 향상 vs 외적 타당도 제한 트레이드오프")

    print("\n※ 본 코드는 교육 목적의 예제입니다.")
    print("   실제 분석 시 다양한 매칭 알고리즘과 민감도 분석이 필요합니다.")

    print("\n분석 완료! 결과가 traditional_psm_analysis.png로 저장되었습니다.")

    return data, psm_results

if __name__ == "__main__":
    data, psm_results = main()
