"""
Chapter 3.1: 전통적 PSM의 한계 - 모델 오설정과 공통지지 문제
============================================================

목적: 로지스틱 회귀 PSM이 비선형 성향점수 모형에서 실패하는 것을 보여줌
- 모델 오설정으로 인한 공변량 불균형
- 공통지지 영역 위반으로 인한 표본 손실
- 편향된 ATT 추정

저자: AI 기반 정책분석방법론
날짜: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """데이터 로드"""
    data_path = Path(__file__).parent / '../data/misspecified_data.csv'
    meta_path = Path(__file__).parent / '../data/misspecified_meta.txt'
    
    data = pd.read_csv(data_path)
    
    # 메타 정보 로드
    with open(meta_path, 'r') as f:
        meta = {}
        for line in f:
            key, val = line.strip().split('=')
            meta[key] = float(val)
    
    true_att = meta['true_att']
    
    print("=" * 70)
    print("제3장 3.1절: 전통적 로지스틱 회귀 PSM의 한계")
    print("=" * 70)
    print(f"\n데이터 로드: {len(data)}개 샘플")
    print(f"처치군: {data['treatment'].sum()}명 ({100*data['treatment'].mean():.1f}%)")
    print(f"대조군: {(1-data['treatment']).sum()}명 ({100*(1-data['treatment'].mean()):.1f}%)")
    print(f"진정한 ATT: {true_att:.3f}")
    
    return data, true_att


def compute_smd(X, treatment, weights=None):
    """표준화 평균 차이 계산"""
    if weights is None:
        weights = np.ones(len(X))
    
    treated_mask = treatment == 1
    control_mask = treatment == 0
    
    t_mean = np.average(X[treated_mask], weights=weights[treated_mask], axis=0)
    c_mean = np.average(X[control_mask], weights=weights[control_mask], axis=0)
    
    t_var = np.average((X[treated_mask] - t_mean)**2, weights=weights[treated_mask], axis=0)
    c_var = np.average((X[control_mask] - c_mean)**2, weights=weights[control_mask], axis=0)
    
    pooled_std = np.sqrt((t_var + c_var) / 2)
    smd = (t_mean - c_mean) / np.where(pooled_std > 0, pooled_std, 1)
    
    return smd


def analyze_before_matching(data, feature_cols):
    """매칭 전 공변량 불균형 분석"""
    print("\n" + "=" * 70)
    print("1. 매칭 전 공변량 분포 분석")
    print("=" * 70)
    
    X = data[feature_cols].values
    T = data['treatment'].values
    
    smd_before = compute_smd(X, T)
    
    print("\n[매칭 전 공변량 불균형]")
    print("-" * 50)
    print(f"{'공변량':<12} {'처치군 평균':>10} {'대조군 평균':>10} {'SMD':>10}")
    print("-" * 50)
    
    for i, col in enumerate(feature_cols):
        t_mean = data.loc[data['treatment']==1, col].mean()
        c_mean = data.loc[data['treatment']==0, col].mean()
        print(f"{col:<12} {t_mean:>10.2f} {c_mean:>10.2f} {smd_before[i]:>10.3f}")
    
    print("-" * 50)
    print(f"{'평균 |SMD|':<12} {'':<10} {'':<10} {np.mean(np.abs(smd_before)):>10.3f}")
    print(f"{'최대 |SMD|':<12} {'':<10} {'':<10} {np.max(np.abs(smd_before)):>10.3f}")
    
    # SMD > 0.25 경고
    severe_imbalance = np.sum(np.abs(smd_before) > 0.25)
    print(f"\n⚠️  심각한 불균형 (|SMD| > 0.25): {severe_imbalance}/{len(feature_cols)} 공변량")
    
    return smd_before


def estimate_propensity_score(data, feature_cols):
    """로지스틱 회귀로 성향점수 추정"""
    print("\n" + "=" * 70)
    print("2. 성향점수 추정 (로지스틱 회귀)")
    print("=" * 70)
    
    X = data[feature_cols].values
    T = data['treatment'].values
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 로지스틱 회귀
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, T)
    
    # 예측 성향점수
    ps_logit = model.predict_proba(X_scaled)[:, 1]
    
    # 진정한 성향점수와 비교
    ps_true = data['true_ps'].values
    
    print(f"\n[추정 성향점수 vs 진정한 성향점수]")
    print(f"로지스틱 회귀 PS 범위: [{ps_logit.min():.3f}, {ps_logit.max():.3f}]")
    print(f"진정한 PS 범위:        [{ps_true.min():.3f}, {ps_true.max():.3f}]")
    print(f"상관계수 (Pearson):    {np.corrcoef(ps_logit, ps_true)[0,1]:.3f}")
    
    # 모델 오설정 진단
    ps_diff = np.abs(ps_logit - ps_true)
    print(f"\n[모델 오설정 진단]")
    print(f"평균 |PS 차이|:        {ps_diff.mean():.3f}")
    print(f"최대 |PS 차이|:        {ps_diff.max():.3f}")
    print(f"|PS 차이| > 0.2:       {(ps_diff > 0.2).sum()} ({100*(ps_diff > 0.2).mean():.1f}%)")
    
    return ps_logit, ps_true, X_scaled, scaler


def analyze_common_support(ps_logit, treatment):
    """공통지지 영역 분석"""
    print("\n" + "=" * 70)
    print("3. 공통지지 영역 분석")
    print("=" * 70)
    
    ps_treated = ps_logit[treatment == 1]
    ps_control = ps_logit[treatment == 0]
    
    # 공통지지 영역 계산
    overlap_min = max(ps_treated.min(), ps_control.min())
    overlap_max = min(ps_treated.max(), ps_control.max())
    
    # 영역 밖 개체 수
    outside_treated = ((ps_treated < overlap_min) | (ps_treated > overlap_max)).sum()
    outside_control = ((ps_control < overlap_min) | (ps_control > overlap_max)).sum()
    outside_total = outside_treated + outside_control
    
    print(f"\n[성향점수 분포]")
    print(f"처치군 PS 범위: [{ps_treated.min():.3f}, {ps_treated.max():.3f}]")
    print(f"대조군 PS 범위: [{ps_control.min():.3f}, {ps_control.max():.3f}]")
    print(f"\n[공통지지 영역]")
    print(f"공통지지 범위: [{overlap_min:.3f}, {overlap_max:.3f}]")
    print(f"\n[공통지지 위반]")
    print(f"영역 밖 처치군: {outside_treated} ({100*outside_treated/len(ps_treated):.1f}%)")
    print(f"영역 밖 대조군: {outside_control} ({100*outside_control/len(ps_control):.1f}%)")
    print(f"영역 밖 전체:   {outside_total} ({100*outside_total/len(ps_logit):.1f}%)")
    
    # 극단 성향점수 (trimming 필요성)
    extreme_low = (ps_logit < 0.05).sum()
    extreme_high = (ps_logit > 0.95).sum()
    print(f"\n[극단 성향점수 (trimming 권장)]")
    print(f"PS < 0.05:  {extreme_low} ({100*extreme_low/len(ps_logit):.1f}%)")
    print(f"PS > 0.95:  {extreme_high} ({100*extreme_high/len(ps_logit):.1f}%)")
    
    return overlap_min, overlap_max, outside_treated, outside_control


def perform_matching(data, ps_logit, feature_cols, caliper=0.05):
    """최근접 이웃 매칭 수행"""
    print("\n" + "=" * 70)
    print("4. 최근접 이웃 매칭 (1:1, caliper=0.05)")
    print("=" * 70)
    
    T = data['treatment'].values
    Y = data['outcome'].values
    
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    
    ps_treated = ps_logit[treated_idx]
    ps_control = ps_logit[control_idx]
    
    # 최근접 이웃 매칭
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(ps_control.reshape(-1, 1))
    distances, indices = nn.kneighbors(ps_treated.reshape(-1, 1))
    
    # caliper 적용
    valid_matches = distances.flatten() <= caliper
    n_matched = valid_matches.sum()
    n_unmatched = len(treated_idx) - n_matched
    
    print(f"\n[매칭 결과]")
    print(f"처치군 전체:     {len(treated_idx)}명")
    print(f"매칭 성공:       {n_matched}명 ({100*n_matched/len(treated_idx):.1f}%)")
    print(f"매칭 실패:       {n_unmatched}명 ({100*n_unmatched/len(treated_idx):.1f}%)")
    print(f"매칭 거리 중위수: {np.median(distances[valid_matches]):.4f}")
    print(f"매칭 거리 최대:   {np.max(distances[valid_matches]):.4f}")
    
    # 매칭된 샘플 추출
    matched_treated_idx = treated_idx[valid_matches]
    matched_control_idx = control_idx[indices[valid_matches].flatten()]
    
    return matched_treated_idx, matched_control_idx, n_matched, n_unmatched


def analyze_after_matching(data, feature_cols, matched_treated_idx, matched_control_idx, smd_before):
    """매칭 후 공변량 균형 분석"""
    print("\n" + "=" * 70)
    print("5. 매칭 후 공변량 균형 분석")
    print("=" * 70)
    
    # 매칭된 데이터
    matched_idx = np.concatenate([matched_treated_idx, matched_control_idx])
    X_matched = data.loc[matched_idx, feature_cols].values
    T_matched = data.loc[matched_idx, 'treatment'].values
    
    smd_after = compute_smd(X_matched, T_matched)
    
    print("\n[매칭 후 공변량 균형]")
    print("-" * 70)
    print(f"{'공변량':<12} {'매칭 전 SMD':>12} {'매칭 후 SMD':>12} {'개선율':>10} {'균형 달성':>10}")
    print("-" * 70)
    
    balance_achieved = 0
    for i, col in enumerate(feature_cols):
        before = smd_before[i]
        after = smd_after[i]
        if abs(before) > 0.001:
            improvement = 100 * (1 - abs(after)/abs(before))
        else:
            improvement = 0
        balanced = "✓" if abs(after) < 0.1 else "✗"
        if abs(after) < 0.1:
            balance_achieved += 1
        print(f"{col:<12} {before:>12.3f} {after:>12.3f} {improvement:>9.1f}% {balanced:>10}")
    
    print("-" * 70)
    print(f"{'평균 |SMD|':<12} {np.mean(np.abs(smd_before)):>12.3f} {np.mean(np.abs(smd_after)):>12.3f}")
    print(f"{'최대 |SMD|':<12} {np.max(np.abs(smd_before)):>12.3f} {np.max(np.abs(smd_after)):>12.3f}")
    print(f"\n균형 달성 공변량: {balance_achieved}/{len(feature_cols)} (SMD < 0.1 기준)")
    
    return smd_after


def estimate_att(data, matched_treated_idx, matched_control_idx, true_att):
    """ATT 추정 및 편향 분석"""
    print("\n" + "=" * 70)
    print("6. 처치효과 추정 (ATT)")
    print("=" * 70)
    
    Y = data['outcome'].values
    
    Y_treated = Y[matched_treated_idx]
    Y_control = Y[matched_control_idx]
    
    att_est = np.mean(Y_treated - Y_control)
    att_se = np.std(Y_treated - Y_control) / np.sqrt(len(Y_treated))
    
    bias = att_est - true_att
    bias_pct = 100 * bias / true_att
    
    ci_low = att_est - 1.96 * att_se
    ci_high = att_est + 1.96 * att_se
    
    t_stat = att_est / att_se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(Y_treated)-1))
    
    print(f"\n[ATT 추정 결과]")
    print(f"추정 ATT:        {att_est:.3f}")
    print(f"표준오차:        {att_se:.3f}")
    print(f"95% 신뢰구간:    [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"t-통계량:        {t_stat:.3f}")
    print(f"p-value:         {p_value:.4f}")
    
    print(f"\n[편향 분석]")
    print(f"진정한 ATT:      {true_att:.3f}")
    print(f"추정 편향:       {bias:.3f}")
    print(f"편향 비율:       {bias_pct:.1f}%")
    
    ci_contains_true = ci_low <= true_att <= ci_high
    print(f"\n95% CI가 진정한 값 포함: {'✓' if ci_contains_true else '✗'}")
    
    return att_est, att_se, bias


def naive_estimate(data, true_att):
    """Naive 추정 (매칭 없음)"""
    print("\n" + "=" * 70)
    print("7. 비교: Naive 추정 (단순 평균 차이)")
    print("=" * 70)
    
    Y = data['outcome'].values
    T = data['treatment'].values
    
    naive_att = Y[T==1].mean() - Y[T==0].mean()
    naive_bias = naive_att - true_att
    naive_bias_pct = 100 * naive_bias / true_att
    
    print(f"\nNaive 추정값:    {naive_att:.3f}")
    print(f"Naive 편향:      {naive_bias:.3f}")
    print(f"Naive 편향 비율: {naive_bias_pct:.1f}%")
    
    return naive_att


def summary_results(data, true_att, smd_before, smd_after, att_est, att_se, naive_att, 
                    n_matched, n_unmatched, outside_treated, outside_control):
    """결과 요약"""
    print("\n" + "=" * 70)
    print("분석 결과 요약")
    print("=" * 70)
    
    print("\n[표 3-1: 전통적 로지스틱 회귀 PSM 추정 결과]")
    print("-" * 50)
    print(f"{'평가 지표':<25} {'매칭 전':>10} {'매칭 후':>10}")
    print("-" * 50)
    print(f"{'평균 |SMD|':<25} {np.mean(np.abs(smd_before)):>10.3f} {np.mean(np.abs(smd_after)):>10.3f}")
    print(f"{'최대 |SMD|':<25} {np.max(np.abs(smd_before)):>10.3f} {np.max(np.abs(smd_after)):>10.3f}")
    balanced_before = np.sum(np.abs(smd_before) < 0.1)
    balanced_after = np.sum(np.abs(smd_after) < 0.1)
    print(f"{'SMD < 0.1 달성':<25} {f'{balanced_before}/5':>10} {f'{balanced_after}/5':>10}")
    print(f"{'처치군 표본 크기':<25} {len(data[data.treatment==1]):>10} {n_matched:>10}")
    print(f"{'공통지지 영역 손실':<25} {'-':>10} {f'{100*n_unmatched/(n_matched+n_unmatched):.1f}%':>10}")
    
    print("\n[표 3-2: 처치효과 추정 비교]")
    print("-" * 50)
    bias_psm = att_est - true_att
    bias_naive = naive_att - true_att
    print(f"{'방법':<15} {'추정값':>10} {'편향':>10} {'편향 비율':>10}")
    print("-" * 50)
    print(f"{'진정한 ATT':<15} {true_att:>10.3f} {'-':>10} {'-':>10}")
    print(f"{'로지스틱 PSM':<15} {att_est:>10.3f} {bias_psm:>10.3f} {100*bias_psm/true_att:>9.1f}%")
    print(f"{'Naive 추정':<15} {naive_att:>10.3f} {bias_naive:>10.3f} {100*bias_naive/true_att:>9.1f}%")
    
    print("\n" + "=" * 70)
    print("주요 발견: 전통적 PSM의 한계")
    print("=" * 70)
    print(f"""
1. 모델 오설정 문제:
   - 로지스틱 회귀는 비선형 성향점수 함수를 포착하지 못함
   - 매칭 후에도 평균 SMD {np.mean(np.abs(smd_after)):.3f}로 불균형 잔존
   - 특히 {['나이', '소득', '교육', '경력', '자산'][np.argmax(np.abs(smd_after))]} 변수에서 
     SMD {np.max(np.abs(smd_after)):.3f}의 심각한 불균형

2. 공통지지 영역 문제:
   - 처치군의 {100*n_unmatched/(n_matched+n_unmatched):.1f}%가 매칭 실패
   - 외적 타당도 저하: 추정된 효과는 매칭된 하위집단에만 일반화 가능
   - 고성향점수 처치군이 제외되어 추정 대상이 변화

3. 편향된 ATT 추정:
   - 추정 ATT ({att_est:.3f})가 진정한 ATT ({true_att:.3f})에서 {abs(100*bias_psm/true_att):.1f}% 벗어남
   - 불완전한 균형과 표본 선택으로 인한 잔여 편향 존재
   - 기계학습 기반 방법론이 필요한 근거 제시
""")


def main():
    # 데이터 로드
    data, true_att = load_data()
    
    feature_cols = ['age', 'income', 'education', 'experience', 'assets']
    
    # 1. 매칭 전 분석
    smd_before = analyze_before_matching(data, feature_cols)
    
    # 2. 성향점수 추정
    ps_logit, ps_true, X_scaled, scaler = estimate_propensity_score(data, feature_cols)
    
    # 3. 공통지지 영역 분석
    overlap_min, overlap_max, outside_treated, outside_control = analyze_common_support(
        ps_logit, data['treatment'].values
    )
    
    # 4. 매칭 수행
    matched_treated_idx, matched_control_idx, n_matched, n_unmatched = perform_matching(
        data, ps_logit, feature_cols
    )
    
    # 5. 매칭 후 균형 분석
    smd_after = analyze_after_matching(
        data, feature_cols, matched_treated_idx, matched_control_idx, smd_before
    )
    
    # 6. ATT 추정
    att_est, att_se, bias = estimate_att(data, matched_treated_idx, matched_control_idx, true_att)
    
    # 7. Naive 추정
    naive_att = naive_estimate(data, true_att)
    
    # 8. 결과 요약
    summary_results(data, true_att, smd_before, smd_after, att_est, att_se, naive_att,
                   n_matched, n_unmatched, outside_treated, outside_control)
    
    print("\n※ 본 코드는 교육 목적의 시뮬레이션입니다.")
    print("_전체 코드는 practice/chapter03/code/3-1-traditional-psm-limitations.py 참고_")


if __name__ == "__main__":
    main()
