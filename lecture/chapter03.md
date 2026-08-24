# 3장. 비슷한 사람 찾기 - 성향점수매칭(PSM)에서 딥러닝 PSM으로

**🎯 학습 목표: "비슷한 사람끼리 비교"하는 성향점수매칭의 원리를 이해하고, 전통적 방법의 한계를 AI(머신러닝, 딥러닝)로 극복하는 최신 기법을 배우기**

---

## 🌟 이 장에서 배울 내용 미리보기

- **전통적 성향점수매칭(PSM)**: 로지스틱 회귀로 "비슷한 사람"을 찾아 비교하는 기본 방법
- **PSM의 한계**: 현실 데이터에서 왜 전통적 방법이 틀릴 수 있는지
- **머신러닝 PSM**: 랜덤 포레스트와 그래디언트 부스팅으로 더 정확하게 매칭하기
- **딥러닝 PSM (LBC-Net)**: 신경망이 "균형 잡기"를 직접 학습하는 최신 방법
- **공변량 균형 최적화**: 손실 함수를 조정해서 더 공정한 비교를 만드는 기법
- **종합 비교**: 네 가지 방법(로지스틱·RF·GB·LBC-Net)을 한눈에 비교

---

## 3.1 전통적 Propensity Score Matching 📊

### "비슷한 쌍둥이 찾기" - 성향점수의 기본 아이디어

2장에서 배운 것처럼, 정책 효과를 정확히 측정하려면 처치를 받은 그룹과 안 받은 그룹이 **처음부터 비슷해야** 한다. 무작위 배정(RCT)을 할 수 없을 때, Rosenbaum & Rubin(1983)이 제안한 방법이 바로 **성향점수매칭(PSM)**이다.

**🎲 핵심 아이디어를 비유로 이해하기**

소개팅 앱을 생각해 보자. 여러 조건(나이, 직업, 취미, 성격)이 비슷한 사람끼리 연결해 준다. 성향점수매칭도 마찬가지다. 나이, 소득, 교육수준 등 여러 특성을 하나의 숫자(성향점수)로 요약하고, 이 숫자가 비슷한 사람끼리 짝 지어 비교하는 것이다.

**성향점수(Propensity Score)**란 "이 사람이 정책을 받을 확률"을 0~1 사이의 숫자 하나로 나타낸 것이다. 예를 들어 성향점수가 0.7이면 "이 사람이 정책에 참여할 확률이 70%"라는 뜻이다. 이 점수가 비슷한 사람끼리 비교하면, 마치 무작위 배정처럼 공정한 비교가 가능해진다.

비즈니스에서는 **"유료 멤버십 가입 유도 캠페인의 효과"**를 분석할 때 유용하다. 캠페인 대상자(처치군)는 주로 '충성도가 높은 고객' 위주로 선정되므로, 단순히 비대상자(대조군)와 비교하면 캠페인 효과가 과대평가된다. PSM은 구매 이력, 방문 빈도 등 공변량을 바탕으로 '캠페인 대상자가 될 뻔했던' 유사한 비대상자를 찾아 매칭함으로써 이 문제를 해결한다.

### PSM의 3단계 과정

성향점수매칭은 크게 세 단계로 진행된다.

**1단계 - 성향점수 추정**: 로지스틱 회귀를 이용해서 "이 사람이 정책을 받을 확률"을 계산한다. 나이, 소득, 교육수준 같은 여러 변수를 넣으면 각 사람에게 0~1 사이의 점수가 하나씩 나온다.

**2단계 - 매칭**: 정책을 받은 사람(처치군)과 성향점수가 비슷한 안 받은 사람(대조군)을 짝 지어 준다. 방법으로는 가장 가까운 사람을 찾는 최근접 이웃 매칭(nearest neighbor matching), 일정 범위 안에서만 매칭하는 캘리퍼 매칭(caliper matching) 등이 있다.

**3단계 - 효과 추정**: 매칭된 짝들의 결과 차이를 평균내면 그것이 정책의 평균처치효과(ATT)가 된다.

### 전통적 PSM의 한계 - 왜 완벽하지 않은가?

로지스틱 회귀를 사용한 PSM은 구현이 쉽고 빠르지만, 근본적인 문제가 있다.

**첫째, "직선"만 그릴 수 있다.** 로지스틱 회귀는 변수들의 관계가 직선형이라고 가정한다. 그런데 현실에서는 "나이가 많을수록 효과가 커지다가 어느 시점부터 줄어든다"처럼 곡선 관계가 흔하다. 이런 비선형 관계를 잡아내지 못하면, 성향점수가 부정확해진다.

**둘째, 연구자가 직접 정해야 할 것이 너무 많다.** 어떤 변수 간 상호작용을 넣을지, 제곱항을 넣을지 등을 연구자가 미리 결정해야 한다. 잘못 선택하면 **모델 오설정(model misspecification)**이 발생하고, 결과가 틀어진다.

**셋째, 공통지지 영역 문제.** 처치군과 대조군의 성향점수가 겹치지 않는 영역이 있으면, 그 부분의 사람들은 매칭 상대를 찾을 수 없어 분석에서 빠진다. 겹치는 영역이 좁을수록 분석의 일반화 가능성이 떨어진다.

![성향점수매칭(PSM)의 3단계 프로세스와 핵심 가정](../diagrams/3-1.png)

### 💻 실습: 전통적 PSM의 한계 확인하기

아래 코드로 로지스틱 회귀 기반 PSM이 비선형 데이터에서 얼마나 힘든지 직접 확인해 보자. 진짜 처치효과(ATT)를 미리 알고 있는 시뮬레이션 데이터를 사용한다.

※ 본 실습 코드는 실습 목적의 시뮬레이션 데이터를 사용합니다. 진정한 ATT가 5.0으로 설정된 통제된 환경에서 각 방법론의 성능을 정확히 비교 평가하기 위함입니다.

```python
# 전통적 로지스틱 회귀 PSM (핵심 알고리즘)

# 1단계: 성향점수 추정

ps_model = LogisticRegression(max_iter=500).fit(X, treatment)
ps = ps_model.predict_proba(X)[:, 1]

# 2단계: 최근접 이웃 매칭 (1:1, caliper=0.05)
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(ps[treatment==0].reshape(-1,1))
distances, indices = nn.kneighbors(ps[treatment==1].reshape(-1,1))
matched_controls = np.where(treatment==0)[0][indices.flatten()]

# 3단계: ATT 추정
att = np.mean(outcomes[treatment==1] - outcomes[matched_controls])
```

<표 3-1: 전통적 로지스틱 회귀 PSM 추정 결과>

| 평가 지표                | 매칭 전 | 매칭 후 | 개선율 |
| ------------------------ | ------- | ------- | ------ |
| 평균 SMD (모든 공변량)   | 0.660   | 0.156   | -76.4% |
| 최대 SMD                 | 0.836   | 0.231   | -72.4% |
| SMD < 0.1 달성 공변량 수 | 0/5     | 2/5     | -      |
| 처치군 표본 크기         | 621     | 621     | 0%     |
| 공통지지 영역 위반       | -       | 9.9%    | -      |

<표 3-2: 공변량 균형 세부 분석 (매칭 후)>

| 공변량   | 매칭 전 SMD | 매칭 후 SMD | 균형 달성 |
| -------- | ----------- | ----------- | --------- |
| 나이     | 0.836       | 0.231       | ✗         |
| 소득     | 0.623       | -0.164      | ✗         |
| 교육년수 | -0.368      | 0.069       | ✓         |
| 경력년수 | 0.786       | 0.224       | ✗         |
| 자산     | 0.687       | -0.091      | ✓         |

<표 3-3: 모델 오설정 진단>

| 진단 지표                    | 값    | 해석                 |
| ---------------------------- | ----- | -------------------- |
| 추정 PS vs 진정한 PS 상관    | 0.665 | 낮음 (심각한 오설정) |
| \|PS 차이\| > 0.2 비율       | 40.2% | 높음                 |
| 극단 PS (< 0.05 또는 > 0.95) | 7.8%  | trimming 권장        |

처치효과 추정:

- ATT (평균처치효과): 4.924 (95% CI: [3.579, 6.268])
- 표준오차: 0.686
- 진정한 ATT: 6.412
- 추정 편향: -1.488 (-23.2%)
- 95% 신뢰구간이 진정한 값 포함: ✗

**🔍 결과를 쉽게 읽어보면:**

첫째, **모델이 심하게 빗나갔다.** 표 3-3을 보면, 로지스틱 회귀가 추정한 성향점수와 진짜 성향점수의 상관관계가 0.665밖에 안 된다. 100명 중 40명(40.2%)은 성향점수가 0.2 이상 차이 나서 완전히 다른 점수를 받은 셈이다. 이유는 간단하다. 데이터에는 곡선 관계(제곱항, 상호작용, 주기함수)가 있는데, 로지스틱 회귀는 직선만 그릴 수 있기 때문이다.

둘째, **매칭 후에도 불균형이 남아있다.** 표 3-2를 보면, 매칭 후에도 5개 변수 중 3개(나이, 소득, 경력년수)에서 SMD가 0.1을 넘는다. SMD 0.1은 "두 그룹이 충분히 비슷하다"고 볼 수 있는 기준인데, 이를 넘으면 아직 편향이 남아있다는 뜻이다.

셋째, **처치효과가 크게 과소추정되었다.** 진짜 ATT는 6.412인데 추정값은 4.924로, **-23.2%나 빗나갔다.** 더 심각한 것은 95% 신뢰구간 [3.579, 6.268]이 진짜 값을 포함하지 못한다는 점이다. 이는 통계적 결론 자체가 잘못될 수 있다는 뜻이다.

※ 이 코드는 교육 목적의 예제입니다. 실제 정책 분석에서는 더 정교한 공변량 설계와 민감도 분석이 필요합니다.
*전체 코드는 practice/chapter03/code/3-1-traditional-psm-limitations.py 참고*

---

## 3.2 Machine Learning for Propensity Scores 🤖

### "직선 대신 자유로운 곡선" - 머신러닝이 PSM을 어떻게 개선하는가

앞 절에서 전통적 PSM의 가장 큰 문제는 "직선만 그릴 수 있다"는 것이었다. 그렇다면 **자유로운 곡선을 그릴 수 있는 머신러닝**을 사용하면 어떨까? 이것이 바로 기계학습 기반 PSM의 핵심 아이디어다.

**🌳 랜덤 포레스트 - "여러 전문가의 평균 의견"**

랜덤 포레스트(Random Forest)는 Breiman(2001)이 제안한 방법으로, 수백 개의 의사결정나무(decision tree)를 만들어 그 평균을 사용한다. 비유하자면 **100명의 전문가에게 물어보고 다수결을 따르는 것**과 비슷하다. 각 전문가가 조금씩 다른 관점에서 판단하기 때문에, 한 사람이 실수해도 전체 결과에는 큰 영향이 없다. 비선형 관계(곡선 패턴)와 변수 간 상호작용을 연구자가 직접 지정하지 않아도 자동으로 찾아낸다는 장점이 있다.

**📈 그래디언트 부스팅 - "실수를 계속 고쳐나가기"**

그래디언트 부스팅(Gradient Boosting)은 Friedman(2001)이 제안한 방법으로, 약한 모델을 순서대로 쌓아가면서 **이전 모델이 틀린 부분을 집중적으로 개선**하는 방식이다. 비유하자면 **시험 오답노트**와 비슷하다. 처음에 대충 풀고, 틀린 문제만 모아서 다시 공부하고, 또 틀린 것만 모아서 공부하면 점점 정확해지는 것이다.

**⚠️ 중요한 포인트: 예측 정확도 ≠ 좋은 매칭**

McCaffrey et al.(2004)은 성향점수 추정에서 중요한 것은 "처치를 잘 예측하는 것"이 아니라 **"두 그룹을 비슷하게 만드는 것(공변량 균형)"**이라고 강조했다. 그래서 부스팅의 반복 횟수를 정할 때 예측 정확도가 아니라 **공변량 균형(ASAM)**을 기준으로 선택하는 방법을 제안했다.

### 💻 실습: 머신러닝으로 더 정확한 PSM 만들기

아래 코드는 랜덤 포레스트와 그래디언트 부스팅으로 성향점수를 추정하고, 로지스틱 회귀와 비교한다.

```python
# 기계학습 PSM (핵심 알고리즘)

# 랜덤 포레스트 성향점수 추정
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10,

                                  min_samples_leaf=20).fit(X, treatment)
ps_rf = rf_model.predict_proba(X)[:, 1]

# 그래디언트 부스팅 성향점수 추정 (균형 기반 최적화)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,

                                      max_depth=3).fit(X, treatment)
ps_gb = gb_model.predict_proba(X)[:, 1]

# 공변량 균형 최대화를 위한 반복 횟수 선택

best_n_iter = select_iterations_by_balance(X, treatment, outcomes)
```

<표 3-4: 기계학습 PSM 성능 비교>

| 방법론            | 평균 SMD | 최대 SMD | 균형 달성 (SMD<0.1) | ATT   | 표준오차 | 매칭률 |
| ----------------- | -------- | -------- | ------------------- | ----- | -------- | ------ |
| 로지스틱 회귀     | 0.071    | 0.163    | 4/5 (80%)           | 5.065 | 0.212    | 97.2%  |
| 랜덤 포레스트     | 0.086    | 0.162    | 3/5 (60%)           | 5.068 | 0.201    | 100%   |
| 그래디언트 부스팅 | 0.039    | 0.081    | 5/5 (100%)          | 4.862 | 0.204    | 99.5%  |

<표 3-5: 변수 중요도 분석 (랜덤 포레스트)>

| 순위 | 특성                 | 중요도 |
| ---- | -------------------- | ------ |
| 1    | Feature 0 (나이)     | 0.354  |
| 2    | Feature 3 (경력년수) | 0.332  |
| 3    | Feature 2 (교육년수) | 0.147  |
| 4    | Feature 1 (소득)     | 0.042  |
| 5    | Feature 4 (자산)     | 0.031  |

<표 3-6: 그래디언트 부스팅 반복 횟수 최적화>

| 반복 횟수 | 학습률 | 평균 SMD | 선택 기준                |
| --------- | ------ | -------- | ------------------------ |
| 100       | 0.01   | 0.039    | 최적 (균형-정확도 Joint) |

<표 3-7: 공변량 균형의 질적 개선>

| 공변량   | 로지스틱 SMD | RF SMD | GB SMD | 최우수 방법 |
| -------- | ------------ | ------ | ------ | ----------- |
| 평균 SMD | 0.071        | 0.086  | 0.039  | GB          |
| 최대 SMD | 0.163        | 0.162  | 0.081  | GB          |
| 매칭률   | 97.2%        | 100%   | 99.5%  | RF          |

**🔍 결과를 쉽게 읽어보면:**

첫째, **그래디언트 부스팅이 가장 균형을 잘 맞춘다.** 표 3-4에서 GB의 평균 SMD는 0.039로, 로지스틱 회귀(0.071)보다 45% 낮고, 5개 공변량 모두에서 SMD < 0.1을 달성했다. "오답노트" 전략이 "비슷한 사람 찾기"에서도 효과적이라는 뜻이다. ATT 추정치 4.862는 진짜 값 5.0과 불과 2.8% 차이다.

둘째, **랜덤 포레스트는 매칭률이 100%로 가장 높다.** 모든 처치군이 매칭 상대를 찾은 것이다. ATT 추정치 5.068도 진짜 값에 아주 가까워 실용적으로 훌륭하다.

셋째, **변수 중요도로 "왜 정책에 참여했는지" 알 수 있다.** 표 3-5를 보면 나이(0.354)와 경력년수(0.332)가 가장 중요하다. 이는 비즈니스에서 "마케팅팀이 주로 어떤 기준으로 고객을 선정했는지"를 역추적하는 것과 비슷하다. 소득(0.042)과 자산(0.031)은 중요도가 낮아, 경제력보다 활동 이력이 타겟팅의 주요 기준이었음을 보여준다.

※ 이 코드는 교육 목적의 예제입니다. 실제 정책 분석에서는 교차검증을 통한 하이퍼파라미터 튜닝이 필요합니다.
*전체 코드는 practice/chapter03/code/3-2-ml-psm.py 참고*

---

## 3.3 Deep Nonparametric Propensity Scores (2024) 🧠

### "균형 맞추기를 직접 학습하는 딥러닝" - LBC-Net

앞에서 머신러닝이 비선형 관계를 잘 잡아내는 것을 확인했다. 그런데 한 가지 아쉬운 점이 있다. 기존 방법들은 **"처치를 잘 예측하는 것"을 먼저 하고**, 균형이 잘 맞았는지는 **나중에 확인**한다. 순서가 뒤바뀌어 있는 셈이다.

Peng et al.(2024)은 이 순서를 뒤집었다. **"균형 맞추기"를 신경망의 학습 목표에 직접 넣는** LBC-Net(Local Balance with Calibration Network)이라는 방법을 제안했다.

**🎯 기존 방법과 LBC-Net의 차이를 비유로 이해하기**

- **기존 방법**: 시험을 먼저 보고(예측), 나중에 오답을 확인(균형 체크) → 이미 시험이 끝났으므로 고칠 수 없음
- **LBC-Net**: 시험을 보면서 동시에 오답을 줄여나감(예측 + 균형을 동시에 최적화)

### LBC-Net의 두 가지 핵심 조건

LBC-Net은 성향점수가 만족해야 하는 두 가지 조건을 **손실 함수**에 직접 넣어서 학습한다.

**1️⃣ 국소 균형(Local Balance)** - "모든 구간에서 비슷해야 한다"

성향점수가 0.3 근처인 사람들끼리만 따로 보면 처치군과 대조군의 특성이 비슷해야 한다. 0.5 근처, 0.7 근처도 마찬가지다. **전체 평균만 비슷하면 안 되고, 각 구간마다 비슷해야** 한다는 것이다. 비유하자면 "반 평균이 같다고 해서 모든 학생이 비슷한 수준인 것은 아니다"라는 것과 같다.

**2️⃣ 국소 보정(Local Calibration)** - "예측한 확률이 진짜여야 한다"

성향점수가 0.7이라고 추정된 사람들 100명 중에서 실제로 정책을 받은 사람이 약 70명이어야 한다. 예측 확률이 실제와 맞아야 한다는 뜻이다. 일기예보에서 "비 올 확률 70%"라고 했을 때 실제로 70% 정도 비가 와야 좋은 예보인 것과 같다.

**🔧 LBC-Net의 손실 함수 구조**

LBC-Net은 세 가지를 동시에 최적화한다.

1. **이진 교차 엔트로피(BCE)**: 처치를 정확히 예측하기 (기본 예측 손실)
2. **국소 균형 패널티**: 성향점수를 10개 구간으로 나눠서, 각 구간에서 처치군과 대조군의 공변량 차이를 줄이기
3. **국소 보정 패널티**: 각 구간에서 예측 확률과 실제 처치 비율의 차이를 줄이기

다만 이 논문은 2025년 현재 arXiv preprint로 아직 동료 심사(peer-review)를 거치지 않은 상태이므로, 실무에서는 다른 방법들과 함께 비교 검증하는 것이 권장된다.

### 💻 실습: LBC-Net 구현하기

아래 코드는 LBC-Net을 PyTorch로 구현하고, 기존 방법들과 성능을 비교한다.

```python
# Deep Nonparametric PSM (LBC-Net 핵심 알고리즘)

# 딥러닝 기반 비모수 성향점수 추정 (Peng et al., 2024)
class LBCNet(nn.Module):
    def forward(self, x):
        # 은닉층을 통한 비선형 변환
        x = relu(dropout(hidden_layers(x)))
        return sigmoid(output_layer(x))

# 손실 함수: BCE + Local Balance + Calibration
def lbc_loss(ps_pred, treatment, covariates):
    bce_loss = BCELoss(ps_pred, treatment)
    balance_penalty = compute_local_balance(ps_pred, treatment, covariates, n_bins=10)
    calib_penalty = compute_calibration(ps_pred, treatment, n_bins=10)  # 각 구간에서 예측 성향점수와 실제 처치 비율 차이
    return bce_loss + λ_balance * balance_penalty + λ_calib * calib_penalty

# 학습

model.train(X, treatment, epochs=200, optimizer=Adam(lr=0.001))
```

<표 3-8: Deep PSM (LBC-Net) 추정 결과>

| 평가 지표                | 로지스틱       | RF             | GB             | LBC-Net        |
| ------------------------ | -------------- | -------------- | -------------- | -------------- |
| 평균 SMD                 | 0.071          | 0.086          | 0.039          | 0.065          |
| 최대 SMD                 | 0.163          | 0.162          | 0.081          | 0.092          |
| 균형 달성 비율 (SMD<0.1) | 80%            | 60%            | 100%           | 100%           |
| ATT 추정                 | 5.065          | 5.068          | 4.862          | 4.535          |
| 표준오차                 | 0.212          | 0.201          | 0.204          | 0.216          |
| 95% 신뢰구간             | [4.650, 5.480] | [4.673, 5.462] | [4.461, 5.262] | [4.112, 4.959] |
| 매칭률                   | 97.2%          | 100.0%         | 99.5%          | 97.0%          |

<표 3-9: 공변량별 균형 비교 (LBC-Net 상세)>

| 공변량        | LBC-Net SMD |
| ------------- | ----------- |
| 나이          | -0.092      |
| 소득          | -0.064      |
| 교육년수      | -0.088      |
| 경력년수      | -0.067      |
| 자산          | -0.012      |
| 평균 (절댓값) | 0.065       |

**🔍 결과를 쉽게 읽어보면:**

첫째, **LBC-Net은 5개 공변량 모두에서 균형을 달성했다(SMD < 0.1).** 표 3-9를 보면 자산(-0.012)은 거의 완벽한 균형이고, 나머지 변수들도 모두 기준을 통과했다. "균형 맞추기"를 직접 학습 목표에 넣은 효과가 확실히 나타난 것이다.

둘째, **ATT 추정은 약간 빗나갔다.** LBC-Net의 ATT 4.535는 진짜 값 5.000에서 -9.3% 차이가 난다. 로지스틱(+1.3%)이나 RF(+1.4%)보다 편향이 크다. 이는 "균형 맞추기"에 너무 집중하면 "정확한 효과 추정"이 살짝 희생될 수 있다는 것을 보여준다. 마치 시험에서 "고르게 맞추려고" 하면 "총점"은 약간 떨어질 수 있는 것과 비슷하다.

셋째, **LBC-Net은 대규모 데이터에서 더 빛을 발한다.** 학습 시간이 약 6초로 로지스틱(0.02초)이나 GB(0.32초)보다 길지만, 데이터가 5,000건 이상으로 많아지고 비선형 관계가 복잡할수록 LBC-Net의 장점이 커진다. 따라서 LBC-Net은 로지스틱과 그래디언트 부스팅 사이의 중간적 선택지로서 유용하다.

※ 이 코드는 교육 목적의 예제입니다. 실제 정책 분석에서는 네트워크 구조와 균형 가중치의 교차검증이 필요합니다.
*전체 코드는 practice/chapter03/code/3-3-deep-psm-pytorch.py 참고*

---

## 3.4 Optimized Covariate Balance ⚖️

### "균형 맞추기를 손실 함수에 직접 넣기" - CBPS와 손실 함수 보정

3.3절에서 LBC-Net이 "균형 맞추기"를 학습 목표에 넣는 아이디어를 보았다. 이 절에서는 이 아이디어를 더 넓은 관점에서 살펴본다. 핵심 질문은 이것이다: **"예측을 잘 하는 것"과 "균형을 잘 맞추는 것", 둘 다 동시에 할 수 있는 최적의 방법이 있을까?**

**🎯 왜 예측 정확도 ≠ 좋은 균형인가?**

비유하자면 이렇다. GPS 내비게이션이 "현재 위치"를 아무리 정확히 맞춰도, 목적지까지의 "경로"가 안 좋으면 소용없다. 마찬가지로 성향점수가 "누가 정책을 받을지" 예측을 잘 해도, 그 점수로 매칭했을 때 "두 그룹이 실제로 비슷해지는지"는 별개의 문제다. 모델이 잘못 설정되면 예측은 꽤 정확해 보이는데 균형은 엉망일 수 있다.

### CBPS - "예측과 균형을 동시에 최적화"

Imai & Ratkovic(2014)이 제안한 **CBPS(Covariate Balancing Propensity Score)**는 이 문제를 해결한 선구적 방법이다. 기존 로지스틱 회귀가 "예측만 잘하겠다"는 목표로 학습하는 것과 달리, CBPS는 두 가지 목표를 동시에 추구한다.

1. **점수 조건**: 처치를 잘 예측한다 (기존과 동일)
2. **균형 조건**: 처치군과 대조군의 공변량 분포를 비슷하게 만든다 (새로 추가)

### 손실 함수 보정 - 신경망에 "균형 패널티" 추가하기

이 아이디어를 신경망에 적용하면 더 강력해진다. 기본 예측 손실에 **"균형이 안 맞으면 벌점을 준다"**는 패널티를 추가하는 것이다.

**결합 손실 함수**: L_total = L_prediction + λ × L_balance

여기서 λ(람다)는 "예측과 균형 사이의 균형"을 조절하는 하이퍼파라미터다. λ=0이면 예측만, λ가 클수록 균형에 더 집중한다.

신경망은 비선형 관계를 자동으로 학습하므로, 연구자가 상호작용항이나 제곱항을 직접 지정할 필요가 없다. 여기에 균형 패널티까지 추가하면, **"복잡한 패턴을 자동으로 학습"하면서 동시에 "두 그룹을 비슷하게 만드는"** 강력한 도구가 된다.

### 💻 실습: 최적의 λ 찾기

아래 코드는 λ를 0.0~1.0까지 바꿔가며 균형 성능을 비교한다.

```python
# Covariate Balance Optimization (핵심 알고리즘)

# 균형 패널티: 역확률 가중 후 SMD 계산
def balance_penalty(ps_pred, treatment, covariates):
    ε = 1e-6  # 분모 안정화 상수
    # ATT 추정용 IPW 가중치: 대조군에 ps/(1-ps) 적용
    weights = ps_pred / (1 - ps_pred + ε)
    treat_mean = covariates[treatment==1].mean(0)
    control_weights = weights[treatment==0]
    control_mean_weighted = (covariates[treatment==0] * control_weights.unsqueeze(1)).sum(0) / control_weights.sum()
    smd = torch.abs(treat_mean - control_mean_weighted)  # 표준화 데이터 가정
    return smd.mean()

# 결합 손실 함수
def combined_loss(ps_pred, treatment, covariates, λ=0.5):
    return BCELoss(ps_pred, treatment) + λ * balance_penalty(ps_pred, treatment, covariates)

# 하이퍼파라미터 최적화 (λ ∈ [0, 1])
for λ in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:

    model.train(X, treatment, loss_fn=combined_loss, lambda_balance=λ)
    final_smd = evaluate_balance(model, X, treatment)
```

<표 3-10: 하이퍼파라미터 λ 최적화 결과 (3-4 실행)>

| λ (균형 가중치) | BCE 손실 | 평균 SMD |
| --------------- | -------- | -------- |
| 0.0             | 0.534    | 0.033    |
| 0.1             | 0.576    | 0.044    |
| 0.3             | 0.563    | 0.017    |
| 0.5             | 0.584    | 0.026    |
| 0.7             | 0.596    | 0.021    |
| 1.0             | 0.569    | 0.016    |

**🔍 결과를 쉽게 읽어보면:**

첫째, **균형 패널티를 넣으면 확실히 효과가 있다.** λ=0(예측만)일 때 평균 SMD가 0.033이었는데, λ=1.0에서는 0.016으로 **절반 이하**로 줄었다. 즉, "균형이 안 맞으면 벌점"이라는 신호를 넣어주면 신경망이 실제로 균형을 더 잘 맞추게 된다.

둘째, **대신 예측 정확도는 살짝 떨어진다.** BCE 손실이 0.534(λ=0)에서 0.569(λ=1.0)로 약간 올라갔다. 이것은 "두 마리 토끼를 동시에 잡으려면 각각 조금씩 양보해야 한다"는 트레이드오프를 보여준다. 하지만 균형 개선 효과가 예측 손실을 충분히 상쇄한다.

셋째, **실무에서는 λ=0.3~1.0 범위를 추천한다.** 이 범위에서 평균 SMD가 0.016~0.021로 매우 낮다. 최종적으로 교차검증을 통해 ATT 추정의 안정성을 확인하면 가장 좋다.

※ 이 코드는 교육 목적의 예제입니다. 실제 정책 분석에서는 교차검증을 통한 최적 λ 선택이 필요합니다.
*전체 코드는 practice/chapter03/code/3-4-covariate-balance.py 참고*

---

## 3.5 실습: PyTorch 기반 Deep PSM 구현 🔬

### 네 가지 방법 종합 비교 - "강한 비선형 교란" 상황에서

이 절에서는 앞에서 배운 네 가지 방법(로지스틱 회귀, 랜덤 포레스트, 그래디언트 부스팅, LBC-Net)을 **같은 데이터에서 한꺼번에 비교**한다. 특히 **"강한 비선형 교란"**이 존재하는 까다로운 상황을 설정하여, 각 방법의 장단점을 확실히 드러낸다.

**🧪 실험 설계**

- **데이터**: 3,000명, 7개 변수 (핵심 교란변수 2개 + 추가 공변량 5개)
- **처치 배정**: 의도적으로 강한 비선형 설계 (제곱항, 가우시안 범프, 상호작용, 주기함수 등)
- **진짜 ATT**: 6.0으로 미리 설정
- **왜 이렇게 어려운 데이터?**: 전통적 로지스틱 회귀가 실패하는 상황을 재현하기 위해서

이런 비선형 패턴은 현실에서도 흔하다. 예를 들어 "특정 연령대에서만 정책 효과가 급증"하거나, "소득과 교육의 상호작용이 복잡하게 얽힌" 경우가 그렇다.

### 💻 실습: 4가지 방법 통합 비교

```python
# 통합 PSM 비교 (핵심 알고리즘)
# 4가지 방법으로 성향점수 추정
ps_lr = LogisticRegression(max_iter=500).fit(X, treatment).predict_proba(X)[:, 1]
ps_rf = RandomForestClassifier(n_estimators=200, max_depth=10).fit(X, treatment).predict_proba(X)[:, 1]
ps_gb = GradientBoostingClassifier(n_estimators=100, max_depth=5).fit(X, treatment).predict_proba(X)[:, 1]
ps_lbc = LBCNet(input_dim=X.shape[1]).train(X, treatment, epochs=300).predict(X)

# 매칭 및 ATT 추정
def match_and_estimate_att(ps, treatment, outcome, caliper=0.01):
    """
    성향점수 기반 최근접 이웃 매칭 및 ATT 추정
    ps: 성향점수 배열, treatment: 처치 변수, outcome: 결과 변수, caliper: 매칭 거리 임계값
    반환값: att(평균처치효과), se(표준오차), n_matched(매칭 쌍 수)
    """
    matches = [(t, c) for t in treated if min_distance(ps[t], ps[controls]) <= caliper]
    att = mean([outcome[t] - outcome[c] for t, c in matches])
    return att, se, n_matched

# 공변량 균형 평가

def calculate_smd(X, treatment, matched_idx):
    """

    표준화 평균 차이(Standardized Mean Difference) 계산
    X: 공변량 행렬, treatment: 처치 변수, matched_idx: 매칭된 인덱스
    반환값: 각 공변량의 SMD 배열
    """

    smd = abs(X[treatment==1].mean(0) - X[matched_idx].mean(0)) / pooled_std
    return smd

# 4가지 방법 비교 실행
for method, ps in [('LR', ps_lr), ('RF', ps_rf), ('GB', ps_gb), ('LBC', ps_lbc)]:
    att, se, smd = evaluate_method(ps, treatment, outcome, X)
```

<표 3-11: 최종 방법론 비교 종합>

| 평가 지표     | 진정한 값 | 로지스틱       | RF           | GB           | LBC-Net      |
| ------------- | --------- | -------------- | ------------ | ------------ | ------------ |
| 공변량 균형   |           |                |              |              |              |
| 평균 SMD      | 0.000     | 0.066          | 0.075        | 0.182        | 0.077        |
| 최대 SMD      | 0.000     | 0.094          | 0.132        | 0.364        | 0.145        |
| 처치효과 추정 |           |                |              |              |              |
| ATT           | 6.00      | 11.063         | 5.705        | 4.737        | 5.832        |
| 편향 (Bias)   | 0.00      | +5.063         | -0.295       | -1.263       | -0.168       |
| 편향율        | 0%        | +84.4%         | -4.9%        | -21.1%       | -2.8%        |
| 표준오차 (SE) | -         | 0.231          | 0.113        | 0.129        | 0.102        |
| 95% CI        | -         | [10.61, 11.52] | [5.48, 5.93] | [4.48, 4.99] | [5.63, 6.03] |
| 계산 성능     |           |                |              |              |              |
| 학습 시간     | -         | 0.008s         | 0.736s       | 1.228s       | 9.132s       |
| 해석 용이성   | -         | 높음           | 중간         | 중간         | 낮음         |
| 구현 복잡도   | -         | 낮음           | 낮음         | 중간         | 높음         |

**🔍 결과를 쉽게 읽어보면:**

첫째, **로지스틱 회귀는 비선형 교란에서 완전히 실패한다.** ATT 추정이 11.063으로 진짜 값 6.0보다 **+84.4%나 과대추정**되었다. 균형(SMD 0.066)은 나쁘지 않아 보이지만, 비선형 관계를 전혀 포착하지 못해서 처치효과 추정이 엉뚱하게 나온 것이다. 이것은 "GPS가 현재 위치는 맞는데, 곡선 도로를 직선으로 안내해서 목적지를 완전히 잘못 찍은 것"과 비슷하다.

둘째, **LBC-Net이 가장 정확하다.** 편향이 -2.8%로 가장 작고, 95% 신뢰구간 [5.63, 6.03]이 **진짜 값 6.0을 포함**한다. 균형 손실(λ_balance=5.0)과 보정 손실(λ_calib=2.0)을 직접 학습 목표에 넣은 것이 결정적이다.

셋째, **랜덤 포레스트가 실용적으로 가장 균형 잡힌 선택이다.** 편향 -4.9%로 LBC-Net에 근접하면서, 학습 시간은 0.7초로 LBC-Net(9초)보다 10배 이상 빠르고 구현도 간단하다.

넷째, **상황에 따라 최적 방법이 다르다.** 아래 표를 참고하자.

<표 3-12: 인과추론 실무 권장사항>

| 상황                                     | 권장 방법                  | 이유                                     |
| ---------------------------------------- | -------------------------- | ---------------------------------------- |
| 비선형 교란 의심 시 (대부분 실제 데이터) | LBC-Net 또는 랜덤 포레스트 | 최소 편향 (-2.8~-4.9%), 비선형 패턴 포착 |
| 빠른 탐색적 분석                         | 랜덤 포레스트              | 학습 시간 0.7초, 편향 -4.9%, 구현 간단   |
| 선형 관계 확신 시                        | 로지스틱 회귀              | 0.008초로 가장 빠름, 해석 용이           |
| 최고 정확도 필요 시                      | LBC-Net (균형 손실 포함)   | 편향 -2.8%, 95% CI가 진정한 값 포함      |
| 하이퍼파라미터 튜닝 불가 시              | 랜덤 포레스트              | 기본 설정으로 안정적 성능                |
| 민감도 분석                              | LBC-Net + RF + GB 병행     | 세 방법 비교로 강건성 확인               |

※ 이 코드는 교육 목적의 예제입니다. 실제 정책 분석에서는 데이터 규모와 교란 구조에 따라 방법을 선택해야 합니다.
*전체 코드는 practice/chapter03/code/3-5-comprehensive-comparison.py 참고*

---

## 실습 후 제출 과제

이번 주 과제물은 수업 중 실행한 실습 결과만 제출한다.

- 실습 결과 화면 또는 결과 파일을 제출한다.
- 별도의 해석 보고서와 추가 분석은 제출하지 않는다.
- 제출 여부로 수업 중 실습 실시 여부를 확인한다.

---

## 📝 핵심 정리

| 방법                | 핵심 아이디어                          | 장점                       | 한계                     |
| ------------------- | -------------------------------------- | -------------------------- | ------------------------ |
| 로지스틱 회귀 PSM   | 직선(선형) 모델로 성향점수 추정        | 빠르고 해석 쉬움           | 비선형 관계를 못 잡음    |
| 랜덤 포레스트 PSM   | 여러 나무의 평균으로 곡선 관계 포착    | 비선형 자동 학습, 안정적   | 균형 최적화가 간접적     |
| 그래디언트 부스팅 PSM | 오답노트식으로 계속 개선               | 균형 기준 최적화 가능      | 하이퍼파라미터 민감      |
| LBC-Net (딥러닝)    | 균형+보정을 손실 함수에 직접 포함      | 가장 정확한 효과 추정      | 학습 시간 길고 구현 복잡 |
| CBPS/균형 최적화    | 예측과 균형을 동시에 최적화            | 트레이드오프 조절 가능     | 최적 λ 선택 필요         |

---

## 참고문헌

Athey, S., Imbens, G. W., & Wager, S. (2018). Approximate residual balancing: Debiased inference of average treatment effects in high dimensions. *Journal of the Royal Statistical Society Series B*, 80(4), 597-623.

Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research*, 46(3), 399-424.

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189-1232.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.

Hirano, K., & Imbens, G. W. (2004). The propensity score with continuous treatments. In A. Gelman & X.-L. Meng (Eds.), *Applied Bayesian modeling and causal inference from incomplete-data perspectives* (pp. 73-84). Wiley.

Imai, K., & Ratkovic, M. (2014). Covariate balancing propensity score. *Journal of the Royal Statistical Society Series B*, 76(1), 243-263.

Johansson, F., Shalit, U., & Sontag, D. (2016). Learning representations for counterfactual inference. *Proceedings of the 33rd International Conference on Machine Learning*, PMLR 48, 3020-3029.

Kallus, N. (2020). Deepmatch: Balancing deep covariate representations for causal inference using adversarial training. *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 5067-5077.

Lee, B. K., Lessler, J., & Stuart, E. A. (2010). Improving propensity score weighting using machine learning. *Statistics in Medicine*, 29(3), 337-346.

McCaffrey, D. F., Ridgeway, G., & Morral, A. R. (2004). Propensity score estimation with boosted regression for evaluating causal effects in observational studies. *Psychological Methods*, 9(4), 403-425.

Peng, M., Wu, C., Li, L., & Diao, G. (2024). A Deep Learning Approach to Nonparametric Propensity Score Estimation with Optimized Covariate Balance. arXiv preprint arXiv:2404.04794.

Pirracchio, R., Petersen, M. L., & van der Laan, M. (2015). Improving propensity score estimators' robustness to model misspecification using super learner. *American Journal of Epidemiology*, 181(2), 108-119.

Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.

Setoguchi, S., Schneeweiss, S., Brookhart, M. A., Glynn, R. J., & Cook, E. F. (2008). Evaluating uses of data mining techniques in propensity score estimation: A simulation study. *Pharmacoepidemiology and Drug Safety*, 17(6), 546-555.

Shalit, U., Johansson, F. D., & Sontag, D. (2017). Estimating individual treatment effect: Generalization bounds and algorithms. *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 3076-3085.

Wang, Y., & Zubizarreta, J. R. (2020). Minimal dispersion approximately balancing weights: Asymptotic properties and practical considerations. *Biometrika*, 107(1), 93-105.

Zubizarreta, J. R. (2015). Stable weights that balance covariates for estimation with incomplete outcome data. *Journal of the American Statistical Association*, 110(511), 910-922.
