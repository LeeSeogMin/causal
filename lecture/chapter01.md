# 1장. 의사결정 과학과 AI 기반 인과 데이터 사이언스

**🎯 학습 목표: AI와 데이터가 의사결정을 어떻게 바꾸는지 이해하고, 인과적 데이터 사이언스의 기본 개념과 분석 프레임워크를 익히기**

---

## 🌟 이 장에서 배울 내용 미리보기

- **인과적 데이터 사이언스란**: "무슨 일이 일어날까?" 예측을 넘어 "어떻게 해야 원하는 결과를 얻을까?"에 답하는 학문
- **에스토니아와 한국 사례**: 디지털 정부와 AI 기반 의사결정의 실제 모습
- **5단계 프레임워크**: 문제 정의 → 대안 설계 → 의사결정 → 적응형 실행 → 지속적 평가
- **전통 vs AI 방법론**: 각각 언제, 왜 쓰는지

---

## 1.1 AI 시대의 인과적 데이터 사이언스 🔍

### 1.1.1 인과추론과 AI의 결합

**인과 데이터 사이언스(Causal Data Science)**란 단순 예측을 넘어 **"무엇을 해야 원하는 결과를 얻을 수 있는가?"**라는 인과적 질문에 답하는 의사결정 과학이다(Hernán & Robins, 2020).

**💡 예측과 인과의 차이:**

```
예측: "이 고객이 이탈할 확률은 70%다"
     → 알겠는데, 그래서 어떻게 해야 하나?

인과: "쿠폰을 보내면 이탈 확률이 70%에서 30%로 줄어든다"
     → 구체적으로 무엇을 해야 하는지 알 수 있다!
```

이 학문이 작동하려면 세 가지 기반이 필요하다:

1. **데이터 기반 분석**: 행정 데이터, 소셜 미디어, IoT 센서 등 대규모 데이터를 통합 활용
2. **AI 기반 설계와 예측**: 머신러닝과 시뮬레이션으로 대안을 비교하고 최적 방안 도출
3. **실시간 평가와 적응적 학습**: 효과를 즉시 측정하고, 전략이 스스로 학습하며 진화

### 1.1.2 의사결정 프레임워크의 진화

전통적으로 의사결정은 "계획 → 실행 → 나중에 평가"의 순서였다. 인과적 데이터 사이언스에서는 실행하면서 동시에 평가하고, 그 결과를 바로 반영하는 **순환 구조**로 바뀐다.

![인과적 데이터 사이언스 순환 모델의 핵심 구성요소](../diagrams/1.png)

이 모델의 핵심 특징 세 가지:

- **실시간 적응성**: 교통 데이터를 실시간 분석해서 신호를 동적으로 조정하는 것처럼, 효과를 즉시 측정하고 전략을 바로 조정
- **예측적 의사결정**: 단순 추세 분석을 넘어, 인과관계를 파악하고 개입 결과를 사전 시뮬레이션
- **다중 이해관계자 통합**: 시민, 기업, 정부 등 각 이해관계자별로 효과가 어떻게 달라지는지 분석하여 의사결정에 반영

### 1.1.3 에스토니아 전자정부 시스템

에스토니아는 인구 130만의 작은 나라이지만, 세계에서 가장 발전된 디지털 정부를 구축했다. 1991년 독립 이후 30년간의 디지털 투자로, 정부 서비스의 **99%가 온라인**으로 제공된다(e-Estonia, 2024a).

**핵심 시스템들:**

- **X-Road**: 정부 기관 간 데이터를 안전하게 주고받는 플랫폼. 연간 22억 건 이상의 데이터 교환, 약 1,345년 분의 근무 시간을 절감(e-Estonia, 2024b)
- **Bürokratt**: AI 가상 공무원. 시민 문의를 자동 처리하고, 복잡한 문제는 담당자에게 연결
- **디지털 ID**: 99% 보급률. 모든 정부 서비스에 원스톱 접근 가능

### 1.1.4 한국의 데이터 기반 의사결정 현황

한국은 2024년 UN 전자정부 평가에서 193개국 중 **4위**를 기록했다(United Nations, 2024). 에스토니아가 2위인 가운데, 한국은 온라인 서비스와 통신 인프라에서 세계 최고 수준이다.

**주요 성과:**

- **정부24**: 중앙부처·지자체 서비스를 통합한 원스톱 플랫폼. AI 챗봇 '구삐'가 민원 처리
- **공공데이터포털**: 10만 개 이상의 데이터를 개방. 코로나19 때 마스크 재고 실시간 제공이 세계적 모범 사례로 인정
- **AI 정부 추진**: 예산 배분 최적화, 복지 사각지대 발굴, 재난 예측 등에 AI 적극 도입 중

---

## 1.2 인과적 데이터 사이언스 프레임워크 📋

### 1.2.1 전통적 분석·평가 모델의 한계

전통적 분석 방법은 "문제 정의 → 대안 설계 → 의사결정 → 실행 → 평가"를 순서대로 밟는 **선형 모델**이다. 이 방식은 반세기 넘게 잘 쓰였지만, 현대 환경에서 세 가지 근본적 한계를 보인다:

1. **시간 지연 문제**: 평가가 실행 후 한참 뒤에야 이루어져, 실패를 조기에 수정하기 어려움
2. **정보 비대칭**: 의사결정자, 실행자, 수혜자 간 정보 격차로 현장 상황이 제대로 반영되지 않음
3. **단절적 피드백**: 피드백이 평가 단계에서만 발생해서, 실행 중 문제를 즉시 해결하기 불가능

### 1.2.2 인과적 데이터 사이언스의 5단계 구조

이러한 한계를 극복하기 위해, 인과적 데이터 사이언스는 **동적이고 순환적인** 5단계 구조를 제안한다:

**Stage 1: 지능형 문제 정의** 🔍
- AI가 다양한 데이터에서 문제를 자동 감지하고 우선순위를 매김
- 비즈니스 예시: 고객 이탈 징후를 AI가 경영진보다 먼저 포착

**Stage 2: 데이터 기반 대안 설계** 📐
- 과거 유사 사례 데이터를 ML로 학습하여 최적 설계를 제안
- 시뮬레이션으로 다양한 시나리오의 효과를 사전 검토

**Stage 3: 알고리즘적 의사결정** ⚖️
- 다기준 평가와 최적화 알고리즘으로 최선의 대안 선택
- 다양한 이해관계자의 가중치를 반영

**Stage 4: 적응형 실행** 🔄
- 실시간 반응에 따라 전략을 즉시 조정
- 비즈니스 예시: A/B 테스트 대신 MAB(Multi-Armed Bandit)로 실시간 최적화

**Stage 5: 지속적 평가와 학습** 📊
- **실시간 평가**: 효과를 실행과 동시에 측정
- **인과적 평가**: DML, Causal Forest 등으로 정책의 순수 효과를 정확히 측정
- **적응적 학습**: 평가 결과가 다음 순환의 문제 정의로 즉시 피드백

### 1.2.3 실시간 피드백 루프

이 프레임워크의 핵심은 각 단계가 **실시간 피드백 루프**로 연결된다는 점이다:

- **마이크로 루프**: 각 단계 내에서 짧은 주기로 즉각 조정 (예: 실행 중 이상 징후 감지 → 즉시 수정)
- **매크로 루프**: 전체 순환을 거쳐 장기적으로 전략을 근본 개선
- **크로스 피드백**: 순차적 단계를 건너뛰어, 실행 데이터가 설계 단계로 직접 전달되어 즉각 수정 가능

---

## 1.3 전통적 방법론과 Causal ML의 비교 🔄

### 1.3.1 전통적 방법론의 특징과 한계

비용편익분석, 회귀분석, 설문조사, 델파이 기법 등 전통적 방법론은 분석의 과학화에 크게 기여했지만, 빅데이터 시대에 한계가 드러난다:

- 소규모 표본(수백~수천)에 의존하여 이질적 효과를 포착하기 어려움
- 선형성·정규성 등 강한 가정에 의존하며, 비선형 관계 처리에 한계
- 무작위 실험이 불가능한 상황에서 인과관계 식별에 근본적 제약
- 사후적 일괄 평가 구조로 실시간 적응이 어려움

### 1.3.2 Causal ML 방법론의 혁신

AI 기반 방법론은 이러한 한계를 극복한다. 본 교재에서 다루는 핵심 기법:

- **2장**: DML(이중 머신러닝) — 고차원 교란변수를 ML로 통제하며 처치효과 추정
- **3장**: 성향점수 매칭과 딥러닝 확장
- **4장**: 이중차분법과 합성통제법
- **5장**: 회귀불연속설계
- **6장**: 도구변수법과 DeepIV
- **7장**: 분위회귀를 통한 분포별 효과 분석
- **8장**: RCT와 적응형 시험설계
- **10장**: 자연어처리(NLP)를 활용한 텍스트 분석
- **12장**: 강화학습을 활용한 전략 최적화

<표 1-1: 방법론 비교 분석표>

| 비교 항목        | 전통적 방법론      | Causal ML 방법론               |
| ---------------- | ------------------ | ------------------------------ |
| 데이터 처리 규모 | 수백~수천 개 표본  | 수백만~수십억 개 데이터 포인트 |
| 분석 속도        | 일~주 단위         | 실시간~분 단위                 |
| 비선형성 처리    | 제한적 (변환 필요) | 자동 학습                      |
| 해석가능성       | 높음 (명확한 계수) | 낮음~중간 (블랙박스)           |
| 전문성 요구      | 통계/경제학 지식   | 프로그래밍/ML 지식             |
| 비용             | 낮음~중간          | 초기 높음, 운영 낮음           |
| 정확도*          | 70-80%             | 85-95%                         |
| 적응성           | 정적 모델          | 동적 학습                      |
| 편향 위험        | 분석자 편향        | 데이터/알고리즘 편향           |
| 규제 준수        | 확립된 기준        | 발전 중인 기준                 |

<small>*정확도 수치는 문헌에서 보고된 일반적 경향이며, 실제 성능은 데이터 특성과 문제 유형에 따라 크게 달라질 수 있음(Athey & Imbens, 2019; Stanford HAI, 2024).</small>

### 1.3.3 방법론 선택 원칙

**전통적 방법론이 적합한 경우:**
- 법적·규제적 맥락에서 명확한 근거 제시가 필요할 때
- 데이터가 제한적이거나 표본 크기가 작을 때
- 이해관계자 설득에 직관적 해석이 중요할 때

**AI 기반 방법론이 적합한 경우:**
- 수백만 건 이상의 대규모 데이터를 처리할 때
- 교통 관리, 재난 대응 등 실시간 의사결정이 필수인 영역
- 텍스트, 이미지 등 비정형 데이터에서 복잡한 패턴을 인식해야 할 때

실무에서는 양쪽 방법론을 **상호보완적으로** 활용하는 하이브리드 접근이 최선인 경우가 많다.

---

## 💡 1장 전체 핵심 요약

```
1. 인과적 데이터 사이언스 = "무엇을 해야 원하는 결과를 얻을까?"에 답하는 학문
   → 예측(무슨 일이 일어날까?)을 넘어 최적 개입을 설계

2. 5단계 프레임워크: 문제 정의 → 대안 설계 → 의사결정 → 적응형 실행 → 지속적 평가
   → 선형이 아닌 순환 구조, 실시간 피드백으로 연결

3. 에스토니아(99% 온라인 정부)와 한국(UN 4위)이 디지털 정부의 모범

4. 전통적 방법론: 소규모, 선형, 해석 쉬움 / 한계: 비선형·대규모·실시간 대응 어려움
   AI 방법론: 대규모, 비선형, 적응적 / 한계: 블랙박스, 초기 비용

5. 실무에서는 두 방법론을 상황에 맞게 조합하는 것이 최선
```

---

## 실습 환경 구축 🛠️

이 교재의 실습 코드를 실행하려면 아래 프로그램들을 설치해야 한다. 각 단계를 순서대로 따라하면 된다.

### 사전 준비: 윈도우 사용자 계정 분리 (권장) 🔒

실습실 PC에서 여러 과목을 수강하는 경우, **과목별로 윈도우 로컬 계정을 분리**하면 다른 과목과의 혼선을 확실히 방지할 수 있다. 계정 간에는 바탕화면, 문서 폴더, 프로그램 설정이 완전히 분리되므로, 한 과목에서 설정을 변경해도 다른 과목에는 전혀 영향을 주지 않는다.

**설정 방법:**

1. **설정 > 계정 > 다른 사용자**로 이동한다
2. "계정 추가"를 클릭하고, 과목별 계정을 생성한다 (예: 'AI정책분석', 'B과목' 등)
3. 계정 유형을 반드시 **'표준 사용자'**로 설정한다

**⚠️ 왜 '표준 사용자'여야 하는가?**

표준 사용자로 설정하면 시스템 중요 파일을 삭제하거나 허가 없이 새 프로그램을 설치하는 것이 차단된다. 실습 중 실수로 시스템을 망가뜨리는 사고를 예방할 수 있으므로, 관리자 계정이 아닌 표준 사용자 계정을 사용하는 것이 안전하다.

> **💡 Tip**: 개인 노트북을 사용하는 경우에도, 이 교재 전용 계정을 하나 만들어두면 실습 환경이 깔끔하게 유지된다.

### 1단계: VS Code 설치

VS Code 공식 사이트(https://code.visualstudio.com/)에서 운영체제에 맞는 최신 버전을 다운로드하여 설치한다. 이미 설치되어 있다면 "Help > Check for Updates"로 갱신한다.

설치 후 터미널에서 버전을 확인한다:

```bash
code --version
```

### 2단계: Python 설치

이 교재의 모든 실습 코드는 Python으로 작성되어 있다. Python 3.10 이상이 필요하다.

- **다운로드**: https://www.python.org/
- **macOS**: `brew install python` (Homebrew 사용 시)
- **Windows**: 설치 시 "Add Python to PATH" 체크 필수

설치 확인:

```bash
python3 --version    # 3.10 이상 필요
pip3 --version       # 패키지 관리자 확인
```

### 3단계: Node.js 설치

일부 도구(MCP 서버 등)에 Node.js가 필요하다.

- **다운로드**: https://nodejs.org/ (LTS 버전 권장)
- **macOS**: `brew install node`

설치 확인:

```bash
node --version       # 18 이상 필요
npm --version        # Node.js와 함께 설치됨
```

### 4단계: Git 설치

코드 버전 관리와 GitHub 연동에 필요하다.

- **다운로드**: https://git-scm.com/
- **macOS**: Xcode Command Line Tools에 포함 (`xcode-select --install`)

설치 확인:

```bash
git --version
```

### 5단계: GitHub Copilot 설정 (선택)

AI 코딩 보조 도구를 활용하고 싶다면 GitHub Copilot을 설정한다.

1. GitHub 계정이 없으면 https://github.com 에서 생성
2. 학생은 GitHub Student Developer Pack(https://education.github.com/pack)에 등록하면 Copilot Pro 무료 사용 가능
3. VS Code 확장(Extensions)에서 "GitHub Copilot"과 "GitHub Copilot Chat"을 설치
4. VS Code 좌측 하단 계정 아이콘으로 GitHub 로그인

### 환경 구축 완료 체크리스트

| 항목 | 확인 명령 | 기대 결과 |
|------|----------|----------|
| VS Code | `code --version` | 최신 버전 |
| Python | `python3 --version` | 3.10 이상 |
| Node.js | `node --version` | 18 이상 |
| Git | `git --version` | 설치 확인 |
| pip | `pip3 --version` | 설치 확인 |

### 주요 Python 패키지 설치

실습에 필요한 핵심 패키지들을 한 번에 설치한다:

```bash
pip3 install numpy pandas scipy matplotlib scikit-learn
pip3 install econml xgboost
```

- **numpy, pandas**: 데이터 처리
- **scipy**: 통계 함수
- **matplotlib**: 시각화
- **scikit-learn**: 머신러닝 기본
- **econml**: Microsoft의 인과 머신러닝 패키지 (2장부터 사용)
- **xgboost**: Gradient Boosting 모형 (2장부터 사용)

---

## 참고문헌

과학기술정보통신부. (2021). 신뢰할 수 있는 인공지능 실현 전략. 정부간행물.

과학기술정보통신부. (2024). 국가인공지능 전략 정책방향. 정책보고서.

대한민국 국회. (2025). 인공지능 발전과 신뢰 기반 조성 등에 관한 기본법. 법률 제20676호.

행정안전부. (2024). 정부24 및 공공데이터 개방 현황. https://www.gov.kr; https://www.data.go.kr

Athey, S., & Imbens, G. W. (2019). Machine Learning Methods That Economists Should Know About. *Annual Review of Economics*, 11, 685-725.

Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning*. MIT Press.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.

e-Estonia. (2024a). Facts and Figures. Retrieved from https://e-estonia.com/facts-and-figures/

e-Estonia. (2024b). X-Road: Interoperability Services. Retrieved from https://e-estonia.com/solutions/interoperability-services/x-road/

European Union. (2024). Regulation on Artificial Intelligence (AI Act). *Official Journal of the EU*.

Government of Estonia. (2019). National AI Strategy (Kratt Strategy). Retrieved from https://e-estonia.com/new-e-estonia-factsheet-national-ai-kratt-strategy/

Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Boca Raton: Chapman & Hall/CRC.

Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.

Kleinberg, J., Ludwig, J., Mullainathan, S., & Obermeyer, Z. (2015). Prediction Policy Problems. *American Economic Review*, 105(5), 491-495.

Lazer, D., et al. (2009). Computational Social Science. *Science*, 323(5915), 721-723.

Mitchell, M. (2019). *Artificial Intelligence: A Guide for Thinking Humans*. Farrar, Straus and Giroux.

OECD. (2019). *Artificial Intelligence in Society*. OECD Publishing.

Pearl, J., & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books.

Salganik, M.J. (2017). *Bit by Bit: Social Research in the Digital Age*. Princeton University Press.

Stanford HAI. (2024). *AI Index Report 2024*. Stanford University Human-Centered AI Institute.

United Nations. (2024). *UN E-Government Survey 2024*. United Nations DESA.

Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *Journal of the American Statistical Association*, 113(523), 1228-1242.
