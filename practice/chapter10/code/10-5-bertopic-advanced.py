"""
제10장: BERTopic 기반 고급 토픽 모델링
======================================

목적: 정책 문서 토픽 분석의 흐름을 간소화 버전으로 실습한다
- 형태소 분석 → TF-IDF → K-평균 군집 → 대표 단어 → 정답 대조
- 문서별 토픽 번호, 달별 토픽 비중(동적 토픽)

저자: AI 기반 정책분석방법론
날짜: 2025-11-20

수정 이력
  2026-11-18  단어를 세기 전에 형태소 분석(kiwipiepy)으로 명사만 뽑는다.
    - 문제: 문장을 띄어쓰기로만 잘라 "발전을", "추진한다"처럼 조사·어미가
      붙은 어절을 그대로 세었다.
    - 필요 패키지: pip install kiwipiepy
  2026-11-18  3단계에 정답 주제와의 교차표, 4단계에 달별 문서 수(비중의 분모)를 출력한다.
  2026-11-18  8단계 "동적 토픽 모델링 인사이트"를 지웠다. 계산하지 않고 코드에 직접 적어 둔
    수치("AI 관련 키워드 8개 → 15개", "'디지털 전환' 토픽 비중 18% → 22%" 등)였다.
  2026-11-18  실습 문서를 200건으로 바꿨다: 주제 8개 × 20건 + 주제 없는 문서 40건.
    - 문제: 기존 120건은 모든 문서가 같은 틀 문장 하나를 써서 묶기가 항상 100% 맞았다.
    - 문장 묶음에서 주제 문장 2~3개와 공통 문장 1~2개를 뽑아 섞는다. 날짜는 무작위로 붙인다.
    - 단어 세기(상위 50단어) 대신 TF-IDF(전체 단어)를 쓴다. 상위 50단어가 공통 문장의
      단어로 채워져 묶기가 무너졌기 때문이다(3단계에서 비교 출력).
    - 묶음 수를 8과 10으로 각각 돌려 정답과 대조한다.
  2026-11-18  문서를 1,200건(24개월 × 50건)으로 늘리고, 주제별 비중이 시간에 따라 바뀌게 만들었다.
    - 문제: 날짜를 무작위로 붙여 동적 토픽 그림에 실제 변화가 없었다.
    - 증가(디지털 전환·탄소 중립·R&D 투자), 감소(보건의료·일자리 창출), 유지(사회 안전망·교육 혁신),
      소멸(지역 균형, 2025.07부터 0건)으로 만든다. 주제 없는 문서는 매달 약 10%.
    - 4단계: 분기별 비중을 계산하고, 직선 추세로 2026Q1·Q2 비중을 예측한다.
    - 달마다 기대 건수를 정수로 맞춰 배정한다(뽑기 잡음 없음). 묶음 수는 8과 9를 비교하고 9를 쓴다.
"""

# == RESULTS LOG (auto): 콘솔 출력과 그림을 results/ 에 저장 ==
import sys as _sys
from pathlib import Path as _Path

_results_dir = _Path(__file__).resolve().parent.parent / "results"
_results_dir.mkdir(exist_ok=True)

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for _s in self._streams:
            _s.write(data)
            _s.flush()
    def flush(self):
        for _s in self._streams:
            _s.flush()

_sys.stdout = _Tee(_sys.stdout, open(_results_dir / (_Path(__file__).stem + ".log"), "w", encoding="utf-8"))

# 그림 저장을 results/ 로 돌린다 (강의자료용 diagrams/ 경로는 그대로 둔다)
try:
    from matplotlib.figure import Figure as _Fig
    if not getattr(_Fig, "_results_patched", False):
        _orig_savefig = _Fig.savefig
        def _savefig_to_results(self, fname, *a, **k):
            try:
                _p = _Path(fname)
                if "diagrams" not in [s.lower() for s in _p.parts]:
                    fname = _results_dir / _p.name
                    print(f"[그림 저장 → {fname}]")
            except TypeError:
                pass
            return _orig_savefig(self, fname, *a, **k)
        _Fig.savefig = _savefig_to_results
        _Fig._results_patched = True
except ImportError:
    pass
# == /RESULTS LOG ==

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from kiwipiepy import Kiwi

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 80)
print("BERTopic 기반 고급 토픽 모델링 (간소화 버전)")
print("=" * 80)

# 주의: 실제 BERTopic은 sentence-transformers, umap-learn, hdbscan 필요
# 본 구현은 개념 시연용 간소화 버전

# 1. 가상 정책 문서 생성
print("\n[1단계] 가상 정책 문서 생성")
print("-" * 80)

TOPIC_SENTENCES = {
    '디지털 전환': [
        "행정 서류를 온라인으로 신청할 수 있도록 민원 플랫폼을 개편한다.",
        "공공기관이 보유한 데이터를 개방해 민간이 새로운 서비스를 만들 수 있게 한다.",
        "인공지능 상담원이 야간에도 주민의 질문에 답하도록 시스템을 구축했다.",
        "종이 문서를 전자 문서로 바꾸면서 처리 기간이 짧아졌다.",
        "고령층이 스마트폰 앱을 쉽게 쓰도록 디지털 교육 강좌를 연다.",
        "클라우드로 전산 시스템을 옮겨 유지 비용을 줄인다.",
        "개인정보가 새지 않도록 데이터 접근 권한을 엄격하게 관리한다.",
        "AI 기반 분석으로 민원이 많은 지역을 미리 찾아낸다.",
    ],
    '탄소 중립': [
        "공공건물 옥상에 태양광 패널을 설치해 전력 사용량을 낮춘다.",
        "노후 경유차를 전기차로 바꾸는 시민에게 보조금을 준다.",
        "2050년까지 온실가스 배출을 0으로 만드는 계획을 세웠다.",
        "공장의 탄소 배출량을 측정하고 기준을 넘으면 부담금을 물린다.",
        "도심 곳곳에 숲을 조성해 여름철 기온을 떨어뜨린다.",
        "재생에너지 발전소를 늘려 석탄 발전 비중을 줄인다.",
        "일회용품 사용을 줄이는 매장에는 인증 표시를 붙여 준다.",
        "건물의 단열을 보강하면 난방에 드는 에너지가 적게 든다.",
    ],
    '사회 안전망': [
        "혼자 사는 노인을 방문해 건강과 생활 형편을 살핀다.",
        "갑자기 소득이 끊긴 가구에 긴급 생계비를 지급한다.",
        "장애인 활동 지원 시간을 늘려 가족의 돌봄 부담을 덜어 준다.",
        "기초생활 수급 기준을 완화해 사각지대에 놓인 가구를 찾아낸다.",
        "아동 학대가 의심되면 전담 공무원이 즉시 현장에 나간다.",
        "취약계층 가구의 난방비와 전기료 일부를 지원한다.",
        "돌봄이 필요한 가정에 요양보호사를 보내 준다.",
        "복지 상담 창구를 하나로 합쳐 여러 번 찾아오지 않아도 되게 했다.",
    ],
    '교육 혁신': [
        "학생마다 학습 속도가 달라 수준별 수업을 확대한다.",
        "교사가 수업 자료를 공유하는 온라인 공간을 만든다.",
        "방과 후에 아이들이 머물며 공부할 수 있는 교실을 늘린다.",
        "기초 학력이 부족한 학생에게 보충 수업을 제공한다.",
        "진로 체험 프로그램으로 학생들이 다양한 직업을 경험하게 한다.",
        "대학 입시 위주의 수업을 토론과 프로젝트 중심으로 바꾼다.",
        "학교 급식에 지역 농산물을 쓰고 영양 교육을 함께 한다.",
        "교원 연수를 늘려 새로운 교육 과정을 가르칠 수 있게 돕는다.",
    ],
    '지역 균형': [
        "수도권에 몰린 공공기관을 지방으로 옮긴다.",
        "인구가 줄어드는 군 지역에 정착 지원금을 준다.",
        "농촌과 도시를 잇는 버스 노선을 새로 만든다.",
        "지방 대학과 지역 기업이 함께 인재를 키우는 사업을 벌인다.",
        "낙후된 구도심을 재생해 상권을 되살린다.",
        "지역마다 다른 특산물을 브랜드로 만들어 판로를 넓힌다.",
        "지방 의료기관이 부족한 곳에 공공병원을 세운다.",
        "청년이 고향에 돌아와 창업하면 임대료를 깎아 준다.",
    ],
    '보건의료': [
        "독감 예방접종을 어르신에게 무료로 놓아 준다.",
        "응급실이 붐비지 않도록 야간 진료 병원을 지정한다.",
        "만성질환자가 집에서 혈압과 혈당을 측정하고 의사와 상담한다.",
        "감염병이 퍼지면 역학 조사반이 접촉자를 추적한다.",
        "정신 건강 상담 센터를 늘려 우울증 치료를 돕는다.",
        "의료비가 많이 나온 가구에 본인 부담금을 돌려준다.",
        "건강 검진 결과가 나쁘면 보건소가 생활 습관 상담을 한다.",
        "의사가 부족한 지역에 공중보건의를 배치한다.",
    ],
    '일자리 창출': [
        "청년을 새로 채용한 중소기업에 인건비를 지원한다.",
        "실직자가 새 기술을 배우는 직업 훈련 과정을 연다.",
        "창업 초기 기업에 사무 공간과 컨설팅을 제공한다.",
        "고용센터에서 구직자와 기업을 직접 연결해 준다.",
        "경력이 끊긴 여성이 다시 일할 수 있도록 맞춤형 교육을 한다.",
        "공공 일자리를 만들어 취업이 어려운 노인을 고용한다.",
        "근로 시간을 줄이면서 일자리를 나누는 기업을 지원한다.",
        "취업 박람회를 열어 지역 기업의 채용을 돕는다.",
    ],
    'R&D 투자': [
        "반도체 핵심 기술 개발에 연구비를 집중 투자한다.",
        "대학 연구실의 성과가 창업으로 이어지도록 기술 이전을 돕는다.",
        "젊은 연구자가 안정적으로 연구하도록 장기 과제를 늘린다.",
        "기업 부설 연구소의 연구 개발비에 세금을 감면한다.",
        "실패 가능성이 큰 도전적 연구에도 예산을 배정한다.",
        "국가 연구 장비를 여러 기관이 함께 쓰도록 개방한다.",
        "특허 출원을 돕고 기술 사업화 자금을 빌려준다.",
        "기초 과학 연구에 대한 정부 투자를 매년 늘린다.",
    ],
}

# 주제와 상관없이 정책 문서에 흔히 붙는 문장 (모든 주제가 공유한다)
COMMON_SENTENCES = [
    "정부는 관계 부처와 협의해 이 정책을 추진한다.",
    "올해 예산은 작년보다 늘어났다.",
    "자세한 내용은 담당 부서에 문의하면 된다.",
    "사업 성과는 연말에 평가해 공개한다.",
    "시민 의견을 듣기 위해 공청회를 연다.",
    "관련 법령을 개정해 제도적 근거를 마련한다.",
]

# 어느 주제에도 해당하지 않는 문서의 문장
OFF_TOPIC_SENTENCES = [
    "청사 주차장은 이번 주말 도색 공사로 이용할 수 없다.",
    "다음 달 첫째 주에 직원 체육대회가 열린다.",
    "구내식당 운영 시간이 오전 11시 반으로 바뀐다.",
    "인사 발령에 따라 일부 부서의 전화번호가 달라졌다.",
    "엘리베이터 정기 점검으로 오후에 운행을 멈춘다.",
    "회의실 예약은 내부 게시판에서 하면 된다.",
    "분실물은 1층 안내 데스크에 보관하고 있다.",
    "정전에 대비해 비상 발전기를 시험 가동한다.",
    "새해 시무식은 대강당에서 오전 9시에 시작한다.",
    "보안을 위해 출입증을 반드시 패용해야 한다.",
    "사무실 냉방 온도는 26도로 맞춘다.",
    "문서 보존 기간이 지난 서류를 일괄 폐기한다.",
]


# 달마다 주제별 문서 비중이 달라지도록 만든다 (2024.01~2025.12, 24개월 × 50건)
#   증가: 디지털 전환, 탄소 중립, R&D 투자 / 감소: 보건의료, 일자리 창출
#   유지: 사회 안전망, 교육 혁신 / 소멸: 지역 균형(2025.07부터 0건)
N_MONTHS, DOCS_PER_MONTH, OFF_TOPIC_SHARE = 24, 50, 0.10


def topic_weights(t):
    s = t / (N_MONTHS - 1)                     # 0(첫 달) ~ 1(마지막 달)
    return {
        '디지털 전환': 6 + 16 * s,
        '탄소 중립': 8 + 10 * s,
        '사회 안전망': 12,
        '교육 혁신': 11,
        '지역 균형': 14 * (1 - t / 17) if t < 18 else 0,
        '보건의료': 16 - 10 * s,
        '일자리 창출': 13 - 3 * s,
        'R&D 투자': 8 + 4 * s,
    }


def make_document(topic):
    if topic == '주제 없음':
        sents = list(rng.choice(OFF_TOPIC_SENTENCES, size=rng.integers(2, 4), replace=False))
        if rng.random() < 0.5:
            sents.append(rng.choice(COMMON_SENTENCES))
    else:
        pool = TOPIC_SENTENCES[topic]
        sents = list(rng.choice(pool, size=rng.integers(2, 4), replace=False))            # 주제 문장 2~3개
        sents += list(rng.choice(COMMON_SENTENCES, size=rng.integers(1, 3), replace=False))  # 공통 문장 1~2개
    rng.shuffle(sents)
    return ' '.join(sents)


rows = []
for t in range(N_MONTHS):
    w = topic_weights(t)
    names = list(w) + ['주제 없음']
    probs = np.array(list(w.values())) / sum(w.values()) * (1 - OFF_TOPIC_SHARE)
    probs = np.append(probs, OFF_TOPIC_SHARE)
    # 기대 건수를 정수로 맞춘다(소수점이 큰 순서로 1건씩 더 준다). 뽑기 잡음 없이 설계한 추세가 그대로 나온다
    expected = probs * DOCS_PER_MONTH
    counts = np.floor(expected).astype(int)
    counts[np.argsort(-(expected - counts))[:DOCS_PER_MONTH - counts.sum()]] += 1
    stamp = f"{2024 + t // 12}.{t % 12 + 1:02d}"
    for name, k in zip(names, counts):
        for _ in range(k):
            rows.append((make_document(name), name, stamp))

df = pd.DataFrame(rows, columns=['document', 'true_topic', 'timestamp'])

print(f"총 문서 수: {len(df)}")
print("정답 주제별 문서 수: " + ", ".join(f"{k} {v}" for k, v in df['true_topic'].value_counts().items()))
print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print(f"예({df['true_topic'][0]}): {df['document'][0]}")
off_example = df.index[df['true_topic'] == '주제 없음'][0]
print(f"예(주제 없음): {df['document'][off_example]}")

# 2. 형태소 분석과 수치화
print("\n[2단계] 형태소 분석과 수치화")
print("-" * 80)

# 형태소 분석: 문장을 형태소로 나누고 명사(일반명사 NNG, 고유명사 NNP, 영문 SL)만 남긴다
kiwi = Kiwi()
NOUN_TAGS = {'NNG', 'NNP', 'SL'}


def extract_nouns(text):
    return ' '.join(t.form for t in kiwi.tokenize(text) if t.tag in NOUN_TAGS)


df['nouns'] = df['document'].apply(extract_nouns)
print(f"원문    : {df['document'][0]}")
print(f"명사만  : {df['nouns'][0]}")

# TF-IDF: 많은 문서에 나오는 단어(공통 문장의 단어)의 비중을 낮춘다
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(df['nouns'])
feature_names = vectorizer.get_feature_names_out()
print(f"서로 다른 명사 수: {len(feature_names)}")


def check_against_truth(labels, title):
    """묶음 번호와 정답 주제를 교차표로 대조하고, 섞인 묶음 수를 센다."""
    table = pd.crosstab(df['true_topic'], labels + 1, rownames=['정답 주제'], colnames=['토픽'])
    topic_rows = table.drop('주제 없음')
    mixed = [c for c in table.columns if (topic_rows[c] > 0).sum() > 1]
    off_only = [c for c in table.columns if topic_rows[c].sum() == 0]
    off_mixed = [c for c in table.columns if topic_rows[c].sum() > 0 and table.loc['주제 없음', c] > 0]
    print(f"\n{title} — 정답 주제와 대조 (행: 정답 주제, 열: 토픽 번호, 칸: 문서 수)")
    print(table.to_string())
    print(f"두 주제 이상이 섞인 토픽: {mixed if mixed else '없음'}")
    print(f"주제 없는 문서만 모인 토픽: {off_only if off_only else '없음'}")
    print(f"주제 문서와 주제 없는 문서가 섞인 토픽: {off_mixed if off_mixed else '없음'}")


def top_keywords(model, n=5):
    return {c: [feature_names[i] for i in model.cluster_centers_[c].argsort()[-n:][::-1]]
            for c in range(model.n_clusters)}


# 3. 묶기, 대표 단어, 정답 대조
print("\n[3단계] 묶기와 대표 단어, 정답 대조")
print("-" * 80)

# 참고: 단어 세기 상위 50단어로 수치화하면 무엇이 남는가
count50 = CountVectorizer(max_features=50).fit(df['nouns'])
freq50 = np.asarray(count50.transform(df['nouns']).sum(axis=0)).ravel()
top15 = [count50.get_feature_names_out()[i] for i in freq50.argsort()[::-1][:15]]
print("단어 세기 상위 50단어 중 많이 나온 15개: " + ", ".join(top15))
labels_c50 = KMeans(n_clusters=10, random_state=SEED, n_init=10).fit_predict(count50.transform(df['nouns']))
table_c50 = pd.crosstab(df['true_topic'], labels_c50).drop('주제 없음')
print(f"단어 세기 상위 50단어로 10묶음: 정답 주제 문서 {table_c50.values.sum()}건 중 묶음별 다수 주제와 같은 문서 "
      f"{table_c50.max(axis=0).sum()}건")

for n_topics in (8, 9):
    model = KMeans(n_clusters=n_topics, random_state=SEED, n_init=10).fit(doc_vectors)
    keywords = top_keywords(model)
    print(f"\n--- 묶음 수 {n_topics} ---")
    for c in range(n_topics):
        print(f"토픽 {c + 1} ({(model.labels_ == c).sum()}개 문서): {', '.join(keywords[c])}")
    check_against_truth(model.labels_, f"묶음 수 {n_topics}")

# 이후 단계는 묶음 수 9의 결과를 쓴다
n_topics = 9
kmeans = KMeans(n_clusters=n_topics, random_state=SEED, n_init=10).fit(doc_vectors)
df['predicted_topic'] = kmeans.labels_
topics_keywords = top_keywords(kmeans)

# 4. 시간적 토픽 변화 (동적 토픽)
print(f"\n[4단계] 분기별 토픽 비중과 다음 두 분기 예측 (묶음 수 {n_topics})")
print("-" * 80)

# 달별 문서는 50건 안팎이라 한 달 단위 비중은 출렁인다. 분기(3개월)로 묶어 분모를 키운다
df['quarter'] = [f"{ts[:4]}Q{(int(ts[5:7]) - 1) // 3 + 1}" for ts in df['timestamp']]
q_counts = df.groupby(['quarter', 'predicted_topic']).size().unstack(fill_value=0)
q_counts = q_counts.reindex(columns=range(n_topics), fill_value=0).sort_index()
q_totals = q_counts.sum(axis=1)
q_share = q_counts.div(q_totals, axis=0) * 100
print("분기별 문서 수(비중의 분모): " + ", ".join(f"{q} {n}건" for q, n in q_totals.items()))

# 직선 추세: 분기 번호(0~7)에 비중을 직선으로 맞추고, 같은 직선을 두 분기 더 늘려 예측한다
x = np.arange(len(q_share))
future = ['2026Q1', '2026Q2']
x_future = np.arange(len(q_share), len(q_share) + len(future))
trend = {}
print(f"\n{'토픽':<22}{'첫 분기':>8}{'마지막 분기':>10}{'분기당 변화':>11}{'2026Q1 예측':>12}{'2026Q2 예측':>12}  판정")
for c in range(n_topics):
    y = q_share[c].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    pred = np.clip(intercept + slope * x_future, 0, 100)
    if y[0] >= 5 and y[-2:].max() < 1:
        verdict = '소멸'
    elif slope >= 0.5:
        verdict = '증가'
    elif slope <= -0.5:
        verdict = '감소'
    else:
        verdict = '유지'
    trend[c] = (slope, pred, verdict)
    name = f"토픽 {c + 1} ({', '.join(topics_keywords[c][:2])})"
    print(f"{name:<22}{y[0]:>7.1f}%{y[-1]:>9.1f}%{slope:>+9.2f}%p{pred[0]:>11.1f}%{pred[1]:>11.1f}%  {verdict}")

# 동적 토픽 변화 그래프: 분기별 실제 비중(실선)과 예측(점선)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 8))
colors = plt.cm.tab10(range(n_topics))
labels_x = list(q_share.index) + future
for c in range(n_topics):
    slope, pred, verdict = trend[c]
    ax.plot(x, q_share[c], marker='o', linewidth=2, color=colors[c],
            label=f"토픽 {c + 1}: {', '.join(topics_keywords[c][:2])} ({verdict})")
    ax.plot([x[-1]] + list(x_future), [q_share[c].iloc[-1]] + list(pred), linestyle='--',
            linewidth=1.5, color=colors[c])
ax.axvspan(x[-1] + 0.5, x_future[-1] + 0.3, color='gray', alpha=0.1)
ax.text(x_future.mean(), ax.get_ylim()[1] * 0.95, '예측', ha='center', fontsize=11)
ax.set_xticks(list(x) + list(x_future))
ax.set_xticklabels(labels_x, rotation=45)
ax.set_xlabel('분기', fontsize=12)
ax.set_ylabel('토픽 비중 (%)', fontsize=12)
ax.set_title('동적 토픽 변화 추이 (Topics Over Time)', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../../diagrams/10-bertopic-topics-over-time.png', dpi=300, bbox_inches='tight')
print("\n✓ 그래프 저장: diagrams/10-bertopic-topics-over-time.png")

# 5. 토픽 간 관계 분석
print(f"\n[5단계] 토픽 간 유사도 (묶음 수 {n_topics})")
print("-" * 80)

topic_similarity = cosine_similarity(kmeans.cluster_centers_)
iu = np.triu_indices(n_topics, 1)
print(f"토픽 {len(iu[0])}쌍의 유사도: 최소 {topic_similarity[iu].min():.3f} ~ 최대 {topic_similarity[iu].max():.3f}")
i_max, j_max = iu[0][topic_similarity[iu].argmax()], iu[1][topic_similarity[iu].argmax()]
print(f"가장 비슷한 쌍: 토픽 {i_max + 1} ({', '.join(topics_keywords[i_max][:2])}) <-> "
      f"토픽 {j_max + 1} ({', '.join(topics_keywords[j_max][:2])}): {topic_similarity[i_max, j_max]:.3f}")

# 6. 문서별 토픽 할당
print(f"\n[6단계] 문서별 토픽 할당 예시 (묶음 수 {n_topics})")
print("-" * 80)

for idx in [0, 20, len(df) - 1]:
    row = df.loc[idx]
    t = row['predicted_topic']
    print(f"문서 {idx + 1} (정답: {row['true_topic']}): {row['document'][:45]}...")
    print(f"  → 토픽 {t + 1} ({', '.join(topics_keywords[t][:3])})")

# 7. BERTopic과 LDA 비교
print("\n[7단계] BERTopic vs LDA 비교")
print("-" * 80)

comparison = pd.DataFrame({
    '특성': ['문맥 이해', '토픽 수 결정', '클러스터 품질', '계산 비용', '해석 용이성'],
    'LDA': ['단어 빈도만', '사전 지정 필요', '중간', '낮음', '높음'],
    'BERTopic': ['의미적 이해', '자동 결정', '높음', '높음', '중간']
})
print(comparison.to_string(index=False))

# 8. 최종 요약
print("\n" + "=" * 80)
print(f"토픽 모델링 최종 결과 (묶음 수 {n_topics})")
print("=" * 80)
print(f"분석 문서 수: {len(df)}개")
for c in range(n_topics):
    print(f"  토픽 {c + 1} ({(df['predicted_topic'] == c).sum()}개): {', '.join(topics_keywords[c][:3])}")

print("\n참고: 실제 BERTopic 구현은 다음 라이브러리 필요:")
print("pip install bertopic sentence-transformers umap-learn hdbscan")
print("=" * 80)
