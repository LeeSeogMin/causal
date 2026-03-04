"""
제10장: BERT vs GPT 비교 (간소화 버전)
=======================================

목적: BERT와 GPT 계열 모델의 특성 비교
- BERT: 양방향 인코더 (분류, NER에 적합)
- GPT: 단방향 디코더 (생성에 적합)

주의: 실제 모델 실행은 transformers 라이브러리 필요
본 파일은 개념 시연용 간소화 버전

저자: AI 기반 정책분석방법론
날짜: 2025-11-20
"""

import numpy as np
import pandas as pd

np.random.seed(42)

print("=" * 80)
print("BERT vs GPT 계열 모델 비교")
print("=" * 80)

# 1. 모델 특성 비교
print("\n[1단계] 모델 아키텍처 비교")
print("-" * 80)

model_comparison = pd.DataFrame({
    '특성': ['아키텍처', '학습 방식', '입력 처리', '주요 용도', '문맥 이해', '한국어 모델'],
    'BERT': [
        'Encoder-only',
        'MLM (Masked Language Model)',
        '양방향 (Bidirectional)',
        '분류, NER, QA',
        '양방향 문맥',
        'KoBERT, KLUE BERT'
    ],
    'GPT': [
        'Decoder-only', 
        'Auto-regressive',
        '단방향 (Left-to-right)',
        '텍스트 생성, 요약',
        '이전 문맥만',
        'KoGPT, HyperCLOVA'
    ]
})

print(model_comparison.to_string(index=False))

# 2. 성능 비교 (시뮬레이션)
print("\n[2단계] 태스크별 성능 비교 (가상 데이터)")
print("-" * 80)

task_performance = pd.DataFrame({
    '태스크': ['정책 문서 분류', '감정 분석', '개체명 인식', '요약 생성', '질의응답'],
    'BERT 정확도': ['88.5%', '85.2%', '87.8%', 'N/A', '85.2%'],
    'GPT 정확도': ['85.7%', '82.1%', 'N/A', '90.2%', '83.5%'],
    '적합 모델': ['BERT', 'BERT', 'BERT', 'GPT', 'BERT']
})

print(task_performance.to_string(index=False))

# 3. 컨텍스트 윈도우 비교
print("\n[3단계] 컨텍스트 윈도우 비교")
print("-" * 80)

context_comparison = pd.DataFrame({
    '모델': ['BERT-base', 'BERT-large', 'GPT-3.5', 'GPT-4', 'GPT-4 Turbo'],
    '최대 토큰': [512, 512, 4096, 8192, 128000],
    '페이지 환산': ['약 1페이지', '약 1페이지', '약 8페이지', '약 16페이지', '약 100페이지']
})

print(context_comparison.to_string(index=False))

# 4. 정책 분석 적용 사례
print("\n[4단계] 정책 분석 적용 사례")
print("-" * 80)

use_cases = {
    'BERT 활용': [
        '✓ 정책 문서 자동 분류 (10개 카테고리)',
        '✓ 시민 의견 감정 분석 (긍정/부정/중립)',
        '✓ 정책 주체/대상 추출 (NER)',
        '✓ 정책 간 유사도 계산'
    ],
    'GPT 활용': [
        '✓ 정책 보고서 자동 요약',
        '✓ 민원 답변 초안 생성',
        '✓ 정책 문서 Q&A',
        '✓ 다국어 정책 문서 번역'
    ]
}

print("\n** BERT 활용 사례 **")
for case in use_cases['BERT 활용']:
    print(f"  {case}")

print("\n** GPT 활용 사례 **")
for case in use_cases['GPT 활용']:
    print(f"  {case}")

# 5. 실제 구현 예시 (개념적)
print("\n[5단계] 실제 구현 개념")
print("-" * 80)

print("""
# BERT 기반 정책 문서 분류 (의사코드)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=10)

# 정책 문서 분류
inputs = tokenizer(policy_text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predicted_category = outputs.logits.argmax(dim=-1)

# GPT 기반 정책 요약 (의사코드)
from transformers import pipeline

summarizer = pipeline("summarization", model="lcw99/t5-base-korean-text-summary")
summary = summarizer(policy_document, max_length=150, min_length=50)
""")

# 6. 장단점 비교
print("\n[6단계] 장단점 비교")
print("-" * 80)

pros_cons = pd.DataFrame({
    '구분': ['BERT 장점', 'BERT 단점', 'GPT 장점', 'GPT 단점'],
    '설명': [
        '양방향 문맥 이해, 분류 태스크 우수, 빠른 추론',
        '생성 불가, 512 토큰 제한, 긴 문서 처리 어려움',
        '자연스러운 생성, 대용량 컨텍스트, few-shot 학습',
        '편향 위험, 환각 문제, 높은 계산 비용'
    ]
})

print(pros_cons.to_string(index=False))

# 7. 최종 요약
print("\n" + "=" * 80)
print("BERT vs GPT 비교 요약")
print("=" * 80)
print("✓ BERT: 정책 문서 **분류**, 감정 분석, 개체명 인식에 최적")
print("✓ GPT: 정책 보고서 **요약**, 답변 생성, 번역에 최적")
print("✓ 실무에서는 태스크에 따라 적절한 모델 선택 필요")
print("✓ 한국어 정책 분석: KLUE BERT (분류), KoGPT/HyperCLOVA (생성)")
print("=" * 80)

print("\n참고: 실제 모델 실행은 transformers 라이브러리 설치 필요")
print("pip install transformers torch")
