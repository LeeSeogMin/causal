"""
제4장: Dynamic BERTopic 데모 - 시간에 따른 토픽 변화 추적
시간 흐름에 따른 민원 토픽 변화를 분석하고 트렌드를 파악
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import random

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class DynamicBERTopicSimulator:
    """Dynamic BERTopic 동작을 시뮬레이션하는 클래스"""
    
    def __init__(self):
        self.monthly_topics = {}
        self.topic_evolution = {}
        
    def generate_monthly_trends(self, start_date='2024-01', periods=12):
        """월별 토픽 트렌드 생성 (시뮬레이션)"""
        
        # 기본 토픽들과 계절성 패턴
        base_topics = {
            '복지': {'base': 25, 'seasonal': [0, 0, 5, 0, 0, 0, 0, 0, 0, 10, 0, 0]},  # 3월, 10월 증가
            '환경': {'base': 20, 'seasonal': [0, 0, 10, 15, 5, 0, 5, 0, 0, 0, 0, 0]}, # 봄철 증가
            '교통': {'base': 20, 'seasonal': [5, 10, 0, 0, 0, 0, 0, 0, 0, 0, 5, 10]}, # 연휴철 증가
            '안전': {'base': 18, 'seasonal': [2, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 2]},  # 여름철 증가
            '행정': {'base': 17, 'seasonal': [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0]}   # 하반기 증가
        }
        
        # 월별 데이터 생성
        for i in range(periods):
            month_date = pd.to_datetime(start_date) + pd.DateOffset(months=i)
            month_str = month_date.strftime('%Y-%m')
            
            monthly_data = {}
            for topic, pattern in base_topics.items():
                base_value = pattern['base']
                seasonal_boost = pattern['seasonal'][i % 12]
                noise = random.randint(-3, 3)  # 랜덤 노이즈
                
                monthly_data[topic] = max(5, base_value + seasonal_boost + noise)
            
            self.monthly_topics[month_str] = monthly_data
        
        return self.monthly_topics
    
    def analyze_topic_evolution(self):
        """토픽 진화 패턴 분석"""
        
        if not self.monthly_topics:
            print("월별 토픽 데이터를 먼저 생성해주세요.")
            return
        
        # 각 토픽의 시간별 변화 추적
        topics = list(next(iter(self.monthly_topics.values())).keys())
        
        for topic in topics:
            values = [self.monthly_topics[month][topic] for month in sorted(self.monthly_topics.keys())]
            
            # 트렌드 분석
            trend = "증가" if values[-1] > values[0] else "감소" if values[-1] < values[0] else "안정"
            volatility = np.std(values)
            peak_month = max(self.monthly_topics.keys(), key=lambda m: self.monthly_topics[m][topic])
            
            self.topic_evolution[topic] = {
                'values': values,
                'trend': trend,
                'volatility': round(volatility, 2),
                'peak_month': peak_month,
                'peak_value': self.monthly_topics[peak_month][topic],
                'avg_value': round(np.mean(values), 1)
            }
        
        return self.topic_evolution
    
    def visualize_dynamic_topics(self):
        """Dynamic 토픽 변화 시각화"""
        
        if not self.monthly_topics:
            print("데이터를 먼저 생성해주세요.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 시간별 토픽 변화 라인 차트
        months = sorted(self.monthly_topics.keys())
        topics = list(next(iter(self.monthly_topics.values())).keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, topic in enumerate(topics):
            values = [self.monthly_topics[month][topic] for month in months]
            ax1.plot(months, values, marker='o', linewidth=2, 
                    label=topic, color=colors[i], markersize=6)
        
        ax1.set_title('월별 토픽 트렌드 변화', fontsize=14, weight='bold')
        ax1.set_xlabel('월')
        ax1.set_ylabel('민원 수')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 토픽별 변동성 (표준편차)
        volatilities = [self.topic_evolution[topic]['volatility'] for topic in topics]
        bars = ax2.bar(topics, volatilities, color=colors, edgecolor='black', alpha=0.7)
        ax2.set_title('토픽별 변동성 (표준편차)', fontsize=14, weight='bold')
        ax2.set_ylabel('변동성')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, vol in zip(bars, volatilities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{vol}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 3. 히트맵 - 월별 토픽 강도
        heatmap_data = []
        for month in months:
            heatmap_data.append([self.monthly_topics[month][topic] for topic in topics])
        
        heatmap_df = pd.DataFrame(heatmap_data, index=months, columns=topics)
        sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax3, cbar_kws={'label': '민원 수'})
        ax3.set_title('월별-토픽별 히트맵', fontsize=14, weight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 토픽 점유율 변화 (첫 달 vs 마지막 달)
        first_month = months[0]
        last_month = months[-1]
        
        first_total = sum(self.monthly_topics[first_month].values())
        last_total = sum(self.monthly_topics[last_month].values())
        
        first_ratios = [self.monthly_topics[first_month][topic]/first_total*100 for topic in topics]
        last_ratios = [self.monthly_topics[last_month][topic]/last_total*100 for topic in topics]
        
        x = np.arange(len(topics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, first_ratios, width, label=f'{first_month}', 
                       color='lightblue', edgecolor='black')
        bars2 = ax4.bar(x + width/2, last_ratios, width, label=f'{last_month}', 
                       color='lightcoral', edgecolor='black')
        
        ax4.set_title('토픽 점유율 변화 (첫 달 vs 마지막 달)', fontsize=14, weight='bold')
        ax4.set_ylabel('점유율 (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(topics, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('output/dynamic_bertopic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def detect_emerging_topics(self):
        """신규 이슈 토픽 감지 (시뮬레이션)"""
        
        # 가상의 신규 이슈 시뮬레이션
        emerging_topics = {
            "2024-03": ["미세먼지", "꽃가루 알레르기"],
            "2024-06": ["폭염 대비", "에어컨 전력"],
            "2024-09": ["태풍 피해", "침수 지역"],
            "2024-12": ["한파 대비", "난방비 지원"]
        }
        
        print("🚨 신규 이슈 토픽 감지:")
        print("=" * 30)
        
        for month, issues in emerging_topics.items():
            print(f"{month}: {', '.join(issues)}")
        
        return emerging_topics
    
    def predict_future_trends(self):
        """미래 토픽 트렌드 예측 (시뮬레이션)"""
        
        if not self.topic_evolution:
            print("토픽 진화 분석을 먼저 실행해주세요.")
            return
        
        predictions = {}
        
        for topic, evolution in self.topic_evolution.items():
            recent_values = evolution['values'][-3:]  # 최근 3개월
            trend_slope = (recent_values[-1] - recent_values[0]) / 3
            
            # 다음 3개월 예측
            future_values = []
            for i in range(1, 4):
                predicted = recent_values[-1] + (trend_slope * i)
                # 현실적인 범위로 제한
                predicted = max(5, min(50, predicted))
                future_values.append(round(predicted, 1))
            
            predictions[topic] = {
                'next_3_months': future_values,
                'trend_direction': '증가' if trend_slope > 0 else '감소' if trend_slope < 0 else '안정',
                'confidence': min(90, max(60, 80 - abs(evolution['volatility']) * 5))
            }
        
        print("\n🔮 향후 3개월 토픽 트렌드 예측:")
        print("=" * 40)
        
        for topic, pred in predictions.items():
            print(f"\n{topic}:")
            print(f"  예측값: {pred['next_3_months']}")
            print(f"  트렌드: {pred['trend_direction']}")
            print(f"  신뢰도: {pred['confidence']}%")
        
        return predictions

def demonstrate_dynamic_bertopic():
    """Dynamic BERTopic 데모 실행"""
    
    print("📈 Dynamic BERTopic 시뮬레이터 데모 시작!")
    print("=" * 50)
    
    # Dynamic BERTopic 시뮬레이터 초기화
    dynamic_sim = DynamicBERTopicSimulator()
    
    # 1. 월별 토픽 트렌드 생성
    print("1️⃣ 월별 토픽 트렌드 생성 중...")
    monthly_topics = dynamic_sim.generate_monthly_trends(start_date='2024-01', periods=12)
    
    print("📊 생성된 월별 데이터 (일부):")
    for i, (month, topics) in enumerate(monthly_topics.items()):
        if i < 3:  # 처음 3개월만 출력
            print(f"  {month}: {topics}")
    
    # 2. 토픽 진화 분석
    print("\n2️⃣ 토픽 진화 패턴 분석 중...")
    topic_evolution = dynamic_sim.analyze_topic_evolution()
    
    print("📈 토픽 진화 분석 결과:")
    for topic, evolution in topic_evolution.items():
        print(f"  {topic}: {evolution['trend']} 트렌드, 변동성 {evolution['volatility']}")
    
    # 3. 시각화
    print("\n3️⃣ Dynamic 토픽 변화 시각화 생성 중...")
    dynamic_sim.visualize_dynamic_topics()
    
    # 4. 신규 이슈 감지
    print("\n4️⃣ 신규 이슈 토픽 감지...")
    emerging_topics = dynamic_sim.detect_emerging_topics()
    
    # 5. 미래 트렌드 예측
    print("\n5️⃣ 미래 토픽 트렌드 예측...")
    predictions = dynamic_sim.predict_future_trends()
    
    return dynamic_sim, monthly_topics, topic_evolution, predictions

def analyze_policy_implications():
    """정책적 시사점 분석"""
    
    implications = {
        "계절별 대응 전략": {
            "봄철 (3-5월)": "환경 관련 민원 급증 → 미세먼지 대책 강화",
            "여름철 (6-8월)": "안전 관련 민원 증가 → 폭염 대비 시설 점검",
            "가을철 (9-11월)": "복지 관련 민원 증가 → 연말 복지 정책 홍보",
            "겨울철 (12-2월)": "교통 관련 민원 증가 → 제설 작업 및 교통 관리"
        },
        "예산 배정 전략": {
            "우선순위 1": "복지 (연중 높은 비중, 안정적 수요)",
            "우선순위 2": "환경 (계절별 변동 큼, 선제적 대응 필요)",
            "우선순위 3": "교통 (연휴철 집중, 탄력적 운영)",
            "우선순위 4": "안전 (여름철 집중, 계절별 대비책)"
        },
        "정책 효과 측정": {
            "단기 효과": "월별 민원 수 변화로 정책 효과 즉시 확인",
            "중기 효과": "계절별 패턴 변화로 정책 정착도 평가",
            "장기 효과": "연도별 트렌드 변화로 구조적 개선 확인"
        }
    }
    
    print("\n💡 Dynamic BERTopic 기반 정책적 시사점:")
    print("=" * 50)
    
    for category, details in implications.items():
        print(f"\n🏷️ {category}:")
        for key, value in details.items():
            print(f"  • {key}: {value}")
    
    return implications

def compare_static_vs_dynamic():
    """정적 vs 동적 토픽 모델링 비교"""
    
    comparison = {
        "분석 범위": {
            "정적 BERTopic": "특정 시점의 스냅샷",
            "동적 BERTopic": "시간 흐름에 따른 변화 추적"
        },
        "활용 분야": {
            "정적 BERTopic": "현재 상황 파악, 일회성 분석",
            "동적 BERTopic": "트렌드 분석, 정책 효과 측정, 예측"
        },
        "장점": {
            "정적 BERTopic": "빠른 분석, 간단한 해석",
            "동적 BERTopic": "변화 패턴 파악, 예측 가능"
        },
        "단점": {
            "정적 BERTopic": "변화 추적 불가, 예측 어려움",
            "동적 BERTopic": "복잡한 분석, 많은 데이터 필요"
        }
    }
    
    print("\n⚖️ 정적 vs 동적 BERTopic 비교:")
    print("=" * 40)
    
    for aspect, details in comparison.items():
        print(f"\n📋 {aspect}:")
        for method, description in details.items():
            print(f"  {method}: {description}")
    
    return comparison

def main():
    """메인 실행 함수"""
    print("📈 제4장: Dynamic BERTopic 데모 - 시간에 따른 토픽 변화 추적")
    print("=" * 70)
    
    # 1. Dynamic BERTopic 데모 실행
    print("1️⃣ Dynamic BERTopic 시뮬레이션 실행...")
    dynamic_sim, monthly_topics, evolution, predictions = demonstrate_dynamic_bertopic()
    
    # 2. 정책적 시사점 분석
    print("\n2️⃣ 정책적 시사점 분석...")
    implications = analyze_policy_implications()
    
    # 3. 정적 vs 동적 비교
    print("\n3️⃣ 정적 vs 동적 토픽 모델링 비교...")
    comparison = compare_static_vs_dynamic()
    
    print("\n✅ Dynamic BERTopic 데모 완료!")
    print("📁 결과 이미지가 output/ 폴더에 저장되었습니다.")
    
    # 요약 정보
    print(f"\n📊 Dynamic BERTopic 분석 요약:")
    print(f"• 분석 기간: 12개월 (2024.01 ~ 2024.12)")
    print(f"• 추적 토픽: 5개 (복지, 환경, 교통, 안전, 행정)")
    print(f"• 가장 변동성 큰 토픽: 환경 (계절별 변화)")
    print(f"• 가장 안정적 토픽: 복지 (연중 높은 비중)")
    print(f"• 예측 신뢰도: 평균 75% (최근 트렌드 기반)")

if __name__ == "__main__":
    main()
