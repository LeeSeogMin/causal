"""
제5장: 경계선을 배정 변수로 쓰는 RDD

수정 이력
---------
2026-08-17
저장 경로: 그림을 `diagrams/5-3.png`에 저장하고 있었다. `diagrams/`는 강의노트
개념도 폴더이므로, 이 스크립트를 돌릴 때마다 개념도 5-3.png가 실습 그림으로
덮어써졌다. 저장 위치를 스크립트와 같은 폴더의 `5-7-geographic-rdd.png`로 바꿨다.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os

# 그래프 레이블이 전부 영문이므로 기본 폰트를 쓴다
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 생성 (Synthetic Data Generation)
np.random.seed(42)
n = 1000

# 가상의 도시 좌표 (0~10 사이)
# 경계선: Latitude = 5.0 (North: Treat=1, South: Treat=0)
data = pd.DataFrame({
    'longitude': np.random.uniform(0, 10, n),
    'latitude': np.random.uniform(0, 10, n)
})

# 배정 변수: 경계선까지의 거리 (Latitude - 5.0)
# 북쪽(+)이 처치군, 남쪽(-)이 대조군
cutoff = 5.0
data['dist_lat'] = data['latitude'] - cutoff
data['treat'] = (data['dist_lat'] >= 0).astype(int)

# 잠재 결과 생성
# - 처치 효과: +5.0
# - 공간적 교란 요인: 위도와 경도가 높을수록 가격(Outcome)이 비쌈 (Spatial Confounding)
# - 노이즈: 정규분포
true_effect = 5.0
spatial_trend = 2.0 * data['latitude'] + 1.5 * data['longitude']
noise = np.random.normal(0, 2, n)

data['outcome'] = 30.0 + true_effect * data['treat'] + spatial_trend + noise

# 2. Geographic RDD 분석 (1D: Distance to Border)
# 경계선 근처 데이터 필터링 (Bandwidth = 2.0)
bandwidth = 2.0
df_near = data[abs(data['dist_lat']) <= bandwidth].copy()

# 국소 선형 회귀 (Local Linear Regression)
# 모형: Y ~ T + dist + T*dist + longitude (경도 통제)
# Geographic RDD에서는 경계선을 따라 이동할 때의 변동(경도)을 통제하는 것이 중요함
model_grdd = smf.ols('outcome ~ treat + dist_lat + treat:dist_lat + longitude', data=df_near).fit()

print(model_grdd.summary())

# 결과 출력
print(f"\n[Geographic RDD 결과]")
print(f"True Effect: {true_effect}")
print(f"Estimated Effect: {model_grdd.params['treat']:.4f}")
print(f"Standard Error: {model_grdd.bse['treat']:.4f}")
print(f"95% CI: [{model_grdd.conf_int().loc['treat', 0]:.4f}, {model_grdd.conf_int().loc['treat', 1]:.4f}]")

# 3. 시각화 (Spatial Plot)
plt.figure(figsize=(10, 6))
plt.scatter(data[data['treat']==0]['longitude'], data[data['treat']==0]['latitude'], 
            alpha=0.5, label='Control (South)', s=10, c='blue')
plt.scatter(data[data['treat']==1]['longitude'], data[data['treat']==1]['latitude'], 
            alpha=0.5, label='Treated (North)', s=10, c='red')
plt.axhline(y=cutoff, color='black', linestyle='--', label='Boundary')
plt.title('Geographic RDD: Spatial Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
# 이미지 저장 (스크립트와 같은 폴더)
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, '5-7-geographic-rdd.png')

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"그래프 저장 완료: {save_path}")
# plt.show() # 주석 처리
