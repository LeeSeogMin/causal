"""
제1장 실습 1: 개발환경 점검
============================

목적: 이 교재의 실습을 돌릴 수 있는 상태인지 확인한다.
- 파이썬 버전 확인
- 필요한 패키지가 설치되어 있는지 확인
- 한글 그래프가 깨지지 않는지 확인
- 간단한 회귀를 한 번 돌려 전체 흐름이 작동하는지 확인

저자: AI 기반 정책분석방법론
날짜: 2026-08-17
"""

import sys
import importlib

print("=" * 70)
print("개발환경 점검")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. 파이썬 버전
# ---------------------------------------------------------------------------
print("\n[1단계] 파이썬 버전")
print("-" * 70)

v = sys.version_info
print(f"설치된 버전: {v.major}.{v.minor}.{v.micro}")
if (v.major, v.minor) >= (3, 10):
    print("판정: 통과 (3.10 이상 필요)")
else:
    print("판정: 실패 - 3.10 이상으로 올려야 한다")

# ---------------------------------------------------------------------------
# 2. 패키지 확인
# ---------------------------------------------------------------------------
print("\n[2단계] 패키지 확인")
print("-" * 70)

packages = [
    ('numpy', '수치 계산'),
    ('pandas', '표 형태 데이터 처리'),
    ('scipy', '통계 함수'),
    ('matplotlib', '그래프'),
    ('sklearn', '머신러닝'),
    ('statsmodels', '회귀분석'),
    ('linearmodels', '도구변수 추정 (6장)'),
]

missing = []
for name, purpose in packages:
    try:
        m = importlib.import_module(name)
        ver = getattr(m, '__version__', '버전 정보 없음')
        print(f"  {name:<14} {ver:<12} {purpose}")
    except ImportError:
        print(f"  {name:<14} {'없음':<12} {purpose}")
        missing.append(name)

if missing:
    install = ' '.join('scikit-learn' if p == 'sklearn' else p for p in missing)
    print(f"\n설치가 필요하다:  pip install {install}")
else:
    print("\n판정: 통과 (필요한 패키지가 모두 있다)")

# ---------------------------------------------------------------------------
# 3. 한글 폰트 확인
# ---------------------------------------------------------------------------
print("\n[3단계] 한글 그래프 확인")
print("-" * 70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

installed = {f.name for f in font_manager.fontManager.ttflist}
for cand in ['Malgun Gothic', 'AppleGothic', 'NanumGothic']:
    if cand in installed:
        korean_font = cand
        break
else:
    korean_font = None

if korean_font:
    print(f"사용할 한글 폰트: {korean_font}")
    print("모든 실습 스크립트 맨 위에 아래 두 줄을 넣는다.")
    print(f"  plt.rcParams['font.family'] = '{korean_font}'")
    print("  plt.rcParams['axes.unicode_minus'] = False")
    plt.rcParams['font.family'] = korean_font
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("한글 폰트를 찾지 못했다. 그래프의 한글이 네모로 나온다.")
    print("윈도우는 'Malgun Gothic', macOS는 'AppleGothic'이 기본 설치되어 있다.")

# ---------------------------------------------------------------------------
# 4. 회귀 한 번 돌려 보기
# ---------------------------------------------------------------------------
print("\n[4단계] 회귀 한 번 돌려 보기")
print("-" * 70)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(0)
n = 300
x = np.random.normal(10, 2, n)
y = 5 + 2.0 * x + np.random.normal(0, 1, n)   # 참 기울기 2.0
df = pd.DataFrame({'x': x, 'y': y})

fit = smf.ols('y ~ x', data=df).fit()
slope = fit.params['x']
se = fit.bse['x']
lo, hi = fit.conf_int().loc['x']

print(f"참 기울기: 2.000")
print(f"추정 기울기: {slope:.3f} (표준오차 {se:.3f})")
print(f"95% 신뢰구간: [{lo:.3f}, {hi:.3f}]")
print(f"신뢰구간이 참값을 포함하는가: {'예' if lo <= 2.0 <= hi else '아니오'}")

# ---------------------------------------------------------------------------
# 5. 그래프 저장
# ---------------------------------------------------------------------------
print("\n[5단계] 그래프 저장")
print("-" * 70)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x, y, alpha=0.4, s=18, color='steelblue', label='관측값')
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, fit.params['Intercept'] + slope * xs, 'r-', linewidth=2,
        label=f'추정 직선 (기울기 {slope:.2f})')
ax.set_xlabel('설명변수 x')
ax.set_ylabel('결과변수 y')
ax.set_title('환경 점검용 회귀', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('1-1-environment-check.png', dpi=150, bbox_inches='tight')
print("그래프 저장: 1-1-environment-check.png")
print("이 파일을 열어 한글이 네모로 나오지 않으면 환경 구축이 끝났다.")

print("\n" + "=" * 70)
print("점검 요약")
print("=" * 70)
print(f"파이썬 {v.major}.{v.minor}.{v.micro} / 부족한 패키지 {len(missing)}개 / "
      f"한글 폰트 {korean_font or '없음'}")
print("네 항목이 모두 통과면 2장 실습으로 넘어가도 된다.")
print("=" * 70)
