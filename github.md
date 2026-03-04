# Git/GitHub + Copilot + 실습환경 가이드 (본 과목)

이 문서는 이 과목에서 과제 제출과 실습을 위해 필요한 Git/GitHub 설정, Copilot 사용, Python 실습환경 구축 방법을 정리한다.

## 0. 이 레포 구조 (중요)

- `lecture/`: 강의자료(학부생용, 쉽고 직관적으로 정리된 버전)
- `practice/`: 장별 실습 코드와 데이터

## 1. Git 설치 및 기본 설정

### Git 설치

- Windows: https://git-scm.com/download/win
- macOS: 터미널에서 `xcode-select --install`

설치 확인:

```bash
git --version
```

### 사용자 정보 등록 (처음 1회)

```bash
git config --global user.name "홍길동"
git config --global user.email "학교이메일@ac.kr"
```

확인:

```bash
git config --global --list
```

## 2. GitHub 계정, 과제 제출 워크플로우

### GitHub 계정

1. https://github.com 가입
2. 가능하면 학교 이메일을 추가하고 인증(verified) 상태로 만든다

### 기본 작업 흐름 (매주 반복)

아래 흐름만 지켜도 대부분의 과제 제출이 깔끔해진다.

```bash
git pull
git status
# 파일 편집
git add -A
git commit -m "week-01: update lecture notes"
git push
```

> 과제 제출 방식이 fork/clone, GitHub Classroom, 또는 특정 브랜치/PR 제출인지 여부는 강의 공지에 따른다.

## 3. GitHub Copilot (선택, 권장)

- 학생이면 GitHub Education(Student Developer Pack)을 통해 Copilot 혜택을 받을 수 있다.
- 신청 페이지: https://education.github.com/pack

VS Code에서는 아래 두 확장을 설치해서 사용한다.

- `GitHub Copilot`
- `GitHub Copilot Chat`

## 4. 초기 설치 절차 (Windows 기준)

### 1단계: VSCode 설치

1. https://code.visualstudio.com/ 에서 다운로드 후 설치한다.
2. 설치 시 `Add to PATH` 옵션 체크를 권장한다.
3. 설치 후 실행하여 아래 확장(Extensions)을 설치한다.
   - `Python` (Microsoft): Python 개발 지원
   - `Jupyter` (Microsoft): 노트북(`.ipynb`) 실행 지원

설치 방법:
- 좌측 사이드바의 확장 아이콘 클릭
- 검색창에 `Python`, `Jupyter` 입력 후 `Install`

### 2단계: Python 설치

1. https://www.python.org/downloads/ 에서 Python 3.12 이상 다운로드
2. 설치 프로그램 실행
3. 첫 화면에서 반드시 `Add python.exe to PATH` 체크
4. `Install Now` 클릭

설치 확인(Windows: `Win+R` -> `cmd`):

```bash
python --version
```

`Python 3.x.x`가 출력되면 성공이다.  
`'python' is not recognized...` 오류가 나오면 PATH 등록이 안 된 것이므로 Python을 제거한 뒤 3번 옵션을 확인하여 재설치한다.

### 3단계: 저장소 클론

명령 프롬프트 열기(Windows: `Win+R` -> `cmd`) 후 실행:

```bash
cd C:\\Dev
git clone <저장소_URL> "causal(2026)"
```

`C:\\Dev\\causal(2026)` 폴더가 생성되면 성공이다.  
이미 폴더가 있다면 클론 단계는 건너뛰고 4단계부터 진행한다.

### 4단계: 환경 자동 설정

같은 명령 프롬프트에서 이어서 실행:

```bash
cd "C:\\Dev\\causal(2026)"
python setup_env.py
```

이 스크립트가 자동으로 처리하는 항목:

- `.venv` 가상환경 생성
- 필요한 패키지 일괄 설치
- Jupyter 커널 등록

`설정 완료!` 메시지가 나오면 성공이다.

### 5단계: VSCode에서 프로젝트 열기

반드시 4단계가 `설정 완료!` 메시지로 끝난 뒤 진행한다.

1. VSCode 실행
2. `파일 -> 폴더 열기` (`Ctrl+K, Ctrl+O`)에서 `C:\\Dev\\causal(2026)` 선택
3. 왼쪽 탐색기에서 `practice -> chapter02 -> code -> 2-1-potential-outcomes.py` 같은 실습 파일을 연다

### 6단계: 커널 선택 및 실행

1. 노트북 우측 상단 `커널 선택` 클릭
2. `Python 환경...` -> `AI 기획 강의 (Python 3)` 선택
3. 노트북(`.ipynb`) 또는 Python 파일 실행 시 동일한 가상환경 커널/인터프리터를 선택해 실행

커널 목록에 `AI 기획 강의 (Python 3)`가 보이지 않으면 4단계(`python setup_env.py`)가 정상 완료되었는지 확인한다.

## 5. Python 실습환경 자동 설정 (`setup_env.py`) - 요약

프로젝트 루트에서 다음 명령으로 자동 설정할 수 있다.

```bash
python setup_env.py
```

설치 확인만 하고 싶을 때:

```bash
python setup_env.py --check
```

옵션: 추가 패키지 설치

```bash
python setup_env.py --extras torch
python setup_env.py --extras tensorflow
```

## 6. 자주 생기는 문제

- PowerShell에서 실행 정책 오류가 나면:
  - 임시 허용: `Set-ExecutionPolicy -Scope Process RemoteSigned`
- 설치가 꼬였으면:
  - `.venv/`를 삭제하고 `python setup_env.py`를 다시 실행
