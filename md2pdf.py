"""
강의노트 Markdown -> PDF 변환
=============================

사용법:
    python md2pdf.py lecture/chapter06.md      # 한 장만
    python md2pdf.py lecture/chapter06.md lecture/chapter07.md
    python md2pdf.py --all                     # lecture/chapter*.md 전부

출력: pdf/chapterNN.pdf

필요 도구:
    pandoc  (https://pandoc.org)
    typst   (https://typst.app) - pandoc의 PDF 엔진으로 사용

확인:
    pandoc --version
    typst --version
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, 'pdf')

# 한글 폰트. 윈도우 기본값을 쓰고, 없으면 아래 후보를 차례로 시도한다.
FONT_CANDIDATES = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Noto Sans KR']


def check_tools():
    missing = [t for t in ('pandoc', 'typst') if shutil.which(t) is None]
    if missing:
        print(f"필요한 도구가 없다: {', '.join(missing)}")
        print("  pandoc: https://pandoc.org/installing.html")
        print("  typst : https://github.com/typst/typst/releases")
        sys.exit(1)


def pick_font():
    try:
        out = subprocess.run(['typst', 'fonts'], capture_output=True,
                             text=True, encoding='utf-8', errors='ignore').stdout
        available = {line.strip() for line in out.splitlines()}
        for f in FONT_CANDIDATES:
            if f in available:
                return f
    except Exception:
        pass
    return FONT_CANDIDATES[0]


def convert(md_path, font):
    md_path = os.path.abspath(md_path)
    if not os.path.exists(md_path):
        print(f"건너뜀 (파일 없음): {md_path}")
        return False

    os.makedirs(OUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(md_path))[0]
    pdf_path = os.path.join(OUT_DIR, f'{name}.pdf')
    md_dir = os.path.dirname(md_path)

    # typst 템플릿의 margin은 맵이라 -V로 못 넘긴다. 메타데이터 파일로 전달한다.
    # 여러 장을 동시에 변환할 수 있으므로 파일명에 장 이름을 넣어 충돌을 막는다.
    meta_path = os.path.join(OUT_DIR, f'_meta_{name}.yaml')
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write('---\n'
                f'mainfont: "{font}"\n'
                'fontsize: 10pt\n'
                'papersize: a4\n'
                'margin:\n  x: 2cm\n  y: 2cm\n'
                '---\n')

    cmd = [
        'pandoc', md_path,
        '--metadata-file', meta_path,
        '-o', pdf_path,
        '--pdf-engine=typst',
        '--toc', '--toc-depth=2',
        '--resource-path', os.pathsep.join([md_dir, ROOT]),
    ]

    print(f"변환: {os.path.relpath(md_path, ROOT)} -> {os.path.relpath(pdf_path, ROOT)}")
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding='utf-8', errors='ignore')
    os.remove(meta_path)
    if r.returncode != 0:
        print("  실패")
        print((r.stderr or '').strip()[:1500])
        return False

    size_kb = os.path.getsize(pdf_path) / 1024
    print(f"  완료 ({size_kb:,.0f} KB)")
    return True


def main():
    ap = argparse.ArgumentParser(description='강의노트 Markdown을 PDF로 변환한다')
    ap.add_argument('files', nargs='*', help='변환할 .md 파일')
    ap.add_argument('--all', action='store_true',
                    help='lecture/chapter*.md 전부 변환')
    args = ap.parse_args()

    check_tools()

    targets = list(args.files)
    if args.all:
        targets = sorted(glob.glob(os.path.join(ROOT, 'lecture', 'chapter*.md')))
    if not targets:
        ap.print_help()
        sys.exit(1)

    font = pick_font()
    print(f"본문 폰트: {font}\n")

    ok = sum(convert(t, font) for t in targets)
    print(f"\n{ok}/{len(targets)}개 변환 완료 -> {os.path.relpath(OUT_DIR, ROOT)}/")


if __name__ == '__main__':
    main()
