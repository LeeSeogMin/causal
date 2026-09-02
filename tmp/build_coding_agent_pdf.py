from pathlib import Path
import html, re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle, Preformatted

root = Path(__file__).resolve().parents[1]
src, dst = root / 'coding-agent-setting.md', root / 'coding-agent-setting.pdf'
pdfmetrics.registerFont(TTFont('Malgun', r'C:\Windows\Fonts\malgun.ttf'))
pdfmetrics.registerFont(TTFont('Malgun-Bold', r'C:\Windows\Fonts\malgunbd.ttf'))
s = getSampleStyleSheet()
s.add(ParagraphStyle('k_title', parent=s['Title'], fontName='Malgun-Bold', fontSize=20, leading=26, textColor=colors.HexColor('#17365D'), spaceAfter=10))
s.add(ParagraphStyle('k_h2', parent=s['Heading2'], fontName='Malgun-Bold', fontSize=13, leading=18, textColor=colors.HexColor('#17365D'), spaceBefore=12, spaceAfter=6, keepWithNext=True))
s.add(ParagraphStyle('k_h3', parent=s['Heading3'], fontName='Malgun-Bold', fontSize=10.5, leading=15, textColor=colors.HexColor('#2F5597'), spaceBefore=8, spaceAfter=4, keepWithNext=True))
s.add(ParagraphStyle('k_body', parent=s['BodyText'], fontName='Malgun', fontSize=9.2, leading=14, spaceAfter=5, wordWrap='CJK'))
s.add(ParagraphStyle('k_bullet', parent=s['BodyText'], fontName='Malgun', fontSize=9.1, leading=13.5, leftIndent=13, firstLineIndent=-8, spaceAfter=3, wordWrap='CJK'))
s.add(ParagraphStyle('k_quote', parent=s['BodyText'], fontName='Malgun', fontSize=9, leading=13.5, leftIndent=13, borderPadding=7, borderColor=colors.HexColor('#A6A6A6'), borderWidth=.5, backColor=colors.HexColor('#F2F5F9'), spaceAfter=7, wordWrap='CJK'))
s.add(ParagraphStyle('k_code', parent=s['Code'], fontName='Malgun', fontSize=8.2, leading=11, leftIndent=10, borderPadding=6, backColor=colors.HexColor('#F4F4F4'), borderColor=colors.HexColor('#D9D9D9'), borderWidth=.5, spaceAfter=7))
s.add(ParagraphStyle('k_cell', parent=s['BodyText'], fontName='Malgun', fontSize=7.4, leading=10, wordWrap='CJK'))
s.add(ParagraphStyle('k_cellhead', parent=s['BodyText'], fontName='Malgun-Bold', fontSize=7.5, leading=10, textColor=colors.white, wordWrap='CJK'))

def fmt(t):
    t = html.escape(t, quote=False)
    t = re.sub(r'`([^`]+)`', r'<font name="Courier" color="#7A1F1F">\1</font>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#0563C1"><u>\1</u></link>', t)
    t = re.sub(r'<(https?://[^>]+)>', r'<link href="\1" color="#0563C1"><u>\1</u></link>', t)
    return t

def table(rows):
    data = [[Paragraph(fmt(c.strip()), s['k_cellhead' if r == 0 else 'k_cell']) for c in row] for r, row in enumerate(rows)]
    t = Table(data, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#4472C4')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#B7C9E2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F7F9FC')])]))
    return t

def build_story():
    lines = src.read_text(encoding='utf-8').splitlines(); story=[]; i=0
    while i < len(lines):
        line=lines[i]
        if not line.strip(): i+=1; continue
        if line.strip()=='---': story += [Spacer(1,3), HRFlowable(width='100%', thickness=.5, color=colors.HexColor('#B7C9E2'), spaceAfter=7)]; i+=1; continue
        if line.startswith('```'):
            i+=1; buf=[]
            while i<len(lines) and not lines[i].startswith('```'): buf.append(lines[i]); i+=1
            i+=1; story.append(Preformatted('\n'.join(buf), s['k_code'])); continue
        if line.startswith('|') and i+1<len(lines) and re.match(r'^\s*\|?\s*:?-+', lines[i+1]):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                if not re.match(r'^\s*\|?\s*:?-+', lines[i]): rows.append(lines[i].strip().strip('|').split('|'))
                i+=1
            story += [table(rows), Spacer(1,5)]; continue
        m=re.match(r'^(#{1,3})\s+(.*)', line)
        if m: story.append(Paragraph(fmt(m.group(2)), s[{1:'k_title',2:'k_h2',3:'k_h3'}[len(m.group(1))]])); i+=1; continue
        if line.startswith('>'): story.append(Paragraph(fmt(line[1:].strip()), s['k_quote'])); i+=1; continue
        m=re.match(r'^\s*([-*○□–])\s+(.*)', line) or re.match(r'^\s*(\d+\.)\s+(.*)', line)
        if m: story.append(Paragraph(fmt(m.group(1)+' '+m.group(2)), s['k_bullet'])); i+=1; continue
        story.append(Paragraph(fmt(line), s['k_body'])); i+=1
    return story

class Document(BaseDocTemplate): pass
doc=Document(str(dst), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=17*mm, bottomMargin=17*mm, title='무료 코딩 에이전트 설치 가이드')
frame=Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
def footer(canvas, doc):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#D9E2F3')); canvas.line(doc.leftMargin,12*mm,A4[0]-doc.rightMargin,12*mm); canvas.setFont('Malgun',7.5); canvas.setFillColor(colors.HexColor('#666666')); canvas.drawString(doc.leftMargin,8*mm,'무료 코딩 에이전트 설치 가이드'); canvas.drawRightString(A4[0]-doc.rightMargin,8*mm,str(doc.page)); canvas.restoreState()
doc.addPageTemplates([PageTemplate(id='main', frames=[frame], onPage=footer)])
doc.build(build_story())
print(dst)
