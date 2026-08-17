from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path


OUT = Path(r"F:\papercutter\教辅试卷自动切题_无训练方案精度与云算力预算.docx")


def set_east_asia(font, name='等线'):
    font.name = name
    rpr = font._element.rPr
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.insert(0, rfonts)
    rfonts.set(qn('w:eastAsia'), name)


def set_cell_margin(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in('w:tcMar')
    if tcMar is None:
        tcMar = OxmlElement('w:tcMar')
        tcPr.append(tcMar)
    for side, value in [('top', 80), ('start', 120), ('bottom', 80), ('end', 120)]:
        tag = qn(f'w:{side}')
        node = tcMar.find(tag)
        if node is None:
            node = OxmlElement(f'w:{side}')
            tcMar.append(node)
        node.set(qn('w:w'), str(value))
        node.set(qn('w:type'), 'dxa')


def set_table_geometry(table, widths):
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl = table._tbl
    tblPr = tbl.tblPr
    tblW = tblPr.first_child_found_in('w:tblW')
    if tblW is None:
        tblW = OxmlElement('w:tblW')
        tblPr.insert(0, tblW)
    tblW.set(qn('w:w'), str(sum(widths)))
    tblW.set(qn('w:type'), 'dxa')
    tblInd = tblPr.first_child_found_in('w:tblInd')
    if tblInd is None:
        tblInd = OxmlElement('w:tblInd')
        tblPr.append(tblInd)
    tblInd.set(qn('w:w'), '120')
    tblInd.set(qn('w:type'), 'dxa')
    for i, grid_col in enumerate(tbl.tblGrid.gridCol_lst):
        grid_col.set(qn('w:w'), str(widths[i]))
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = Inches(widths[i] / 1440)
            cell._tc.tcPr.tcW.set(qn('w:w'), str(widths[i]))
            cell._tc.tcPr.tcW.set(qn('w:type'), 'dxa')
            set_cell_margin(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement('w:tblHeader')
    el.set(qn('w:val'), 'true')
    trPr.append(el)


def text(p, value, bold=False, size=None, color=None):
    run = p.add_run(value)
    run.bold = bold
    set_east_asia(run.font)
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    return run


def paragraph(doc, value, bold_prefix=None):
    p = doc.add_paragraph(style='Normal')
    if bold_prefix and value.startswith(bold_prefix):
        text(p, bold_prefix, bold=True)
        text(p, value[len(bold_prefix):])
    else:
        text(p, value)
    return p


def bullet(doc, value):
    p = doc.add_paragraph(style='List Bullet')
    text(p, value)
    return p


def table(doc, headers, rows, widths, fs=9.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    set_table_geometry(t, widths)
    repeat_header(t.rows[0])
    for i, val in enumerate(headers):
        p = t.rows[0].cells[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        text(p, val, bold=True, size=fs)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            text(p, str(val), size=fs)
    return t


def add_page_number(p):
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    text(p, '第 ')
    begin = OxmlElement('w:fldChar'); begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText'); instr.set(qn('xml:space'), 'preserve'); instr.text = 'PAGE'
    end = OxmlElement('w:fldChar'); end.set(qn('w:fldCharType'), 'end')
    p.runs[-1]._r.append(begin); p.runs[-1]._r.append(instr); p.runs[-1]._r.append(end)
    text(p, ' 页')


doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.85); sec.bottom_margin = Inches(0.8)
sec.left_margin = Inches(0.85); sec.right_margin = Inches(0.85)
sec.header_distance = Inches(0.35); sec.footer_distance = Inches(0.35)

normal = doc.styles['Normal']
normal.font.size = Pt(10.5); set_east_asia(normal.font)
normal.paragraph_format.space_after = Pt(6); normal.paragraph_format.line_spacing = 1.15
for name, size, before, after in [('Heading 1', 15, 14, 7), ('Heading 2', 12.5, 10, 5)]:
    st = doc.styles[name]
    st.font.size = Pt(size); st.font.bold = True; st.font.color.rgb = RGBColor(0, 0, 0)
    set_east_asia(st.font)
    st.paragraph_format.space_before = Pt(before); st.paragraph_format.space_after = Pt(after)
    st.paragraph_format.keep_with_next = True
for name in ['List Bullet']:
    st = doc.styles[name]
    st.font.size = Pt(10.5); set_east_asia(st.font)
    st.paragraph_format.space_after = Pt(3); st.paragraph_format.line_spacing = 1.15

hp = sec.header.paragraphs[0]
text(hp, '教辅与试卷自动切题项目', size=8.5, color=(100, 100, 100))
fp = sec.footer.paragraphs[0]
add_page_number(fp)
for r in fp.runs:
    r.font.size = Pt(8.5); r.font.color.rgb = RGBColor(100, 100, 100); set_east_asia(r.font)

p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(4)
text(p, '自动切题无训练方案：精度与云算力预算', bold=True, size=21)
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(14)
text(p, '前提：不训练模型、不做训练标注，只优化现有程序逻辑并使用现成开源模型。', size=10, color=(85, 85, 85))

table(doc, ['项目', '本方案口径'], [
    ('目标', '在不训练的前提下，尽量减少多裁、少裁和目录误切；不承诺达到人工级高精度。'),
    ('计算规模', '1 万套教材，按平均 100 页/套估算，合计约 100 万页。'),
    ('云算力', '起源云 RTX 3090 24GB，0.77 元/小时。'),
    ('重要说明', '本方案的核心是“自动预切 + 低置信度页筛出”，不是全自动高精度交付。'),
], [2100, 7260], 10)

doc.add_heading('一、简单结论', level=1)
paragraph(doc, '不训练时，程序只能依据现成模型识别到的文字、版面和题号来“推断”题目范围。它能比当前单纯 OCR 规则更好，但无法真正理解不同教材中“哪一段内容算完整一道题”。')
paragraph(doc, '因此，本方案的合理目标是：常规页面尽量自动切，复杂页面自动标记出来，后续再由人决定是否使用。')
for v in [
    '不训练、不人工标注，也可以升级程序逻辑并提高一部分准确率。',
    '不能把公开模型的 OCR 或文档解析指标直接当作切题准确率。',
    '若要求所有教材都达到 90% 以上整题准确率，无训练方案不现实。',
]: bullet(doc, v)

doc.add_heading('二、建议采用的无训练程序逻辑', level=1)
paragraph(doc, '下面的模型都直接使用现成能力，不对教材数据进行训练。它们各自做擅长的部分，再由程序合并判断。')
table(doc, ['模块', '使用方式', '解决的问题'], [
    ('版面识别', 'PP-DocLayout-L / PP-StructureV3', '先识别正文、标题、表格、公式、页眉页脚和目录区域，避免把整页当作普通文字。'),
    ('文字与题号', 'PP-OCRv6-medium', '找出“1.”、“一、”、“（1）”等题号，作为切题开始位置。'),
    ('切题规则', '题号层级 + 阅读顺序 + 相邻内容合并', '题干向下合并到下一同级题号；把紧邻的图、表、公式纳入题目；支持单栏和双栏。'),
    ('误切过滤', '目录/页眉页脚识别 + 异常框检查', '目录页、页码、页眉、过小框、重叠框不直接输出为题目。'),
    ('难页复核', 'PaddleOCR-VL-1.6（可选）', '仅处理程序没有把握的页面，辅助判断跨栏、图文混排和总题干带小题。'),
], [1450, 3100, 4810], 9.2)

doc.add_heading('三、不训练时可预期的切题准确率', level=1)
paragraph(doc, '以下是“题目是否被完整裁出”的保守经验目标，不是模型厂商承诺。实际结果受教材版式、扫描质量、双栏比例、图表和公式数量影响很大。')
table(doc, ['页面类型', '整题可用准确率（保守估计）', '说明'], [
    ('排版规则、单栏、题号清晰', '70%～82%', '题号和题目边界较明显时，规则合并效果较好。'),
    ('普通混合教材', '60%～75%', '含知识点、例题、题组、图片或不规则留白时，仍会出现多裁或少裁。'),
    ('复杂版式、双栏、跨栏、图表公式密集', '45%～65%', '无训练模型难以稳定判断内容归属，应优先进入复核队列。'),
    ('只放行高置信度页面', '85%～92%', '可以提高已放行结果的可靠性，但会有约 30%～50% 页面被筛出，不等于全部自动完成。'),
], [2500, 2200, 4660], 9.2)
paragraph(doc, '最关键的区别：85%～92% 是“筛选后可直接使用的页面”的准确率；所有页面一律自动输出时，整体结果通常仍接近 60%～75%。')

doc.add_heading('四、1 万套教材的时间与费用（保守估计）', level=1)
paragraph(doc, '按 1 万套、平均 100 页/套、共 100 万页计算；使用一张 RTX 3090 24GB，0.77 元/小时。费用只包括 GPU 时费，不包括云硬盘、流量和人工核对。')
table(doc, ['工作内容', '保守耗时', 'GPU 费用', '说明'], [
    ('基础切题：版面 + OCR + 规则', '80～230 小时', '62～178 元', '所有页面都跑；包含图片读取、规则判断和裁图写盘的预留。'),
    ('10% 难页视觉复核', '140～560 小时', '108～431 元', '只对低置信度页使用视觉大模型；不建议全量使用。'),
    ('模型组合实际合计', '220～790 小时', '170～608 元', '基础切题加 10% 难页复核。'),
    ('建议预算：加 30% 缓冲', '290～1,030 小时', '223～793 元', '用于重试、数据读写波动和临时补跑。'),
], [2600, 1800, 1650, 2910], 9.2)
paragraph(doc, '单张 3090 连续运行约需 12～43 天。若同时租用 4 张 3090 并行，日历时间通常可缩短到约 4～13 天，纯 GPU 总费用基本不变。')
paragraph(doc, '如果平均每套为 150 页，以上时间和费用乘以 1.5；如果为 200 页，则乘以 2。')

doc.add_heading('五、这条路线适合什么情况', level=1)
for v in [
    '适合：先把大量教材做成“可初步使用、可筛选、可抽查”的切题结果，控制技术投入。',
    '不适合：要求几乎所有教材、所有题型都完整无误且无需人工检查。',
    '建议做法：先选 100 页最容易出错的教材实测，统计整题完整率和难页比例；达到接受标准后再处理 1 万套。',
]: bullet(doc, v)

doc.add_heading('六、最终判断', level=1)
paragraph(doc, '在不训练的前提下，最好的策略不是承诺“全自动高精度”，而是让程序把容易处理的页面自动完成，把明显不可靠的页面自动筛出来。这样能减少人工查找错误的时间，但不能完全取消人工判断。')

p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(0)
text(p, '方案版本：无训练 V1.0 | 估算日期：2026 年 8 月 | 费用仅含 GPU 时费。', size=8.5, color=(100, 100, 100))

doc.core_properties.title = '自动切题无训练方案：精度与云算力预算'
doc.core_properties.subject = '教辅与试卷自动切题项目'
doc.core_properties.author = 'Codex'
doc.save(OUT)
print(OUT)
