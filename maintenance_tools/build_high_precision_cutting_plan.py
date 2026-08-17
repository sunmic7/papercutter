from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE
from pathlib import Path


OUT = Path(r"F:\papercutter\教辅试卷自动切题_高精度模型组合与云算力预算方案.docx")


def set_east_asia(font, name="等线"):
    font.name = name
    rpr = font._element.rPr
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.insert(0, rfonts)
    rfonts.set(qn('w:eastAsia'), name)


def set_cell_margin(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in('w:tcMar')
    if tcMar is None:
        tcMar = OxmlElement('w:tcMar')
        tcPr.append(tcMar)
    for side, value in [('top', top), ('start', start), ('bottom', bottom), ('end', end)]:
        node = tcMar.find(qn(f'w:{side}'))
        if node is None:
            node = OxmlElement(f'w:{side}')
            tcMar.append(node)
        node.set(qn('w:w'), str(value))
        node.set(qn('w:type'), 'dxa')


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), color)
    tcPr.append(shd)


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
    grid = tbl.tblGrid
    for i, grid_col in enumerate(grid.gridCol_lst):
        grid_col.set(qn('w:w'), str(widths[i]))
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = Inches(widths[i] / 1440)
            tcW = cell._tc.tcPr.tcW
            tcW.set(qn('w:w'), str(widths[i]))
            tcW.set(qn('w:type'), 'dxa')
            set_cell_margin(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement('w:tblHeader')
    tblHeader.set(qn('w:val'), 'true')
    trPr.append(tblHeader)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run('第 ')
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'PAGE'
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    paragraph.add_run(' 页')


def add_text(p, text, bold=False, size=None, color=None):
    r = p.add_run(text)
    r.bold = bold
    set_east_asia(r.font)
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor(*color)
    return r


def add_body(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='Normal')
    if bold_prefix and text.startswith(bold_prefix):
        add_text(p, bold_prefix, bold=True)
        add_text(p, text[len(bold_prefix):])
    else:
        add_text(p, text)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style='List Bullet')
    add_text(p, text)
    return p


def add_number(doc, text):
    p = doc.add_paragraph(style='List Number')
    add_text(p, text)
    return p


def add_table(doc, headers, rows, widths, font_size=9.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    set_table_geometry(table, widths)
    set_repeat_table_header(table.rows[0])
    for i, text in enumerate(headers):
        c = table.rows[0].cells[i]
        shade(c, 'F2F2F2')
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        add_text(p, text, bold=True, size=font_size)
    for row in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            add_text(p, str(text), size=font_size)
    return table


doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.85)
section.bottom_margin = Inches(0.8)
section.left_margin = Inches(0.85)
section.right_margin = Inches(0.85)
section.header_distance = Inches(0.35)
section.footer_distance = Inches(0.35)

# Standard-business-brief token map, made deliberately plain: no backgrounds/coloured bands.
styles = doc.styles
normal = styles['Normal']
normal.font.size = Pt(10.5)
set_east_asia(normal.font)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15

for name, size, before, after in [('Heading 1', 15, 14, 7), ('Heading 2', 12.5, 10, 5), ('Heading 3', 11, 8, 4)]:
    st = styles[name]
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    set_east_asia(st.font)
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(after)
    st.paragraph_format.keep_with_next = True

for name in ['List Bullet', 'List Number']:
    st = styles[name]
    st.font.size = Pt(10.5)
    set_east_asia(st.font)
    st.paragraph_format.space_after = Pt(3)
    st.paragraph_format.line_spacing = 1.15

# Header/footer.
header = section.header.paragraphs[0]
header.alignment = WD_ALIGN_PARAGRAPH.LEFT
add_text(header, '教辅与试卷自动切题项目', size=8.5, color=(100, 100, 100))
footer = section.footer.paragraphs[0]
add_page_number(footer)
for run in footer.runs:
    set_east_asia(run.font)
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(100, 100, 100)

# Opening.
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(10)
p.paragraph_format.space_after = Pt(4)
add_text(p, '高精度自动切题模型组合与云算力预算方案', bold=True, size=21)
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(14)
add_text(p, '适用范围：教辅教材、同步训练、试卷及答案页 | 估算口径：保守估计', size=10, color=(85, 85, 85))

meta = add_table(doc, ['项目', '本次方案口径'], [
    ('目标', '提高“完整一道题”裁切的完整率，重点降低多裁、少裁、目录误切和图表公式截断。'),
    ('部署方式', '起源云 RTX 3090 24GB，按 0.77 元/小时估算；模型训练与批处理在云端执行。'),
    ('规模假设', '1 万套教材，平均 100 页/套，合计约 100 万页。实际页数变化时按比例调整。'),
    ('精度口径', '只统计“整题内容完整、边界可用”的题目；不以 OCR 字符准确率或通用 mAP 代替切题准确率。'),
], [2200, 7160], 10)

doc.add_heading('一、结论摘要', level=1)
add_body(doc, '本方案不依赖单一 OCR 或单一 YOLO 模型。采用“页面结构识别 + 题号 OCR + 自训练题目检测 + 难页视觉审核”的组合，才能适应教材版式差异。')
add_body(doc, '在已有足够高质量标注、并完成多轮实测训练的前提下，保守目标是：', bold_prefix='在已有足够高质量标注、并完成多轮实测训练的前提下，保守目标是：')
for t in [
    '所有页面直接自动输出的整题可用准确率：约 86%～92%。',
    '仅放行高置信度结果、复杂页进入审核队列后的交付准确率：约 94%～97%。',
    '上述不是现成模型的承诺值；必须以本项目未参与训练的新教材测试集验收。',
]: add_bullet(doc, t)

doc.add_heading('二、高精度模型组合配置', level=1)
add_body(doc, '模型按职责组合，而非简单叠加。普通页走快速模型；低置信度、跨栏、总题干带小题、图表公式混排等页面才进入视觉大模型审核。')
add_table(doc, ['环节', '建议模型/方法', '作用与使用方式'], [
    ('页面预处理', 'UVDoc / 方向分类', '修正拍照倾斜、弯曲、旋转或光照问题；扫描质量正常的页面可跳过。'),
    ('版面结构', 'PP-DocLayout-L + PP-StructureV3', '识别文本、标题、表格、公式、页眉页脚、目录等区域，生成带坐标的页面结构。'),
    ('题号与文字锚点', 'PP-OCRv6-medium', '识别“1.”、“一、”、“（1）”等题号和题目层级，辅助题目开始/结束判断。'),
    ('最终题目框', '自训练 RT-DETR-L 与 PP-YOLOE+ L', '训练 question_block、answer_block、material_block 等类别；两个模型结果交叉验证，高置信度一致时直接输出。'),
    ('复杂页审核', 'PaddleOCR-VL-1.6', '只处理低置信度页面，辅助判断跨栏、图表归属、总题干与小题的合并关系；不直接替代最终裁剪框。'),
    ('结果质检', '规则 + OCR 回读 + 置信度分流', '发现无题号、框内文字过少、相邻框重叠、疑似目录等情况，自动转入审核队列。'),
], [1350, 2700, 5310], 9.2)

doc.add_heading('三、切题精度预期与验收方式', level=1)
add_body(doc, '下表为项目目标区间，不是各模型的官方指标。模型公开的 mAP、OCR 识别率和文档解析分数不能直接等同于“完整切题准确率”。')
add_table(doc, ['阶段', '数据与处理条件', '保守整题可用准确率', '说明'], [
    ('未训练阶段', '仅使用现成版面模型、OCR 和规则', '50%～75%', '能生成候选框，但不同教材差异大，不能作为大规模交付结果。'),
    ('第一轮训练', '约 2,000～5,000 页高质量标注；覆盖多学科、多版式', '78%～86%', '能够明显减少典型多裁和漏题；仍需较多页面审核。'),
    ('成熟训练模型', '约 10,000～20,000 页高质量标注；包含目录、双栏、图表、公式、答案页等难例', '86%～92%', '指所有页面直接自动输出的结果；这是本方案的保守自动化目标。'),
    ('高置信度交付', '检测模型一致 + OCR/规则质检；低置信度页不直接放行', '94%～97%', '准确率提高来自“拒绝不确定页面”，并非所有页面都无需审核。'),
], [1350, 3200, 1800, 3010], 9.2)
add_body(doc, '验收建议：固定保留 500～1,000 页从未参与训练的教材页，按“整题完整率、漏题率、多裁率、目录误切率”统计。只有该测试集达到目标，才进入 1 万套批量处理。')

doc.add_heading('四、训练数据与人工工作量原则', level=1)
add_body(doc, '不建议让人工逐框修改一套几乎全错的自动结果。若预标注质量很差，应支持“清空本页框后重新标注”，避免在错误框上耗费大量时间。')
for t in [
    '统一标注规则：明确总题干和小题如何计算、图表和公式归属、跨页题处理、答案对应关系。',
    '先做 50～100 页高难样本试验：确认预标注可被少量修正后，再扩大标注规模。',
    '审核人员不需要懂算法：只需按规则做删除、补框、合并、拆分和确认。',
    '所有训练数据必须经过核对；错误框不能直接用于训练。',
]: add_bullet(doc, t)

doc.add_heading('五、1 万套教材云算力时间与费用估算', level=1)
add_body(doc, '估算前提：平均 100 页/套，即约 100 万页；使用 1 张起源云 RTX 3090 24GB，价格 0.77 元/小时；高分辨率页面处理；不把所有页面都交给视觉大模型。')
add_table(doc, ['工作内容', '保守耗时（GPU 小时）', '按 0.77 元/小时费用', '说明'], [
    ('主生产线批量切题', '220～500 小时', '170～385 元', '版面检测 + OCR + 已训练题目检测 + 裁图写盘。'),
    ('10% 难页视觉审核', '140～550 小时', '110～425 元', '仅对低置信度页面调用 PaddleOCR-VL；实际比例应由试运行确定。'),
    ('模型组合实际运行合计', '360～1,050 小时', '280～810 元', '未计意外中断、重试、数据读写波动。'),
    ('建议预算：加 30% 缓冲', '470～1,370 小时', '360～1,055 元', '建议作为 1 万套正式批处理的保守云卡预算。'),
], [2600, 2100, 1900, 2760], 9.2)
add_body(doc, '对应日历时间：单张 3090 连续运行约 20～57 天；若租用 4 张同型号显卡并行，通常可缩短至约 6～17 天，纯 GPU 总费用基本不变。')
add_body(doc, '若实际平均页数不是 100 页，时间和 GPU 费用按页数同比例变化。例如平均 150 页/套，以上批处理预算约为 540～1,580 元。')

doc.add_heading('六、前期试验与训练预算（不含在 1 万套批处理内）', level=1)
add_table(doc, ['项目', '建议内容', '保守 GPU 时间', '估算费用'], [
    ('模型基准测试', '用 500～1,000 页难例测试 PP-DocLayout、RT-DETR、PP-YOLOE+ 与视觉审核分流规则', '80～200 小时', '62～154 元'),
    ('首轮训练与调参', '训练 2～3 个候选检测模型，保留验证集并反复评估', '150～400 小时', '116～308 元'),
    ('合计建议预留', '用于证明路线是否可行，不应省略', '230～600 小时', '177～462 元'),
], [2000, 4100, 1700, 1560], 9.2)

doc.add_heading('七、关键风险与建议决策', level=1)
add_body(doc, '最大风险不是云显卡费用，而是没有经过核对的训练数据。云算力只能提高训练和批量运行速度，不能让模型自动理解“什么范围算完整一道题”。')
for t in [
    '先批准小规模验证：标注 500～1,000 页并训练首轮模型，而不是直接处理 1 万套。',
    '设立明确准入门槛：未见过的新教材测试集整题准确率未达到 86%，不进入全量批处理。',
    '采用高置信度放行机制：目标是 94%～97% 的可交付结果，低置信度页进入后续审核，不强行全自动。',
    '全量处理前用 1,000 页实测实际秒数、显存占用和难页比例，再最终锁定云卡数量和预算。',
]: add_number(doc, t)

doc.add_heading('八、公开模型资料说明', level=1)
add_body(doc, '本方案参考的公开资料包括：PaddleOCR 的 PP-OCRv6、PP-StructureV3、PaddleOCR-VL-1.6 与 PP-DocLayout-L 官方说明，以及 PaddleDetection 的目标检测训练能力。公开资料中 PP-DocLayout-L 面向书籍、试卷等版面；PP-OCRv6 用于文字识别；PaddleOCR-VL 用于文档结构化理解。')
add_body(doc, '重要说明：上述公开模型指标来自通用文档或 OCR 基准，并非本项目教材切题指标。本文件中的切题准确率均为方案级保守目标，必须以项目自有测试集实测验收。')

doc.add_paragraph()
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(0)
add_text(p, '方案版本：V1.0 | 估算日期：2026 年 8 月 | 本文费用仅含 GPU 时费，不含云硬盘、流量、标注、审核和人工管理成本。', size=8.5, color=(100, 100, 100))

doc.core_properties.title = '高精度自动切题模型组合与云算力预算方案'
doc.core_properties.subject = '教辅与试卷自动切题项目'
doc.core_properties.author = 'Codex'
doc.save(OUT)
print(OUT)
