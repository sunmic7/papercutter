from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


OUT = "教辅试卷自动切题项目_简明汇报版.docx"


def set_cell_shading(cell, fill="FFFFFF"):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="BFBFBF", size="4"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        elem = borders.find(tag)
        if elem is None:
            elem = OxmlElement(f"w:{edge}")
            borders.append(elem)
        elem.set(qn("w:val"), "single")
        elem.set(qn("w:sz"), size)
        elem.set(qn("w:space"), "0")
        elem.set(qn("w:color"), color)


def set_table_widths(table, widths_cm):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    grid = table._tbl.tblGrid
    for col, width_cm in zip(grid.gridCol_lst, widths_cm):
        col.set(qn("w:w"), str(int(Cm(width_cm).twips)))
        col.set(qn("w:type"), "dxa")
    for row in table.rows:
        for cell, width_cm in zip(row.cells, widths_cm):
            cell.width = Cm(width_cm)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(Cm(width_cm).twips)))
            tc_w.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_widths(table, widths)
    set_table_borders(table)
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for i, value in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = value
        set_cell_shading(cell, "FFFFFF")
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        run = p.runs[0]
        run.bold = True
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(0, 0, 0)
    for row_values in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row_values):
            cell = cells[i]
            cell.text = value
            set_cell_shading(cell, "FFFFFF")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            for run in p.runs:
                run.font.size = Pt(10.5)
                run.font.color.rgb = RGBColor(0, 0, 0)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.1
    r = p.add_run(text)
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


doc = Document()
section = doc.sections[0]
section.page_width = Cm(21)
section.page_height = Cm(29.7)
section.top_margin = Cm(1.8)
section.bottom_margin = Cm(1.8)
section.left_margin = Cm(1.85)
section.right_margin = Cm(1.85)
section.header_distance = Cm(1.25)
section.footer_distance = Cm(1.25)

# Plain Word-default style: black text, no colored fills or page furniture.
normal = doc.styles["Normal"]
normal.font.size = Pt(11)
normal.font.color.rgb = RGBColor(0, 0, 0)
normal.paragraph_format.space_after = Pt(5)
normal.paragraph_format.line_spacing = 1.1

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(4)
r = title.add_run("教辅与试卷自动切题项目")
r.bold = True
r.font.size = Pt(18)
r.font.color.rgb = RGBColor(0, 0, 0)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.paragraph_format.space_after = Pt(14)
r = subtitle.add_run("投入、难点与实施建议（简明汇报版）")
r.font.size = Pt(12)
r.font.color.rgb = RGBColor(0, 0, 0)

add_heading(doc, "一、结论")
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(6)
r = p.add_run("这不是一个简单的 OCR 项目。")
r.bold = True
r.font.color.rgb = RGBColor(0, 0, 0)
p.add_run("要把教材和试卷中的题目、答案准确裁开并正确配对，实际涉及页面识别、题目边界、答案判断、题答匹配、错误复核和持续训练。")

add_bullet(doc, "一名开发人员可以做出可演示的原型，但无法在短期内同时承担开发、数据整理、标注、测试、质检和持续维护。")
add_bullet(doc, "如果要求不同出版社、不同版式的教材都能稳定使用，不能承诺“一个人一个月完成、全部自动且准确率 97% 以上”。")
add_bullet(doc, "开源软件可以降低许可费用，但不能省掉数据标注、人工核对、训练算力和后续维护。")

add_heading(doc, "二、为什么一个人完成不现实")
add_table(doc,
          ["工作", "实际要做的内容"],
          [
              ("程序开发", "图像处理、OCR、切题、答案配对、批处理、断点续跑和结果导出。"),
              ("数据工作", "整理教材、统计版式、选择代表样本、制作训练集和测试集。"),
              ("标注与质检", "在整页上标出完整题目和答案；检查漏题、多切、少切、错配。"),
              ("模型训练", "训练题目/答案区域模型，分析失败原因，反复补充困难样本。"),
              ("生产维护", "处理新出版社、新题型、双栏、图文题、跨页题和扫描质量问题。"),
          ], [3.0, 13.3])

add_heading(doc, "三、合理的工作量与周期")
add_table(doc,
          ["阶段", "累计工作量", "一人独立完成的大致时间", "可达到的状态"],
          [
              ("原型 / MVP", "3-5 人月", "约 3-5 个月", "只验证少数固定版式，建立测试和复核流程。"),
              ("生产版", "8-14 人月", "约 8-14 个月", "主要已知教材可用，但仍需人工核对困难页。"),
              ("稳定版", "16-26 人月", "通常超过 1 年", "覆盖更多版式并持续降低人工率。"),
          ], [2.4, 3.0, 4.2, 6.7])

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(5)
r = p.add_run("建议的团队配置：")
r.bold = True
p.add_run("至少 1-2 名算法/视觉人员、1 名后端或数据人员，以及 1-3 名阶段性标注与质检人员。")

doc.add_page_break()
add_heading(doc, "四、数据和标注是精度的关键")
p = doc.add_paragraph("模型要从整页上学习“哪一块是一道完整题”，必须有原始页面和正确的题目框坐标。只保存已经裁好的小图，不能直接训练整页切题模型。")
p.paragraph_format.space_after = Pt(5)
add_table(doc,
          ["目标", "建议标注量", "说明"],
          [
              ("先验证可行性", "300-600 页", "覆盖主要版式，先找出最常见的错误。"),
              ("训练首个可用模型", "800-2,000 页", "约 5,000-15,000 组题目/答案关系。"),
              ("提高跨教材稳定性", "3,000-8,000 页以上", "需要覆盖新出版社、双栏、图文题、跨页题等难例。"),
          ], [3.8, 3.4, 9.1])

add_heading(doc, "五、建议的管理目标")
add_bullet(doc, "第一阶段只承诺：选定主要版式的试点、固定测试样本、可视化错误报告和人工复核流程。")
add_bullet(doc, "不要承诺：一个模型覆盖所有教材、一个月完成全部历史资料、无需抽检即可稳定达到 97% 以上。")
add_bullet(doc, "以“高置信结果自动输出，低置信结果进入复核”为原则，先建立数据闭环，再逐步提高自动化比例。")

add_heading(doc, "六、最终建议")
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(6)
r = p.add_run("建议先以团队项目方式立项。")
r.bold = True
p.add_run("一个人可以负责原型和技术验证，但如果目标是大规模、多教材、高精度生产，必须配套数据标注、测试质检和持续模型训练资源。否则即使程序能够运行，也很难保证结果质量和交付周期。")

note = doc.add_paragraph()
note.paragraph_format.space_before = Pt(8)
note.paragraph_format.space_after = Pt(0)
r = note.add_run("说明：上述人力、周期和数据量为工程估算，正式立项前应使用公司自有教材样本进行验证。")
r.italic = True
r.font.size = Pt(9.5)
r.font.color.rgb = RGBColor(0, 0, 0)

doc.core_properties.title = "教辅与试卷自动切题项目 - 简明汇报版"
doc.core_properties.subject = "项目投入、难点与实施建议"
doc.core_properties.author = ""
doc.core_properties.comments = ""
doc.save(OUT)
print(OUT)
