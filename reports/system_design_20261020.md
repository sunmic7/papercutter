# 切分算法改造设计：B → V 重构（v1.0）

日期：2026-10-20
作者：高见远（架构师）
状态：设计定稿，待工程侧实现；**本文不含任何代码改动**
上位口径：`reports/切分粒度统一规则_20261020.md`（产品已定版 v1.0）
适用范围：全部版式（exam / workbook / drill / holiday / olympiad / instructional / general）

---

## 0. 摘要：本次改造的一句话

**把「先按 x 切分、再补编号」改成「先按 B 系列算出编号辖域、再按 V 系列决定这条 x 线能不能切」，并给「证据不足」一个显式的第三态（水平带 + `needs_review`），而不是默认允许或默认合并。**

改造发生在 `split_major_into_minors` 内部的**决策顺序**上，不改版面裁判、不改策略层、不改任何已冻结文件。

---

## 1. 实测复核：根因与 briefing 描述有偏差（重要）

briefing 说根因是 `_side_by_side_columns` 把两列口算的列间隙当成栏缝。我按产品口径实测了 `_shot_br.png` 对应页
（`F:/2008-2015/2012/金版卷王名师面对面大试卷数学四年级下册北师大版/20210126153256260565.jpg`，4897×3377，对开页），
结论是：**该页的 `minor_extra=3` 并不是竖切造成的，竖切路径根本没触发**。实测证据：

```
# split_page_into_majors → 「六、计算。（30分）」块：_x_start=2434, _y_start=2085, 裁块 2463×1292
# find_minor_boundaries(local, 2463, 1292) 返回：
   num 1 primary y 112 x 232  ← '1.直接写得数。(6分）'   ✅ 真编号
   num 2 primary y 211 x 809  ← '2.7+0.2='             ❌ 假编号（口算）
   num 2 primary y 522 x 228  ← '2.用竖式计算。（12分）' ✅ 真编号
   num 3 primary y 608 x 285  ← '3.08+6.32='           ❌ 假编号（竖式）
# → split_major_into_minors 切出 4 段：cut=[1,2,2,3]，印刷=[1,2] ⇒ minor_extra=2

# 竖切路径实测：
any(boundary.get('side_by_side')) == False
_split_side_by_side_minors(...) -> None          # 未触发
_side_by_side_columns(...) 确实返回 4 栏 [(0,862),(785,1317),(1318,1919),(1838,2463)]   # 证据存在但没用上
```

假编号的**唯一来源**是 `find_minor_boundaries` 里的 `suspects` 救援通道（`cut_questions_v4.py:1771-1780`）：
`leading = re.match(r'^([0-9]{1,2})[\.．]\s*\S', text)` 会把 `2.7+0.2=` / `3.08+6.32=` / `10.67-4.07=` 全部
登记为「疑似题号」；而 `side_by_side_layout = crop_height < crop_width * 0.6`（1292 < 1477.8 → True）把
`left_limit` 从 `块宽×0.28` 放开到 `块宽`，于是这些 x=809/892/1518 的算式全部越过左限；
最后 `:1839-1853` 的「正好补上序列空缺就采纳」把它们插进边界序列。

**因此真实缺陷是两条独立缺陷，不是一条：**

| 编号 | 缺陷 | 命中规则 | 现状归属 |
|---|---|---|---|
| **F-1** | 小数算式被 `suspects` 通道提升为印刷编号，在同一小题内部制造假边界 | **D7**（小数 `3.5` 禁止作边界） | 未被 §9 差距清单列出 |
| **F-2** | `_side_by_side_columns` 用局部 OCR 左缘聚类当栏缝；`_split_side_by_side_minors` 缺 V-B/V-C/V-DENY(i) | **V-A / V-DENY(i)(iii)** | §9 第 1、2 条 |

F-1 是本案 `minor_extra≠0` 的**唯一直接原因**，且它同时制造了截图里「`10.67−4.07=` 被截断」的**真实机理**：
假编号 `2.7+0.2=`（y=211）把 `1.直接写得数` 与其 8 个口算切成两段，假编号 `3.08+6.32=`（y=608）把
`2.用竖式计算` 与其 3 个竖式切成两段；竖式行的三段算式跨了两个不同 `num` 的辖域，
后续 `_recover_missed_minor_boundaries` 与检测器精修再叠加 x 收窄，最终裁图左缘落在 `10.67−4.07=` 中间。

**这不改变本次改造的方向，但改变了改造点清单**：只做 V 系列会漏掉 F-1，只做编号辖域会漏掉 F-2。
下面 §2 的调用链同时覆盖两者，且 **F-1 的修法天然就是「先定编号归属」**——B 系列要求「编号必须是直读到的印刷编号」，
`D7` 明写小数不得作边界，所以 B 系列前置一步就够，不需要额外机制。

> 该结论已同步给 team-lead 与产品；下文按「F-1 + F-2 一并修」设计。

---

## 2. 改造点（函数级）

### 2.1 改造前后调用链对比

**改造前（现状，`_depth==0` 的正常路径）**

```
split_page_into_majors(img, ocr_items)
└─ _split_single_col_majors()                       # 产出大题块 crop_img + _column_bounds
   └─ split_major_into_minors(major_crop, ocr_items, padding, profile, _depth=0)
      ├─ find_minor_boundaries()                     # ① suspects 通道在此把 2.7+0.2= 提升为 num=2 【F-1 源头】
      ├─ find_unnumbered_boundaries()                # （无编号时）
      ├─ _recover_missed_minor_boundaries() ×3       # ② 原生分辨率重识别补跳号
      ├─ if boundaries and _depth == 0:
      │    _split_side_by_side_minors(...)           # ③ 先按 x 切，再补编号  【F-2 源头】
      │    ├─ _side_by_side_columns()                #    局部 OCR 左缘聚类 → 栏缝（违反 V-A）
      │    ├─ split_major_into_minors(_depth=1) × N  #    各列递归切分
      │    └─ 事后校验 max(num)-min(num)+1 == len     #    唯一门禁，放过 V-DENY(i)
      └─ for i, bd in enumerate(boundaries):         # ④ 按 y 一刀一刀切
            y_start = 0 if i==0 else bd['y'] - padding
            y_end   = _find_clean_horizontal_cut()
```

**改造后（B → V）**

```
split_page_into_majors(img, ocr_items)
└─ _split_single_col_majors()
   └─ split_major_into_minors(major_crop, ocr_items, padding, profile, _depth=0,
                              column_evidence=⟨新增⟩, scope=⟨新增⟩)
      ├─ ① find_minor_boundaries_strict()            # 【改】D7 前置：算式小数不算编号
      │      └─ 只认直读印刷编号；小数/页码/选项一律不登记为候选
      ├─ ② build_number_territories()               # 【新】B 系列：先算「编号辖域」
      │      ├─ 按 scope（L1 标题 / 练习 / 例题）分组，编号从 1 起算
      │      ├─ 每个最内层编号 → 一条 territory{y0,y1,level,num,number_path}
      │      ├─ L3：(1)(2) / ①②③ / 例N-M 归属到父级 territory
      │      ├─ D2 的 P 区判据：P 是指令 → 不生成记录；P 是材料 → region_role=group
      │      └─ D6/D7/D8 判定，产出 boundary_source 与 minor_number_inferred
      ├─ ③ find_unnumbered_boundaries()              # 无编号时的兜底（D3/D4）
      ├─ ④ _recover_missed_minor_boundaries() ×3    # 保留，但只在 territory 序列内补跳号
      └─ ⑤ plan_vertical_cuts()                     # 【新】V 系列：对每条候选 x 跑判定
             ├─ 候选来源 = _column_evidence 里的页面级栏缝（不再来自 OCR 左缘聚类）
             ├─ 逐条跑 V-ALLOW(A~E) 全满足 且 V-DENY(i~v) 全不命中 → 允许竖切
             ├─ 允许 → 该 territory 一分为二，各自重算 y 边界（§2.4 收窄版）
             └─ 不允许 → territory 保持整块；若「有并排编号证据但 V-A 不可用」
                        → 标记 needs_review + vertical_reason，**按水平带出样本**
```

### 2.2 逐函数改造清单

| 函数 | 处置 | 说明 |
|---|---|---|
| `find_minor_boundaries()` | **收窄 + 新增严格版** | 保留原函数签名与行为供 `_primary_number_boundaries`（`manual_annotation_workspace.py:551`）等既有调用方使用；新增 `strict=True` 参数，`split_major_into_minors` 走严格版。严格版在登记候选前先跑 D7 拦截：`text` 以「数字 + `.`/`．` + 数字」开头即整条丢弃（`2.7+0.2=`、`3.08+6.32=`、`10.67-4.07=` 全中），`A.`~`D.` 选项、`（ ）`、小数、`P12`、`第N题` 同样丢弃。**这一处即修掉 F-1。** |
| `_recover_missed_minor_boundaries()` | 保留，收窄入参 | 补回的编号必须落进某个已存在的 territory 缺口；补不回就保持现状。不改内部逻辑。 |
| `_side_by_side_columns()` | **降级为纯几何工具，删除切分职责** | 保留函数（`_experiments/draw_columns.py`、`tests/test_smart_cutter.py::SideBySideMinorTests` 引用它），但其返回值**只能**作为 `plan_vertical_cuts` 的候选线来源之一，且必须过 V-DENY(i)(iii)(iv)(v) 才能被采用。**不再有任何代码路径直接把它的 spans 当裁切区间。** |
| `_split_side_by_side_minors()` | **删除**，能力上移并收窄到 `plan_vertical_cuts` | 它做的事被拆成三处：①候选线枚举 → `plan_vertical_cuts`；②"V 系列门禁" → 新增；③"各列递归切分" → 仅在 V 判定通过后由调用方执行，函数本身不再递归（避免 `_depth` 双入口）。删除理由：它的存在前提「先切后补编号」已被规则判死，且它内部 `column_of()` 会把 `3.` 判给第 2 列（`reports/横排小问切分_20261019.md` §3.4 已记录），与 B 系列的辖域归属冲突。**单测 `SideBySideMinorTests` 需要随之调整（见 §7 T04）。** |
| `split_major_into_minors()` | **改造为薄壳** | 签名新增两个**可选关键字参数**（默认 `None`，老调用方零行为变化）：`column_evidence=None`、`scope=None`。函数体只剩「建辖域 → 判竖切 → 出 cuts」三步，不再自己实现编号序列校验。 |
| `attach_question_source_metadata()` | 扩展 | 新增把 territory 的 `_answer_slots` / `_bbox_parts` / `_context_ref` 透传到 `minor` 字典。 |
| `build_question_metadata()` | 扩展 | `prediction` 段新增 4 个字段（见 §4），`schema_version` 2 → 3。 |
| `write_paperflow_project()` | 扩展 | 写 `section_role`（沿用现有 `region_role` 取值，不新增枚举值）、`answer_granularity`、`answer_slots`、`bbox_parts`、`context_ref`。 |
| `layout_strategy.py` | **新增 3 个开关**（不改动任何现有字段） | `strict_minor_boundary`（F-1 门禁，默认全版式 `True`）、`vertical_cut_default_deny`（V 默认禁止，默认 `True`）、`slot_order`（`column_first`/`row_first`，默认 `column_first`）。全部进 `DESCRIPTIONS`，符合该文件"新增开关必须给中文说明"的约定。 |
| `layout_arbiter.py` | **不动**（冻结） | 只作为 `column_evidence` 的**供给方**被读取（`detect()['columns']` / `['gutters']`，二者都是既有键）。 |
| `review_tools/cut_quality_audit.py` | 扩展 3 个指标 | `vertical_cut_false` / `frame_intersected_cuts` / `text_truncated`（见 §6）。 |

### 2.3 B → V 判定流程（核心）

```
                     ┌─────────────────────────────────────────┐
                     │ 输入：major_crop, ocr_items, column_evidence│
                     └──────────────────┬──────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ 阶段 B：编号辖域（唯一权威：谁拥有哪块像素）                      │
   ├────────────────────────────────────────────────────────────────┤
   │ B0 scope 划分：L1 标题 / 练习·做一做·举一反三 / 例N  各开新 scope  │
   │ B1 严格编号候选（D7 拦截小数/选项/页码/题内引用）                │
   │ B2 圈号 D6：行首 + 题干≥4汉字 才算编号，否则是图注                │
   │ B3 例题 D5：例N 正文独立成样本；其后练习另起 scope               │
   │ B4 D2 P 区判据：P_h<2.5·line_h ∧ slots(P)==0 ∧ 命中指令词 ∧ ≤20字│
   │                 → 指令：不生成记录，文本进 subsection_title      │
   │                 → 否则：材料，生成 region_role='group'            │
   │ B5 L3 归属：(1)(2)/①②③ 挂父级 territory，number_path 加深一层    │
   │ B6 锚点（B1~B6）：y_start/y_end，末题=块高，空白带吸附，          │
   │                 括号/横线/图不得跨切（B5/B6）                    │
   └──────────────────────────┬─────────────────────────────────────┘
                              ▼  territories = [{y0,y1,num,path,level}, ...]
   ┌────────────────────────────────────────────────────────────────┐
   │ 阶段 V：每条候选竖切线独立判定（默认不竖切）                      │
   ├────────────────────────────────────────────────────────────────┤
   │ 候选线来源（按证据强度排序，全部必须是页面级）：                  │
   │   1) layout_arbiter.detect()['columns'] 的相邻列边界            │
   │   2) _visual_column_split() 的连续纯留白（≥60% 页高）            │
   │   3) _visual_vertical_divider() 的实线                          │
   │   ✗ 不接受 _side_by_side_columns 的 OCR 左缘聚类                │
   │                                                                │
   │ 逐条跑（顺序固定，任一 DENY 立即否决）：                          │
   │   V-DENY(i)   切线任一侧在 territory 内无编号行 → 否决           │
   │   V-DENY(ii)  切线上下各 1 行高内无编号行 → 否决                 │
   │   V-DENY(iii) 缝宽 < max(行高, 页宽3%) ∧ 两侧块宽均 < 列宽45%      │
   │              → 否决（两列口算列间隙 20~60px 命中此条）           │
   │   V-DENY(iv)  切线与闭合矩形框相交（竖式框/表格线）→ 否决         │
   │   V-DENY(v)   缺口两侧同属一个 number_path 辖区 → 否决           │
   │   ─────────── 以下全过才继续 ───────────                        │
   │   V-A 栏缝有页面级独立视觉证据                                    │
   │   V-B 每列列顶 25% 内有自己的编号行，且左缘 ≤ 列宽×28%            │
   │   V-C 各列编号不同且连续（左列最大号+1=右列最小号）                │
   │   V-D 切线与任何闭合矩形框的竖直边交点数 = 0                      │
   │   V-E 无横跨切线的文本行（复用 has_spanning_body_text）           │
   └──────────────────────────┬─────────────────────────────────────┘
                              ▼
              ┌───────────────┴───────────────┐
        V 全通过                        任一 DENY / 证据不足
              ▼                               ▼
    territory 一分为二                ┌──────────────────────────┐
    各自重算 y 边界（B 系列）          │ 三态判定：                │
    _x_start/_x_end 写窄              │  ① 有并排编号证据        │
                                       │     (territory 内≥2 个  │
                                       │      编号 + 列聚类有缝) │
                                       │     但 V-A 无页面级证据 │
                                       │     → 水平带 + 竖直虚线   │
                                       │       标记 needs_review  │
                                       │       + vertical_reason  │
                                       │       ='no_page_evidence'│
                                       │       （见 §3）           │
                                       │  ② 无并排编号证据        │
                                       │     → 整块 1 条（默认）  │
                                       │  ③ 部分满足（如 V-A 仅   │
                                       │     OCR 聚类证据）        │
                                       │     → 同 ①，且进 L2      │
                                       └──────────────────────────┘
```

**关键设计点**：V 判定的输入是「territory（纵向）⊕ 候选线（横向）」，
而现状是「列（横向）⊕ 编号（事后补）」。同一批 OCR 数据，只是**先算哪个**不同。

### 2.4 需要哪些几何量：现有 vs 新增

| 几何量 | 用途 | 现状 | 处置 |
|---|---|---|---|
| 缝隙宽度 `gutter_w` | V-DENY(iii)、V-A | `_side_by_side_columns` 内部算但**不返回**；`_local_column_split_candidates` 返回 `width` | 从 `column_evidence['gutters']` 直取；无则由 `_visual_column_split` 的返回中线 ± 留白宽度估算 |
| 中位行高 `line_h` | V-DENY(ii)(iii)、B4/B5/D2 的 P 区判据 | `find_minor_boundaries` 内部 `typical_height`（局部变量）；`find_unnumbered_boundaries` 内部同名变量 | **新增**：`_median_line_height(ocr_items) -> int`，三处改为调用它（消除重复计算，口径统一） |
| 两侧文本块宽 `left_w/right_w` | V-DENY(iii)、V-C | `_side_by_side_columns` 内部 `narrow_limit` | **新增**：`_column_content_width(items, x0, x1) -> int` |
| 闭合矩形框检测 | V-DENY(iv)、V-D | **完全没有**。`cut_questions_v4` 无 `findContours`/矩形检测；`layout_arbiter` 有 `table` label 但语义不同 | **新增**：`_closed_rect_boxes(crop) -> list[[x0,y0,x1,y1]]`，用 `_content_mask` + `cv2.findContours(RETR_EXTERNAL)` + 面积/长宽比/闭合度三重过滤。这是唯一必须新写的图像算子 |
| 行宽跨线检测 | V-E | `has_spanning_body_text(ocr_items, split_x, height, width)` **已有**，口径与规则一致 | 直接复用，只调整入参为块内局部坐标 |
| 页面级栏位证据 | V-A | `layout_arbiter.detect()['columns'/'gutters']`、`_visual_column_split()`、`_visual_vertical_divider()` 均已有 | **新增一个取数函数** `_collect_column_evidence(img, profile) -> dict`，把三个来源统一成 `{columns, gutters, source}`，`source ∈ {'arbiter','visual','none'}` |
| 编号行 y / 左缘 | V-B、V-DENY(i)(ii) | `find_minor_boundaries` 返回的 `y`/`x` 已有 | 直接用 |

**新增函数签名建议**（共 6 个，全部纯函数、无 IO、可单测）：

```python
def _median_line_height(ocr_items) -> int
    """块内中位行高。三个调用点共用，消除口径分裂。"""

def _column_content_width(ocr_items, x0, x1) -> int
    """给定列区间，返回该区间内文本块的代表性宽度（取窄块中位数）。"""

def _closed_rect_boxes(crop) -> list
    """检出块内闭合矩形（竖式框/作答框/表格线外框）。
    返回 [[x0,y0,x1,y1], ...]；检不到返回 []。
    实现：_content_mask → findContours(RETR_EXTERNAL) → 面积≥块面积0.15%
    且 长宽比∈[0.15,6] 且 4 边墨迹覆盖率≥0.7。"""

def _collect_column_evidence(img, profile='general') -> dict
    """页面级栏缝证据。返回
    {'columns': [(x0,x1)], 'gutters': [{'x','width'}], 'source': 'arbiter'|'visual'|'none'}。
    裁判不可用时降级到 _visual_column_split，再不行 source='none'。"""

def build_number_territories(major_crop, ocr_items, *, scope=None, profile='general',
                             padding=4, strict=True) -> list
    """【B 系列核心】先定编号归属。
    返回 [{'y0','y1','level','num','number_path','minor_number_inferred',
           'boundary_source','instruction','material_box','slots'}]
    内部跑 D1~D8 + B1~B6。strict=True 时启用 D7 小数拦截。"""

def plan_vertical_cuts(territories, ocr_items, column_evidence, major_crop, *,
                       line_h, page_width) -> list
    """【V 系列核心】对每条候选竖切线跑 V-ALLOW/V-DENY。
    返回 [{'x','allowed','deny_reason','needs_review','reason_detail'}]
    —— 不做任何裁切，只出判定结果（纯判定，好测）。"""
```

**为何把 V 判定做成纯函数**：规则 §7 有 4 条红线指标全靠它。
纯函数 `plan_vertical_cuts` 可以直接对着构造的 `ocr_items` 做表驱动单测，
不需要跑 OCR、不需要模型、不需要图片——单测成本从「分钟级 + 依赖模型」降到「毫秒级」。
这也是把 `_split_side_by_side_minors` 删掉换来的最大收益。

---

## 3. V-A 证据来源与降级的第三种处理

### 3.1 证据来源（严格按规则）

| 优先级 | 来源 | 判定为「页面级独立视觉证据」的条件 |
|---|---|---|
| 1 | `layout_arbiter.detect()['columns']` | 裁判 `available()` 为真且返回 ≥2 栏 |
| 2 | `_visual_column_split(img)` | 返回非 None（内部已要求连续纯留白 ≥ 页高 82% 区间 + 两侧墨量对称 + 缝宽 ≥ 1.8% 页宽） |
| 3 | `_visual_vertical_divider(img)` | 返回非 None（实线 + 两侧正文 + 显著独立于runner-up） |
| ✗ | `_side_by_side_columns(...)` 的 spans | **不算证据**。它是块内局部 OCR 左缘聚类，正是 V-A 明确排除的对象 |

**现状的差距**：`layout_arbiter.detect()` 是**按整页图**推理的，而 V 判定发生在**块级**。
`_collect_column_evidence` 必须拿到整页 img（`split_page_into_majors` 里有），
把页面级栏位换算成块内局部 x，再喂给 `plan_vertical_cuts`。
若只传块图给裁判，裁判看到的是裁剪后的伪页面，`MIN_PAGE_WIDTH=1200` 会让多数块直接返回 None——**这是降级路径最容易被忽略的坑**。

### 3.2 裁判不可用时的降级：第三种处理

规则禁止两种默认：**不能默认允许竖切**，**也不能把并排算式全并成一块**。第三态设计如下：

```
if V 系列全通过:
    → 竖切（现有横排并排小题形态保留）

elif V-DENY 命中（证据充分地否决）:
    → 整块 1 条，不竖切，不标记
      （这是绝大多数情况：口算两列、竖式三框都会命中 V-DENY(i)(iii)(iv)）

else:  # V-ALLOW 部分满足 —— 典型是 V-A 只有 OCR 聚类证据，或压根没有页面级证据
    → 【第三态】水平带 + needs_review
       1. 样本仍按 B 系列出，**保证 1 个编号 = 1 条样本**（与规则 §4.1 一致）
       2. record 上写：
            needs_review        = True
            review_level        = 'L2'
            vertical_cut_status = 'degraded_to_horizontal_band'
            vertical_reason     = 'no_page_evidence' | 'ocr_cluster_only'
            vertical_candidates = [{'x': ..., 'gutter_w': ..., 'left_w': ...,
                                    'right_w': ..., 'deny': 'V-DENY(iii)'}, ...]
       3. 裁图上**画一条竖直虚线**标注"这里原本可能能竖切"，
          让人工一眼看出分歧点，而不是重新框一遍
       4. 进 L2 队列（规则 §8 L2-9）
```

**为什么这是「第三种」而不是妥协**：
- 对**用户**：1 条样本 = 1 道题，答案槽位完整（8 个口算都在），不会得到 7 条无题干的碎片；
- 对**算法**：`minor_extra=0`（没有多切），`minor_missed=0`（没有漏切），两个红线同时干净；
- 对**改进**：降级点被完整记录，下一次规则升版（拿到更可靠的页面级证据、或允许某一版式例外）可以
  **按 `vertical_candidates` 精确回放**哪些页本可以竖切，不需要重跑全量。

**明确禁止的降级姿势**（写进代码注释与单测）：
- ❌ 裁判不可用 → 默认允许竖切（会退回 F-2）；
- ❌ 裁判不可用 → 静默合并（丢失 `needs_review`，人看不到分歧，L2-9 失效）；
- ❌ 裁判不可用 → 退回 `_side_by_side_columns` 当证据（这就是现状，正是缺陷本体）。

---

## 4. 字段扩展

### 4.1 产出与消费

| 字段 | 谁产出 | 存在哪 | 谁消费 | 与现有字段的关系 |
|---|---|---|---|---|
| `answer_granularity`<br>`'slot_list'` / `'single'` | `build_number_territories()` 计算 `slots` 数量后确定；`build_question_metadata()` 与 `write_paperflow_project()` 落盘 | `cut_questions_v4` 的 minor 字典 → `metadata.json` 的 `prediction` 段 → PaperFlow `record` | enrich 阶段算 `slot_count_mismatch`（A6）；导出与交付门禁 | **新增**。不与 `region_role` 冲突：`region_role` 说「这条记录是什么角色」，`answer_granularity` 说「答案怎么组织」 |
| `answer_slots[]`<br>`{slot, expr, xy}` | `build_number_territories()` 从 territory 的 y 区间内按 §5 排序产出 `xy`；答案文本由 enrich 阶段回填 `answer` | 同上 | enrich 回填、答案页配对（A2/A5）、UI 逐槽展示 | **新增**。与 `number_path` 正交：一个 `number_path` 对多个 slot |
| `bbox_parts[]` | `merge_cross_page_question()` / `peel_leading_cross_page_fragment()` 已产出 `_source_segments`，改造时映射为 `bbox_parts` | minor 字典 → `record` | 跨栏/跨页裁图（C4/C5）、UI 分段显示 | **新增公开字段**，**复用现有内部数据**（`_minor_source_segments` 已在 `build_question_metadata` 输出 `source.segments`）。零新增几何计算 |
| `context_ref` | D2 判为材料时，`build_number_territories()` 给子样本写 group 记录的 `_pf_id` | `record` | UI 跳转、enrich 取材料上下文 | **新增**。与现有 `shared_context_segments`（图/表前置区）**分工不同**：`shared_context_*` 是"同一裁图里额外拼进去的像素段"，`context_ref` 是"指向另一条记录"，材料不重复裁进子样本 bbox（规则 §5 形态 6 明确要求） |
| `region_role` | 已有，**不新增枚举值** | 已有 | 已有 | 规则里的 `section_role` 就是现有的 `region_role`（`{question, group, continuation, shared_material}`，`manual_annotation_data.py:23`）。`shared_material` 在 `normalize_record` 里已归一为 `group` |
| `number_path` | `build_number_territories()` 产出任意深度（L1→L2→L3） | 已有 | 已有 | 已有实现已支持任意深度（`_normalize_number_path` 截断到 8 层，`number_path()` 三级回退）。L3 只是让 `path` 多一层，**不改数据结构** |
| `minor_number_inferred` | `build_number_territories()` 在 D3/D4/D8 时置 True | 已有 | `review_levels.py:282` 已消费 → L2 | 已有消费方，**只补产出** |

### 4.2 向后兼容（硬要求）

老工程读出来不能崩，四条措施：

1. **只加不改**：`region_role` / `number_path` / `minor_number_inferred` / `major_title` 的**取值域与语义一律不变**；
   `section_role` 不作为新字段落盘，映射到现有 `region_role`。
2. **默认值由 `manual_annotation_data.normalize_record()` 统一补齐**（该文件**不在冻结清单**内，可改）：
   在现有 `ENRICH_NORMALIZE_DEFAULTS` 旁边**另起**一组
   `CUT_V4_NORMALIZE_DEFAULTS = {'answer_granularity': 'single', 'answer_slots': [],
   'bbox_parts': [], 'context_ref': ''}`，
   并入 `normalize_record` 末尾。`apply_enrich_defaults()` 同样补（老工程读出来字段齐全）。
   **不复用 `ENRICH_PERSISTENT_KEYS`**：那 14 个键受 `PAPERFLOW_ENRICH` 开关控制，
   而 cut 字段必须**无条件**存在，否则关掉 enrich 的工程读出来就缺字段。
3. **标注工作台保存不丢字段**：`paperflow_annotation_api.py:915-930` 是**白名单重建**路径
   （`new_record(...)` 逐字段列举 + `ENRICH_PERSISTENT_KEYS` 回接）。新增字段若不回接，
   人工在 UI 上保存一次就会**抹掉**后端算的槽位。
   → 该文件**已冻结**，故在 `manual_annotation_data.py` 里把四个键加进一个**新的**常量
   `CUT_PERSISTENT_KEYS`，并由 `paperflow_annotation_api.py` 消费——**这构成一条解冻申请（见 §9）**。
   备选（不动冻结文件）：把四个键并入现有 `ENRICH_PERSISTENT_KEYS`。
   **不推荐**：那会让 cut 字段被 `PAPERFLOW_ENRICH=0` 一起关掉，违反上条。
4. **`schema_version` 2 → 3**：`build_question_metadata` 升版；消费方按 `>= 3` 读新字段，
   老版本（`== 2`）走 `.get()` 兜底。`desktop_annotation` 侧已有 `annotation_version = 5` 的
   `setdefault` 模式可照抄。

---

## 5. 答案槽位顺序（唯一答案：栏优先）

**结论：栏优先（column-first）**，即两列四行口算 = 左列 1~4、右列 5~8。规则 §4.2 A1 已定，此处给出**推导方式**与**判据**。

### 5.1 为什么是栏优先（而不是行优先）

| 依据 | 内容 |
|---|---|
| ① 全链路一致 | `split_page_into_majors` 的阅读顺序就是「先左栏到底再进右栏」（`cut_questions_v4.py:4438` `return left_blocks + right_blocks`）。栏优先与它同构。选行优先会让「切分顺序」与「页面上从上到下从左到右的视觉顺序」在两列情形下不一致 |
| ② 规则已定 | A1 明确「与 `split_page_into_majors` 的『先左栏到底再进右栏』阅读顺序一致，全链路只有一种顺序」 |
| ③ 答案页可复算 | A2 要求「同一排序函数作用在答案块上」。答案页若也是两栏四行，栏优先复算即得同一顺序；行优先在答案页排成一行时会错位 |
| ④ 竖式场景一致 | A3 竖式按栏优先；本案竖式是**横排 3 个**（`3.08+6.32=` / `10.67−4.07=` / `18.64+9.32=`），栏 = 3 个框，栏内槽位 = 1，栏优先退化为「从左到右」，与 A3 不冲突 |

### 5.2 如何从 bbox 推导

不需要 OCR 语义，只用**槽位 bbox 的几何聚类**：

```
1. 取该 territory 的全部作答位候选框 S = {slot_bbox}（来自 OCR 项 + 闭合矩形框顶边）
2. 估算列数 k：
     用一维 x 中心聚类，间隙阈值 = max(1.2 × line_h, 列宽 × 6%)
     k = 聚出的列数。k=1 → 单列，顺序即 y 升序
3. 栏优先排序键 = (列序号, 列内 y 升序)
     列序号 = 该槽位 x 中心所属列的升序编号
4. slot 编号 = 排序后从 1 开始连续编号
```

**`78×199≈16000()` 这类场景的判据**：该形态是**单列**（一个算式占满一行，
或两三个算式在同一行内横排但**列间隙 = max(1.2×line_h, 列宽×6%) 达不到**，因为两列之间没有真实栏缝，
只有字间距）。此时 k=1，槽位顺序退化为 y 升序，判据成立。

反之，若 `78×199≈` 与 `16000()` 被 OCR 读成两个独立框且 x 间隙超过阈值，k=2，
会得到「`78×199≈` 在左、`16000()` 在右」的**错误栏序**。
**防御**：栏聚类前先做**行合并**——两个框 y 区间重叠 ≥ 60% 且属于同一行时，
按 x 拼成一个槽位（它们的 x 是连续的，属于同一算式的不同部分）。
这条防御与 §2.4 的「闭合矩形框」检测共用 `_content_mask`，不新增算子。

### 5.3 一致性校验（写进单测）

- 两列四行 8 个口算 → `slot` 1~8，左列 1~4、右列 5~8；
- 同一 `expr` 文本的槽位，`slot` 在两次运行间稳定（几何确定性，无随机）；
- 把排序函数作用在**答案页**对应块上，得到的顺序必须与题面一致，否则 `answer_order='unverified'` → L2（A2）。

---

## 6. 回归与验收方案

### 6.1 基线（已实测确认，`reports/audit_baseline_v2_fixed.jsonl`，60 页 / 20 本 / 6 版式）

| 指标 | 实测值 |
|---|---|
| 大题召回 | **80/82 = 97.6%** |
| 小题召回 | 1 − 27/286 = **90.6%** |
| 有错页 | **17/48** |
| `minor_extra`（现有工具口径：切出集合 − 印刷集合） | **16** |
| `minor_extra`（规则口径：同块内 `max(0, 切出数 − 印刷数)` 正差） | **51**（分布在 9 个过切块 / 140 块） |
| 软指标 | `untitled_with_evidence=4`、`merged_blocks=1`、`duplicate_title_blocks=0`、`collapsed_blocks=2`、`empty_blocks=1`、`undetected=8`、`tiny_cuts=0`、`giant_cuts=1`、`bad_quality=0`、`seq_breaks=4` |

**必须说清的口径事实**：规则 §7.1 写 `minor_extra` 目标 = 0，但**现状是 16（工具口径）/ 51（规则口径），不是 1**。
briefing 里「本案现状 = 1」指的是**单页**（本案页 `extra=[3]`，工具口径计 1）。
因此本次改造的红线**不能**写成「全量 `minor_extra` 归零」——那需要同时消灭 16 个存量 over-cut，
远超本次范围，且会把 B 系列改造的风险面扩大一个数量级。

**本次改造采用的红线定义**（写进验收脚本）：

```
R1  本案专项页 minor_extra = 0            （单页，规则 §7.3 已定）
R2  全量 minor_extra ≤ 16                （不高于现有工具口径基线）
R3  vertical_cut_false = 0
R4  frame_intersected_cuts = 0
R5  text_truncated = 0
R6  major_recall ≥ 97.6%（不得低于 80/82）
R7  minor_recall ≥ 90.6%（不得低于 259/286）
R8  10 项软指标逐项 ≤ 基线
R9  有错页 ≤ 17/48
```

同时**并列上报**（不作为红线，但必须给出数字）：
`minor_extra`（规则口径 51）、`text_contained_rate`（规则 §7.1 要求 ≥ 0.995）。

### 6.2 跑哪些脚本

```bash
# ① 60 页固定审计集（与基线同集合、同版式）
python review_tools/cut_quality_audit.py --pages-from reports/audit_plan_v2.jsonl \
       --out reports/audit_after_v3.jsonl
python review_tools/cut_quality_audit.py --summarize reports/audit_after_v3.jsonl \
       --json-out reports/audit_after_v3_summary.json

# ② 与基线逐页逐字段 diff（必须用固定清单，不能用 --sample）
python _experiments/diff_audit.py reports/audit_baseline_v2_fixed.jsonl reports/audit_after_v3.jsonl

# ③ 本案专项页（规则 §7.3 的 5 条断言）
python review_tools/cut_quality_audit.py --pages-from reports/audit_plan_v3_case.jsonl \
       --out reports/audit_case_v3.jsonl

# ④ 全量单测
python -m unittest discover -s tests -q

# ⑤ 策略层不回归（layout_strategy 改了 3 个开关）
python _experiments/p1_recheck.py reports/audit_baseline_v2_fixed.jsonl reports/strategy_v3.jsonl
python _experiments/compare_audit.py reports/strategy_before.jsonl reports/strategy_v3.jsonl
```

> `reports/audit_plan_v3_case.jsonl` 需新建（1 行，指向 `_shot_br.png` 对应页）。
> 这是**新增数据文件**，不改任何既有审计集，保证基线可比。

### 6.3 比对哪些字段

| 类别 | 字段 | 判据 |
|---|---|---|
| **必须变好** | `minor_extra`(本案页) | → 0 |
| | `blocks[].cut_numbers`（本案页「六、计算」块） | `[1,2,2,3]` → `[1,2]` |
| | `blocks[].diagnostic.extra`（本案页该块） | `[3]` → `[]` |
| | `text_truncated`（本案页） | ≥1 → 0 |
| | `frame_intersected_cuts`（本案页） | ≥1 → 0 |
| **不得变差** | 10 项软指标、R6/R7/R8/R9 | 见 §6.1 |
| **逐页逐字段 diff** | `title` / `title_idx` / `col` / `bounds` / `y_start` / `height` / `printed_majors` / `cut_numbers` / `minor_missed` | 除本案页与被明确豁免的页外，**必须完全一致** |
| **豁免白名单** | 需在 diff 报告里显式列出并说明理由 | 不允许「 unexplained 差异」 |

### 6.4 回退判据（任一条命中 ⇒ 判定本次改造失败，必须回滚）

| # | 触发条件 | 为什么是死线 |
|---|---|---|
| **RB1** | 全量 `minor_missed > 27`（即 `minor_recall` < 90.6%） | 改造把真题切没了，比多切更严重：直接丢样本 |
| **RB2** | 全量 `major_recall < 97.6%` | 大题层被 B 系列改动破坏 |
| **RB3** | `major_recall` 或 `minor_recall` **任一相对基线下降**（即使仍 ≥ 目标值） | 存量换新量的交易不划算；本次改造只许在 `minor_extra` 上赢 |
| **RB4** | 任一软指标**上升** | 说明 B 系列锚点改动外溢到了别的版式 |
| **RB5** | `continuation_blocks`（跨栏/跨页续题）行为变化：跨页续题被拆成两条 `question` | 破坏 C1~C3，既有行为回归 |
| **RB6** | 有错页 > 17/48 | 综合退步 |
| **RB7** | 单页平均耗时增幅 > 40%（基线 `avg_seconds`） | `_closed_rect_boxes` 每块跑一次 `findContours`，必须实测；若超预算则改为「仅对含 `=` 结尾算式的块检测」的短路 |
| **RB8** | 本案页切出**不是恰好 2 条**样本，或样本 1 的槽位数 ≠ 8 | 规则 §7.3 的 5 条断言是本次改造的**唯一目标页**，不达成即改造无效 |
| **RB9** | `vertical_cut_false > 0` | 规则红线，且说明 V-DENY 没拦住 |

**回滚方式**：`layout_strategy` 三个新开关全部可通过 `strategy.with_overrides(...)` 关掉
（`strict_minor_boundary=False` / `vertical_cut_default_deny=False`），
配合 `git revert` 单个提交即可。**设计上保证「关掉开关 = 回到旧行为」，不依赖 `PAPERFLOW_ENRICH` 之类的环境变量。**

---

## 7. 任务分解（4 个，按依赖顺序）

> 硬约束：T01/T02/T03 已冻结的 7 个文件**不得改动**。本方案**不需要解冻它们**即可完成 T01~T03；
> T04 需要一条解冻申请（`paperflow_annotation_api.py`），见 §9。

### T01 · 数据与几何层（可独立验证的最小改动）

**Task ID**: T01
**任务名**: 切分几何算子与页面级栏位证据采集
**源文件**:
- 新增 `cut_geometry.py`（`_median_line_height` / `_column_content_width` / `_closed_rect_boxes` / `_collect_column_evidence`）
- 修改 `layout_strategy.py`（新增 3 个开关 + `DESCRIPTIONS` 条目）
- 新增 `tests/test_cut_geometry.py`
- 修改 `tests/test_layout_strategy.py`（新增开关的默认值与 `with_overrides` 断言）

**依赖**: 无
**优先级**: P0

**验收方式**（可跑的断言）:
```bash
python -m unittest tests.test_cut_geometry tests.test_layout_strategy -q
```
- `_closed_rect_boxes(合成图)`：画一个 200×120 的空心矩形 → 返回 1 个框且坐标误差 ≤ 2px；
  画 5 条随机横线 → 返回 `[]`；空图 → `[]`。
- `_median_line_height`：给定 5 个高度 `[20,22,21,19,200]` → 返回 21（中位数，不被离群值带偏）。
- `_collect_column_evidence(空白图)` → `source == 'none'`；
  `_collect_column_evidence(双栏合成图)` → `source in {'arbiter','visual'}` 且 `len(columns) == 2`。
- `layout_strategy.strategy_for('exam').to_dict()` 新增 3 键且默认值正确；
  `with_overrides(strict_minor_boundary=False)` 抛 `ValueError`（未知键）与不抛（已知键）各测一次。

**注意**: 本任务**不改** `cut_questions_v4.py` 任何一行 —— 新算子先落地但不接线，
保证 T01 单独合入时全量单测与审计结果**逐字节不变**。

---

### T02 · B → V 决策链改造（核心）

**Task ID**: T02
**任务名**: 编号辖域前置 + 竖切门禁 + `_split_side_by_side_minors` 下线
**源文件**:
- 新增 `cut_numbering.py`（`build_number_territories` / `plan_vertical_cuts` / D2 的 P 区判据 / D5~D8 判定）
- 修改 `cut_questions_v4.py`（`find_minor_boundaries(strict=)`；`split_major_into_minors` 改薄壳并新增 2 个可选参数；删除 `_split_side_by_side_minors`；`attach_question_source_metadata` 透传新字段）
- 修改 `_experiments/p1_recheck.py`（`_split_side_by_side_minors` 调用点如有需同步）
- 修改 `tests/test_smart_cutter.py`（`SideBySideMinorTests` 改为断言 V 判定结果）

**依赖**: T01
**优先级**: P0

**验收方式**:
```bash
python -m unittest tests.test_smart_cutter tests.test_cut_numbering -q   # 新增 test_cut_numbering.py
python review_tools/cut_quality_audit.py --pages-from reports/audit_plan_v3_case.jsonl \
       --out reports/audit_case_v3.jsonl
python -m unittest discover -s tests -q
```
- **本案专项断言（规则 §7.3 五条）**：切出恰好 2 条；`number_path == ['6','1']` / `['6','2']`；
  样本 1 有 8 个槽位且 `answer_granularity == 'slot_list'`；样本 2 的 bbox 完整包含 3 个竖式；
  `text_truncated == 0`；`minor_extra == 0`；两条 `major_title` 都以「六、计算」开头且无第三条空壳。
- **V 系列表驱动单测**：构造 8 组 `(territories, ocr_items, column_evidence)` 夹具，
  逐一断言 `plan_vertical_cuts` 的 `allowed` 与 `deny_reason`：
  两列口算→`V-DENY(i)`；竖式三框→`V-DENY(iv)`；横排并排 3 小题（各有编号、有页面级栏缝）→`allowed=True`；
  只有 OCR 聚类证据 → `needs_review=True` 且 `deny_reason == 'ocr_cluster_only'`。
- **回归硬门**：全量单测通过；60 页审计集 `minor_missed ≤ 27`、`major_recall ≥ 97.6%`、
  10 项软指标不上升、耗时增幅 ≤ 40%。
- **旧调用方零影响**：`split_major_into_minors(crop)` 与
  `split_major_into_minors(crop, items, padding, profile)` 两个老签名调用
  （`review_tools/cut_quality_audit.py:201`、`evaluate_cutter.py:36`、
  `manual_annotation_workspace.py:807`、`process_book` 内 2 处）行为按 §2.1 新链路等价。

---

### T03 · 字段扩展与向后兼容

**Task ID**: T03
**任务名**: 槽位/多段 bbox/材料引用四字段落盘 + 老工程兼容
**源文件**:
- 修改 `cut_questions_v4.py`（`build_question_metadata` 升 `schema_version=3`；`write_paperflow_project` 写 4 个新字段；`_minor_source_segments` → `bbox_parts` 映射）
- 修改 `review_tools/manual_annotation_data.py`（`CUT_V4_NORMALIZE_DEFAULTS` + `CUT_PERSISTENT_KEYS` + `normalize_record` / `apply_enrich_defaults` 接入）
- 修改 `tests/test_manual_annotation_data.py`（老工程读出字段齐全；关 `PAPERFLOW_ENRICH` 仍齐全）
- 修改 `tests/test_annotation_api.py`（UI 保存一次后 4 个字段不丢）

**依赖**: T02
**优先级**: P0

**验收方式**:
```bash
python -m unittest tests.test_manual_annotation_data tests.test_annotation_api -q
python -m unittest discover -s tests -q
```
- **老工程兼容**：构造一个不含 4 个新字段的最小工程 JSON → `load_project()` 后
  `answer_granularity == 'single'`、`answer_slots == []`、`bbox_parts == []`、`context_ref == ''`，
  **不抛异常**；`PAPERFLOW_ENRICH=0` 与 `=1` 两种取值下结果一致。
- **UI 保存不丢字段**：走 `paperflow_annotation_api` 的保存路径往返一次，
  断言 `answer_slots` 与 `bbox_parts` 逐元素相等（这是**唯一**能证明白名单重建路径没漏字段的测试）。
- **槽位顺序**：两列四行 8 个口算 → `slot == [1..8]`，且 `1~4` 的 x 中心全小于 `5~8` 的 x 中心。
- `schema_version == 3`；老版本消费方走 `.get()` 不炸（用 `tests/test_model_contract.py` 的契约测试覆盖）。

---

### T04 · 审计指标与回归门禁

**Task ID**: T04
**任务名**: 三项新指标 + 专项审计集 + 门禁脚本
**源文件**:
- 修改 `review_tools/cut_quality_audit.py`（新增 `vertical_cut_false` / `frame_intersected_cuts` / `text_truncated` / `text_contained_rate`，并进 `summarize` 的统计）
- 新增 `reports/audit_plan_v3_case.jsonl`（本案专项页 1 行）
- 新增 `review_tools/cut_granularity_gate.py`（R1~R9 九条红线的判定脚本，输出 pass/fail + 逐条数字）
- 修改 `tests/test_cut_quality_audit.py`（三个新指标的合成图单测）

**依赖**: T02（需要 T02 的 `_side_by_side` 标记与 `bbox` 才能算）、T03（`slot_count_mismatch` 需要槽位）
**优先级**: P0

**验收方式**:
```bash
python -m unittest tests.test_cut_quality_audit -q
python review_tools/cut_granularity_gate.py \
       --baseline reports/audit_baseline_v2_fixed.jsonl \
       --after    reports/audit_after_v3.jsonl \
       --case     reports/audit_case_v3.jsonl
# 退出码 0 = R1~R9 全过；非 0 = 触发 §6.4 的某条回退判据
```
- `vertical_cut_false`：合成 1 个块，`cut_numbers` 长度 > 印刷小题数且带 `_side_by_side` → 计 1；不带 → 计 0。
- `frame_intersected_cuts`：合成 1 个竖式框 + 1 条穿过它的切线 → 计 1。
- `text_truncated`：真值行右缘超出样本 bbox 右缘 +4px → 计 1（用本案页 `10.67−4.07=` 的真实坐标做夹具）。
- 门禁脚本在**故意破坏**的场景下必须返回非 0（负向测试：把 `after` 换成 `baseline` 自身，
  R3/R8 应触发——基线的 `minor_extra=16 > 0` 说明该脚本对存量问题敏感）。

---

## 8. 风险清单

| # | 风险 | 触发条件 | 影响 | 缓解 |
|---|---|---|---|---|
| **RK-1** | **对开页/跨页续题既有行为被破坏** | `build_number_territories` 在 scope 划分时把下一页首块误判为「新 scope」，或 `continuation_target_major` 依赖的 `_col` 语义被 `_x_start/_x_end` 收窄改变 | 跨页续题被拆成两条 `question`，`number_path` 重复，`duplicate_number_path_count` 上升 → 交付门禁拦下 | ①territory 构建**只看块内**，绝不跨块；跨页归属仍由既有 `merge_cross_page_question` / `should_merge_cross_page_question` 处理<br>②`_col` 与 `_column_bounds` 的写入点**一行不改**<br>③审计集含 6 本对开扫描书，RB5 专门盯这一条<br>④单测：`_experiments` 现有跨页用例全量重跑 |
| **RK-2** | **`_column_bounds` 不可用时降级把真题并成一块（漏切）** | `layout_arbiter.available()` 为 False（模型缺失 / `PAPERFLOW_LAYOUT_ARBITER=0` / 哨兵文件存在）且 `_visual_column_split` 也返回 None | 形如 `横排小问切分_20261019.md` 记录的 `56260570` 页（三小问并排、`1.` 被 OCR 漏读）会退回 1 块 → `collapsed_blocks` 上升 | ①该页在 `audit_plan_v2.jsonl` 内，RB1/RB4 会立刻抓到<br>②降级第三态给的是「水平带 + `needs_review`」而不是「静默合并」，人能看到<br>③`_recover_missed_minor_boundaries` 保留（原生分辨率重识别补跳号），它对这类页面有效<br>④**预留**：`layout_strategy.vertical_cut_default_deny=False` 开关，允许按版式例外放开，但必须显式打开 |
| **RK-3** | **词表与答案槽位字段在旧工程上炸** | 老工程 JSON 无新字段；或 `answer_slots` 里有非 JSON 原生类型（numpy int / tuple） | UI 打开老工程白屏或保存时崩 | ①默认值走 `normalize_record`，**无条件**补齐（不挂 `PAPERFLOW_ENRICH`）<br>②槽位坐标在落盘前统一 `int()`/列表化（复用 `_json_safe_crop_info` 的既有做法）<br>③T03 的两个测试专门覆盖「老工程读出」与「UI 保存往返」<br>④`write_json_atomic` 已有 `json_safe` 包装，字段类型异常会被拦在写盘前 |
| **RK-4** | **D7 小数拦截过宽，误伤真编号** | `strict=True` 把 `1.5元` 之类的**题干开头**误当小数丢弃，或把 `2. 计算`（编号后跟空格+汉字）误拦 | 真编号丢失 → `minor_missed` 上升 | ①D7 判据只拦「数字 + `.` + **数字**」的**紧邻**形态（`2.7`、`3.08`），`2. 计算`（点后是空格/汉字）不受影响<br>②先在 60 页审计集上跑「只开 D7、其余不动」的**单开关实验**，确认 `minor_missed` 不上升再往下走 |
| **RK-5** | **`_closed_rect_boxes` 性能** | 每块都跑一次 `findContours`；对开页 4897×3377 有 6~8 块 | 单页耗时上升，可能触发 RB7 | ①只在「块内含 `=` 结尾算式 **或** `(`/`）` 作答位」时检测（短路），纯文字块直接返回 `[]`<br>②mask 复用 `split_major_into_minors` 已算的 `_content_mask`（该函数已在 `y_end` 计算时调用过），不重复构建<br>③`findContours` 前先做 `cv2.pyrDown` 降采样<br>④RB7 硬门：增幅 > 40% 即回滚 |
| **RK-6** | **`_split_side_by_side_minors` 删除引发测试与脚本连锁失败** | `tests/test_smart_cutter.py::SideBySideMinorTests` 4 个用例、`_experiments/draw_columns.py` 可能引用 | CI 红 | ①`_side_by_side_columns` **保留不删**（只降级为几何工具），4 个用例中 3 个仍可原样通过<br>②只有 1 个用例（`test_three_side_by_side_questions_are_cut_and_numbered`）需要改写为「断言 V 判定结果」<br>③合入前先 `grep -rn "_split_side_by_side_minors"` 确认全部调用点（已知 2 处：定义 4913、调用 5047） |
| **RK-7** | **规则口径与审计口径不一致导致验收扯皮** | `minor_extra` 工具口径 16 / 规则口径 51 / 单页 1，三个数都在文档里出现过 | 验收结论不可信 | ①§6.1 把三个口径并列写清并各自给基线数字<br>②门禁脚本 R2 明确用**工具口径 ≤ 16**，规则口径 51 只上报不设门<br>③规则 §7.1 的「`minor_extra` 目标 0」在本次改造范围内**明确不达成**，需产品书面确认（§10 待确认项 Q1） |
| **RK-8** | **`build_number_territories` 成为新的大复杂度热点** | 一个函数里塞进 D1~D8 + B1~B6 + P 区判据 | 难以单测、难以维护、回归定位困难 | ①按 §2.4 拆成独立小函数（`d7_filter` / `scope_split` / `d2_p_zone` / `d6_circled` / `territory_anchor`），`build_number_territories` 只做编排<br>②放进**新文件** `cut_numbering.py`，不继续膨胀 8617 行的 `cut_questions_v4.py`<br>③`plan_vertical_cuts` 做成纯函数（无 IO），表驱动单测 |
| **RK-9** | **`_minor_source_segments` → `bbox_parts` 映射改变既有交付物形状** | `build_question_metadata` 的 `source.segments` 已有消费者（`review_queue.py` / 训练导出） | 老消费方字段缺失 | ①`source.segments` **保留不动**，`bbox_parts` 是**新增平铺字段**，两者并存<br>②映射只在 `write_paperflow_project` 一处发生<br>③`tests/test_answer_training.py` / `test_training_export.py` 全量重跑 |

---

## 9. 解冻申请（单列，且**只申请 1 个文件**）

### 解冻申请 #1：`desktop_annotation/paperflow_annotation_api.py`

**为什么必须动**
该文件的 `save_page_records`（约 915~930 行）是**白名单重建**路径：
`record = annotation_data.new_record(...逐字段列举...)` 之后，只从 `previous` 回接
`delivery_id` 与 `ENRICH_PERSISTENT_KEYS` 两组键。白名单之外的字段**会被丢弃**。
`answer_slots` / `bbox_parts` 是后端算出来的、人工在 UI 上看不见也改不了的数据，
一旦标注员在界面上点一次保存，整批槽位就被抹掉——而且是**静默**丢失。

**要改什么**（共 2 行）
```python
# 915 行之后，与 ENRICH_PERSISTENT_KEYS 同构，加一个 CUT_PERSISTENT_KEYS 循环
for cut_key in annotation_data.CUT_PERSISTENT_KEYS:
    if cut_key in previous:
        record[cut_key] = _json_copy(previous[cut_key])
```
`CUT_PERSISTENT_KEYS` 定义在**未冻结**的 `review_tools/manual_annotation_data.py`（与 T03 同一个文件）。

**为什么不能走备选路**
备选是把 4 个键并入现有 `ENRICH_PERSISTENT_KEYS`。但那 14 个键受 `PAPERFLOW_ENRICH`
开关统一控制（`enrich_fields_enabled()`），一旦现场关掉 enrich，切分字段也会消失，
违反「切分字段无条件存在」的兼容要求。

**影响面**
- 只影响「保存」路径的字段保留，**不影响**任何判定逻辑（不碰 `review_levels` / `enrich_snapshot`）。
- `_save_page_records` 的其他白名单键、`delivery_id`、重复 id 校验、`pairs` 清理逻辑全部不动。

**回归代价**
- 新增 1 个单测（T03 已含）：UI 保存往返后 4 字段逐元素相等。
- 现有 `tests/test_annotation_api.py` 中**唯一**相关的是 `:909` 的
  `for key in api.annotation_data.ENRICH_PERSISTENT_KEYS: stale_record.pop(key)`，
  它遍历常量本身，加新常量不影响该测试（cut 字段不在 `ENRICH_PERSISTENT_KEYS` 里）。
- 预估回归面：< 1% 的测试文件，需重跑 `test_annotation_api.py`（约 40 个用例）。

**若不批准**：T03 降级为「4 个字段只写 `metadata.json`、不写 PaperFlow `record`」，
槽位数据不进交付物，规则 §4 的答案粒度无法落地。**这是唯一需要解冻的点，其余全部在未冻结文件内完成。**

---

## 10.  Anything UNCLEAR（不猜的地方，需要产品/团队确认）

| # | 问题 | 我的默认假设 | 若假设错的后果 |
|---|---|---|---|
| **Q1** | 规则 §7.1 写 `minor_extra` 目标 = 0，但实测基线是 16（工具口径）/ 51（规则口径）。本次是否要求全量归零？ | **否**。本次只要求「不高于基线」+ 本案页归零（§6.1 R1/R2） | 若要求全量归零，需额外一轮针对 9 个过切块的专项修复，任务量翻倍，且与 B→V 改造混在一起会让回退判据失效 |
| **Q2** | 规则 §4.1 允许拆分的唯一例外（`(1)(2)` 独立编号 + 独立作答空白框）是否要求本次实现？ | **不要求**。本次只实现「不拆分 + 有序槽位」；例外留给后续 | 若要求，需在 `build_number_territories` 里加 L3 拆分判定 + 父级 D2 的 `group` 分支，+1~2 天且需新单测 |
| **Q3** | 规则 §2.2 V-B 要求「编号左缘 ≤ 列宽×28%」。但 `reports/横排小问切分_20261019.md` §3.1 现状在横排长条里**故意**放开了左限（`left_limit = crop_width`）。两者冲突时谁优先？ | **V-B 优先**（规则是定版口径）。放开左限是为了让 `2.`/`3.` 不被丢，但那是**识别**问题；V-B 是**竖切许可**问题，识别放开了不等于许可放开 | 若以现状为准，`56260570` 那类页面可以竖切，本案也会被误切——与规则 §0 直接矛盾 |
| **Q4** | 规则 §5 形态 3「选择题四选项横排」要求 `options` 齐 4 项，缺项 → L2。`options` 字段现状是否已有产出？ | 假设已有（`paperflow_annotation_api.py:899` 有 `options=item.get("options", [])`） | 若实际不产出，需额外补 `options` 抽取，超出本次范围 |
| **Q5** | `answer_slots[].answer` 由谁回填？本次只产出 `slot`/`expr`/`xy`（题面几何），答案文本回填属 enrich 阶段 | 假设 enrich 回填 | 若要求切分阶段就带答案，需切分侧读答案页，`process_book` 的 `pairings` 流程要改，影响面显著变大 |
| **Q6** | 本案页 `20210126153256260565.jpg` 不在现有 60 页审计集内（审计集里是四年级**上**/三年级等）。是否需要把它**加入**主审计集？ | **不加入**，另建 `audit_plan_v3_case.jsonl` 专项集（保持主集 60 页可比） | 加入主集会让所有基线数字失效，60 页的历史对比全部作废 |

---

## 11. 交付物清单

| 文件 | 状态 |
|---|---|
| `reports/切分算法改造设计_20261020.md` | 本文档 |
| `reports/system_design_20261020.md` | 同一内容的英文名副本（按架构师标准输出路径） |
| `reports/sequence-diagram.mermaid` | B→V 判定时序图 |
| `reports/class-diagram.mermaid` | 新增数据结构与接口类图 |
| 代码改动 | **无**（本次只出设计） |
| 测试改动 | **无** |