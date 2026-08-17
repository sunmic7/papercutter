# PaperFlow 人工标注工作台交接文档

最后更新：2026-08-14

> 新开 Codex 窗口后，先完整阅读本文件，再检查代码。不要根据旧聊天记录猜测当前实现。

2026-08-13 全流程审计、真实教材实测和整改详情另见：

```text
F:\papercutter\PAPERFLOW_AUDIT_20260813.md
```

## 1. 项目与用户目标

工作区：

```text
F:\papercutter
```

这是面向非技术标注人员的教材切题工具，主要流程为：

1. 打开一本教材图片目录。
2. 人工确认题目框并填写题目字段。
3. 人工寻找答案页、切答案框。
4. 人工确认题目—答案匹配。
5. 从“文件 → 生成裁剪图片和 JSON”选择匹配分组，输出题目与答案成套的数据包。

基本原则：人工标注为主，自动预标注/OCR/候选匹配只作辅助；界面必须直接、稳定、低操作成本。

## 2. 启动方式与实际加载代码

启动入口：

```text
F:\papercutter\启动切题工具.vbs
```

它启动：

```text
F:\papercutter\desktop_annotation\PaperFlow-Annotation\PaperFlow人工标注工作台.exe
```

Electron 没有使用 `app.asar`，前端实际读取：

```text
F:\papercutter\desktop_annotation\PaperFlow-Annotation\resources\app
```

本地服务：

- `desktop_annotation\paperflow_annotation_api.py`：127.0.0.1:8601。
- `resources\app\electron\backend\mock_server.exe`：RapidOCR 兼容服务，127.0.0.1:5000。

修改后必须彻底退出旧 Electron 进程，再通过 VBS 重启。旧窗口不会热加载源码。

## 3. 核心代码

```text
desktop_annotation/PaperFlow-Annotation/resources/app/index.html
desktop_annotation/PaperFlow-Annotation/resources/app/js/main.js
desktop_annotation/PaperFlow-Annotation/resources/app/js/employee-workbench.js
desktop_annotation/PaperFlow-Annotation/resources/app/js/ui-manager.js
desktop_annotation/PaperFlow-Annotation/resources/app/js/image-manager.js
desktop_annotation/PaperFlow-Annotation/resources/app/css/employee-workbench.css

desktop_annotation/PaperFlow-Annotation/resources/app/electron/main.js
desktop_annotation/PaperFlow-Annotation/resources/app/electron/preload.js
desktop_annotation/PaperFlow-Annotation/resources/app/electron/exit-cache.js

desktop_annotation/paperflow_annotation_api.py
review_tools/manual_annotation_data.py
review_tools/manual_annotation_workspace.py
review_tools/delivery_workspace.py
review_tools/group_delivery.py
review_tools/export_ocr_delivery.py
review_tools/workspace_store.py
cut_questions_v4.py
```

本目录当前不是 Git 仓库。不得清理用户文件、删除 `runtime` 数据或使用破坏性命令；修改文件使用 `apply_patch`。

## 4. 当前保存规则（重要）

### 4.1 标注缓存的真实位置

当前项目缓存不再默认放在 `runtime/manual_annotation_projects`。每本教材只读取并保存它自己目录内的：

```text
<教材目录>\PaperFlow标注项目.json
```

打开教材时只检查这一个精确路径，不扫描其他目录，也不为全部页预建输出文件夹。

### 4.2 保存时机

- 编辑框、题号、OCR 文字等：先写入当前运行内存并标记“已修改，退出时保存”。
- 翻页：把当前页编辑器内容提交到内存，不做整本磁盘写入。
- 点击“确认本页”：显式保存当前页并标记为 `accepted`，当前页全部框立即显示为绿色，左侧页面保留对号；不再生成或刷新任何裁图/JSON。
- 页面归类、答案匹配：立即保存相关修改。
- 正常退出：保存当前教材全部内存状态到教材旁的缓存文件。
- “文件→保存教材信息和标注框”或 `Ctrl+S`：立即保存当前编辑器内容、全部页面的框、页面分类、题答配对、匹配分组、旋转状态和教材级信息，不生成裁图。保存前会合并尚未回页的预标注结果，并取消待触发的自动保存计时器。
- 若“设置教材信息”窗口正在打开，文件保存会读取窗口内尚未点“保存”的语言、学段、年级、学科和答案处理方式；必填项完整时一并提交，未填完整时仍先安全保存全部框和原有教材信息，并明确提示未覆盖教材信息。

当前保存不再允许旧 renderer 快照无条件覆盖新状态：

- 项目带单调递增的 `project_revision`。
- `/api/project/save-cache` 要求 `base_revision` 与磁盘版本一致；过期快照直接拒绝，不能删掉后端新框、稳定 ID 或匹配关系。
- 正常退出优先调用受 `PROJECT_WRITE_LOCK` 保护的后端保存。
- 仅当本地服务不可用时，才启用 Electron 容灾直写；容灾路径也会校验磁盘 revision、原子替换 JSON、递增并回传新 revision。
- 切换教材前必须成功保存旧教材；失败时阻止切换。
- 设置交付目录、修改教材信息、页面归类后会同步后端 revision。修改教材信息前先落盘当前页，避免用后端项目对象覆盖未保存编辑。

已知边界：Electron 容灾直写的“revision 检查完成到 rename 之间”仍有极窄 TOCTOU 窗口。正常路径不走该分支；若要彻底消除，应改为写独立 `.recovery.json` 或使用跨进程文件锁。

关键实现：

```text
main.js                         savePaperFlowBeforeExit()
electron/main.js                close / finishAppClose()
electron/exit-cache.js          saveProjectCacheOnExit()
```

## 5. 教材统一信息

“打开教材”右边有“教材信息”按钮，用于一次性设置整本教材：

- language：`zh` / `en`
- stage：小学 / 初中 / 高中
- grade：如六年级、七年级、高一
- subject：数学、语文、英语等
- answer_mode：`textbook`（教材自带答案）/ `generate_later`（无教材答案，后续技术生成）/ `mixed`（混合模式）

这些字段属于教材级元数据，不要求每题重复选择。题目级字段仍包括题型、难度、知识点、选项、答案、解析等。

## 6. 题目切分与字段规则

- “编辑→顺时针旋转当前页面 90°”只允许用于完全空白页；当前画布或项目中已有题框/答案框/配图框，或页面已被其他题当作共享材料时，前后端都拒绝旋转。
- 每次顺时针旋转 90°，四次回到 0°。原教材图片不改写；旋转视图保存在教材内的 `_paperflow_page_views` 隐藏工作目录，项目持久保存 `rotation` / `rotated_image`。后续预标注、OCR、人工画框、匹配预览和交付裁图都使用旋转后页面及其坐标。
- 画布右键支持“续框”和“下一个框”操作，题目页与答案页均可用：右键已有框选择“复制此框”，移动到目标左上角后右键可选择“粘贴为…的续框”或“粘贴为下一个框”。后者会只递增最后一个层级，例如 `七-1→七-2`、`七→八`、`七-1-1→七-1-2`。新框继承题号层级、题型等结构信息但使用新 ID；续框写入 `region_role=continuation` 与 `continuation_of=原框ID`，下一个框为普通题。两者都不复制 OCR 原文、题干/答案/解析和模型来源，须独立核对。
- 主框、续框、跨页片段及共享题干生成裁图时按上下边缘直接贴合，不再插入固定 12 px 白色分隔条。重新生成对应分组后，历史带白缝的该分组题图会被整体覆盖。
- 层级编号以 `-` 连接，系统不再自动添加括号：输入中间层级 `1` 显示并导出为 `1`，而非 `(1)`。用于显示和精准匹配时，中文数词、圆圈数字、括号数字、全角数字会规范为阿拉伯数字，例如 `七-1-（1）→7-1-1`、`八-（二）→8-2`；原始栏目标题和正文文字不改写。
- 续框粘贴位置以第二次右键位置为左上角，边界超出页面时自动收进页面内；可再用现有拖拽/缩放调整。刚画出的源框尚未保存时，右侧“续接到上一题/答案”也能正确保留该关联。
- 小题号允许为空；部分题目只有大题号，不能因缺少小题号阻止“确认本页”。
- 支持大题号、最多 8 层中间层级和小题号。
- 中间层级不能使用 `<details>`/`<summary>`；当前是固定标题、添加按钮和直接显示的输入行。
- 中间层级按钮通过 `data-paperflow-add-level=<annotation id>` 定位，不依赖易丢失的右侧焦点。
- 题卡按版面阅读顺序排列：先左栏从上到下，再右栏从上到下，不按 OCR 题号排序。
- 只有选择题显示并保存选项输入框；填空、判断、计算、应用等题型不显示无意义的选项框。
- `确认本页` 会先刷新当前仍在编辑的表单，再序列化保存，避免最后一个输入值遗漏。
- 题目页没有任何题框时，前后端都禁止将其确认为 `accepted`；应先画框，或将页面改分类为答案/忽略页。
- 页面保存严格拒绝非对象记录、非法 bbox、空 ID、页内重复 ID 和全书重复 ID；校验失败时整页不写盘。
- 保存会往返保留 `answer_split_status`、`ocr`、`boundary_source`、`detector_assisted` 和 `detector_confidence`。

中间层级实现：

```text
resources/app/js/ui-manager.js
  setupPaperFlowLevelControls()
  addPaperFlowNumberLevel()
  removePaperFlowNumberLevel()
```

## 7. 答案页与答案切分（最新流程）

答案页完全由标注人员人工寻找，不再依赖目录名或 OCR 自动判断。

新教材创建项目时：

- 所有源图片先进入可浏览的教材页列表。
- `answer_pages` 初始为空。
- 不按“答案”“answer”等文件夹名称自动分类。

操作流程：

1. 点击顶部流程“答案切分”。
2. 如果尚无答案页，程序保持教材可浏览并进入“人工查找答案页”状态。
3. 人工翻到真实答案页。
4. 点击“当前页设为答案页”。
5. 在该页绘制/生成答案框，OCR 后核对答案与解析。
6. 点击“确认本页”。
7. 需要继续寻找时，点击“查找更多答案页”。

误分类可通过“页面分类”移回题目页或忽略页。人工分类结果立即写入该教材缓存，重新打开不应丢失。

答案 OCR 文本会保守拆分：仅在出现“答案：”“解析：”等明确标记，或短文本明显就是答案时自动分开；不确定内容进入待人工核对，不覆盖已经手填的答案/解析。

答案页右侧编辑器采用精简布局：不显示“栏目标题”“标准答案”“详细解析”，只保留一份“答案页 OCR 原文”并放在编号字段之后；已有的 `answer` / `analysis` 数据仍保留在项目中，不因隐藏而删除。“续接到上一答案”固定在答案页表单最底部。

题目页和答案页的“续接到上一题/答案”均固定在各自编辑表单最底部。画布滚轮改变缩放后，顶部缩放工具旁显示“还原大小”；点击会恢复当前页面的 100% 初始大小及初始位置。

“还原大小”必须一次性应用缩放与平移的最终状态后再重绘标注画布；不可先重绘、再重设平移，否则第二次画布尺寸更新会清空框线。题目页题型与难度使用同一行双列选择；右侧题卡的坐标单独显示为较小、低透明度文字，完整坐标仍可悬停查看。

画布与右侧题卡实行双向定位：点击画布中的框，右侧会自动补齐分批加载的题卡、展开该卡并滚到其编辑菜单；点击右侧题卡，会选中左侧对应框并将长页画布平移到框的可见位置。删除选中框（顶部按钮、快捷键或右侧删除入口）都会先显示统一确认提示“确定要删除这个框吗？”，确认后才删除。

“绘制题目框 / 绘制答案框”是显式开关：点击后按钮变为“停止绘制”，再次点击才停止并恢复对应的绘制文字。画完一个框后仍保持绘制状态，方便连续画框；停止时会清掉尚未完成的两点绘制临时状态。保存 API 可接收当前绘制开关用于诊断，但该状态是临时 UI 状态，绝不写入教材项目；后端只保存已完成且坐标合法的框。

人工新画题框或答案框时，右侧不再等待延迟刷新：创建事件会先立即渲染包含新框的题卡，再将选中编辑器挂到该题卡并滚动定位；尤其答案页新增框必须当场显示其编号、OCR 原文和续接菜单。

答案页允许在答案文字尚未填写时点击“确认本页”，以便继续页面流转和题答匹配；确认不再被空文字硬拦截。若存在普通答案框的 `answer` 为空，或续框的原文/解析均为空，后端会在确认结果中返回数量警告，前端以黄色提示“本页已确认，但有 N 个答案框尚无答案文字”；此提示不改变已确认状态，但交付前仍须补充并复核。

右侧题卡排序优先使用已识别或人工填写的层级号：在页面中至少半数框具有完整编号时，按“单元/大题号 → 小题号 → 后续层级”升序排列，支持阿拉伯数字、中文数字及常见罗马数字，因此 `一-7` 会排在 `二-1` 前。编号不足的页面保留原来的多栏版面阅读顺序；无法识别编号的少量框排在已编号框之后并保持其相对位置。

顶部“教材信息”按钮显示文字为“设置教材信息”，功能和已保存数据不变。系统菜单“帮助 → 操作说明”会打开完整、可滚动的员工操作指引，覆盖打开教材、页面浏览与缩放、题目/答案切分、续框、旋转、分类、保存、匹配、交付、快捷键及注意事项；Esc、关闭按钮或点击遮罩均可关闭。

主工具栏已移除冗余的“题目页标注/答案页标注”下拉框；题目与答案工作流仍由窗口顶部步骤切换。当前按钮顺序为“页面分类 → 匹配分组 → 生成裁剪图片和 JSON → 绘制题目框/答案框 → 删除选中”，其中生成按钮占用原下拉框位置。

旧版“印刷/手写属性”已从全局删除：新建标注不再创建该属性，编辑表单、旧版 JSON 生成与解析、质检展示及相关 CSS 均不再包含它。历史 JSON 中已有同名字段时直接忽略，不传播到新保存结果。

答案页预标注不对高密度结果静默截断或合并。当候选过多（当前诊断阈值 120）、缺题号或同上下文题号重复时，API 返回 `requires_review=true` 和 `bulk_accept_safe=false`，界面提示职员分批复核。题卡每批最多渲染 40 张，避免一次解码整页图数百次。

关键实现：

```text
main.js
  loadPaperFlowKind()
  startPaperFlowManualAnswerSearch()
  moveCurrentPaperFlowPage()
  splitPaperFlowAnswerText()

manual_annotation_data.py
  create_book_project()
  reconcile_book_pages()
  reclassify_pages()
```

## 8. 答案匹配

- 只有已经确认的题目页进入匹配列表。
- 只有已经确认的答案页/答案框可成为有效候选。
- 匹配窗口使用可搜索、可按状态筛选的题目卡片。
- 候选按完整题号、层级、文字和版面顺序提供辅助分数，但最终由人员确认。
- 批量确认只接受高置信度、完整题号一致且明显领先第二候选的唯一结果。
- 可标记“暂缺答案”。
- 前端必须采用后端返回的完整 `pair`，保留稳定的关系 `id` / `delivery_id`；不再自行构造简化关系。
- 删除题框会移除失效关系；多答案关系删掉其中一个答案时，仅移除该答案 ID，剩余答案仍有效则保留关系。
- 顶部“匹配分组”用于处理不同单元、不同试卷重复题号：分组是整本教材共享的，可在题目页和答案页选择同一个分组。
- 分组支持三种赋值方式：拖动框选当前页区域（以各框中心点是否落入区域判定）、当前页全部设置、按页码范围勾选多页后全部设置。框选区域会持久化，之后在该区域新画的框自动继承该组；整页设置也会成为该页新框的默认组。
- “执行安全自动匹配”建立 `(分组 ID, 规范题号路径)` 索引，只处理已确认页面中“题目恰好 1 个、答案恰好 1 个、双方均未占用”的组合。关系来源记录为 `scope_number_auto`。
- 中文数词、圆圈数字、括号数字和全角数字会按阿拉伯数字规范比较；没有分组、题号缺失、同组重号、多对一、一对多或已经匹配的组合全部放弃自动处理并提示冲突数量，绝不静默猜测。
- 分组窗口选择具体分组后，会同时显示该组覆盖的题目页和答案页。每个页面卡片明确标记“整页分组”“框选区域 N 处”或“单独归组 N 个框”，并显示归组框数量和源图片名；框选页使用小型区域示意图，与整页分组有明显区别。仅切换分组或勾选页面不会改变项目，只有明确点击移出按钮并确认后才保存修改。
- 覆盖概览中的题目页和答案页均可勾选后“移出此分组”。该操作只清除所选页面的整页默认组、该组框选区域及属于该组的框字段，不删除教材页面、图片、题框或答案框。下方批量选择区同时显示题目页和答案页，两侧分别支持全选、取消全选和页码范围，一次批量操作可同时设置两类页面。

匹配窗口有四种退出方式：

- 底部“返回标注界面”。
- 顶部“返回教材查找答案页”。
- 右上角 ×。
- `Esc`。

`closePaperFlowModal()` 会同时移除 `show`、`modal-open` 并清理当前匹配题 ID，防止点击答案匹配后界面锁死。

## 9. 交付输出位置与目录结构

默认交付根目录：

```text
F:\papercutter\runtime\delivery_packages\B<教材稳定ID>\
```

用户可在“文件”菜单选择：

- “生成裁剪图片和 JSON…”
- “设置交付输出位置…”
- “查看当前交付输出位置…”

选择共享输出根目录后，每本教材仍放入独立目录：

```text
<用户选择目录>\B<教材稳定ID>\
```

新员工流程使用按分组成套输出：

```text
B<教材ID>\groups\<界面中创建的分组名称>\
├─ images\
│  ├─ questions\Q<题目ID>.jpg
│  └─ answers\Q<题目ID>\answer_01.jpg
└─ json\
   ├─ manifest.json
   ├─ dataset.jsonl
   └─ items\Q<题目ID>.json
```

- 每次只选择一个匹配分组生成；题目与答案在同一条 JSON 记录中关联，裁图分别进入该分组的题目图和答案图子目录。
- 分组目录直接使用职员在界面中创建的分组名称；仅当名称含 Windows 禁止的路径字符时替换为 `_`。分组改名后再次生成，会依据内部 scope ID 清理旧名称目录。
- 所有裁剪图片统一位于 `images`，所有 JSON/JSONL/manifest 统一位于 `json`，两类文件不再混放。
- 再次生成同一分组时，先在临时目录完整构建，再原子替换整个旧分组目录；旧图、旧 JSON 和残留文件不会混入新结果。
- `textbook` 模式若存在未匹配题目，界面询问是否继续；确认后只生成已匹配题目。
- `generate_later` 模式会把已确认但未匹配的题目写为 `answer_status: pending_generation`，保留题图和结构，供后续技术生成答案。
- `mixed` 模式可在生成窗口选择是否把未匹配题目纳入 `pending_generation`。
- 已匹配题目的 `answer_status` 为 `matched`。未来生成答案流程预留 `generated_pending_review`、`generated_accepted`、`no_answer_confirmed` 状态，不应把机器答案直接当人工已确认答案。
- 未确认页面上的题不会生成，窗口会显示排除数量。

旧的整本增量交付接口仍保留供兼容和专项工具使用，其主要结构为：

```text
B<教材ID>\
├─ manifest.json
├─ database_import.json
├─ images\questions\P0001\Q<题目ID>.jpg
├─ images\answers\P0001\A<答案ID>.jpg
├─ json\questions\P0001\Q<题目ID>.json
├─ json\answers\P0001\A<答案ID>.json
└─ k12_dataset\
   ├─ dataset.jsonl
   ├─ quality_report.json
   └─ pages\P0001\
      ├─ dataset.jsonl
      └─ images\Q001_img_01.jpg
```

旧结构页目录采用 `P0001`、`P0002` 顺序。日常员工流程不再由确认页、配对、教材信息或旋转操作隐式更新该结构。

稳定 ID：

- `B...` 教材
- `Q...` 题目
- `A...` 答案
- `R...` 题答关系

同一个框反复编辑时对外 ID 保持稳定；删除后重新画框才生成新 ID。

交付安全与一致性规则：

- `manifest.json` 不可信。清理旧交付文件前必须校验相对路径、交付根边界、允许的目录前缀和扩展名，禁止 `../` 或绝对路径越界删除。
- K12 `dataset.jsonl` 写入前必须物化所有被引用插图；增量清理依据完整活动索引，不得删除别页仍在引用的图片。
- 每本教材的 workspace SQLite 按规范项目路径 SHA-256 标识命名，不能因项目文件都叫 `PaperFlow标注项目.json` 而串数据。

## 10. K12 JSONL 交付要求

目标核心结构：

```json
{
  "metainfo": {
    "language": "zh",
    "stage": "小学",
    "grade": "六年级",
    "subject": "数学",
    "question_type": "填空",
    "difficulty": "中等",
    "knowledge_points": ["正负数"]
  },
  "data": {
    "question_images": ["pages/P0001/images/Q001_img_01.jpg"],
    "option_images": [],
    "question": "题干",
    "options": [],
    "answer": "答案",
    "analysis": "解析"
  }
}
```

正式合格样本至少要求完整题干、题型、答案、详细解析、教材元数据、难度、知识点和有意义题图/选项图；选择题至少两个选项。公式、序号、整页截图、答案过程截图等不能冒充题图。不合格数据写入 `quality_report.json`，不能混入正式 `dataset.jsonl`。

## 11. 性能与兼容性现状

- 打开教材只读取教材旁缓存；不会搜索旧 runtime 或其他教材路径。
- 不在打开时同步整本 SQLite 或生成全书输出目录。
- 图片只预载当前页附近，并限制预载缓存。
- 本地服务延后启动，避免启动阶段阻塞界面。
- RapidOCR 已存在监听进程时直接复用，避免重复启动导致内存压力。
- 顶部教材/图片长名称使用省略显示，避免与缩放、标注按钮重叠。
- 动态隐藏按钮有 `[hidden] { display:none !important; }`，避免 `.btn {display:inline-flex}` 破坏隐藏状态。
- `image-manager.js` 使用图片集合代次、请求令牌和导航锁，真正等待解码及 DOM 显示。过期 `onload` 不能将 A 页图写到 B 页状态，坏图加载失败会回滚。
- 预标注任务以项目会话、项目路径、教材根、页面类型、页号和源图组成完整身份；切书主动取消，超时 120 秒，迟到结果不能串入另一本教材。
- Paddle/PaddleX 预测器是进程级单例，API 已加全局串行推理锁；同页同参数并发请求合并为同一个 in-flight 任务。
- `cut_questions_v4.py` 的内容掩码与水平切线扫描已向量化并复用中间结果。`G:\测试2` 3 个已有框题页的暖缓存总耗时由约 194.9 秒降到约 3.1 秒；隔离冷页由约 98.1 秒降到约 50.9 秒，框结果保持一致。
- 整页密封栏检测已兼容 3～6 px 的细内边界：只有同时出现外侧独立长线/竖排标签、贯穿大部分页高的内边界、正文前空白沟和大量右侧正文时才裁切。OCR 水平文本保护会将边界回退到正文之前。`DM_20240429140249_009.JPEG` 的左栏题框已由 `x=0` 修正为 `x=172`，同套 41 页几何扫描仅在 14 张存在该密封栏的页面触发。
- 题目页预标注已支持 1～4 栏的递归栏位识别。三栏/四栏页会先从视觉栏缝与 OCR 两侧正文共同确认每个栏，再严格按“左栏到底 → 中栏到底 → 右栏到底”的阅读顺序逐栏切题；每个题框的来源边界限制在所属栏内，不能横跨相邻栏。若没有足够的双重证据，仍保守退回既有单栏/双栏规则，避免将宽题干或表格误拆。试卷页的结构 OCR 已覆盖整张页，不再只抽取双页扫描的两侧窄条；`周测`、`单元测`、`达标检测` 等书名也优先归为试卷档，避免“训练”字样误归为通用练习册。`G:\测试2\原教材3` 的 `image0001042A.jpg`、`image0001042B.jpg` 实测均识别为 3 栏；B 页重新预标注得到 33 个框，所有框均落在 `[0,1625]`、`[1625,3124]`、`[3124,4735]` 三个独立栏位内。
- 冷 OCR 仍是主要瓶颈：真实页无缓存结构 OCR 约 39–51 秒；热缓存题页约 1.02–1.10 秒。
- `runtime/ocr_cache` 于审计时约 8520 个文件、66 MB；进程内 OCR cache 仍是无上限字典，尚需 LRU、磁盘配额和启动预热。
- 已部署题块检测器只以 `PAPERCUTTER_MANUAL_PRELABEL_DETECTOR=1` 特性开关接入，默认关闭；仅做一对一高 IoU 边界辅助，异常时回退规则链。该模型在测试教材真实数学页检测为 0 框，现阶段不可替代规则预标注。

## 12. 后端接口

```text
/api/project/open-book
/api/project/save-cache
/api/project/save-page
/api/project/reclassify-pages
/api/project/set-book-metadata
/api/project/set-delivery-output
/api/project/delivery-location
/api/project/match-candidates
/api/project/set-pair
/api/project/accept-confident-matches
/api/project/upsert-match-scope
/api/project/assign-match-scope
/api/project/remove-match-scope-pages
/api/project/auto-match-scopes
/api/project/group-export-preflight
/api/project/export-match-scope
/api/project/sync-delivery
/api/project/workspace-page
/api/prelabel
```

后端写操作有 `PROJECT_WRITE_LOCK`，避免保存、匹配、交付同步并发破坏项目 JSON。

额外边界：

- `source_path` 仅接受教材根目录内的相对图片，拒绝绝对路径和 `../` 越界。
- `/api/prelabel` 的 `project_path` / `kind` / `page_index` / `image_path` 必须彼此一致。
- 打开历史缓存时会识别并剔除交付包回流形成的伪教材页，再按相对图片路径保留真实页标注。

## 13. 已验证测试

最近完整回归：

```text
python -m unittest discover -s tests -p "test_*.py"
Ran 294 tests
OK
```

JavaScript 语法检查已通过：

```text
node --check desktop_annotation/PaperFlow-Annotation/resources/app/js/main.js
node --check desktop_annotation/PaperFlow-Annotation/resources/app/js/image-manager.js
node --check desktop_annotation/PaperFlow-Annotation/resources/app/js/employee-workbench.js
node --check desktop_annotation/PaperFlow-Annotation/resources/app/js/ui-manager.js
node --check desktop_annotation/PaperFlow-Annotation/resources/app/js/latex-renderer.js
node --check desktop_annotation/PaperFlow-Annotation/resources/app/electron/main.js
node --check desktop_annotation/PaperFlow-Annotation/resources/app/electron/preload.js
node --check desktop_annotation/PaperFlow-Annotation/resources/app/electron/exit-cache.js
node tests/test_exit_cache.js
```

重点测试文件：

```text
tests/test_employee_workbench_ui.py
tests/test_annotation_api.py
tests/test_manual_annotation_data.py
tests/test_ocr_delivery_export.py
tests/test_group_delivery.py
tests/test_book_metadata_api.py
tests/test_exit_cache.js
tests/test_prelabel_concurrency.py
tests/test_manual_prelabel_detector.py
```

## 14. `G:\测试2` 审计基线（2026-08-13）

实测时：

- 目录共 315 个文件、173 张图片。
- 主数学项目有 42 个题目页、2 个答案页、48 个题框、531 个答案框、1 条匹配关系。
- 4 个题目页标记为 accepted，其中 3 页有框；存在 1 个历史“accepted 但零题框”页面。新代码会阻止再产生此状态，不会自动改写历史人工状态。
- 3 个有框题页预标注输出 21 / 6 / 21 框，与已存 48 个规则框 48/48 对应，代理均值 IoU 约 0.996。这些已存框也来自 `suggested_rule`，不是独立人工真值，不得宣传为真实准确率。
- 两张答案页已存/新推理数量为 288/288、243/243，热缓存约 0.367 / 0.212 秒；两页均需高密度和重复题号人工复核。
- 项目教材元数据不完整、答案页未 accepted，且只有 1 条关系，当前不应直接当作正式 K12 交付。
- 579 条记录的 `ocr` 都是空字典，尚无法按字符置信度做质量分层。
- 历史根级缓存中有 81 页来自交付输出回流污染；新打开流程会根据交付标记剔除它们。

测试原则：不改写原教材图片。真实教材回归以读取图片、读取现有项目和临时目录复现为主；启动/退出程序可能按产品既有规则更新教材旁缓存的时间和 revision。

## 15. 尚未关闭的技术债

按优先级排序：

1. Electron 容灾直写仍有极窄 TOCTOU 窗口；彻底修复需独立 recovery 文件或跨进程锁。
2. preload 仍暴露较宽的通用文件 IPC。OCR/LaTeX 主注入链、sandbox、导航和外链已收紧，但主进程仍应按教材根/交付根建立读写删白名单，并继续清查其他 `innerHTML` 数据流。
3. 框内文字 OCR 尚未完整使用项目/页面身份令牌和真正的 `AbortController`；取消不能可靠终止底层请求与批次。
4. 冷 OCR 仍需 39–51 秒；需继续做模型预热、内存 LRU、磁盘配额、阶段耗时和排队状态。
5. 题块检测器训练域不足，答案检测模型尚未部署。先建立独立人工真值，不得使用算法框自我对比证明精度。
6. 训练闭环尚未直接消费 PaperFlow accepted 项目。建议导出“首次预标注 + 职员最终框”，按整本教材划分 train/validation/test。
7. 新的分组精准匹配已使用规范题号索引，复杂度接近线性；旧的文本/分数候选接口仍近似 O(题目数×答案数)，长教材若继续使用旧候选模式，仍应补文本倒排索引。
8. `database_import` 仍需补齐答案、解析、难度、知识点、选项、插图和教材元数据的 schema 合同测试。
9. 1024/1366 小屏布局、模态框焦点陷阱、ARIA、键盘框微调、以鼠标为中心缩放仍需专项整改。
10. `G:\测试2` 中有 14 张任一边小于 100 px 的装饰图；教材刷新流程应提供尺寸/内容过滤预览，不要无提示自动删除。

## 16. 新窗口优先验证事项

用一套真实教材完成以下闭环：

1. 打开教材，确认旧框从 `<教材目录>\PaperFlow标注项目.json` 恢复。
2. 修改一个框后翻页，再返回确认内存状态未丢失。
3. 点击“确认本页”，确认左侧页面出现对号、当前页所有框变绿，且不创建或刷新裁图/JSON。
4. 进入答案切分，人工查找并把当前页设为答案页。
5. 绘制答案框、填写答案和解析、确认本页。
6. 打开答案匹配，确认四种退出方式都有效。
7. 匹配一题后确认不会自动改交付目录；再从“文件→生成裁剪图片和 JSON”选择该分组，检查题图、答案图、`item.json` 和 `dataset.jsonl` 成套生成。
8. 正常退出并重新打开，确认全部框、教材信息和页面分类均恢复。
9. A 书编辑未确认页后打开 B 书，确认 A 书先成功落盘；再打开 A 书复核内容。
10. 快速连续翻页、预标注中途切书和首图坏路径，确认无错页、无伪确认和无数据丢失；再修改多个页面，在“文件→保存教材信息和标注框”和 `Ctrl+S` 两个入口分别保存，重开教材确认全部页面框和教材信息均已落盘。
11. 用 288/243 框的答案页检查分批卡片、复核警告和内存占用。
12. 找一张倒置且无框页，从“编辑”菜单每次顺时针旋转 90°；确认重开教材后角度保留，旋转后新框及交付裁图正常。再在有框页点旋转，确认程序明确拒绝。
13. 在题目页和答案页各选一个已有框：右键“复制此框”后，在下一行续段左上角右键“粘贴为续框”；确认新框题号层级保留、内容留空、右侧续接关系指向原框，保存重开后交付裁图仍可读取续框。
14. 在超过 40 个框的答案页点击首批以外的画布框，确认右侧自动加载、展开并定位到其编辑菜单；再点击任一右侧题卡，确认左侧对应框被选中且自动滚入可见区域。分别从顶部删除按钮、右侧删除入口和 Delete 快捷键删除，均应先出现“确定要删除这个框吗？”确认提示。
15. 在题目页和答案页各点击一次绘制按钮，确认按钮立即变为“停止绘制”；画完一个框后仍为停止状态。再次点击确认十字光标消失、按钮恢复“绘制题目框 / 绘制答案框”，两点绘制尚未完成的临时框也被取消；保存重开后只保留已完成框。
16. 在答案页人工新画一个框，松开鼠标后无需翻页或点击其他位置，右侧必须立即出现并定位到这个新框的编辑菜单；题目页也应保持同样行为。
17. 在答案页新画一个不填写答案文字的框后点击“确认本页”，确认页面仍进入已确认状态并可继续后续流程，同时界面显示该页尚无答案文字的框数量警告；补上文字后再次确认，警告应消失。
18. 在同一页存在 `一-7`、`二-1` 等跨单元编号时，确认右侧题卡按层级号排序而不是仅按物理坐标；在无编号或多数编号无法识别的页，确认仍按正常多栏阅读顺序显示。
19. 新建“第一单元”等匹配分组，在题目页框选一块区域、在答案页框选对应区域；确认命中的框右侧显示分组名称，新画在区域内的框也自动继承。再用当前页和 `1-3,8` 页码范围批量设置复核多页操作。
20. 准备同组题目 `七-1-（1）` 与答案 `7-1-1`，两页均确认后执行安全自动匹配，确认自动建立唯一关系；再增加同组重复题号，确认程序报告冲突并且不自动匹配重复项。不同分组中的相同题号不得互相匹配。
21. 在教材信息中分别测试“教材自带答案”“无教材答案，后续技术生成”“混合模式”。前者遇未匹配题须询问并只生成已匹配项；后两者按选择生成 `pending_generation`。在分组目录手工放入一个旧文件后再次生成，确认旧文件被清除且新包完整。

注意：2026-08-13 自动化执行会话中，Electron 本地服务、OCR 桥和 renderer 初始化日志正常，但执行环境无法取得可见窗口句柄。打包或交付前仍必须在交互式 Windows 桌面完成上述点击式验收。

若用户反馈界面仍是旧版本，第一步先确认是否彻底结束了旧 Electron 进程并通过 `启动切题工具.vbs` 重启；不要立即重复修改同一功能。

## 17. 给新窗口的建议开场指令

```text
请先完整阅读 F:\papercutter\PROJECT_HANDOFF.md，然后结合当前源码继续处理我的最新问题。保留现有用户数据，不要删除 runtime 或教材目录内的 PaperFlow标注项目.json；完成修改后运行相关定向测试和完整测试。
```
