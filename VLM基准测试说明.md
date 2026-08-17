# VLM 完整题目坐标基准测试

此工具用于验证视觉大模型能否识别“完整一道题”，不会修改正式 OCR/规则切题链路。

## 前置条件

先使用“精度标注”工具为原始教材页创建并完成真值标注。只有 `status=labeled` 的页面会参与正式评分。每页题目框必须按最终业务标准标注：题干、选项、公式、插图和子问是否属于同一题，必须保持一致。

共享阅读材料或一图多问需要在测试前明确业务标准：最终每道题图是否重复包含共享内容。当前 VLM 输出已支持 `shared_regions` 与 `shared_region_ids`，但现有真值清单仍只统计问题框；共享关系需作为第二阶段标注扩展后再评分。

## 运行 Qwen3-VL-Plus 测试

在 PowerShell 中设置密钥（不会写入日志或报告）：

```powershell
$env:DASHSCOPE_API_KEY = '你的百炼 API Key'
python review_tools\vlm_benchmark.py `
  --annotations 'G:\标注\golden_200.json' `
  --output 'G:\基准结果\qwen3_vl_plus_20260808' `
  --model qwen3-vl-plus `
  --repeats 2
```

默认使用阿里云 OpenAI 兼容地址、`json_schema` 结构化输出、最长边 2200 像素，以及每百万输入/输出 token ¥1/¥10 的规划价格。若供应商的严格 JSON Schema 兼容性不符合预期，可临时使用：

```powershell
python review_tools\vlm_benchmark.py --annotations 'G:\标注\golden_200.json' --output 'G:\基准结果\qwen_json_object' --structured-mode json_object
```

先做本地预检、绝不发出 API 请求：

```powershell
python review_tools\vlm_benchmark.py --annotations 'G:\标注\golden_200.json' --output 'G:\基准结果\preflight' --dry-run
```

## 结果

输出目录包含：

- `config.json`：模型、提示词策略、输入尺寸、价格假设和运行配置。
- `responses/`：每页、每次调用的原始响应、解析结果、映射回原图的坐标、耗时和 token 用量；不保存 API 密钥。
- `previews/`：原图上的红色题框、黄色共享材料框，供快速人工目检。
- `crops/`：按 VLM 首次预测坐标裁出的题图；共享材料暂不合成进小题图，仍以预览图和 JSON 关系为准。
- `report.json`：逐页结果和汇总指标。

重点观察 `question_recall`、`strict_complete_crop_rate`、`perfect_page_rate`、多切/漏题、裁少/裁多，以及启用 `--repeats 2` 后的 `repeat_stability`。模型生成的 `confidence` 仅作为诊断信息，不能单独用于自动放行。

## 使用边界

200 页 Golden Set 适合筛掉明显不合格的路线，不能以统计意义证明百万页下的 99.5% 召回。通过初筛后，应增加未参与任何提示调优的跨出版社、跨年级、跨版式页面再做验收。
