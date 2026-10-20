# -*- coding: utf-8 -*-
"""HUMAN-PENDING 机械核对：把"人工点一次"变成机器可比的证据。

背景
----
补全流水线的人工验收（HUMAN-PENDING）是"确认题页 → 确认答案页 → 查快照回填"。
纯人工记录容易出现两类问题：一是**看漏**（没发现没回填），二是**记错**
（报告里写的数与代码实际返回的不一致）。本脚本只做一件事：把该看的字段
逐条断言下来，打印 PASS/FAIL/SKIP 并给出退出码，让人点完之后有机器痕迹可查。

它**只读**：不写工程、不改快照、不调任何写接口。它读 `project.json`，
并调用 `enrich-status`（只读路由）。

断言清单（QA Round 13 规格，逐条对应）
------------------------------------
| # | 断言 | 数据来源 | SKIP 条件 |
|---|------|----------|-----------|
| 1 | ``enabled == true``（补全开关已开） | enrich-status 响应 | 不跳过：关着就是 FAIL |
| 2 | ``bridge_revision == 'enrich-snapshot-20261020'`` | 同上 | 不跳过 |
| 3 | ``enrich_summary`` 存在，且 ``status_counts`` **不存在** | 同上 | 不跳过 |
| 4 | ``review_summary.L0+L1+L2 == level_stats.total`` | 同上 | total 为 0 时 SKIP |
| 5 | ``todo[*].enrich_status`` ∈ 6 种待办状态 | 同上 | todo 为空时 SKIP |
| 6 | ``todo[*].snapshot_id`` 非空 | 同上 | todo 为空时 SKIP |
| 7 | 至少一条**已配对**题：``snapshot.answer`` 与答案页文本一致（规范化相等或包含）且 ``answer_source == 'paired'`` | project.json | 无已配对题时 SKIP |
| 8 | 至少一条**未配对但人工填了答案**的题：``answer_source == 'manual'`` | 同上 | 无此类题时 SKIP |

第 7 条**独立重算**期望答案：按 ``pairs`` 取出 ``answer_ids``，按"答案页顺序 +
页内顺序"拼接答案文本，**不调用** ``enrich_snapshot.answer_segment`` ——
否则就是拿被测函数自己证明自己。

退出码
------
``0`` 全部 PASS/SKIP；``1`` 有 FAIL；``2`` 无法核对（工程路径不合规 / 桥服务不可达 /
响应不是预期结构），它与"核对失败"必须区分开，否则会被误读成产品缺陷。

跑法
----
    python reports/qa_check_enrich_human.py --project runtime/manual_annotation_projects/<id>.json
    python reports/qa_check_enrich_human.py --project <id>.json --json out.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 打开教材会在后台抽样正文判定版式；核对脚本不需要它
os.environ.setdefault("PAPERFLOW_DISABLE_PROFILE_DETECT", "1")

API_PATH = ROOT / "desktop_annotation" / "paperflow_annotation_api.py"
EXPECTED_BRIDGE_REVISION = "enrich-snapshot-20261020"
TODO_STATUSES = ("pending", "stale", "kp_stale", "analysis_stale", "failed", "rejected")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_WHITESPACE = re.compile(r"\s+")


class Verdict:
    """一条断言的结果。``SKIP`` 不算失败，但要显式出现在报告里。"""

    def __init__(self, number: int, name: str, status: str, detail: str):
        self.number = number
        self.name = name
        self.status = status
        self.detail = detail

    def line(self) -> str:
        return f"[{self.status}] #{self.number} {self.name} —— {self.detail}"


def _load_api():
    """按 tests/test_annotation_api.py 的方式加载桥服务（desktop_annotation 不是包）。"""
    spec = importlib.util.spec_from_file_location("paperflow_annotation_api_human", API_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fetch_status(api, project_path: Path, base_url: str = "") -> dict:
    """取 ``enrich-status`` 响应。给了 ``base_url`` 就走 HTTP，否则进程内调用。

    两种方式读的是同一个只读路由；进程内调用不需要桥服务在跑，适合脚本自检。
    """
    if base_url:
        url = base_url.rstrip("/") + "/api/project/enrich-status"
        request = urllib.request.Request(
            url,
            data=json.dumps({"project_path": str(project_path)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    return api._enrich_status({"project_path": str(project_path)})


def _normalize(text: object) -> str:
    """规范化比较：去掉所有空白与常见标点，避免"多一个空格"被判失败。"""
    value = _WHITESPACE.sub("", str(text or ""))
    return re.sub(r"[，。；：、,.;:]", "", value)


def _answer_texts_by_id(project: dict) -> dict[str, str]:
    """按"答案页顺序 + 页内顺序"收集答案记录文本（独立于产品代码重算）。"""
    texts: dict[str, str] = {}
    for page in project.get("answer_pages") or []:
        if not isinstance(page, dict):
            continue
        for record in page.get("answers") or []:
            if not isinstance(record, dict):
                continue
            record_id = str(record.get("id") or "").strip()
            if not record_id:
                continue
            text = str(record.get("answer") or "").strip()
            if text:
                texts.setdefault(record_id, text)
    return texts


def _paired_answer_ids(project: dict) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for pair in project.get("pairs") or []:
        if not isinstance(pair, dict):
            continue
        question_id = str(pair.get("question_id") or "").strip()
        answer_ids = [str(value).strip() for value in pair.get("answer_ids") or []
                      if str(value or "").strip()]
        if question_id and answer_ids:
            mapping.setdefault(question_id, []).extend(answer_ids)
    return mapping


def _question_records(project: dict) -> list[dict]:
    records: list[dict] = []
    for page in project.get("pages") or []:
        if not isinstance(page, dict):
            continue
        for record in page.get("questions") or []:
            if isinstance(record, dict) \
                    and str(record.get("region_role", "question")) == "question":
                records.append(record)
    return records


def check_status_payload(payload: dict) -> list[Verdict]:
    """断言 1~6：全部来自 ``enrich-status`` 响应体。"""
    out: list[Verdict] = []

    enabled = payload.get("enabled")
    out.append(Verdict(
        1, "补全开关已开（enabled == true）",
        PASS if enabled is True else FAIL,
        f"enabled={enabled!r}"
        + ("；若为 False 请检查环境变量 PAPERFLOW_ENRICH" if enabled is not True else ""),
    ))

    revision = payload.get("bridge_revision")
    out.append(Verdict(
        2, f"bridge_revision == {EXPECTED_BRIDGE_REVISION}",
        PASS if revision == EXPECTED_BRIDGE_REVISION else FAIL,
        f"bridge_revision={revision!r}",
    ))

    has_summary = isinstance(payload.get("enrich_summary"), dict)
    has_wrong_key = "status_counts" in payload
    out.append(Verdict(
        3, "enrich_summary 存在，且不存在误写的 status_counts",
        PASS if has_summary and not has_wrong_key else FAIL,
        f"enrich_summary={'存在' if has_summary else '缺失'}，"
        f"status_counts={'存在（键名写错了）' if has_wrong_key else '不存在'}",
    ))

    level_stats = payload.get("level_stats") if isinstance(payload.get("level_stats"), dict) else {}
    review_summary = payload.get("review_summary") \
        if isinstance(payload.get("review_summary"), dict) else {}
    total = int(level_stats.get("total", 0) or 0)
    counted = sum(int(review_summary.get(key, 0) or 0) for key in ("L0", "L1", "L2"))
    out.append(Verdict(
        4, "review_summary.L0+L1+L2 == level_stats.total",
        PASS if total and counted == total else (SKIP if not total else FAIL),
        f"三档之和={counted}，level_stats.total={total}",
    ))

    todo = payload.get("todo") if isinstance(payload.get("todo"), list) else []
    if not todo:
        out.append(Verdict(5, "todo[*].enrich_status ∈ 6 种待办状态", SKIP, "todo 为空（本批无待办）"))
        out.append(Verdict(6, "todo[*].snapshot_id 非空", SKIP, "todo 为空（本批无待办）"))
        return out

    bad_status = [
        f"{item.get('question_id')}={item.get('enrich_status')!r}"
        for item in todo if isinstance(item, dict)
        and str(item.get("enrich_status") or "") not in TODO_STATUSES
    ]
    out.append(Verdict(
        5, "todo[*].enrich_status ∈ 6 种待办状态",
        PASS if not bad_status else FAIL,
        f"todo 共 {len(todo)} 条，越界 {len(bad_status)} 条"
        + (f"：{bad_status[:5]}" if bad_status else ""),
    ))

    empty_ids = [
        str(item.get("question_id"))
        for item in todo if isinstance(item, dict) and not str(item.get("snapshot_id") or "")
    ]
    out.append(Verdict(
        6, "todo[*].snapshot_id 非空",
        PASS if not empty_ids else FAIL,
        f"空 snapshot_id {len(empty_ids)} 条" + (f"：{empty_ids[:5]}" if empty_ids else ""),
    ))
    return out


def check_project_file(project: dict) -> list[Verdict]:
    """断言 7~8：直接读 project.json，验证"答案后到"的回填与来源标注。"""
    out: list[Verdict] = []
    texts = _answer_texts_by_id(project)
    paired = _paired_answer_ids(project)
    records = _question_records(project)

    paired_hits: list[str] = []
    paired_problems: list[str] = []
    manual_hits: list[str] = []
    for record in records:
        snap = record.get("snapshot")
        if not isinstance(snap, dict):
            continue
        source = str(snap.get("answer_source") or "")
        if source == "manual":
            manual_hits.append(str(record.get("id") or ""))
            continue
        if source != "paired":
            continue
        answer_ids = paired.get(str(record.get("id") or ""), []) \
            or [str(value) for value in snap.get("answer_ids") or []]
        expected = "\n".join(texts.get(value, "") for value in answer_ids
                             if texts.get(value, ""))
        actual = str(snap.get("answer") or "")
        if not expected:
            paired_problems.append(f"{record.get('id')}: 答案页里找不到配对答案记录")
            continue
        left, right = _normalize(actual), _normalize(expected)
        if left and right and (left == right or left in right or right in left):
            paired_hits.append(str(record.get("id") or ""))
        else:
            paired_problems.append(
                f"{record.get('id')}: snapshot.answer={actual[:24]!r} "
                f"与答案页文本={expected[:24]!r} 不一致"
            )

    if not paired_hits and not paired_problems:
        out.append(Verdict(7, "已配对题：snapshot.answer 与答案页文本一致且 answer_source=='paired'",
                           SKIP, "本工程没有已配对的题（无法核对回填）"))
    else:
        out.append(Verdict(
            7, "已配对题：snapshot.answer 与答案页文本一致且 answer_source=='paired'",
            PASS if paired_hits and not paired_problems else FAIL,
            f"一致 {len(paired_hits)} 条" + (f"；不一致 {paired_problems[:3]}" if paired_problems else ""),
        ))

    out.append(Verdict(
        8, "未配对但人工填了答案的题：answer_source == 'manual'",
        PASS if manual_hits else SKIP,
        f"命中 {len(manual_hits)} 条" + (f"：{manual_hits[:5]}" if manual_hits else "（本工程无此类题）"),
    ))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="HUMAN-PENDING 机械核对（只读）")
    parser.add_argument("--project", required=True, help="PaperFlow标注项目.json 的路径")
    parser.add_argument("--base-url", default="", help="桥服务地址；留空则进程内调用只读路由")
    parser.add_argument("--json", default="", help="把结果写成 JSON，便于归档比对")
    args = parser.parse_args()

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_file():
        print(f"[FATAL] 标注项目不存在：{project_path}", file=sys.stderr)
        return 2
    try:
        with project_path.open("r", encoding="utf-8") as handle:
            project = json.load(handle)
    except (OSError, ValueError) as error:
        print(f"[FATAL] 工程文件无法解析：{error}", file=sys.stderr)
        return 2

    try:
        api = _load_api()
        payload = fetch_status(api, project_path, args.base_url)
    except (ValueError, urllib.error.URLError, OSError) as error:
        print(f"[FATAL] 无法取到 enrich-status 响应（桥服务未起或工程路径不合规）：{error}",
              file=sys.stderr)
        print("        工程路径需位于 runtime/manual_annotation_projects/ "
              "或教材目录下的 PaperFlow标注项目.json", file=sys.stderr)
        return 2

    verdicts = check_status_payload(payload)
    verdicts += check_project_file(project)

    print(f"核对工程：{project_path}")
    print(f"响应来源：{args.base_url or '进程内调用 _enrich_status'}")
    print("-" * 78)
    for verdict in verdicts:
        print(verdict.line())
    print("-" * 78)
    failed = [item for item in verdicts if item.status == FAIL]
    skipped = [item for item in verdicts if item.status == SKIP]
    blocked = payload.get("blocked") if isinstance(payload.get("blocked"), list) else []
    print(f"合计 {len(verdicts)} 条：PASS {len(verdicts) - len(failed) - len(skipped)}"
          f" / SKIP {len(skipped)} / FAIL {len(failed)}")
    print(f"[INFO] enrich_status.blocked（无快照、不计入待办）：{len(blocked)} 条"
          "  ← rebuild_missing 补建后应下降")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "project": str(project_path),
            "verdicts": [item.line() for item in verdicts],
            "failed": len(failed),
            "skipped": len(skipped),
            "blocked": len(blocked),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] 结果已写入 {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())