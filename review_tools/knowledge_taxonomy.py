# -*- coding: utf-8 -*-
"""知识点词表：加载 / 归一键 / 越界校验（纯文件读取，无网络）。

为什么必须是**封闭**词表
------------------------
知识点是交付字段，一错就是全书一致地错。所以 ``resolve()`` 只有三步归一层级：

1. 精确匹配 ``name``
2. 别名匹配 ``aliases``
3. 规范化后匹配（全角→半角、去空格、去括号补充说明、简繁归一）

**没有第四步**：不做编辑距离模糊、不做向量相似、不做同义词生成、不做"就近挂靠"。
三步都不命中 → 返回 ``None`` → 调用方计入 ``rejected`` 并把该题升 L2。
宁可让 AI 的越界词被拒收并暴露出来，也不能静默接受一个不在词表里的说法。

加载校验（任一不通过即抛 ``ValueError``，不静默降级）
--------------------------------------------------
``version`` 非空且与文件名一致；``code`` 唯一；``name`` 非空且同学科唯一；
``parent`` 要么为空要么存在；``grades`` 是 1~12 的整数。

进程内缓存 + ``mtime`` 校验：批处理是常驻进程，不能每题重读文件。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(ROOT_DIR, 'runtime', 'taxonomy')

# 文件名用 ascii slug，JSON 内 subject 用中文名；两个方向都要能 load()
SUBJECT_SLUGS = {
    '数学': 'math', 'math': 'math',
    '语文': 'chinese', 'chinese': 'chinese',
    '英语': 'english', 'english': 'english',
    '物理': 'physics', 'physics': 'physics',
    '化学': 'chemistry', 'chemistry': 'chemistry',
    '生物': 'biology', 'biology': 'biology',
    '历史': 'history', 'history': 'history',
    '地理': 'geography', 'geography': 'geography',
}

GRADE_NAMES = {
    1: '一年级', 2: '二年级', 3: '三年级', 4: '四年级', 5: '五年级', 6: '六年级',
    7: '七年级', 8: '八年级', 9: '九年级', 10: '高一', 11: '高二', 12: '高三',
}
_GRADE_ALIASES = {name: index for index, name in GRADE_NAMES.items()}
_GRADE_ALIASES.update({
    '初一': 7, '初二': 8, '初三': 9,
    '初中一年级': 7, '初中二年级': 8, '初中三年级': 9,
})

# 简繁归一：只覆盖数学教材高频字，够用且可审。不做整表转换（会引入新依赖）。
_TRADITIONAL_TO_SIMPLIFIED = {
    '數': '数', '學': '学', '邊': '边', '長': '长', '寬': '宽', '圓': '圆',
    '圖': '图', '計': '计', '統': '统', '練': '练', '習': '习', '時': '时',
    '間': '间', '與': '与', '個': '个', '幾': '几', '減': '减', '除': '除',
    '積': '积', '麵': '面', '體': '体', '積': '积', '轉': '转', '稱': '称',
    '對': '对', '稱': '称', '軸': '轴', '線': '线', '點': '点', '邊': '边',
    '應': '应', '題': '题', '單': '单', '複': '复', '雜': '杂', '認識': '认识',
    '認': '认', '識': '识', '開': '开', '關': '关', '係': '系', '進': '进',
    '隻': '只', '場': '场', '範': '范', '圍': '围', '總': '总', '簡': '简',
    '數': '数', '縱': '纵', '橫': '横', '豎': '竖', '徑': '径', '圓': '圆',
    '夠': '够', '條': '条', '嗎': '吗', '麼': '么', '爲': '为', '為': '为',
}

# 括号内的补充说明（"三角形面积（1）" → "三角形面积"）
_PARENTHETICAL_RE = re.compile(r'[（(\[【].*?[）)\]】]')
_PUNCTUATION_RE = re.compile(r'[·・,，。;；:：!！?？"\'“”‘’《》<>—\-_/\\|]+')
# 行首序号：必须带分隔符才剥。"20以内数的认识" 里的 20 是数量不是序号，
# 剥掉会把 "20以内…" 与 "100以内…" 归成同一个键（真实事故）。
_LEADING_PAREN_INDEX_RE = re.compile(r'^[（(]\s*\d{1,3}\s*[）)]\s*')
_LEADING_DOTTED_INDEX_RE = re.compile(r'^\d{1,3}\s*[、.．)]\s*')
# 括号组（用于抽"（四年级）"这种年级限定）
_PAREN_GROUP_RE = re.compile(r'[（(]([^（）()]{1,12})[）)]')
# 尾部年级限定（无括号写法："旋转 五年级"）
_GRADE_SUFFIX_RE = re.compile(
    r'[（(]?\s*([一二三四五六七八九]|1[0-2]|[1-9]|十)\s*年级\s*[）)]?\s*$'
)
_FILENAME_RE = re.compile(
    r'^knowledge_points\.(?P<slug>[A-Za-z0-9_\-]+)\.(?P<version>\d[A-Za-z0-9._\-]*)\.json$'
)

_CACHE: dict[str, tuple[float, 'KnowledgeTaxonomy']] = {}


class TaxonomyMissing(ValueError):
    """词表文件不存在。继承 ValueError，便于调用方统一按"加载失败"处理。"""


class TaxonomyInvalid(ValueError):
    """词表内容不合法（code/name 冲突、grades 越界、parent 悬空等）。"""


def _subject_slug(subject: str) -> str:
    text = str(subject or '').strip()
    if not text:
        raise TaxonomyInvalid('subject 不能为空')
    return SUBJECT_SLUGS.get(text, SUBJECT_SLUGS.get(text.lower(), text.lower()))


def _version_key(version: str) -> tuple:
    """'2026.10' → (2026, 10)；非数字段退化为字符串比较，保证可排序。"""
    parts = []
    for chunk in str(version or '').split('.'):
        parts.append((0, int(chunk)) if chunk.isdigit() else (1, chunk))
    return tuple(parts)


def _to_simplified(text: str) -> str:
    return ''.join(_TRADITIONAL_TO_SIMPLIFIED.get(char, char) for char in text)


def normalize_name(text: str) -> str:
    """规范化：NFKC（全角→半角）→ 简繁归一 → 去序号 → 去括号补充 → 去空格与标点。"""
    value = unicodedata.normalize('NFKC', str(text or ''))
    value = _to_simplified(value)
    value = _LEADING_PAREN_INDEX_RE.sub('', value)
    value = _LEADING_DOTTED_INDEX_RE.sub('', value)
    value = _PARENTHETICAL_RE.sub('', value)
    value = _PUNCTUATION_RE.sub('', value)
    value = re.sub(r'\s+', '', value)
    return value


def grade_index(value) -> int | None:
    """年级归一：3 / '3' / '三年级' / '七年级上册' / '初一' → 整数 1~12。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = int(value)
        return number if 1 <= number <= 12 else None
    text = unicodedata.normalize('NFKC', str(value or '')).strip()
    if not text:
        return None
    if text.isdigit():
        number = int(text)
        return number if 1 <= number <= 12 else None
    for alias, index in sorted(_GRADE_ALIASES.items(), key=lambda item: -len(item[0])):
        if text.startswith(alias):
            return index
    return None


def split_grade_qualifier(text: str) -> tuple[str, int | None]:
    """把"带年级限定"的写法拆成 (基名, 年级)。

    ``'笔算乘法（四年级）'`` → ``('笔算乘法', 4)``；``'旋转(5年级)'`` → ``('旋转', 5)``；
    无年级限定 → ``(原文, None)``。

    存在的理由：有些知识点写法（笔算乘法/笔算除法/统计表/运算顺序/旋转）
    在课标里**确实跨年级**，同一个写法在二、三年段与四、五年段指的不是同一个节点。
    直接收成一条必然静默归错年级，所以改成"带年级限定才收录"，
    不带限定的写法走 rejected + L2 由人挂靠。
    """
    value = str(text or '')
    for match in _PAREN_GROUP_RE.finditer(value):
        index = grade_index(match.group(1))
        if index:
            base = (value[:match.start()] + value[match.end():]).strip()
            return base, index
    match = _GRADE_SUFFIX_RE.search(value)
    if match:
        base = value[:match.start()].strip()
        if base:
            return base, grade_index(match.group(0))
    return value.strip(), None


class KnowledgeTaxonomy:
    """一学科一版本的封闭词表。所有方法都是纯内存操作，可安全复用与并发读。"""

    def __init__(self, payload: dict, path: str, slug: str, file_version: str):
        if not isinstance(payload, dict):
            raise TaxonomyInvalid('词表根节点必须是对象')
        self.path = path
        self.slug = slug
        self.file_version = file_version
        declared = str(payload.get('version') or '').strip()
        if not declared:
            raise TaxonomyInvalid('词表 version 不能为空')
        # 允许 'math-2026.10'（文档示例写法）或 '2026.10'（文件名写法）
        if declared not in {file_version, f'{slug}-{file_version}'}:
            raise TaxonomyInvalid(
                f'词表 version={declared!r} 与文件名版本 {file_version!r} 不一致'
            )
        self.version = declared if declared.startswith(f'{slug}-') else f'{slug}-{declared}'
        self.subject = str(payload.get('subject') or '').strip()
        if not self.subject:
            raise TaxonomyInvalid('词表 subject 不能为空')
        self.stages = [str(item) for item in payload.get('stages') or []]
        self.domains = [str(item) for item in payload.get('domains') or []]
        self.updated_at = str(payload.get('updated_at') or '')
        self.source = str(payload.get('source') or '')
        self.raw = payload
        self.nodes: list[dict] = [dict(node) for node in payload.get('nodes') or []]
        self._validate_nodes()
        self._validate_alias_uniqueness()
        self._build_index()

    # -------------------------------------------------------- 校验与索引
    def _plain_keys(self, node: dict) -> list[str]:
        """节点的"不带年级限定"的规范键：name + 非年级限定的 alias。

        年级限定的别名（如「笔算乘法（四年级）」）不走这一层 —— 它必须与
        同名的其它年级写法共存，靠 (基名, 年级) 二元键区分。
        """
        keys = [normalize_name(str(node.get('name') or ''))]
        for alias in node.get('aliases') or []:
            base, grade = split_grade_qualifier(str(alias))
            if grade:
                continue
            key = normalize_name(base)
            if key:
                keys.append(key)
        return [key for key in keys if key]

    def _qualified_keys(self, node: dict) -> list[tuple[str, int]]:
        keys = []
        for alias in node.get('aliases') or []:
            base, grade = split_grade_qualifier(str(alias))
            if not grade:
                continue
            key = normalize_name(base)
            if key:
                keys.append((key, grade))
        return keys

    def _validate_alias_uniqueness(self) -> None:
        """跨条目重复的别名/名称一律拒绝加载（fail fast，绝不"取第一条"）。

        为什么必须抛错：同一写法挂在两个年级节点上时（例如「笔算乘法」既是三年级
        两位数乘两位数、又是四年级三位数乘两位数），``setdefault`` 会静默选中其中
        一个，于是四年级题面被判成三年级知识点、``validate()`` 还判 accepted ——
        既不拒收也不降 L2，整本书一致地错。这与"宁可拒收也不静默接受"直接冲突。

        消歧的正规做法有两种，都允许：
        - 加年级限定写成「笔算乘法（四年级）」（本版采用，见 ``_qualified_keys``）；
        - 只在课程唯一时保留不带限定的写法。
        """
        owners: dict[str, set[str]] = {}
        qualified: dict[tuple[str, int], set[str]] = {}
        for node in self.nodes:
            code = str(node.get('code'))
            for key in self._plain_keys(node):
                owners.setdefault(key, set()).add(code)
            for key in self._qualified_keys(node):
                qualified.setdefault(key, set()).add(code)
        conflicts = {
            key: sorted(codes) for key, codes in owners.items() if len(codes) > 1
        }
        conflicts.update({
            f'{key[0]}(年级{key[1]})': sorted(codes)
            for key, codes in qualified.items() if len(codes) > 1
        })
        if conflicts:
            detail = '；'.join(
                f'{key!r} → {codes}' for key, codes in sorted(conflicts.items())[:10]
            )
            raise TaxonomyInvalid(
                f'词表存在跨条目重复的别名/名称（会静默归到错误节点）：{detail}'
            )
        self.alias_conflicts: dict[str, list[str]] = {}

    def _validate_nodes(self) -> None:
        if not self.nodes:
            raise TaxonomyInvalid('词表 nodes 不能为空')
        codes = set()
        names = set()
        for node in self.nodes:
            code = str(node.get('code') or '').strip()
            name = str(node.get('name') or '').strip()
            if not code:
                raise TaxonomyInvalid('节点缺少 code')
            if code in codes:
                raise TaxonomyInvalid(f'code 重复：{code}')
            if not name:
                raise TaxonomyInvalid(f'节点 {code} 缺少 name')
            if name in names:
                raise TaxonomyInvalid(f'name 在同学科内重复：{name}')
            grades = node.get('grades')
            if not isinstance(grades, list) or not grades:
                raise TaxonomyInvalid(f'节点 {code} 的 grades 不能为空')
            for grade in grades:
                if isinstance(grade, bool) or not isinstance(grade, int) \
                        or not 1 <= grade <= 12:
                    raise TaxonomyInvalid(f'节点 {code} 的 grades 必须是 1~12 的整数：{grades}')
            stage = str(node.get('stage') or '').strip()
            if stage and self.stages and stage not in self.stages:
                raise TaxonomyInvalid(f'节点 {code} 的 stage={stage} 不在 stages 内')
            domain = str(node.get('domain') or '').strip()
            if domain and self.domains and domain not in self.domains:
                raise TaxonomyInvalid(f'节点 {code} 的 domain={domain} 不在 domains 内')
            aliases = node.get('aliases')
            if aliases is not None and not isinstance(aliases, list):
                raise TaxonomyInvalid(f'节点 {code} 的 aliases 必须是数组')
            codes.add(code)
            names.add(name)
        for node in self.nodes:
            parent = str(node.get('parent') or '').strip()
            if parent and parent not in codes:
                raise TaxonomyInvalid(f'节点 {node.get("code")} 的 parent={parent} 不存在')

    def _build_index(self) -> None:
        self._by_code: dict[str, dict] = {}
        self._by_name: dict[str, str] = {}
        self._by_alias: dict[str, str] = {}
        self._by_normalized: dict[str, str] = {}
        self._by_qualified: dict[tuple[str, int], str] = {}
        self._parent_codes: set[str] = set()
        for node in self.nodes:
            code = str(node['code'])
            self._by_code[code] = node
            name = str(node['name']).strip()
            self._by_name[name] = code
            self._by_normalized.setdefault(normalize_name(name), code)
            parent = str(node.get('parent') or '').strip()
            if parent:
                self._parent_codes.add(parent)
        for node in self.nodes:
            code = str(node['code'])
            for alias in node.get('aliases') or []:
                text = str(alias).strip()
                if not text:
                    continue
                base, grade = split_grade_qualifier(text)
                if grade:
                    # 年级限定别名：只进 (基名, 年级) 二元键，绝不进通用键，
                    # 否则「笔算乘法（四年级）」的基名会覆盖/被覆盖三年级那条。
                    self._by_qualified.setdefault((normalize_name(base), grade), code)
                    continue
                self._by_alias.setdefault(text, code)
                self._by_normalized.setdefault(normalize_name(text), code)
        self._build_ambiguous_index()

    def _build_ambiguous_index(self) -> None:
        """登记"课标里确实跨年级、且不做年级限定就无法判定"的写法。

        这些写法**一条都不进词表**：``resolve()`` 必须返回 ``None`` 并记 warning，
        由 ``validate()`` 判 rejected → 该题升 L2 由人按年级挂靠。
        """
        declared = self.raw.get('ambiguous_terms') or {}
        self.ambiguous_terms: dict[str, list[str]] = {}
        if not isinstance(declared, dict):
            raise TaxonomyInvalid('ambiguous_terms 必须是 {写法: [code, ...]} 的对象')
        for term, codes in declared.items():
            text = str(term or '').strip()
            if not text:
                continue
            if not isinstance(codes, list) or len(codes) < 2:
                raise TaxonomyInvalid(f'ambiguous_terms[{text!r}] 至少要列 2 个 code')
            resolved = []
            for code in codes:
                key = str(code or '').strip()
                if key not in self._by_code:
                    raise TaxonomyInvalid(f'ambiguous_terms[{text!r}] 指向不存在的 code：{key}')
                resolved.append(key)
            base = normalize_name(split_grade_qualifier(text)[0])
            if base in self._by_normalized or base in self._by_alias:
                raise TaxonomyInvalid(
                    f'ambiguous_terms[{text!r}] 已能被词表唯一解析，登记矛盾'
                )
            self.ambiguous_terms[text] = resolved
        self._ambiguous_by_base = {
            normalize_name(split_grade_qualifier(term)[0]): term
            for term in self.ambiguous_terms
        }

    # -------------------------------------------------------- 查询
    @property
    def leaf_nodes(self) -> list[dict]:
        return [node for node in self.nodes if str(node['code']) not in self._parent_codes]

    def node(self, code: str) -> dict | None:
        return self._by_code.get(str(code or ''))

    def codes(self) -> list[str]:
        return [str(node['code']) for node in self.nodes]

    def resolve(self, text: str, warnings: list | None = None) -> str | None:
        """四步归一：精确 name → 精确 alias → 年级限定 alias → 规范化后匹配。

        歧义写法（课标里跨年级、未加年级限定）**不进词表**：走到最后一步仍在
        ``ambiguous_terms`` 里时返回 ``None``，并向 ``warnings`` 追加一条诊断。
        绝不猜年级 —— 猜错的代价是整本书一致地挂错知识点。
        """
        value = str(text or '').strip()
        if not value:
            return None
        if value in self._by_name:
            return self._by_name[value]
        if value in self._by_alias:
            return self._by_alias[value]

        base_text, grade = split_grade_qualifier(value)
        if grade:
            code = self._by_qualified.get((normalize_name(base_text), grade))
            if code:
                return code

        normalized = normalize_name(value)
        if normalized and normalized in self._by_normalized:
            return self._by_normalized[normalized]

        base_key = normalize_name(base_text)
        if base_key in self._ambiguous_by_base:
            term = self._ambiguous_by_base[base_key]
            if warnings is not None:
                warnings.append(
                    f'知识点 {value!r} 跨多个年级（{self.ambiguous_terms[term]}），'
                    f'不做年级推测，需人工挂靠'
                )
            return None
        return None

    def validate(self, names: list[str],
                 warnings: list | None = None) -> tuple[list[tuple[str, str]], list[str]]:
        """越界拒收。返回 ``([(词表标准名, code)], [越界词原文])``。

        ``knowledge_points`` 只写词表里的标准 name（不是 AI 的写法），
        这样交付出去的字段天然归一，下游不用再清洗。
        歧义词既进 ``rejected`` 也进 ``warnings``（调用方据此把该题升 L2）。
        """
        accepted: list[tuple[str, str]] = []
        rejected: list[str] = []
        seen_codes: set[str] = set()
        seen_rejected: set[str] = set()
        for item in names or []:
            text = str(item or '').strip()
            if not text:
                continue
            code = self.resolve(text, warnings)
            if code is None:
                if text not in seen_rejected:
                    seen_rejected.add(text)
                    rejected.append(text)
                continue
            if code in seen_codes:
                continue
            seen_codes.add(code)
            accepted.append((str(self._by_code[code]['name']), code))
        return accepted, rejected

    def prompt_vocabulary(self, stage: str = '', grade='', limit: int = 120) -> list[str]:
        """给 prompt 用的封闭候选词表（只返回叶子节点，按年级、code 排序）。"""
        stage_text = str(stage or '').strip()
        target_grade = grade_index(grade) if grade not in ('', None) else None
        selected = []
        for node in self.leaf_nodes:
            if stage_text and str(node.get('stage') or '').strip() != stage_text:
                continue
            grades = [grade_index(item) for item in node.get('grades') or []]
            if target_grade is not None and target_grade not in grades:
                continue
            selected.append(node)
        selected.sort(key=lambda node: (
            min([grade or 99 for grade in (grade_index(item) for item in node.get('grades') or [])]),
            str(node.get('code') or ''),
        ))
        names = [str(node['name']) for node in selected]
        if limit is not None and limit > 0:
            return names[:int(limit)]
        return names

    def stats(self) -> dict:
        by_stage: dict[str, int] = {}
        by_domain: dict[str, int] = {}
        by_grade: dict[str, int] = {}
        alias_count = 0
        for node in self.leaf_nodes:
            stage = str(node.get('stage') or '')
            by_stage[stage] = by_stage.get(stage, 0) + 1
            domain = str(node.get('domain') or '')
            by_domain[domain] = by_domain.get(domain, 0) + 1
            for grade in node.get('grades') or []:
                key = GRADE_NAMES.get(grade_index(grade) or 0, str(grade))
                by_grade[key] = by_grade.get(key, 0) + 1
            alias_count += len([item for item in node.get('aliases') or [] if str(item).strip()])
        return {
            'version': self.version,
            'subject': self.subject,
            'path': self.path,
            'nodes': len(self.nodes),
            'leaves': len(self.leaf_nodes),
            'groups': len(self.nodes) - len(self.leaf_nodes),
            'aliases': alias_count,
            'alias_conflicts': len(self.alias_conflicts),
            'grade_qualified_aliases': sum(
                len(self._qualified_keys(node)) for node in self.nodes
            ),
            'ambiguous_terms': len(self.ambiguous_terms),
            'by_stage': dict(sorted(by_stage.items())),
            'by_domain': dict(sorted(by_domain.items())),
            'by_grade': dict(sorted(by_grade.items())),
            'updated_at': self.updated_at,
        }


def _candidate_files(root: str, slug: str) -> list[tuple[str, str]]:
    found = []
    try:
        entries = os.listdir(root)
    except OSError:
        return found
    for entry in entries:
        match = _FILENAME_RE.match(entry)
        if not match or match.group('slug') != slug:
            continue
        found.append((match.group('version'), os.path.join(root, entry)))
    found.sort(key=lambda item: _version_key(item[0]))
    return found


def load(subject: str, version: str | None = None, root: str | None = None) -> KnowledgeTaxonomy:
    """加载词表。``version=None`` 取同学科最高版本；快照里必须记具体版本号。

    旧版本文件必须保留：历史快照的 ``taxonomy_version`` 要能回查。
    """
    slug = _subject_slug(subject)
    base = root or DEFAULT_ROOT
    candidates = _candidate_files(base, slug)
    if not candidates:
        raise TaxonomyMissing(f'未找到 {subject!r} 的知识点词表（目录：{base}）')
    if version:
        wanted = str(version).strip()
        wanted = wanted[len(slug) + 1:] if wanted.startswith(f'{slug}-') else wanted
        matched = [item for item in candidates if item[0] == wanted]
        if not matched:
            raise TaxonomyMissing(
                f'未找到 {subject!r} 词表版本 {version!r}（已知：{[item[0] for item in candidates]}）'
            )
        file_version, path = matched[-1]
    else:
        file_version, path = candidates[-1]

    try:
        mtime = os.path.getmtime(path)
    except OSError as error:
        raise TaxonomyMissing(f'无法读取词表文件：{path}（{error}）') from error
    cached = _CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        with open(path, 'r', encoding='utf-8') as file:
            payload = json.load(file)
    except (OSError, ValueError) as error:
        raise TaxonomyInvalid(f'词表文件解析失败：{path}（{error}）') from error
    taxonomy = KnowledgeTaxonomy(payload, path, slug, file_version)
    _CACHE[path] = (mtime, taxonomy)
    return taxonomy


def clear_cache() -> None:
    """测试与词表运营用：丢弃进程内缓存，强制下次重新读盘。"""
    _CACHE.clear()
