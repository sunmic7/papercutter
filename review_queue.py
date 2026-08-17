#!/usr/bin/env python3
"""题目—答案审核协议、单调配对与审核结果应用。

该模块刻意不依赖 OCR 或 GUI。切题主程序、人工审核工具和未来在线智能体
共享同一份 JSON 协议，避免三套逻辑逐渐不一致。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone


QUEUE_SCHEMA_VERSION = 1
DECISION_SCHEMA_VERSION = 1
VALID_DECISIONS = {
    'accept', 'reject_pair', 'question_bad', 'answer_bad', 'needs_manual',
}


def json_safe(value):
    """Recursively convert NumPy/OpenCV scalar values to JSON primitives.

    OCR and image-layout code can place ``np.int32`` / ``np.float32`` values
    into source boxes.  They compare like ordinary numbers but Python's JSON
    encoder cannot serialize them.  Keeping this conversion at the review
    protocol boundary makes queue fingerprints and persisted review files use
    the same stable, portable representation.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return json_safe(scalar)
    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            converted = list_method()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return json_safe(converted)
    return value


def _clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(upper, float(value)))


def _number(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _candidate_features(question, answer, question_index, answer_index, q_count, a_count):
    question_number = _number(question.get('num'))
    answer_number = _number(answer.get('num'))
    number_detected = bool(question.get('number_detected', True))
    cross_page_merge = bool(question.get('_cross_page_merge', False))
    context_ambiguous = bool(question.get('_answer_context_ambiguous', False))
    question_confidence = _clamp(
        question.get('_number_confidence', question.get('number_confidence', 1.0 if number_detected else 0.0))
    )
    exact_number = question_number is not None and question_number == answer_number
    number_conflict = (
        number_detected and question_number is not None and answer_number is not None
        and question_number != answer_number
    )
    q_position = question_index / max(1, q_count - 1)
    a_position = answer_index / max(1, a_count - 1)
    sequence_score = _clamp(1.0 - abs(q_position - a_position))
    ocr_confidence = _clamp(answer.get('ocr_confidence', 0.65))

    if number_conflict:
        score = 0.04 + 0.10 * sequence_score + 0.04 * ocr_confidence
    elif exact_number and number_detected:
        score = 0.76 + 0.08 * ocr_confidence + 0.08 * question_confidence + 0.08 * sequence_score
    elif exact_number:
        score = 0.61 + 0.09 * ocr_confidence + 0.10 * sequence_score
    else:
        # 无可靠题号时，顺序只能形成待审核候选，永不进入自动通过区间。
        score = 0.37 + 0.09 * ocr_confidence + 0.18 * sequence_score
    score = _clamp(score)
    return {
        'question_number': question_number,
        'answer_number': answer_number,
        'question_number_detected': number_detected,
        'cross_page_merge': cross_page_merge,
        'answer_context_ambiguous': context_ambiguous,
        'question_number_confidence': round(question_confidence, 6),
        'exact_number': exact_number,
        'number_conflict': number_conflict,
        'sequence_score': round(sequence_score, 6),
        'answer_ocr_confidence': round(ocr_confidence, 6),
        'score': round(score, 6),
    }


def _match_utility(features):
    if features['number_conflict']:
        return -3.0
    return features['score'] * 4.0 - 2.0


def _pair_decision(features):
    score = features['score']
    if (
        features['exact_number']
        and features['question_number_detected']
        and not features['number_conflict']
        and score >= 0.90
        and features['question_number_confidence'] >= 0.72
        and features['answer_ocr_confidence'] >= 0.72
        and not features['cross_page_merge']
        and not features['answer_context_ambiguous']
    ):
        return 'auto'
    if not features['number_conflict'] and score >= 0.55:
        return 'review'
    return 'no_answer'


def _pair_reasons(features, band):
    reasons = []
    if features['exact_number']:
        reasons.append('题号一致')
    if not features['question_number_detected']:
        reasons.append('题目题号未可靠识别')
    elif features['question_number_confidence'] < 0.72:
        reasons.append('题目题号 OCR 置信度不足以自动通过')
    if features['number_conflict']:
        reasons.append('题目与答案题号冲突')
    if features.get('cross_page_merge'):
        reasons.append('题目由跨页内容拼接，必须复核边界')
    if features.get('answer_context_ambiguous'):
        reasons.append('答案上下文仅凭课时顺序选择，必须复核')
    if features['sequence_score'] < 0.65:
        reasons.append('题目与答案在序列中的位置偏差较大')
    if features['answer_ocr_confidence'] < 0.72:
        reasons.append('答案题号 OCR 置信度不足以自动通过')
    if band == 'auto':
        reasons.append('满足自动通过阈值')
    elif band == 'review':
        reasons.append('仅作为候选，需人工或智能体确认')
    else:
        reasons.append('证据不足，未自动配对')
    return reasons


def align_monotonic(questions, answers):
    """按阅读顺序做一对一动态规划配对，允许任一侧跳过。

    返回 ``pairs``（与 questions 等长）和未使用答案索引。低置信度候选可供
    审核，但题号冲突不会被强行写成正式答案。
    """
    questions = list(questions or [])
    answers = list(answers or [])
    q_count, a_count = len(questions), len(answers)
    skip_question, skip_answer = -0.52, -0.42
    negative_infinity = float('-inf')
    scores = [[negative_infinity] * (a_count + 1) for _ in range(q_count + 1)]
    previous = [[None] * (a_count + 1) for _ in range(q_count + 1)]
    scores[0][0] = 0.0

    for qi in range(q_count + 1):
        for ai in range(a_count + 1):
            current = scores[qi][ai]
            if current == negative_infinity:
                continue
            if qi < q_count and current + skip_question > scores[qi + 1][ai]:
                scores[qi + 1][ai] = current + skip_question
                previous[qi + 1][ai] = (qi, ai, 'skip_question', None)
            if ai < a_count and current + skip_answer > scores[qi][ai + 1]:
                scores[qi][ai + 1] = current + skip_answer
                previous[qi][ai + 1] = (qi, ai, 'skip_answer', None)
            if qi < q_count and ai < a_count:
                features = _candidate_features(
                    questions[qi], answers[ai], qi, ai, q_count, a_count,
                )
                candidate = current + _match_utility(features)
                if candidate > scores[qi + 1][ai + 1]:
                    scores[qi + 1][ai + 1] = candidate
                    previous[qi + 1][ai + 1] = (qi, ai, 'match', features)

    matched = {}
    used_answers = set()
    qi, ai = q_count, a_count
    while qi or ai:
        step = previous[qi][ai]
        if step is None:
            break
        prev_qi, prev_ai, action, features = step
        if action == 'match':
            band = _pair_decision(features)
            if band != 'no_answer':
                matched[prev_qi] = (prev_ai, features, band)
                used_answers.add(prev_ai)
        qi, ai = prev_qi, prev_ai

    pairs = []
    for question_index, question in enumerate(questions):
        match = matched.get(question_index)
        if match is None:
            pairs.append({
                'question': question,
                'answer': None,
                'answer_index': None,
                'score': 0.0,
                'decision_band': 'no_answer',
                'features': {
                    'question_number': _number(question.get('num')),
                    'question_number_detected': bool(question.get('number_detected', True)),
                },
                'reasons': ['未找到不破坏顺序且证据充分的答案'],
            })
            continue
        answer_index, features, band = match
        pairs.append({
            'question': question,
            'answer': answers[answer_index],
            'answer_index': answer_index,
            'score': features['score'],
            'decision_band': band,
            'features': features,
            'reasons': _pair_reasons(features, band),
        })
    return pairs, [index for index in range(a_count) if index not in used_answers]


def stable_review_id(unit_name, major_idx, sequence, question_source):
    identity = {
        'unit': unit_name,
        'major_idx': major_idx,
        'sequence': sequence,
        'source_image': question_source.get('image', ''),
        'source_box': question_source.get('question_box', []),
        'source_segments': question_source.get('segments', []),
    }
    payload = json.dumps(json_safe(identity), ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return 'review-' + hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]


def queue_fingerprint(items):
    identity = [
        {
            'review_id': item.get('review_id'),
            'question': item.get('question'),
            'answer': item.get('answer'),
            'pairing': item.get('pairing'),
        }
        for item in items
    ]
    payload = json.dumps(json_safe(identity), ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def create_review_queue(unit_name, items, algorithm_version=''):
    items = json_safe(list(items))
    return {
        'schema_version': QUEUE_SCHEMA_VERSION,
        'kind': 'question_answer_review_queue',
        'algorithm_version': algorithm_version,
        'unit_name': unit_name,
        'coordinate_format': 'xyxy_pixels',
        'queue_fingerprint': queue_fingerprint(items),
        'summary': {
            'total': len(items),
            'auto': sum(item.get('pairing', {}).get('decision_band') == 'auto' for item in items),
            'needs_review': sum(item.get('pairing', {}).get('decision_band') != 'auto' for item in items),
            'without_answer': sum(not item.get('answer') for item in items),
        },
        'items': items,
    }


def validate_review_queue(queue):
    if not isinstance(queue, dict) or queue.get('schema_version') != QUEUE_SCHEMA_VERSION:
        raise ValueError('审核清单格式无效：schema_version 必须为 1')
    items = queue.get('items')
    if not isinstance(items, list):
        raise ValueError('审核清单格式无效：缺少 items 数组')
    identifiers = set()
    for item in items:
        review_id = item.get('review_id') if isinstance(item, dict) else None
        if not isinstance(review_id, str) or not review_id:
            raise ValueError('审核项缺少 review_id')
        if review_id in identifiers:
            raise ValueError(f'审核项 review_id 重复：{review_id}')
        identifiers.add(review_id)
        if item.get('pairing', {}).get('decision_band') not in {'auto', 'review', 'no_answer'}:
            raise ValueError(f'审核项状态无效：{review_id}')
    expected = queue_fingerprint(items)
    if queue.get('queue_fingerprint') != expected:
        raise ValueError('审核清单指纹不一致，文件可能被不完整修改')
    return queue


def new_decision_document(queue, reviewer_type='human', reviewer_name=''):
    validate_review_queue(queue)
    return {
        'schema_version': DECISION_SCHEMA_VERSION,
        'kind': 'question_answer_review_decisions',
        'queue_fingerprint': queue['queue_fingerprint'],
        'reviewer': {'type': reviewer_type, 'name': reviewer_name},
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'decisions': [],
    }


def validate_decisions(queue, document):
    validate_review_queue(queue)
    if not isinstance(document, dict) or document.get('schema_version') != DECISION_SCHEMA_VERSION:
        raise ValueError('审核决定格式无效：schema_version 必须为 1')
    if document.get('queue_fingerprint') != queue['queue_fingerprint']:
        raise ValueError('审核决定不属于当前清单，已拒绝应用')
    queue_items = {item['review_id']: item for item in queue['items']}
    queue_ids = set(queue_items)
    seen = set()
    for record in document.get('decisions', []):
        review_id = record.get('review_id')
        decision = record.get('decision')
        if review_id not in queue_ids:
            raise ValueError(f'审核决定引用了未知 review_id：{review_id}')
        if review_id in seen:
            raise ValueError(f'同一审核项出现多次：{review_id}')
        if decision not in VALID_DECISIONS:
            raise ValueError(f'不支持的审核决定：{decision}')
        if decision == 'accept' and not queue_items[review_id].get('answer'):
            raise ValueError(f'没有答案候选的审核项不能标记为 accept：{review_id}')
        for field in ('corrected_question_box', 'corrected_answer_box'):
            box = record.get(field)
            if box is not None and not (
                isinstance(box, list) and len(box) == 4
                and all(isinstance(value, (int, float)) for value in box)
                and box[2] > box[0] and box[3] > box[1]
            ):
                raise ValueError(f'{field} 必须为有效的 [x0,y0,x1,y1]')
            if field == 'corrected_answer_box' and box is not None and not queue_items[review_id].get('answer'):
                raise ValueError(f'没有答案来源，不能设置 corrected_answer_box：{review_id}')
        seen.add(review_id)
    return document


def _safe_relative_path(unit_root, relative_path):
    if not isinstance(relative_path, str) or not relative_path:
        return None
    unit_root = os.path.abspath(unit_root)
    target = os.path.abspath(os.path.join(unit_root, relative_path.replace('/', os.sep)))
    try:
        if os.path.commonpath([unit_root, target]) != unit_root:
            raise ValueError('审核文件路径越过单元目录')
    except ValueError as error:
        raise ValueError('审核文件路径无效') from error
    return target


def apply_decisions(unit_root, queue, document):
    """应用已校验决定；拒绝时保留原文件备份，不做不可恢复删除。"""
    validate_decisions(queue, document)
    by_id = {item['review_id']: item for item in queue['items']}
    summary = {'accepted': 0, 'rejected': 0, 'marked': 0, 'missing_file': 0}
    for record in document.get('decisions', []):
        item = by_id[record['review_id']]
        answer = item.get('answer') or {}
        proposed = _safe_relative_path(unit_root, answer.get('proposed_image'))
        accepted = _safe_relative_path(unit_root, answer.get('accepted_image'))
        decision = record['decision']
        if decision == 'accept':
            if proposed and accepted and os.path.isfile(proposed):
                if os.path.normcase(proposed) != os.path.normcase(accepted):
                    os.makedirs(os.path.dirname(accepted), exist_ok=True)
                    shutil.copy2(proposed, accepted)
                summary['accepted'] += 1
            elif accepted and os.path.isfile(accepted):
                summary['accepted'] += 1
            else:
                summary['missing_file'] += 1
        elif decision in {'reject_pair', 'answer_bad'}:
            if accepted and os.path.isfile(accepted):
                backup = accepted + '.审核前备份'
                suffix = 1
                while os.path.exists(backup):
                    suffix += 1
                    backup = accepted + f'.审核前备份{suffix}'
                os.replace(accepted, backup)
            summary['rejected'] += 1
        else:
            summary['marked'] += 1
    return summary


def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def write_json_atomic(path, data):
    temp_path = path + '.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(temp_path, 'w', encoding='utf-8') as file:
        json.dump(json_safe(data), file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)
