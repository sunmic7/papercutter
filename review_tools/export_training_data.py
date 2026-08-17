#!/usr/bin/env python3
"""从审核清单/决定导出可训练清单，可选生成便携图片副本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import review_queue


def dataset_split(group_name):
    """按整本教材稳定拆分，杜绝同一本书跨训练/验证/测试集。"""
    digest = hashlib.sha256(group_name.encode('utf-8')).digest()
    bucket = int.from_bytes(digest[:4], 'big') % 100
    if bucket < 80:
        return 'train'
    if bucket < 90:
        return 'validation'
    return 'test'


def _absolute(unit_root, path):
    if not path:
        return None
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(unit_root, path.replace('/', os.sep)))


def collect_training_records(root, include_auto=False):
    records = []
    invalid = []
    for base, dirs, files in os.walk(os.path.abspath(root)):
        dirs[:] = sorted(directory for directory in dirs if directory != '_运行中')
        if 'review_queue.json' not in files:
            continue
        queue_path = os.path.join(base, 'review_queue.json')
        try:
            queue = review_queue.load_json(queue_path)
            review_queue.validate_review_queue(queue)
            decision_path = os.path.join(base, 'review_decisions.json')
            decisions = {}
            if os.path.isfile(decision_path):
                document = review_queue.load_json(decision_path)
                review_queue.validate_decisions(queue, document)
                decisions = {
                    record['review_id']: record for record in document.get('decisions', [])
                }
        except (OSError, ValueError, json.JSONDecodeError) as error:
            invalid.append({'queue': queue_path, 'error': str(error)})
            continue

        for item in queue['items']:
            decision = decisions.get(item['review_id'])
            provenance = 'reviewed'
            if decision is None:
                if not include_auto or item.get('pairing', {}).get('decision_band') != 'auto':
                    continue
                decision = {
                    'review_id': item['review_id'],
                    'decision': 'accept',
                    'notes': '算法高置信自动通过，未经人工复核',
                }
                provenance = 'auto'

            question = item.get('question') or {}
            answer = item.get('answer') or {}
            book_name = item.get('book_name') or os.path.basename(os.path.dirname(base))
            label = decision['decision']
            records.append({
                'schema_version': 1,
                'review_id': item['review_id'],
                'dataset_split': dataset_split(book_name),
                'group_id': book_name,
                'unit_name': item.get('unit_name') or queue.get('unit_name', ''),
                'major': item.get('major', {}),
                'label': label,
                'label_provenance': provenance,
                'pair_valid': label == 'accept',
                'question_valid': label not in {'question_bad'},
                'answer_valid': label != 'answer_bad',
                'question': {
                    'crop_image': _absolute(base, question.get('image')),
                    'source_image': _absolute(base, question.get('source_image')),
                    'predicted_box': question.get('source_box'),
                    'target_box': decision.get('corrected_question_box', question.get('source_box')),
                    'number': question.get('number'),
                    'number_detected': question.get('number_detected'),
                    'source_segments': [
                        {
                            **segment,
                            'image': _absolute(base, segment.get('image')),
                        }
                        for segment in question.get('source_segments', [])
                    ],
                },
                'answer': ({
                    'crop_image': _absolute(base, answer.get('proposed_image')),
                    'source_image': _absolute(base, answer.get('source_image')),
                    'predicted_box': answer.get('source_box'),
                    'target_box': decision.get('corrected_answer_box', answer.get('source_box')),
                    'number': answer.get('number'),
                    'context': answer.get('context', ''),
                } if answer else None),
                'pairing': item.get('pairing', {}),
                'risk_flags': item.get('risk_flags', []),
                'notes': decision.get('notes', ''),
                'source_queue': queue_path,
            })
    return records, invalid


def _portable_copy(source, output_root, category, split, cache):
    if not source or not os.path.isfile(source):
        return None
    source = os.path.abspath(source)
    if source in cache:
        return cache[source]
    extension = os.path.splitext(source)[1].lower() or '.jpg'
    digest = hashlib.sha256(os.path.normcase(source).encode('utf-8')).hexdigest()[:24]
    relative = f'images/{split}/{category}/{digest}{extension}'
    destination = os.path.join(output_root, relative.replace('/', os.sep))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)
    cache[source] = relative
    return relative


def make_portable(records, output_root):
    cache = {}
    for record in records:
        split = record['dataset_split']
        question = record['question']
        question['crop_image'] = _portable_copy(
            question.get('crop_image'), output_root, 'question_crops', split, cache,
        )
        question['source_image'] = _portable_copy(
            question.get('source_image'), output_root, 'question_sources', split, cache,
        )
        for segment in question.get('source_segments', []):
            segment['image'] = _portable_copy(
                segment.get('image'), output_root, 'question_sources', split, cache,
            )
        answer = record.get('answer')
        if answer:
            answer['crop_image'] = _portable_copy(
                answer.get('crop_image'), output_root, 'answer_crops', split, cache,
            )
            answer['source_image'] = _portable_copy(
                answer.get('source_image'), output_root, 'answer_sources', split, cache,
            )
    return records


def export_dataset(root, output, include_auto=False, portable=False):
    records, invalid = collect_training_records(root, include_auto=include_auto)
    output = os.path.abspath(output)
    os.makedirs(output, exist_ok=True)
    if portable:
        make_portable(records, output)
    manifest_path = os.path.join(output, 'training_manifest.jsonl')
    with open(manifest_path + '.tmp', 'w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
    os.replace(manifest_path + '.tmp', manifest_path)
    summary = {
        'schema_version': 1,
        'record_count': len(records),
        'reviewed_count': sum(record['label_provenance'] == 'reviewed' for record in records),
        'auto_count': sum(record['label_provenance'] == 'auto' for record in records),
        'splits': {
            split: sum(record['dataset_split'] == split for record in records)
            for split in ('train', 'validation', 'test')
        },
        'labels': {
            label: sum(record['label'] == label for record in records)
            for label in sorted({record['label'] for record in records})
        },
        'invalid_queues': invalid,
        'portable': bool(portable),
        'manifest': manifest_path,
    }
    review_queue.write_json_atomic(os.path.join(output, 'dataset_summary.json'), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description='从审核结果导出训练数据')
    parser.add_argument('root', help='包含切题结果的教材或年份根目录')
    parser.add_argument('--output', required=True, help='训练清单输出目录')
    parser.add_argument('--include-auto', action='store_true', help='同时纳入未经人工复核的自动通过项')
    parser.add_argument('--portable', action='store_true', help='复制裁图和来源原页，生成可迁移数据包')
    args = parser.parse_args()
    summary = export_dataset(
        args.root, args.output, include_auto=args.include_auto, portable=args.portable,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
