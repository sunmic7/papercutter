#!/usr/bin/env python3
"""校验并应用人工或在线智能体生成的答案审核决定。"""

from __future__ import annotations

import argparse
import json
import os
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import review_queue


def main():
    parser = argparse.ArgumentParser(description='校验/应用题目—答案审核决定')
    parser.add_argument('queue', help='review_queue.json')
    parser.add_argument('decisions', nargs='?', help='review_decisions.json')
    parser.add_argument('--template', action='store_true', help='生成空决定模板')
    parser.add_argument('--validate-only', action='store_true', help='只校验，不应用文件变更')
    args = parser.parse_args()

    queue_path = os.path.abspath(args.queue)
    queue = review_queue.load_json(queue_path)
    review_queue.validate_review_queue(queue)
    if args.template:
        output_path = os.path.abspath(
            args.decisions or os.path.join(os.path.dirname(queue_path), 'review_decisions.json')
        )
        document = review_queue.new_decision_document(queue, 'agent', '')
        review_queue.write_json_atomic(output_path, document)
        print(json.dumps({'created': output_path, 'items': len(queue['items'])}, ensure_ascii=False))
        return
    if not args.decisions:
        parser.error('未使用 --template 时必须提供 decisions 文件')

    decision_path = os.path.abspath(args.decisions)
    document = review_queue.load_json(decision_path)
    review_queue.validate_decisions(queue, document)
    if args.validate_only:
        print(json.dumps({
            'valid': True,
            'decision_count': len(document.get('decisions', [])),
            'queue_fingerprint': queue['queue_fingerprint'],
        }, ensure_ascii=False))
        return
    result = review_queue.apply_decisions(
        os.path.dirname(queue_path), queue, document,
    )
    print(json.dumps({'applied': True, **result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
