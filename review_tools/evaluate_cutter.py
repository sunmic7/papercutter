#!/usr/bin/env python3
"""切题诊断与真值评估。

兼容原有 ``profile=图片`` 诊断方式，也可读取原页题框标注，计算多裁、
少裁、合并题和拆分题。标注以原始页面为单位，避免只看成品小图时丢失上下文。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import cut_questions_v4 as cutter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def evaluate_page(path: str, profile: str) -> dict:
    image = cutter.cv2_imread(path)
    if image is None:
        return {'path': path, 'error': 'read_failed'}
    items = cutter.ocr_image(path)
    page_info = cutter.classify_page(image, items, profile)
    blocks = cutter.split_page_into_majors(image, items, profile=profile)
    cuts = []
    predictions = []
    for block in blocks:
        local_items = cutter.ocr_items_for_block(items, block)
        minors = cutter.split_major_into_minors(
            block['crop_img'], ocr_items=local_items, profile=profile,
        )
        cutter.attach_question_source_metadata(
            minors,
            block,
            path,
            source_size=(image.shape[1], image.shape[0]),
            deskew_angle=0.0,
        )
        for minor in minors:
            normalized, normalize_info = cutter.normalize_question_crop(minor['crop_img'])
            issues = cutter.crop_quality_issues(normalized)
            prediction = {
                'major_idx': block['idx'],
                'major_title': block['title'],
                'column': block.get('_col', 'single'),
                'minor_num': minor['num'],
                'number_detected': minor.get('number_detected', True),
                'boundary_source': minor.get('_boundary_source', 'unknown'),
                'bbox': list(minor.get('_source_box', [])),
                'shape': list(normalized.shape[:2]),
                'quality_issues': issues,
                'normalize_warning': normalize_info.get('quality_warning', ''),
            }
            cuts.append(prediction)
            predictions.append({
                key: prediction[key]
                for key in (
                    'major_idx', 'major_title', 'column', 'minor_num',
                    'number_detected', 'boundary_source', 'bbox',
                )
            })
    return {
        'path': path,
        'profile': profile,
        'page': page_info,
        'ocr_items': len(items),
        'major_blocks': len(blocks),
        'question_cuts': len(cuts),
        'bad_cuts': sum(1 for cut in cuts if cut['quality_issues']),
        'cuts': cuts,
        'predictions': predictions,
    }


def _valid_box(box):
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(value, (int, float)) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def bbox_iou(left, right):
    """标准矩形 IoU，用于衡量题框上下边缘和宽度是否贴合。"""
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _association_score(predicted, truth):
    """判断两个题框是否表示同一道题；容许全栏框与紧内容框宽度不同。"""
    overlap_x = max(0.0, min(predicted[2], truth[2]) - max(predicted[0], truth[0]))
    overlap_y = max(0.0, min(predicted[3], truth[3]) - max(predicted[1], truth[1]))
    min_width = min(predicted[2] - predicted[0], truth[2] - truth[0])
    min_height = min(predicted[3] - predicted[1], truth[3] - truth[1])
    if min_width <= 0 or min_height <= 0:
        return 0.0
    return (overlap_x / min_width) * (overlap_y / min_height)


def compare_page_boxes(
    predictions, ground_truth, association_threshold=0.50,
    minimum_content_coverage=0.95, maximum_excess_ratio=0.12,
):
    """比较预测框和真值框，分别统计数量错误与几何裁剪错误。

    ``extra/missed`` 是多切或漏切的数量问题；``under/over_crop`` 才表示
    单个框少包含了题目内容或包含了过多题外区域。保留旧字段供已有脚本兼容。
    """
    predicted_boxes = [item['bbox'] if isinstance(item, dict) else item for item in predictions]
    truth_boxes = [item['bbox'] if isinstance(item, dict) else item for item in ground_truth]
    if not all(_valid_box(box) for box in predicted_boxes + truth_boxes):
        raise ValueError('题框必须为 [x0, y0, x1, y1] 且宽高为正数')

    associations = []
    predicted_links = [[] for _ in predicted_boxes]
    truth_links = [[] for _ in truth_boxes]
    for pred_index, predicted in enumerate(predicted_boxes):
        for truth_index, truth in enumerate(truth_boxes):
            score = _association_score(predicted, truth)
            if score >= association_threshold:
                iou = bbox_iou(predicted, truth)
                associations.append((score, iou, pred_index, truth_index))
                predicted_links[pred_index].append(truth_index)
                truth_links[truth_index].append(pred_index)

    matched_predicted = set()
    matched_truth = set()
    matches = []
    for score, iou, pred_index, truth_index in sorted(associations, reverse=True):
        if pred_index in matched_predicted or truth_index in matched_truth:
            continue
        matched_predicted.add(pred_index)
        matched_truth.add(truth_index)
        predicted = predicted_boxes[pred_index]
        truth = truth_boxes[truth_index]
        intersection = (
            max(0.0, min(predicted[2], truth[2]) - max(predicted[0], truth[0]))
            * max(0.0, min(predicted[3], truth[3]) - max(predicted[1], truth[1]))
        )
        predicted_area = (predicted[2] - predicted[0]) * (predicted[3] - predicted[1])
        truth_area = (truth[2] - truth[0]) * (truth[3] - truth[1])
        content_coverage = intersection / truth_area if truth_area else 0.0
        excess_ratio = max(0.0, predicted_area - intersection) / truth_area if truth_area else 0.0
        matches.append({
            'prediction_index': pred_index,
            'truth_index': truth_index,
            'association': round(score, 6),
            'iou': round(iou, 6),
            'content_coverage': round(content_coverage, 6),
            'excess_ratio': round(excess_ratio, 6),
            'under_crop': content_coverage < minimum_content_coverage,
            'over_crop': excess_ratio > maximum_excess_ratio,
            'top_error_px': round(predicted_boxes[pred_index][1] - truth_boxes[truth_index][1], 3),
            'bottom_error_px': round(predicted_boxes[pred_index][3] - truth_boxes[truth_index][3], 3),
        })

    predicted_count, truth_count, matched_count = (
        len(predicted_boxes), len(truth_boxes), len(matches)
    )
    ious = [match['iou'] for match in matches]
    extra_count = predicted_count - matched_count
    missed_count = truth_count - matched_count
    return {
        'predicted_count': predicted_count,
        'truth_count': truth_count,
        'matched_count': matched_count,
        'precision': matched_count / predicted_count if predicted_count else float(truth_count == 0),
        'recall': matched_count / truth_count if truth_count else float(predicted_count == 0),
        'extra_prediction_count': extra_count,
        'missed_truth_count': missed_count,
        # 兼容旧版：旧字段表达的是数量，不是边界裁多/裁少。
        'over_cut_count': extra_count,
        'under_cut_count': missed_count,
        'under_crop_count': sum(match['under_crop'] for match in matches),
        'over_crop_count': sum(match['over_crop'] for match in matches),
        'merged_prediction_count': sum(len(links) > 1 for links in predicted_links),
        'split_truth_count': sum(len(links) > 1 for links in truth_links),
        'exact_count_page': predicted_count == truth_count,
        'mean_iou': sum(ious) / len(ious) if ious else 0.0,
        'matches': matches,
        'unmatched_prediction_indices': sorted(set(range(predicted_count)) - matched_predicted),
        'unmatched_truth_indices': sorted(set(range(truth_count)) - matched_truth),
    }


def create_annotation_manifest(image_dir, profile='general'):
    """为一个原页目录建立空标注清单；空条目标记为 unlabeled，不参与评估。"""
    root = os.path.abspath(image_dir)
    pages = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(directory for directory in dirs if not directory.startswith('_'))
        for name in sorted(files, key=cutter.image_sort_key):
            if os.path.splitext(name)[1].lower() not in cutter.IMAGE_EXTS:
                continue
            path = os.path.join(base, name)
            pages.append({
                'image': os.path.relpath(path, root).replace('\\', '/'),
                'profile': profile,
                'status': 'unlabeled',
                'questions': [],
            })
    return {
        'schema_version': 1,
        'image_root': root,
        'coordinate_format': 'xyxy_pixels',
        'pages': pages,
    }


def evaluate_annotation_manifest(manifest_path, association_threshold=0.50):
    with open(manifest_path, 'r', encoding='utf-8') as file:
        manifest = json.load(file)
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('pages'), list):
        raise ValueError('标注文件格式无效：需要 schema_version=1 和 pages 数组')
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    image_root = manifest.get('image_root') or manifest_dir
    if not os.path.isabs(image_root):
        image_root = os.path.join(manifest_dir, image_root)

    page_results = []
    for page in manifest['pages']:
        if page.get('status') != 'labeled':
            continue
        image_path = page.get('image', '')
        if not os.path.isabs(image_path):
            image_path = os.path.join(image_root, image_path)
        profile = page.get('profile', 'general')
        diagnostic = evaluate_page(os.path.abspath(image_path), profile)
        if diagnostic.get('error'):
            page_results.append({'image': page.get('image', ''), 'error': diagnostic['error']})
            continue
        comparison = compare_page_boxes(
            diagnostic['predictions'],
            page.get('questions', []),
            association_threshold=association_threshold,
        )
        page_results.append({
            'image': page.get('image', ''),
            'profile': profile,
            **comparison,
            'predictions': diagnostic['predictions'],
        })

    valid_pages = [page for page in page_results if 'error' not in page]
    totals = {
        key: sum(page[key] for page in valid_pages)
        for key in (
            'predicted_count', 'truth_count', 'matched_count', 'over_cut_count',
            'under_cut_count', 'extra_prediction_count', 'missed_truth_count',
            'under_crop_count', 'over_crop_count',
            'merged_prediction_count', 'split_truth_count',
        )
    }
    matched = totals.get('matched_count', 0)
    totals.update({
        'page_count': len(valid_pages),
        'error_page_count': len(page_results) - len(valid_pages),
        'exact_count_page_count': sum(page['exact_count_page'] for page in valid_pages),
        'precision': matched / totals['predicted_count'] if totals['predicted_count'] else float(totals['truth_count'] == 0),
        'recall': matched / totals['truth_count'] if totals['truth_count'] else float(totals['predicted_count'] == 0),
        'mean_iou': (
            sum(page['mean_iou'] * page['matched_count'] for page in valid_pages) / matched
            if matched else 0.0
        ),
    })
    return {
        'schema_version': 1,
        'manifest': os.path.abspath(manifest_path),
        'association_threshold': association_threshold,
        'summary': totals,
        'pages': page_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'samples', nargs='*',
        help='格式：profile=图片路径，例如 exam=runtime/eval_exam.jpg',
    )
    parser.add_argument('--annotations', help='评估带原页题框的 JSON 真值标注')
    parser.add_argument('--init-annotations', metavar='IMAGE_DIR', help='为原页目录生成空标注清单')
    parser.add_argument('--output', help='把标注模板或评估结果写入此 JSON 文件')
    parser.add_argument('--profile', default='general', help='初始化标注时使用的默认教材场景')
    parser.add_argument('--association-threshold', type=float, default=0.50)
    args = parser.parse_args()
    if args.init_annotations:
        manifest = create_annotation_manifest(args.init_annotations, profile=args.profile)
        output_path = args.output or os.path.abspath('cut_annotations.json')
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(cutter.review_queue.json_safe(manifest), file, ensure_ascii=False, indent=2)
        print(f"已生成 {len(manifest['pages'])} 页标注清单：{output_path}")
        return
    if args.annotations:
        result = evaluate_annotation_manifest(
            args.annotations,
            association_threshold=args.association_threshold,
        )
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as file:
                json.dump(cutter.review_queue.json_safe(result), file, ensure_ascii=False, indent=2)
        summary = result['summary']
        print(
            f"[真值评估] pages={summary['page_count']} "
            f"precision={summary['precision']:.3f} recall={summary['recall']:.3f} "
            f"多切={summary['extra_prediction_count']} 漏题={summary['missed_truth_count']} "
            f"裁少={summary['under_crop_count']} 裁多={summary['over_crop_count']} "
            f"合并题={summary['merged_prediction_count']} 拆分题={summary['split_truth_count']}"
        )
        print(json.dumps(cutter.review_queue.json_safe(result), ensure_ascii=False, indent=2))
        return
    if not args.samples:
        parser.error('请提供 profile=图片，或使用 --annotations / --init-annotations')
    results = []
    for spec in args.samples:
        if '=' not in spec:
            raise SystemExit(f'样本参数缺少场景：{spec}')
        profile, path = spec.split('=', 1)
        result = evaluate_page(os.path.abspath(path), profile)
        results.append(result)
        print(
            f"[评估] {os.path.basename(path)} profile={profile} "
            f"kind={result.get('page', {}).get('kind')} "
            f"cuts={result.get('question_cuts')} bad={result.get('bad_cuts')}",
            flush=True,
        )
    print(json.dumps(cutter.review_queue.json_safe(results), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
