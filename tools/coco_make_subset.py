#!/usr/bin/env python3
"""
Create a smaller COCO annotations JSON with only 'person' annotations for a
subset of images (default: 500 images that contain at least one person).

Defaults are set for this repo:
  input:  dataset/coco/annotations/instances_val2017.json
  output: dataset/coco/annotations/simple_instances_val2017.json

Usage examples:
  python tools/coco_make_subset.py
  python tools/coco_make_subset.py --count 500 --category person \
         --input dataset/coco/annotations/instances_val2017.json \
         --output dataset/coco/annotations/simple_instances_val2017.json
  python tools/coco_make_subset.py --count 200 --seed 42
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Wrote subset annotations to: {path}")


def find_category_id(categories: List[Dict[str, Any]], category_name: str) -> int:
    for c in categories:
        if c.get('name') == category_name:
            return int(c['id'])
    raise ValueError(f"Category '{category_name}' not found in categories")


def main():
    parser = argparse.ArgumentParser(description="Make a COCO subset JSON")
    parser.add_argument('--input', default='dataset/coco/annotations/instances_val2017.json', help='Path to input COCO annotations JSON')
    parser.add_argument('--output', default='dataset/coco/annotations/simple_instances_val2017.json', help='Path to write subset JSON')
    parser.add_argument('--category', default='person', help='Category name to filter to')
    parser.add_argument('--count', type=int, default=500, help='Number of images that contain the category to include')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (if set, sample randomly from eligible images)')
    parser.add_argument('--random', action='store_true', help='Enable random sampling (requires seed)')
    args = parser.parse_args()

    data = load_json(args.input)
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])

    cat_id = find_category_id(categories, args.category)

    # Map image_id -> list of person annotations
    ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        if int(ann.get('category_id', -1)) != cat_id:
            continue
        img_id = int(ann['image_id'])
        ann_by_image.setdefault(img_id, []).append(ann)

    # Eligible images: those that have >= 1 person annotation
    eligible_image_ids = sorted(ann_by_image.keys())
    if len(eligible_image_ids) == 0:
        raise RuntimeError(f"No images contain category '{args.category}'")

    if args.random:
        if args.seed is None:
            raise ValueError("--random requires --seed to be set for reproducibility")
        random.seed(args.seed)
        sampled_image_ids = random.sample(eligible_image_ids, k=min(args.count, len(eligible_image_ids)))
    else:
        sampled_image_ids = eligible_image_ids[: min(args.count, len(eligible_image_ids))]

    sampled_image_id_set = set(sampled_image_ids)

    # Filter images
    id_to_image = {int(img['id']): img for img in images}
    sampled_images = [id_to_image[iid] for iid in sampled_image_ids if iid in id_to_image]

    # Filter annotations to only 'person' for selected images
    sampled_annotations: List[Dict[str, Any]] = []
    for iid in sampled_image_ids:
        anns = ann_by_image.get(iid, [])
        sampled_annotations.extend(anns)

    # Keep only the 'person' category entry to simplify downstream
    person_category_entry = next((c for c in categories if int(c['id']) == cat_id), None)
    if not person_category_entry:
        raise RuntimeError("Person category entry missing unexpectedly")

    subset = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': [person_category_entry],
    }

    print(f"Original images: {len(images)} | annotations: {len(annotations)}")
    print(f"Eligible 'person' images: {len(eligible_image_ids)}")
    print(f"Subset images: {len(sampled_images)} | person annotations: {len(sampled_annotations)}")

    save_json(subset, args.output)


if __name__ == '__main__':
    main()
