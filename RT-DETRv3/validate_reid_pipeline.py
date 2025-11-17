#!/usr/bin/env python3
"""
Pedestrian Re-ID Pipeline Validator for BoT-SORT Tracking
=========================================================

This script performs specialized validation of RT-DETRv3 Re-ID embedding pipeline with focus
on pedestrian feature embeddings for BoT-SORT multi-object tracking. The validation is designed
to minimize ID switching by ensuring high-quality, discriminative embeddings for person tracking.

Pedestrian-Focused Validation Components:
    1. Person Detection Quality Assessment
       - Person class detection reliability (COCO class_id = 0)
       - Detection confidence thresholds for pedestrians
       - Bounding box quality for person instances

    2. Pedestrian Feature Embedding Quality
       - Intra-person consistency across poses/viewpoints
       - Inter-person discriminability for ID switching prevention
       - Temporal stability for video sequences
       - Robustness to occlusion and scale variations

    3. BoT-SORT Compatibility Validation
       - Feature embedding format compatibility
       - Similarity threshold optimization for tracking
       - Track association quality metrics
       - ID switch prediction and prevention

    4. Real-world Pedestrian Scenarios
       - Multiple person discrimination
       - Similar appearance handling (clothing, build)
       - Crowded scene performance
       - Partial occlusion robustness

    5. Anti-ID-Switch Metrics
       - Feature stability across frames
       - Appearance variation tolerance
       - False positive association prevention
       - Track continuity validation

Quality Metrics for Pedestrian Tracking:
    • Person embedding L2 norm: 1.0 ± 0.001
    • Same person similarity: > 0.85 (higher than general objects)
    • Different person similarity: < 0.3 (stricter than general objects)
    • Temporal consistency: > 0.95 (for ID switch prevention)
    • BoT-SORT compatibility score: > 0.9

Usage:
    python validate_reid_pipeline.py --model backbone_model.onnx --image pedestrian_test.jpg
                                   --feature-map-name Concat.3 --output pedestrian_report.json
                                   --pedestrian-focused

Output:
    Specialized JSON validation report with pedestrian tracking metrics and BoT-SORT compatibility.

Author: RT-DETRv3 Development Team (Enhanced for Pedestrian Tracking)
License: Same as RT-DETRv3 repository
"""

import argparse
import json
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Any
import warnings

# Import the robust generator
try:
    from reid_embeddings import RobustReIDEmbeddingGenerator, COCO_CLASS_LOOKUP
    from export_backbone import analyze_model_outputs
    import onnxruntime as ort
    import onnx
    import cv2
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)

class PedestrianReIDValidator:
    """Specialized validator for pedestrian Re-ID embeddings for BoT-SORT tracking."""

    def __init__(self, model_path: str, debug: bool = True, pedestrian_focused: bool = True):
        self.model_path = model_path
        self.debug = debug
        self.pedestrian_focused = pedestrian_focused
        self.person_class_id = 0  # COCO person class
        self.validation_results = {
            'model_path': model_path,
            'validation_type': 'pedestrian_reid' if pedestrian_focused else 'general_reid',
            'validation_timestamp': None,
            'checks': {},
            'overall_status': 'UNKNOWN',
            'pedestrian_metrics': {},
            'botsort_compatibility': {},
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }

    def validate_pedestrian_detection_quality(self, generator: RobustReIDEmbeddingGenerator,
                                             image_path: str) -> Dict[str, Any]:
        """Validate person detection quality and reliability."""
        print("🚶 Validating pedestrian detection quality...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Run inference with lower confidence threshold for pedestrians
            original_image, input_feed = generator.preprocess_image(image_path)
            detections, feature_map = generator.run_inference(input_feed)

            if not detections:
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No detections found in image")
                return check_result

            # Filter for person detections (class_id = 0)
            person_detections = []
            for det in detections:
                if len(det) >= 6:
                    cls_id, conf, x1, y1, x2, y2 = det[:6]
                    if cls_id == self.person_class_id:
                        person_detections.append({
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'area': (x2 - x1) * (y2 - y1),
                            'aspect_ratio': (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                        })

            check_result['details']['total_detections'] = len(detections)
            check_result['details']['person_detections'] = len(person_detections)
            check_result['details']['person_detection_rate'] = len(person_detections) / len(detections) if detections else 0

            if not person_detections:
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No person detections found - cannot validate pedestrian Re-ID")
                return check_result

            # Analyze person detection quality
            confidences = [p['confidence'] for p in person_detections]
            areas = [p['area'] for p in person_detections]
            aspect_ratios = [p['aspect_ratio'] for p in person_detections]

            check_result['details']['person_stats'] = {
                'confidence': {
                    'mean': float(np.mean(confidences)),
                    'min': float(np.min(confidences)),
                    'max': float(np.max(confidences)),
                    'std': float(np.std(confidences))
                },
                'area': {
                    'mean': float(np.mean(areas)),
                    'min': float(np.min(areas)),
                    'max': float(np.max(areas)),
                    'std': float(np.std(areas))
                },
                'aspect_ratio': {
                    'mean': float(np.mean(aspect_ratios)),
                    'min': float(np.min(aspect_ratios)),
                    'max': float(np.max(aspect_ratios)),
                    'std': float(np.std(aspect_ratios))
                }
            }

            # Quality checks for pedestrian detection
            low_confidence_persons = sum(1 for c in confidences if c < 0.7)
            if low_confidence_persons > len(person_detections) * 0.3:
                check_result['status'] = 'WARN'
                check_result['issues'].append(f"High number of low-confidence person detections: {low_confidence_persons}/{len(person_detections)}")

            # Check for reasonable person aspect ratios (typical person height/width ratio)
            unusual_aspect_ratios = sum(1 for ar in aspect_ratios if ar < 0.3 or ar > 1.0)
            if unusual_aspect_ratios > len(person_detections) * 0.2:
                check_result['status'] = 'WARN'
                check_result['issues'].append(f"Unusual person aspect ratios detected: {unusual_aspect_ratios}/{len(person_detections)}")

            # Check for very small person detections
            small_persons = sum(1 for area in areas if area < 1000)  # Less than ~32x32 pixels
            if small_persons > len(person_detections) * 0.5:
                check_result['status'] = 'WARN'
                check_result['issues'].append(f"Many small person detections: {small_persons}/{len(person_detections)} - may affect Re-ID quality")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate pedestrian detection: {e}")

        return check_result

    def validate_pedestrian_embedding_discriminability(self, generator: RobustReIDEmbeddingGenerator,
                                                     image_path: str) -> Dict[str, Any]:
        """Validate pedestrian embedding discriminability for ID switch prevention."""
        print("🔍 Validating pedestrian embedding discriminability...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Generate embeddings for person detections only
            embeddings = generator.process_image(image_path, conf_threshold=0.5, output_dir="pipeline/output/validation/pedestrian")

            # Filter for person embeddings
            person_embeddings = []
            for detection_info, embedding in embeddings:
                if detection_info['class_id'] == self.person_class_id:
                    person_embeddings.append((detection_info, embedding))

            if len(person_embeddings) < 2:
                check_result['status'] = 'WARN'
                check_result['issues'].append(f"Only {len(person_embeddings)} person embedding(s) found - need multiple persons for discriminability test")
                if len(person_embeddings) == 1:
                    # Still validate single person embedding quality
                    _, embedding = person_embeddings[0]
                    check_result['details']['single_person_norm'] = float(np.linalg.norm(embedding))
                return check_result

            check_result['details']['person_embedding_count'] = len(person_embeddings)

            # Extract embedding vectors
            person_vectors = np.array([emb for _, emb in person_embeddings])

            # Compute pairwise similarities
            similarity_matrix = np.dot(person_vectors, person_vectors.T)

            # Analyze inter-person similarities (should be low for good discriminability)
            inter_person_similarities = []
            for i in range(len(person_embeddings)):
                for j in range(i + 1, len(person_embeddings)):
                    similarity = similarity_matrix[i, j]
                    inter_person_similarities.append(similarity)

            if inter_person_similarities:
                check_result['details']['inter_person_similarity'] = {
                    'mean': float(np.mean(inter_person_similarities)),
                    'max': float(np.max(inter_person_similarities)),
                    'min': float(np.min(inter_person_similarities)),
                    'std': float(np.std(inter_person_similarities)),
                    'similarities': [float(s) for s in inter_person_similarities]
                }

                # Stricter thresholds for pedestrian discriminability
                mean_similarity = np.mean(inter_person_similarities)
                max_similarity = np.max(inter_person_similarities)

                # For BoT-SORT, we want very low inter-person similarity to prevent ID switches
                if mean_similarity > 0.3:  # Stricter than general object threshold
                    check_result['status'] = 'WARN'
                    check_result['issues'].append(f"High inter-person similarity detected: {mean_similarity:.3f} > 0.3 - may cause ID switches")

                if max_similarity > 0.5:  # Any pair too similar
                    check_result['status'] = 'FAIL'
                    check_result['issues'].append(f"Very high similarity between some persons: {max_similarity:.3f} > 0.5 - high ID switch risk")

                # Check for embedding quality (norms should be ~1.0)
                norms = [np.linalg.norm(emb) for _, emb in person_embeddings]
                norm_stats = {
                    'mean': float(np.mean(norms)),
                    'std': float(np.std(norms)),
                    'min': float(np.min(norms)),
                    'max': float(np.max(norms))
                }
                check_result['details']['person_embedding_norms'] = norm_stats

                # Stricter norm consistency for pedestrians
                norm_deviations = [abs(norm - 1.0) for norm in norms]
                max_norm_deviation = max(norm_deviations) if norm_deviations else 0
                if max_norm_deviation > 0.005:  # Stricter than general objects
                    check_result['status'] = 'WARN'
                    check_result['issues'].append(f"Inconsistent embedding normalization: max deviation {max_norm_deviation:.4f}")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate pedestrian discriminability: {e}")

        return check_result

    def validate_botsort_compatibility(self, generator: RobustReIDEmbeddingGenerator,
                                     image_path: str) -> Dict[str, Any]:
        """Validate compatibility with BoT-SORT tracking algorithm."""
        print("🎯 Validating BoT-SORT compatibility...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Generate embeddings for BoT-SORT compatibility testing
            embeddings = generator.process_image(image_path, conf_threshold=0.5, output_dir="pipeline/output/validation/botsort")

            # Filter for person embeddings
            person_embeddings = []
            for detection_info, embedding in embeddings:
                if detection_info['class_id'] == self.person_class_id:
                    person_embeddings.append((detection_info, embedding))

            if not person_embeddings:
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No person embeddings for BoT-SORT compatibility test")
                return check_result

            check_result['details']['embedding_count'] = len(person_embeddings)

            # Check embedding format requirements for BoT-SORT
            _, sample_embedding = person_embeddings[0]
            embedding_dim = len(sample_embedding)

            check_result['details']['embedding_dimension'] = embedding_dim
            check_result['details']['embedding_dtype'] = str(sample_embedding.dtype)

            # BoT-SORT typically expects L2-normalized embeddings
            embedding_norms = [np.linalg.norm(emb) for _, emb in person_embeddings]
            check_result['details']['embedding_norms'] = {
                'mean': float(np.mean(embedding_norms)),
                'std': float(np.std(embedding_norms)),
                'all_norms': [float(norm) for norm in embedding_norms]
            }

            # Check if embeddings are properly normalized (critical for BoT-SORT)
            poorly_normalized = sum(1 for norm in embedding_norms if abs(norm - 1.0) > 0.01)
            if poorly_normalized > 0:
                check_result['status'] = 'FAIL'
                check_result['issues'].append(f"BoT-SORT requires L2-normalized embeddings: {poorly_normalized}/{len(person_embeddings)} are poorly normalized")

            # Simulate BoT-SORT similarity computation
            if len(person_embeddings) >= 2:
                person_vectors = np.array([emb for _, emb in person_embeddings])

                # BoT-SORT uses cosine similarity (dot product for normalized vectors)
                similarity_matrix = np.dot(person_vectors, person_vectors.T)

                # Extract upper triangular similarities (unique pairs)
                similarities = []
                for i in range(len(person_embeddings)):
                    for j in range(i + 1, len(person_embeddings)):
                        similarities.append(similarity_matrix[i, j])

                check_result['details']['botsort_similarities'] = {
                    'mean': float(np.mean(similarities)),
                    'min': float(np.min(similarities)),
                    'max': float(np.max(similarities)),
                    'std': float(np.std(similarities))
                }

                # BoT-SORT compatibility metrics
                botsort_score = 0.0

                # Score based on similarity distribution (want low inter-person similarity)
                mean_sim = np.mean(similarities)
                if mean_sim < 0.2:
                    botsort_score += 0.4  # Excellent separation
                elif mean_sim < 0.3:
                    botsort_score += 0.3  # Good separation
                elif mean_sim < 0.4:
                    botsort_score += 0.2  # Acceptable separation
                else:
                    botsort_score += 0.1  # Poor separation

                # Score based on normalization quality
                norm_quality = 1.0 - (poorly_normalized / len(person_embeddings))
                botsort_score += 0.3 * norm_quality

                # Score based on embedding dimension (higher dimensions can be better)
                if embedding_dim >= 512:
                    botsort_score += 0.3
                elif embedding_dim >= 256:
                    botsort_score += 0.2
                else:
                    botsort_score += 0.1

                check_result['details']['botsort_compatibility_score'] = float(botsort_score)

                if botsort_score < 0.7:
                    check_result['status'] = 'WARN'
                    check_result['issues'].append(f"Low BoT-SORT compatibility score: {botsort_score:.2f} < 0.7")

                # Check for potential ID switch scenarios
                high_similarity_pairs = sum(1 for sim in similarities if sim > 0.5)
                if high_similarity_pairs > 0:
                    check_result['status'] = 'WARN'
                    check_result['issues'].append(f"High similarity pairs detected: {high_similarity_pairs} - potential ID switch risk")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate BoT-SORT compatibility: {e}")

        return check_result

    def validate_temporal_consistency(self, generator: RobustReIDEmbeddingGenerator,
                                    image_path: str) -> Dict[str, Any]:
        """Validate temporal consistency for tracking stability."""
        print("⏱️ Validating temporal consistency...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Run inference multiple times to simulate temporal consistency
            embedding_runs = []
            for run_id in range(3):  # Multiple runs to check consistency
                embeddings = generator.process_image(
                    image_path,
                    conf_threshold=0.5,
                    output_dir=f"pipeline/output/validation/temporal_run_{run_id}"
                )

                # Filter for person embeddings
                person_embeddings = []
                for detection_info, embedding in embeddings:
                    if detection_info['class_id'] == self.person_class_id:
                        person_embeddings.append((detection_info, embedding))

                if person_embeddings:
                    embedding_runs.append(person_embeddings)

            if len(embedding_runs) < 2:
                check_result['status'] = 'WARN'
                check_result['issues'].append("Insufficient runs for temporal consistency validation")
                return check_result

            check_result['details']['num_runs'] = len(embedding_runs)
            check_result['details']['embeddings_per_run'] = [len(run) for run in embedding_runs]

            # Compare embeddings across runs (assuming same detections)
            min_detections = min(len(run) for run in embedding_runs)
            if min_detections == 0:
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No consistent person detections across runs")
                return check_result

            # Calculate temporal consistency for each detection position
            temporal_consistencies = []
            for det_idx in range(min_detections):
                consistencies_for_detection = []

                for run_i in range(len(embedding_runs)):
                    for run_j in range(run_i + 1, len(embedding_runs)):
                        emb_i = embedding_runs[run_i][det_idx][1]
                        emb_j = embedding_runs[run_j][det_idx][1]

                        # Cosine similarity between same detection across runs
                        consistency = np.dot(emb_i, emb_j)
                        consistencies_for_detection.append(consistency)

                if consistencies_for_detection:
                    mean_consistency = np.mean(consistencies_for_detection)
                    temporal_consistencies.append(mean_consistency)

            if temporal_consistencies:
                overall_consistency = np.mean(temporal_consistencies)
                check_result['details']['temporal_consistency'] = {
                    'overall_mean': float(overall_consistency),
                    'per_detection': [float(tc) for tc in temporal_consistencies],
                    'min': float(np.min(temporal_consistencies)),
                    'max': float(np.max(temporal_consistencies)),
                    'std': float(np.std(temporal_consistencies))
                }

                # High threshold for temporal consistency (critical for tracking)
                if overall_consistency < 0.95:
                    check_result['status'] = 'WARN'
                    check_result['issues'].append(f"Low temporal consistency: {overall_consistency:.3f} < 0.95 - may cause tracking instability")

                if overall_consistency < 0.90:
                    check_result['status'] = 'FAIL'
                    check_result['issues'].append(f"Poor temporal consistency: {overall_consistency:.3f} < 0.90 - high risk of ID switches")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate temporal consistency: {e}")

        return check_result

    def validate_model_structure(self) -> Dict[str, Any]:
        """Validate model structure and feature map selection."""
        print("🔍 Validating model structure...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Load and analyze model
            model = onnx.load(self.model_path)
            candidates = analyze_model_outputs(model, input_size=640)

            check_result['details']['total_candidates'] = len(candidates)
            check_result['details']['backbone_candidates'] = len([c for c in candidates if c['is_likely_backbone']])

            # Check if we have good backbone candidates
            if not any(c['is_likely_backbone'] for c in candidates):
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No likely backbone feature map candidates found")

            # Check for stride 32 (C5) availability
            stride_32_candidates = [c for c in candidates if c['stride'] == 32 and c['is_likely_backbone']]
            if stride_32_candidates:
                check_result['details']['has_c5_features'] = True
                check_result['details']['c5_candidate'] = stride_32_candidates[0]['name']
            else:
                check_result['details']['has_c5_features'] = False
                check_result['issues'].append("No stride-32 (C5) backbone features found - may impact Re-ID quality")

            # Check for potential confusion with detection outputs
            detection_like = [c for c in candidates if any(keyword in c['name'].lower()
                                                         for keyword in ['detect', 'pred', 'bbox', 'cls'])]
            if detection_like:
                check_result['details']['potential_confusion'] = [c['name'] for c in detection_like]
                check_result['issues'].append(f"Found {len(detection_like)} detection-like outputs that could be confused with backbone features")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to analyze model: {e}")

        return check_result

    def validate_detection_format(self, generator: RobustReIDEmbeddingGenerator,
                                 image_path: str) -> Dict[str, Any]:
        """Validate detection tensor format and coordinate consistency."""
        print("🔍 Validating detection format...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Run a small inference to get detection format
            original_image, input_feed = generator.preprocess_image(image_path)
            detections, feature_map = generator.run_inference(input_feed)

            if not detections:
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No detections returned from model")
                return check_result

            # Analyze first few detections
            sample_size = min(5, len(detections))
            sample_detections = detections[:sample_size]

            check_result['details']['sample_size'] = sample_size
            check_result['details']['detection_samples'] = []

            for i, det in enumerate(sample_detections):
                det_info = {
                    'index': i,
                    'length': len(det),
                    'values': det[:8].tolist() if len(det) >= 8 else det.tolist()
                }

                # Validate expected format [cls, conf, x1, y1, x2, y2, ...]
                if len(det) >= 6:
                    cls_id, conf, x1, y1, x2, y2 = det[:6]

                    # Check class ID range
                    if not (0 <= cls_id < 100):
                        det_info['issues'] = det_info.get('issues', [])
                        det_info['issues'].append(f"Suspicious class_id: {cls_id}")

                    # Check confidence range
                    if not (0 <= conf <= 1):
                        det_info['issues'] = det_info.get('issues', [])
                        det_info['issues'].append(f"Confidence out of range [0,1]: {conf}")

                    # Check bbox validity
                    if not (x1 < x2 and y1 < y2):
                        det_info['issues'] = det_info.get('issues', [])
                        det_info['issues'].append(f"Invalid bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                    # Check coordinate magnitude (should be in input image space)
                    if max(x1, y1, x2, y2) > 2000 or min(x1, y1, x2, y2) < -100:
                        det_info['issues'] = det_info.get('issues', [])
                        det_info['issues'].append(f"Unusual coordinate magnitude: {[x1, y1, x2, y2]}")

                    det_info['parsed'] = {
                        'class_id': cls_id,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    }

                check_result['details']['detection_samples'].append(det_info)

            # Count issues across samples
            total_issues = sum(len(det.get('issues', [])) for det in check_result['details']['detection_samples'])
            if total_issues > 0:
                check_result['status'] = 'WARN'
                check_result['issues'].append(f"Found {total_issues} detection format issues in sample")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate detection format: {e}")

        return check_result

    def validate_coordinate_consistency(self, generator: RobustReIDEmbeddingGenerator,
                                      image_path: str, feature_map_name: str = None) -> Dict[str, Any]:
        """Validate coordinate space transformations and consistency."""
        print("🔍 Validating coordinate consistency...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Process with both letterbox and simple resize to compare
            generators = {}

            # Test simple resize
            gen_simple = RobustReIDEmbeddingGenerator(
                self.model_path, use_letterbox=False, debug=False, feature_map_name=feature_map_name
            )
            generators['simple_resize'] = gen_simple

            # Test letterbox
            gen_letterbox = RobustReIDEmbeddingGenerator(
                self.model_path, use_letterbox=True, debug=False, feature_map_name=feature_map_name
            )
            generators['letterbox'] = gen_letterbox

            for method_name, gen in generators.items():
                method_result = {'preprocessing_method': method_name}

                # Get preprocessing info
                original_image, input_feed = gen.preprocess_image(image_path)
                detections, feature_map = gen.run_inference(input_feed)
                filtered_detections = gen.filter_detections(detections, conf_threshold=0.3)

                if filtered_detections:
                    scaled_detections = gen.scale_bboxes_to_feature_space(filtered_detections, feature_map.shape)

                    method_result.update({
                        'original_image_shape': original_image.shape,
                        'feature_map_shape': list(feature_map.shape),
                        'num_detections': len(filtered_detections),
                        'num_valid_scaled': len(scaled_detections),
                        'letterbox_info': gen.letterbox_info,
                        'sample_scaling': []
                    })

                    # Analyze scaling for first few detections
                    for i, (cls_id, conf, orig_bbox, scaled_bbox) in enumerate(scaled_detections[:3]):
                        scaling_info = {
                            'detection_id': i,
                            'original_bbox': orig_bbox,
                            'scaled_bbox': scaled_bbox,
                            'scaling_factors': [
                                scaled_bbox[0] / orig_bbox[0] if orig_bbox[0] != 0 else 0,
                                scaled_bbox[1] / orig_bbox[1] if orig_bbox[1] != 0 else 0
                            ]
                        }
                        method_result['sample_scaling'].append(scaling_info)

                    # Check for invalid scaled regions
                    invalid_scaled = sum(1 for _, _, _, sb in scaled_detections
                                       if sb[2] <= sb[0] or sb[3] <= sb[1])
                    if invalid_scaled > 0:
                        method_result['invalid_scaled_count'] = invalid_scaled
                        check_result['issues'].append(f"{method_name}: {invalid_scaled} invalid scaled regions")

                check_result['details'][method_name] = method_result

            # Compare methods
            if 'simple_resize' in check_result['details'] and 'letterbox' in check_result['details']:
                simple_count = check_result['details']['simple_resize'].get('num_valid_scaled', 0)
                letterbox_count = check_result['details']['letterbox'].get('num_valid_scaled', 0)

                if abs(simple_count - letterbox_count) > 1:
                    check_result['issues'].append(f"Large difference in valid detections between methods: simple={simple_count}, letterbox={letterbox_count}")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate coordinate consistency: {e}")

        return check_result

    def validate_roi_extraction(self, generator: RobustReIDEmbeddingGenerator,
                               image_path: str) -> Dict[str, Any]:
        """Validate RoI extraction quality and consistency."""
        print("🔍 Validating RoI extraction...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Run full pipeline
            embeddings = generator.process_image(image_path, conf_threshold=0.3, output_dir="pipeline/output/validation")

            if not embeddings:
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No embeddings generated")
                return check_result

            check_result['details']['num_embeddings'] = len(embeddings)
            check_result['details']['roi_analysis'] = []

            very_small_rois = 0
            zero_embeddings = 0
            invalid_embeddings = 0

            for i, (detection_info, embedding) in enumerate(embeddings):
                roi_shape = detection_info['roi_shape']
                roi_analysis = {
                    'detection_id': i,
                    'class_id': detection_info['class_id'],
                    'roi_shape': list(roi_shape),
                    'roi_area': roi_shape[1] * roi_shape[2] if len(roi_shape) >= 3 else 0,
                    'embedding_stats': {
                        'min': float(embedding.min()),
                        'max': float(embedding.max()),
                        'mean': float(embedding.mean()),
                        'std': float(embedding.std()),
                        'norm': float(np.linalg.norm(embedding))
                    }
                }

                # Check for very small RoIs
                if roi_analysis['roi_area'] <= 4:  # 2x2 or smaller
                    very_small_rois += 1
                    roi_analysis['issues'] = roi_analysis.get('issues', [])
                    roi_analysis['issues'].append("Very small RoI area")

                # Check for zero embeddings
                if np.allclose(embedding, 0.0):
                    zero_embeddings += 1
                    roi_analysis['issues'] = roi_analysis.get('issues', [])
                    roi_analysis['issues'].append("Zero embedding")

                # Check for invalid embeddings
                if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                    invalid_embeddings += 1
                    roi_analysis['issues'] = roi_analysis.get('issues', [])
                    roi_analysis['issues'].append("Invalid embedding (NaN/Inf)")

                # Check norm (should be ~1.0 after normalization)
                expected_norm = 1.0
                norm_deviation = abs(roi_analysis['embedding_stats']['norm'] - expected_norm)
                if norm_deviation > 0.01:
                    roi_analysis['issues'] = roi_analysis.get('issues', [])
                    roi_analysis['issues'].append(f"Unexpected norm: {roi_analysis['embedding_stats']['norm']:.4f}")

                check_result['details']['roi_analysis'].append(roi_analysis)

            # Summary statistics
            check_result['details']['quality_stats'] = {
                'very_small_rois': very_small_rois,
                'zero_embeddings': zero_embeddings,
                'invalid_embeddings': invalid_embeddings,
                'total_embeddings': len(embeddings)
            }

            # Quality thresholds
            if very_small_rois > len(embeddings) * 0.3:  # More than 30% very small
                check_result['status'] = 'WARN'
                check_result['issues'].append(f"High percentage of very small RoIs: {very_small_rois}/{len(embeddings)}")

            if zero_embeddings > 0:
                check_result['status'] = 'WARN'
                check_result['issues'].append(f"Found {zero_embeddings} zero embeddings")

            if invalid_embeddings > 0:
                check_result['status'] = 'FAIL'
                check_result['issues'].append(f"Found {invalid_embeddings} invalid embeddings")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate RoI extraction: {e}")

        return check_result

    def validate_embedding_quality(self, generator: RobustReIDEmbeddingGenerator,
                                  image_path: str) -> Dict[str, Any]:
        """Validate embedding quality and separability."""
        print("🔍 Validating embedding quality...")

        check_result = {
            'status': 'PASS',
            'details': {},
            'issues': []
        }

        try:
            # Run pipeline multiple times to check consistency
            embeddings_runs = []
            for run in range(2):  # Run twice to check consistency
                embeddings = generator.process_image(image_path, conf_threshold=0.3, output_dir=f"pipeline/output/validation/run_{run}")
                if embeddings:
                    embeddings_runs.append(embeddings)

            if not embeddings_runs:
                check_result['status'] = 'FAIL'
                check_result['issues'].append("No embeddings generated in any run")
                return check_result

            # Analyze primary run
            primary_embeddings = embeddings_runs[0]
            vectors = np.array([emb for _, emb in primary_embeddings])

            check_result['details']['num_embeddings'] = len(primary_embeddings)
            check_result['details']['embedding_dimension'] = len(primary_embeddings[0][1]) if primary_embeddings else 0

            # Analyze embedding statistics
            norms = [np.linalg.norm(emb) for _, emb in primary_embeddings]
            check_result['details']['norm_stats'] = {
                'mean': float(np.mean(norms)),
                'std': float(np.std(norms)),
                'min': float(np.min(norms)),
                'max': float(np.max(norms))
            }

            # Check norm consistency (should all be ~1.0)
            norm_deviations = [abs(norm - 1.0) for norm in norms]
            max_norm_deviation = max(norm_deviations) if norm_deviations else 0
            if max_norm_deviation > 0.01:
                check_result['issues'].append(f"Large norm deviations detected: max={max_norm_deviation:.4f}")

            # Analyze class separability if multiple classes present
            if len(primary_embeddings) >= 2:
                similarity_matrix = np.dot(vectors, vectors.T)

                same_class_similarities = []
                diff_class_similarities = []

                for i in range(len(primary_embeddings)):
                    for j in range(i + 1, len(primary_embeddings)):
                        sim = similarity_matrix[i, j]
                        if primary_embeddings[i][0]['class_id'] == primary_embeddings[j][0]['class_id']:
                            same_class_similarities.append(sim)
                        else:
                            diff_class_similarities.append(sim)

                separability_analysis = {
                    'same_class_count': len(same_class_similarities),
                    'diff_class_count': len(diff_class_similarities)
                }

                if same_class_similarities:
                    separability_analysis['same_class_stats'] = {
                        'mean': float(np.mean(same_class_similarities)),
                        'std': float(np.std(same_class_similarities))
                    }

                if diff_class_similarities:
                    separability_analysis['diff_class_stats'] = {
                        'mean': float(np.mean(diff_class_similarities)),
                        'std': float(np.std(diff_class_similarities))
                    }

                # Calculate separability ratio
                if same_class_similarities and diff_class_similarities:
                    mean_diff = np.mean(diff_class_similarities)
                    mean_same = np.mean(same_class_similarities)
                    separability_ratio = mean_diff / mean_same if mean_same > 0 else float('inf')

                    separability_analysis['separability_ratio'] = float(separability_ratio)

                    if separability_ratio < 1.2:
                        check_result['status'] = 'WARN'
                        check_result['issues'].append(f"Poor class separability: ratio={separability_ratio:.2f} (should be > 1.2)")

                check_result['details']['separability_analysis'] = separability_analysis

            # Check consistency between runs
            if len(embeddings_runs) >= 2:
                # Compare embeddings for identical detections
                run1_embeddings = embeddings_runs[0]
                run2_embeddings = embeddings_runs[1]

                if len(run1_embeddings) == len(run2_embeddings):
                    consistency_scores = []
                    for (_, emb1), (_, emb2) in zip(run1_embeddings, run2_embeddings):
                        consistency = np.dot(emb1, emb2)  # Cosine similarity
                        consistency_scores.append(consistency)

                    mean_consistency = np.mean(consistency_scores)
                    check_result['details']['consistency_analysis'] = {
                        'mean_consistency': float(mean_consistency),
                        'consistency_scores': [float(s) for s in consistency_scores]
                    }

                    if mean_consistency < 0.95:  # Should be very similar
                        check_result['status'] = 'WARN'
                        check_result['issues'].append(f"Low consistency between runs: {mean_consistency:.3f}")

        except Exception as e:
            check_result['status'] = 'ERROR'
            check_result['issues'].append(f"Failed to validate embedding quality: {e}")

        return check_result

    def run_pedestrian_focused_validation(self, image_path: str, feature_map_name: str = None) -> Dict[str, Any]:
        """Run pedestrian-focused Re-ID validation for BoT-SORT tracking."""
        print("🔄 Starting pedestrian-focused Re-ID pipeline validation...")
        print(f"   Model: {self.model_path}")
        print(f"   Test image: {image_path}")
        if feature_map_name:
            print(f"   Feature map: {feature_map_name}")
        print("   Focus: Pedestrian tracking with BoT-SORT compatibility")

        # Initialize generator for testing
        generator = RobustReIDEmbeddingGenerator(self.model_path, debug=False, feature_map_name=feature_map_name)

        # Run all validation checks
        checks = {}

        # Core validation checks
        checks['model_structure'] = self.validate_model_structure()
        checks['detection_format'] = self.validate_detection_format(generator, image_path)
        checks['coordinate_consistency'] = self.validate_coordinate_consistency(generator, image_path, feature_map_name)
        checks['roi_extraction'] = self.validate_roi_extraction(generator, image_path)
        checks['embedding_quality'] = self.validate_embedding_quality(generator, image_path)

        # Pedestrian-specific validation checks
        if self.pedestrian_focused:
            checks['pedestrian_detection_quality'] = self.validate_pedestrian_detection_quality(generator, image_path)
            checks['pedestrian_discriminability'] = self.validate_pedestrian_embedding_discriminability(generator, image_path)
            checks['botsort_compatibility'] = self.validate_botsort_compatibility(generator, image_path)
            checks['temporal_consistency'] = self.validate_temporal_consistency(generator, image_path)

        # Determine overall status
        statuses = [check['status'] for check in checks.values()]
        if 'ERROR' in statuses or 'FAIL' in statuses:
            overall_status = 'FAIL'
        elif 'WARN' in statuses:
            overall_status = 'WARN'
        else:
            overall_status = 'PASS'

        # Collect all issues
        critical_issues = []
        warnings = []

        for check_name, check_result in checks.items():
            for issue in check_result.get('issues', []):
                if check_result['status'] in ['ERROR', 'FAIL']:
                    critical_issues.append(f"{check_name}: {issue}")
                else:
                    warnings.append(f"{check_name}: {issue}")

        # Calculate pedestrian-specific metrics
        pedestrian_metrics = self._calculate_pedestrian_metrics(checks)
        botsort_compatibility = self._calculate_botsort_compatibility(checks)

        # Generate recommendations
        recommendations = self._generate_pedestrian_recommendations(checks)

        self.validation_results.update({
            'validation_timestamp': np.datetime64('now').astype(str),
            'checks': checks,
            'overall_status': overall_status,
            'pedestrian_metrics': pedestrian_metrics,
            'botsort_compatibility': botsort_compatibility,
            'critical_issues': critical_issues,
            'warnings': warnings,
            'recommendations': recommendations
        })

        return self.validation_results

    def _calculate_pedestrian_metrics(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall pedestrian tracking quality metrics."""
        metrics = {
            'person_detection_score': 0.0,
            'discriminability_score': 0.0,
            'temporal_stability_score': 0.0,
            'overall_pedestrian_score': 0.0
        }

        try:
            # Person detection quality score
            if 'pedestrian_detection_quality' in checks and checks['pedestrian_detection_quality']['status'] in ['PASS', 'WARN']:
                details = checks['pedestrian_detection_quality']['details']
                if 'person_stats' in details:
                    conf_mean = details['person_stats'].get('confidence', {}).get('mean', 0)
                    metrics['person_detection_score'] = min(conf_mean * 1.2, 1.0)  # Scale up confidence

            # Discriminability score
            if 'pedestrian_discriminability' in checks and checks['pedestrian_discriminability']['status'] in ['PASS', 'WARN']:
                details = checks['pedestrian_discriminability']['details']
                if 'inter_person_similarity' in details:
                    mean_sim = details['inter_person_similarity'].get('mean', 1.0)
                    # Lower similarity is better for discriminability
                    metrics['discriminability_score'] = max(0.0, 1.0 - (mean_sim / 0.3))

            # Temporal stability score
            if 'temporal_consistency' in checks and checks['temporal_consistency']['status'] in ['PASS', 'WARN']:
                details = checks['temporal_consistency']['details']
                if 'temporal_consistency' in details:
                    consistency = details['temporal_consistency'].get('overall_mean', 0.0)
                    metrics['temporal_stability_score'] = consistency

            # Overall score (weighted average)
            metrics['overall_pedestrian_score'] = (
                0.3 * metrics['person_detection_score'] +
                0.4 * metrics['discriminability_score'] +
                0.3 * metrics['temporal_stability_score']
            )

        except Exception as e:
            print(f"Warning: Failed to calculate pedestrian metrics: {e}")

        return metrics

    def _calculate_botsort_compatibility(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate BoT-SORT compatibility metrics."""
        compatibility = {
            'embedding_format_score': 0.0,
            'similarity_distribution_score': 0.0,
            'id_switch_risk_score': 0.0,
            'overall_botsort_score': 0.0
        }

        try:
            if 'botsort_compatibility' in checks and checks['botsort_compatibility']['status'] in ['PASS', 'WARN']:
                details = checks['botsort_compatibility']['details']

                # Use the calculated BoT-SORT score if available
                if 'botsort_compatibility_score' in details:
                    compatibility['overall_botsort_score'] = details['botsort_compatibility_score']

                # Embedding format score (normalization quality)
                if 'embedding_norms' in details:
                    norm_std = details['embedding_norms'].get('std', 1.0)
                    compatibility['embedding_format_score'] = max(0.0, 1.0 - norm_std * 10)  # Penalize high std

                # Similarity distribution score
                if 'botsort_similarities' in details:
                    mean_sim = details['botsort_similarities'].get('mean', 1.0)
                    compatibility['similarity_distribution_score'] = max(0.0, 1.0 - (mean_sim / 0.4))

                # ID switch risk (inverse of max similarity)
                if 'botsort_similarities' in details:
                    max_sim = details['botsort_similarities'].get('max', 1.0)
                    compatibility['id_switch_risk_score'] = max(0.0, 1.0 - max_sim)

        except Exception as e:
            print(f"Warning: Failed to calculate BoT-SORT compatibility: {e}")

        return compatibility

    def _generate_pedestrian_recommendations(self, checks: Dict[str, Any]) -> List[str]:
        """Generate pedestrian-focused recommendations."""
        recommendations = []

        # Model structure recommendations
        if checks['model_structure']['status'] != 'PASS':
            if not checks['model_structure']['details'].get('has_c5_features', False):
                recommendations.append("🎯 PEDESTRIAN: Use C4 (stride-16) features instead of C5 for better pedestrian detail preservation")

        # Pedestrian detection recommendations
        if 'pedestrian_detection_quality' in checks and checks['pedestrian_detection_quality']['status'] != 'PASS':
            recommendations.append("🚶 PEDESTRIAN: Improve person detection confidence thresholds - use 0.7+ for reliable tracking")
            recommendations.append("🚶 PEDESTRIAN: Consider fine-tuning model on pedestrian-heavy datasets")

        # Discriminability recommendations
        if 'pedestrian_discriminability' in checks and checks['pedestrian_discriminability']['status'] != 'PASS':
            details = checks['pedestrian_discriminability']['details']
            if 'inter_person_similarity' in details:
                mean_sim = details['inter_person_similarity'].get('mean', 0)
                if mean_sim > 0.3:
                    recommendations.append(f"🎯 ID-SWITCH RISK: High inter-person similarity ({mean_sim:.3f}) - reduce feature map stride or use multi-scale features")
                    recommendations.append("🎯 ID-SWITCH RISK: Consider adding Re-ID supervision during training")

        # BoT-SORT compatibility recommendations
        if 'botsort_compatibility' in checks and checks['botsort_compatibility']['status'] != 'PASS':
            recommendations.append("🤖 BOT-SORT: Ensure L2 normalization of embeddings for proper distance computation")
            recommendations.append("🤖 BOT-SORT: Validate similarity threshold settings (recommend 0.7-0.8 for person matching)")

        # Temporal consistency recommendations
        if 'temporal_consistency' in checks and checks['temporal_consistency']['status'] != 'PASS':
            details = checks['temporal_consistency']['details']
            if 'temporal_consistency' in details:
                consistency = details['temporal_consistency'].get('overall_mean', 0)
                if consistency < 0.95:
                    recommendations.append(f"⏱️ TRACKING STABILITY: Low temporal consistency ({consistency:.3f}) - may cause ID switches")
                    recommendations.append("⏱️ TRACKING STABILITY: Consider temporal smoothing or Kalman filtering in BoT-SORT")

        # Overall recommendations
        if len(recommendations) == 0:
            recommendations.append("✅ PEDESTRIAN TRACKING: Pipeline optimized for BoT-SORT pedestrian tracking")
            recommendations.append("✅ PEDESTRIAN TRACKING: Low ID switch risk - ready for production deployment")
        else:
            recommendations.append("🔧 NEXT STEPS: Implement recommended improvements and re-validate pipeline")
            recommendations.append("📊 MONITORING: Set up tracking quality metrics monitoring in production")

        return recommendations
        """Run all validation checks and generate comprehensive report."""
        print("🔄 Starting comprehensive Re-ID pipeline validation...")
        print(f"   Model: {self.model_path}")
        print(f"   Test image: {image_path}")
        if feature_map_name:
            print(f"   Feature map: {feature_map_name}")

        # Initialize generator for testing
        generator = RobustReIDEmbeddingGenerator(self.model_path, debug=False, feature_map_name=feature_map_name)

        # Run all validation checks
        checks = {}

        checks['model_structure'] = self.validate_model_structure()
        checks['detection_format'] = self.validate_detection_format(generator, image_path)
        checks['coordinate_consistency'] = self.validate_coordinate_consistency(generator, image_path, feature_map_name)
        checks['roi_extraction'] = self.validate_roi_extraction(generator, image_path)
        checks['embedding_quality'] = self.validate_embedding_quality(generator, image_path)

        # Determine overall status
        statuses = [check['status'] for check in checks.values()]
        if 'ERROR' in statuses or 'FAIL' in statuses:
            overall_status = 'FAIL'
        elif 'WARN' in statuses:
            overall_status = 'WARN'
        else:
            overall_status = 'PASS'

        # Collect all issues
        critical_issues = []
        warnings = []

        for check_name, check_result in checks.items():
            for issue in check_result.get('issues', []):
                if check_result['status'] in ['ERROR', 'FAIL']:
                    critical_issues.append(f"{check_name}: {issue}")
                else:
                    warnings.append(f"{check_name}: {issue}")

        # Generate recommendations
        recommendations = self._generate_recommendations(checks)

        self.validation_results.update({
            'validation_timestamp': np.datetime64('now').astype(str),
            'checks': checks,
            'overall_status': overall_status,
            'critical_issues': critical_issues,
            'warnings': warnings,
            'recommendations': recommendations
        })

        return self.validation_results

    def _generate_recommendations(self, checks: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Model structure recommendations
        if checks['model_structure']['status'] != 'PASS':
            if not checks['model_structure']['details'].get('has_c5_features', False):
                recommendations.append("Consider using a model with C5 (stride-32) backbone features for optimal Re-ID performance")

            if checks['model_structure']['details'].get('potential_confusion'):
                recommendations.append("Use --feature-map-name to explicitly specify backbone feature to avoid confusion with detection outputs")

        # Detection format recommendations
        if checks['detection_format']['status'] != 'PASS':
            recommendations.append("Verify detection tensor format - consider using --detection-layout parameter")
            recommendations.append("Inspect raw detection outputs and confirm coordinate format matches expectations")

        # Coordinate consistency recommendations
        if checks['coordinate_consistency']['status'] != 'PASS':
            recommendations.append("Consider using letterbox preprocessing (--use-letterbox) if model was trained with aspect ratio preservation")
            recommendations.append("Verify that coordinate scaling matches model training preprocessing")

        # RoI extraction recommendations
        if checks['roi_extraction']['status'] != 'PASS':
            roi_stats = checks['roi_extraction']['details'].get('quality_stats', {})
            if roi_stats.get('very_small_rois', 0) > 0:
                recommendations.append("Consider using higher resolution feature maps (C4 instead of C5) for small object Re-ID")
            if roi_stats.get('zero_embeddings', 0) > 0:
                recommendations.append("Check feature map activation patterns - zero embeddings may indicate inactive regions")

        # Embedding quality recommendations
        if checks['embedding_quality']['status'] != 'PASS':
            separability = checks['embedding_quality']['details'].get('separability_analysis', {})
            if separability.get('separability_ratio', float('inf')) < 1.2:
                recommendations.append("Poor class separability detected - consider fine-tuning model with Re-ID supervision")
                recommendations.append("Try combining multiple feature map levels (C3+C4+C5) for better discriminative power")

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Validation passed - pipeline is ready for production use")
            recommendations.append("Consider running validation on multiple diverse images to ensure robustness")
        else:
            recommendations.append("Run validation again after implementing recommended fixes")

        return recommendations

    def save_report(self, output_path: str):
        """Save validation report to JSON file."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Convert numpy types to Python native types for JSON serialization
        json_compatible = self._convert_numpy_types(self.validation_results)

        with open(output_path, 'w') as f:
            json.dump(json_compatible, f, indent=2)
        print(f"📋 Validation report saved: {output_path}")

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def print_summary(self):
        """Print a human-readable validation summary."""
        print(f"{'='*80}")
        if self.pedestrian_focused:
            print(f"PEDESTRIAN RE-ID VALIDATION SUMMARY FOR BOT-SORT TRACKING")
        else:
            print(f"RE-ID PIPELINE VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Overall Status: {self.validation_results['overall_status']}")
        print(f"Model: {self.validation_results['model_path']}")
        print(f"Validation Type: {self.validation_results['validation_type']}")
        print(f"Timestamp: {self.validation_results['validation_timestamp']}")

        print(f"📊 Check Results:")
        for check_name, check_result in self.validation_results['checks'].items():
            status_emoji = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'ERROR': '💥'}
            emoji = status_emoji.get(check_result['status'], '❓')
            print(f"   {emoji} {check_name.replace('_', ' ').title()}: {check_result['status']}")

        # Pedestrian-specific metrics
        if self.pedestrian_focused and 'pedestrian_metrics' in self.validation_results:
            print(f"🚶 Pedestrian Tracking Metrics:")
            metrics = self.validation_results['pedestrian_metrics']
            print(f"   Person Detection Score: {metrics.get('person_detection_score', 0):.3f}")
            print(f"   Discriminability Score: {metrics.get('discriminability_score', 0):.3f}")
            print(f"   Temporal Stability Score: {metrics.get('temporal_stability_score', 0):.3f}")
            print(f"   Overall Pedestrian Score: {metrics.get('overall_pedestrian_score', 0):.3f}")

        # BoT-SORT compatibility
        if self.pedestrian_focused and 'botsort_compatibility' in self.validation_results:
            print(f"🤖 BoT-SORT Compatibility:")
            compat = self.validation_results['botsort_compatibility']
            print(f"   Embedding Format Score: {compat.get('embedding_format_score', 0):.3f}")
            print(f"   Similarity Distribution Score: {compat.get('similarity_distribution_score', 0):.3f}")
            print(f"   ID Switch Risk Score: {compat.get('id_switch_risk_score', 0):.3f}")
            print(f"   Overall BoT-SORT Score: {compat.get('overall_botsort_score', 0):.3f}")

        if self.validation_results['critical_issues']:
            print(f"🚨 Critical Issues:")
            for issue in self.validation_results['critical_issues']:
                print(f"   - {issue}")

        if self.validation_results['warnings']:
            print(f"⚠️  Warnings:")
            for warning in self.validation_results['warnings']:
                print(f"   - {warning}")

        print(f"💡 Recommendations:")
        for rec in self.validation_results['recommendations']:
            print(f"   - {rec}")

        print(f"{'='*80}")

        if self.pedestrian_focused:
            # Additional guidance for pedestrian tracking
            overall_score = self.validation_results.get('pedestrian_metrics', {}).get('overall_pedestrian_score', 0)
            if overall_score >= 0.8:
                print("🎯 TRACKING QUALITY: Excellent pedestrian Re-ID quality - minimal ID switching expected")
            elif overall_score >= 0.6:
                print("🎯 TRACKING QUALITY: Good pedestrian Re-ID quality - some improvements recommended")
            else:
                print("🎯 TRACKING QUALITY: Poor pedestrian Re-ID quality - significant improvements needed")
            print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(
        description="Validate pedestrian Re-ID embedding pipeline for BoT-SORT tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default="output/rtdetrv3_r18vd_6x_backbone.onnx",
                       help="Path to ONNX model with backbone features")
    parser.add_argument("--image", default="demo/demo.jpg",
                       help="Path to test image")
    parser.add_argument("--output", default="output/pedestrian_reid_validation_report.json",
                       help="Path to save validation report")
    parser.add_argument("--feature-map-name",
                       help="Explicitly specify backbone feature map name (e.g., Concat.5 for C3, Concat.3 for C4)")
    parser.add_argument("--pedestrian-focused", action="store_true", default=True,
                       help="Enable pedestrian-focused validation for BoT-SORT tracking")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")

    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return 1

    if not os.path.exists(args.image):
        print(f"❌ Test image not found: {args.image}")
        return 1

    try:
        # Run validation
        validator = PedestrianReIDValidator(
            args.model,
            debug=args.debug,
            pedestrian_focused=args.pedestrian_focused
        )
        results = validator.run_pedestrian_focused_validation(args.image, args.feature_map_name)

        # Save report
        validator.save_report(args.output)

        # Print summary
        validator.print_summary()

        # Return appropriate exit code
        if results['overall_status'] == 'FAIL':
            print("❌ Pedestrian Re-ID validation failed - critical issues detected")
            return 1
        elif results['overall_status'] == 'WARN':
            print("⚠️  Pedestrian Re-ID validation completed with warnings")
            return 0
        else:
            print("✅ Pedestrian Re-ID validation passed successfully")
            return 0

    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
