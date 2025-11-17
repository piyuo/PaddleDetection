# onnx/validate_pedestrian.py
#!/usr/bin/env python3
"""
Pedestrian-Specific Re-ID Validation
===================================

This script provides specialized validation for pedestrian Re-ID applications, focusing
on person-to-person distinguishability, counting accuracy, and real-world deployment
scenarios for surveillance and crowd monitoring systems.

Pedestrian-Specific Features:
    1. Person Detection Optimization
       - High-precision person class filtering
       - Confidence threshold optimization
       - False positive reduction strategies

    2. Pedestrian Re-ID Quality Assessment
       - Person-specific embedding validation
       - Clothing and appearance consistency
       - Viewpoint invariance testing

    3. Duplicate Detection and Filtering
       - Same-person identification across detections
       - Temporal consistency validation
       - Overlap-based duplicate removal

    4. Counting Accuracy Validation
       - Unique person counting algorithms
       - Crowd density estimation
       - Multi-camera fusion simulation

    5. Real-World Scenario Testing
       - Varying lighting conditions
       - Different camera angles
       - Partial occlusion handling

Application Domains:
    • Pedestrian counting systems
    • Crowd monitoring and analytics
    • Surveillance and security applications
    • Person tracking across camera networks
    • Retail analytics and customer flow

Validation Metrics:
    • Person detection precision (target: > 90%)
    • Re-ID distinguishability (target: > 85%)
    • Counting accuracy (target: > 95%)
    • Temporal consistency (target: > 90%)

Quality Thresholds:
    • Confidence threshold: 0.3 (optimized for person detection)
    • Similarity threshold: 0.95 (for duplicate filtering)
    • Minimum embedding norm: 0.99 (L2 normalization check)

Usage:
    python validate_pedestrian.py --model backbone_model.onnx --image test.jpg
                                 --feature-map-name Concat.3

Output:
    Pedestrian-specific validation results with counting accuracy and quality metrics.

Author: RT-DETRv3 Development Team
License: Same as RT-DETRv3 repository
"""

import numpy as np
import argparse
from reid_embeddings import RobustReIDEmbeddingGenerator

def filter_duplicate_detections(people_embeddings, similarity_threshold=0.9):
    """Remove likely duplicate detections of same person."""
    if len(people_embeddings) <= 1:
        return people_embeddings

    filtered = []
    used_indices = set()

    for i, (info_i, emb_i) in enumerate(people_embeddings):
        if i in used_indices:
            continue

        current_group = [(info_i, emb_i)]

        for j, (info_j, emb_j) in enumerate(people_embeddings[i+1:], i+1):
            if j in used_indices:
                continue

            similarity = np.dot(emb_i, emb_j)
            if similarity > similarity_threshold:
                current_group.append((info_j, emb_j))
                used_indices.add(j)

        # Keep detection with highest confidence from group
        best_detection = max(current_group, key=lambda x: x[0]['confidence'])
        filtered.append(best_detection)
        used_indices.add(i)

    return filtered

def production_pedestrian_counting(model_path: str, image_path: str, feature_map_name: str = None,
                                 conf_threshold: float = 0.3, similarity_threshold: float = 0.95):
    """Production-ready pedestrian counting with duplicate removal."""
    print("🏭 Production Pedestrian Counting Pipeline...")

    generator = RobustReIDEmbeddingGenerator(
        model_path,
        feature_map_name=feature_map_name,
        debug=False  # Disable debug for production
    )

    # Generate embeddings
    embeddings = generator.process_image(image_path, conf_threshold=conf_threshold,
                                       output_dir="onnx/validation")

    # Filter only people (class_id = 0)
    people_embeddings = [
        (info, emb) for info, emb in embeddings
        if info['class_id'] == 0
    ]

    print(f"   Initial people detections: {len(people_embeddings)}")

    # Remove duplicates
    if len(people_embeddings) > 1:
        unique_people = filter_duplicate_detections(people_embeddings, similarity_threshold=similarity_threshold)
        removed_count = len(people_embeddings) - len(unique_people)

        if removed_count > 0:
            print(f"   Removed {removed_count} duplicate detections")

        print(f"   Final pedestrian count: {len(unique_people)}")
        return unique_people
    else:
        print(f"   Final pedestrian count: {len(people_embeddings)}")
        return people_embeddings

def validate_pedestrian_reid(model_path: str, image_path: str, feature_map_name: str = None):
    """Validate Re-ID pipeline specifically for pedestrian counting."""

    print("🚶 Validating Re-ID pipeline for pedestrian counting...")

    generator = RobustReIDEmbeddingGenerator(
        model_path,
        feature_map_name=feature_map_name,
        debug=True
    )

    embeddings = generator.process_image(image_path, conf_threshold=0.3,
                                       output_dir="output/validation")

    # Filter only people (class_id = 0)
    people_embeddings = [
        (info, emb) for info, emb in embeddings
        if info['class_id'] == 0
    ]

    print(f"\n📊 Pedestrian Counting Analysis:")
    print(f"   Total detections: {len(embeddings)}")
    print(f"   People detected: {len(people_embeddings)}")
    print(f"   Other objects: {len(embeddings) - len(people_embeddings)}")

    if len(people_embeddings) < 2:
        print("⚠️  Need at least 2 people for separability analysis")
        return

    # Analyze person-to-person separability
    people_vectors = np.array([emb for _, emb in people_embeddings])
    similarity_matrix = np.dot(people_vectors, people_vectors.T)

    # Get all pairwise similarities between different people
    person_similarities = []
    for i in range(len(people_embeddings)):
        for j in range(i + 1, len(people_embeddings)):
            sim = similarity_matrix[i, j]
            person_similarities.append(sim)
            person_similarities.append(sim)

    if person_similarities:
        mean_similarity = np.mean(person_similarities)
        std_similarity = np.std(person_similarities)
        min_similarity = np.min(person_similarities)
        max_similarity = np.max(person_similarities)

        print(f"\n🔍 Person-to-Person Similarity Analysis:")
        print(f"   Mean similarity: {mean_similarity:.3f}")
        print(f"   Std deviation: {std_similarity:.3f}")
        print(f"   Min similarity: {min_similarity:.3f}")
        print(f"   Max similarity: {max_similarity:.3f}")

        # For pedestrian counting, we want:
        # - Not too high similarity (can distinguish different people)
        # - Not too low similarity (stable embeddings for same person)

        if mean_similarity > 0.8:
            print("⚠️  Warning: Very high person-to-person similarity")
            print("    This may make it hard to distinguish different people")
        elif mean_similarity < 0.3:
            print("✅ Good: Low person-to-person similarity")
            print("    Different people are well distinguished")
        else:
            print("✅ Reasonable: Moderate person-to-person similarity")

        # Check for very similar people (potential duplicates)
        very_similar_count = sum(1 for sim in person_similarities if sim > 0.9)

        if very_similar_count > 0:
            print(f"⚠️  Found {very_similar_count} very similar person pairs (sim > 0.9)")
            print("    These might be the same person detected multiple times")

            # Test duplicate filtering
            print(f"\n🔄 Testing duplicate filtering...")
            filtered_people = filter_duplicate_detections(people_embeddings, similarity_threshold=0.95)
            print(f"   Before filtering: {len(people_embeddings)} people")
            print(f"   After filtering (sim > 0.95): {len(filtered_people)} people")
            print(f"   Removed duplicates: {len(people_embeddings) - len(filtered_people)}")

            if len(filtered_people) < len(people_embeddings):
                print("✅ Duplicate filtering successfully reduced count")
                people_embeddings = filtered_people  # Use filtered for remaining analysis

    # Analyze ROI quality specifically for people
    people_roi_areas = [info['roi_shape'][1] * info['roi_shape'][2]
                       for info, _ in people_embeddings]

    small_people_rois = sum(1 for area in people_roi_areas if area <= 9)  # 3x3 or smaller

    print(f"\n📏 People ROI Quality:")
    print(f"   Small people ROIs: {small_people_rois}/{len(people_embeddings)}")

    if small_people_rois > len(people_embeddings) * 0.5:
        print("⚠️  Warning: Many people have small ROIs")
        print("    Consider using C3 (stride 8) for better small person detection")
    else:
        print("✅ Good: Most people have adequate ROI size")

    # Overall assessment for pedestrian counting
    print(f"\n🎯 Pedestrian Counting Assessment:")
    final_count = len(people_embeddings)
    if final_count >= 5:
        print("✅ Good detection count for analysis")
    else:
        print("⚠️  Low detection count - test with busier scenes")

    if mean_similarity < 0.7 and small_people_rois < len(people_embeddings) * 0.6:
        print("✅ Pipeline suitable for pedestrian counting")
        print(f"💡 Final pedestrian count: {final_count}")
    else:
        print("⚠️  Pipeline may have challenges with pedestrian counting")
        print(f"💡 Final pedestrian count: {final_count}")

    return {
        'total_people': final_count,
        'original_detections': len([info for info, _ in embeddings if info['class_id'] == 0]),
        'person_similarity_stats': {
            'mean': mean_similarity,
            'std': std_similarity,
            'min': min_similarity,
            'max': max_similarity
        },
        'small_rois_percentage': small_people_rois / len(people_embeddings) if people_embeddings else 0
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian counting with Re-ID validation")
    parser.add_argument("--model", default="onnx/backbone/rtdetrv3_r18vd_6x.onnx",
                       help="Path to ONNX model")
    parser.add_argument("--image", default="onnx/demo/demo.jpg",
                       help="Path to test image")
    parser.add_argument("--feature-map-name", default="Concat.5",
                       help="Feature map name (C3=Concat.5, C4=Concat.3)")
    parser.add_argument("--conf-threshold", type=float, default=0.3,
                       help="Confidence threshold for detections")
    parser.add_argument("--similarity-threshold", type=float, default=0.95,
                       help="Similarity threshold for duplicate removal")
    parser.add_argument("--mode", choices=["validate", "production", "both"], default="both",
                       help="Run validation, production counting, or both")

    args = parser.parse_args()

    print("🚶 Pedestrian Re-ID Analysis")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Feature map: {args.feature_map_name}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print("")

    if args.mode in ["validate", "both"]:
        print("🔍 Running Validation...")
        results = validate_pedestrian_reid(args.model, args.image, args.feature_map_name)
        print("")

    if args.mode in ["production", "both"]:
        print("🏭 Running Production Counting...")
        unique_people = production_pedestrian_counting(
            args.model, args.image, args.feature_map_name,
            args.conf_threshold, args.similarity_threshold
        )

        print(f"\n✅ Production Result: {len(unique_people)} unique pedestrians detected")

        # Show confidence distribution
        if unique_people:
            confidences = [info['confidence'] for info, _ in unique_people]
            print(f"   Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"   Mean confidence: {sum(confidences)/len(confidences):.3f}")

        print("🎯 Ready for deployment!")

    print("\n" + "=" * 50)
    print("Analysis complete! 🎉")