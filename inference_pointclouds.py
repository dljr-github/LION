#!/usr/bin/env python3

import argparse
import glob
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Add tools directory to path and set working directory
tools_dir = project_root / 'tools'
sys.path.append(str(tools_dir))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils
from scipy.spatial.distance import cdist


class SimpleTracker:
    def __init__(self, max_distance=5.0, max_frames_lost=5):
        self.tracks = {}  # track_id -> track_info
        self.next_track_id = 0
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        self.frame_count = 0
    
    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of dicts with keys: 'location', 'dimension', 'rotation_y', 'object', 'score'
        Returns: list of tracks with assigned track_ids
        """
        self.frame_count += 1
        
        if not detections:
            # Mark all tracks as lost
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['frames_lost'] += 1
                if self.tracks[track_id]['frames_lost'] > self.max_frames_lost:
                    del self.tracks[track_id]
            return []
        
        # Extract positions for distance calculation
        det_positions = np.array([[det['location'][0], det['location'][1]] for det in detections])
        
        if not self.tracks:
            # No existing tracks, create new ones
            result = []
            for det in detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    'location': det['location'],
                    'last_seen': self.frame_count,
                    'frames_lost': 0
                }
                det['track_id'] = track_id
                result.append(det)
            return result
        
        # Get active tracks and their positions
        active_tracks = {tid: track for tid, track in self.tracks.items() if track['frames_lost'] == 0}
        if active_tracks:
            track_positions = np.array([[track['location'][0], track['location'][1]] 
                                      for track in active_tracks.values()])
            track_ids = list(active_tracks.keys())
            
            # Calculate distance matrix
            distances = cdist(det_positions, track_positions)
            
            # Hungarian algorithm would be better, but for simplicity use greedy matching
            matched_detections = set()
            matched_tracks = set()
            result = []
            
            # Greedy assignment: for each detection, find closest track within threshold
            for det_idx, detection in enumerate(detections):
                best_track_idx = None
                best_distance = float('inf')
                
                for track_idx, track_id in enumerate(track_ids):
                    if track_idx in matched_tracks:
                        continue
                    
                    distance = distances[det_idx, track_idx]
                    if distance < self.max_distance and distance < best_distance:
                        best_distance = distance
                        best_track_idx = track_idx
                
                if best_track_idx is not None:
                    # Match found
                    track_id = track_ids[best_track_idx]
                    matched_detections.add(det_idx)
                    matched_tracks.add(best_track_idx)
                    
                    # Update track
                    self.tracks[track_id]['location'] = detection['location']
                    self.tracks[track_id]['last_seen'] = self.frame_count
                    self.tracks[track_id]['frames_lost'] = 0
                    
                    detection['track_id'] = track_id
                    result.append(detection)
            
            # Create new tracks for unmatched detections
            for det_idx, detection in enumerate(detections):
                if det_idx not in matched_detections:
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    self.tracks[track_id] = {
                        'location': detection['location'],
                        'last_seen': self.frame_count,
                        'frames_lost': 0
                    }
                    detection['track_id'] = track_id
                    result.append(detection)
            
            # Mark unmatched tracks as lost
            for track_idx, track_id in enumerate(track_ids):
                if track_idx not in matched_tracks:
                    self.tracks[track_id]['frames_lost'] += 1
        else:
            # All tracks are lost, treat as new detections
            result = []
            for det in detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    'location': det['location'],
                    'last_seen': self.frame_count,
                    'frames_lost': 0
                }
                det['track_id'] = track_id
                result.append(det)
        
        # Remove tracks that have been lost too long
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['frames_lost'] > self.max_frames_lost:
                del self.tracks[track_id]
        
        return result


class InferenceDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root_path, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=False, 
            root_path=root_path, logger=logger
        )
        self.root_path = Path(root_path)
        self.sample_file_list = sorted(glob.glob(str(self.root_path / '*.bin')))
        
    def __len__(self):
        return len(self.sample_file_list)
    
    def __getitem__(self, index):
        bin_file = self.sample_file_list[index]
        # Load point cloud with 5 dimensions (x,y,z,intensity,lidar_index)
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
        # Convert to 4D format expected by model (x,y,z,intensity)
        points = points[:, :4]
        
        input_dict = {
            'points': points,
            'frame_id': Path(bin_file).stem,
        }
        
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def convert_predictions_to_json(predictions, frame_id, timestamp=None, score_threshold=0.3, nms_threshold=0.5, tracker=None, cross_class_nms=0.2, use_mapped_classes=False):
    """Convert model predictions to JSON format similar to annotations."""
    result = {
        "transform": {
            "from": "base_link",
            "to": "map", 
            "mat4": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0], 
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            "ego_pose_addon_done": False,
            "curb_frame_selected": False,
            "ego_pose_addon": {
                "heading": 0.0,
                "location": [0.0, 0.0, 0.0]
            }
        },
        "timestamp": timestamp or int(frame_id) * 100000000,  # Generate fake timestamp
        "timestamp_exact_match": False,
        "track": [],
        "semantics": {
            "polylines": None
        }
    }
    
    if len(predictions) > 0:
        pred_boxes = predictions['pred_boxes'].cpu().numpy()
        pred_scores = predictions['pred_scores'].cpu().numpy() 
        pred_labels = predictions['pred_labels'].cpu().numpy()
        
        # Apply score threshold to filter low-confidence detections
        valid_mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[valid_mask]
        pred_scores = pred_scores[valid_mask]
        pred_labels = pred_labels[valid_mask]
        
        # Apply Non-Maximum Suppression per class to remove duplicate detections
        from pcdet.ops.iou3d_nms import iou3d_nms_utils
        
        final_boxes = []
        final_scores = []
        final_labels = []
        
        # Class-specific NMS thresholds based on evaluation config: [0.75, 0.6, 0.55] for [Vehicle, Pedestrian, Static]
        class_nms_thresholds = {0: 0.75, 1: 0.6, 2: 0.55}  # Vehicle, Pedestrian, Static
        
        # Process each class separately
        for class_id in np.unique(pred_labels):
            class_mask = pred_labels == class_id
            class_boxes = pred_boxes[class_mask]
            class_scores = pred_scores[class_mask]
            
            if len(class_boxes) > 0:
                # Convert to tensor for NMS
                class_boxes_tensor = torch.from_numpy(class_boxes).cuda()
                class_scores_tensor = torch.from_numpy(class_scores).cuda()
                
                # Use class-specific NMS threshold, fallback to default
                class_threshold = class_nms_thresholds.get(class_id, nms_threshold)
                keep_idx, _ = iou3d_nms_utils.nms_gpu(class_boxes_tensor, class_scores_tensor, thresh=class_threshold)
                keep_idx = keep_idx.cpu().numpy()
                
                final_boxes.append(class_boxes[keep_idx])
                final_scores.append(class_scores[keep_idx])
                final_labels.append(np.full(len(keep_idx), class_id))
        
        if final_boxes:
            pred_boxes = np.concatenate(final_boxes, axis=0)
            pred_scores = np.concatenate(final_scores, axis=0)
            pred_labels = np.concatenate(final_labels, axis=0)
            
            # Apply cross-class NMS to handle same object detected as different classes
            if len(pred_boxes) > 1:
                boxes_tensor = torch.from_numpy(pred_boxes).cuda()
                scores_tensor = torch.from_numpy(pred_scores).cuda()
                
                # More aggressive cross-class NMS with lower threshold
                cross_class_threshold = cross_class_nms
                keep_idx, _ = iou3d_nms_utils.nms_gpu(boxes_tensor, scores_tensor, thresh=cross_class_threshold)
                keep_idx = keep_idx.cpu().numpy()
                
                pred_boxes = pred_boxes[keep_idx]
                pred_scores = pred_scores[keep_idx]
                pred_labels = pred_labels[keep_idx]
        else:
            pred_boxes = np.array([]).reshape(0, 7)
            pred_scores = np.array([])
            pred_labels = np.array([])
        
        # Prepare detections for tracker
        detections = []
        for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            # Extract box parameters: [x, y, z, l, w, h, rotation_y]
            x, y, z, l, w, h, ry = box
            
            # Model outputs center coordinates, but JSON format expects bottom center
            # So we need to adjust z coordinate to move from center to bottom
            z_bottom = z - h / 2.0
            
            # Map label index to object type
            if use_mapped_classes:
                # Mapped classes: ['Vehicle', 'Pedestrian', 'Static'] (3 classes)
                object_type_map = {
                    0: "car",        # Vehicle -> car for JSON format
                    1: "pedestrian", # Pedestrian
                    2: "static"      # Static objects
                }
            else:
                # Unmapped classes: 18 detailed classes
                object_type_map = {
                    0: "bicycle",
                    1: "bus", 
                    2: "car",
                    3: "car.golfcar",
                    4: "crane.armg",
                    5: "crane.quay",
                    6: "crane.rtg",
                    7: "eng_veh",
                    8: "eng_veh.construct",
                    9: "eng_veh.construct.ext",
                    10: "eng_veh.forklift",
                    11: "eng_veh.forklift.ext",
                    12: "motorcycle",
                    13: "pedestrian",
                    14: "tractor",
                    15: "traffic_cone",
                    16: "trailer",
                    17: "truck"
                }
            object_type = object_type_map.get(label - 1, "car")  # Subtract 1 if labels are 1-indexed
            
            detection = {
                "location": [float(x), float(y), float(z_bottom)],
                "rotation_y": float(ry),
                "dimension": [float(l), float(w), float(h)],
                "object": object_type,
                "score": float(score)
            }
            detections.append(detection)
        
        # Update tracker with detections
        if tracker is not None:
            tracked_detections = tracker.update(detections)
        else:
            # Fallback: assign sequential track IDs
            tracked_detections = detections
            for i, det in enumerate(tracked_detections):
                det['track_id'] = i
        
        # Convert to track format
        for detection in tracked_detections:
            track = {
                "track_id": detection['track_id'],
                "object": detection['object'],
                "location": detection['location'],
                "rotation_y": detection['rotation_y'],
                "dimension": detection['dimension'],
                "vehicle_activity": "stopped",
                "occlusion": "0-20"
            }
            
            # Add object-specific fields
            object_type = detection['object']
            if object_type in ["car", "tractor", "truck"]:
                track["door_opened"] = False
                if object_type == "car":
                    track["trunk_opened"] = False
            elif object_type == "trailer":
                track["has_container"] = False
                
            result["track"].append(track)
    
    return result


def find_bundle_directories(root_dir):
    """Find all subdirectories containing a gta.yaml file."""
    root_path = Path(root_dir)
    bundle_dirs = []
    
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            gta_file = subdir / 'gta.yaml'
            if gta_file.exists():
                bundle_dirs.append(subdir)
    
    return sorted(bundle_dirs)


def process_bundle(bundle_path, model, cfg, tracker, args, logger):
    """Process a single bundle directory."""
    # Get data and output paths
    data_path = bundle_path / 'pcd'
    if not data_path.exists():
        logger.warning(f"Data path {data_path} does not exist. Skipping bundle {bundle_path.name}.")
        return False
    
    output_dir = bundle_path / 'annotations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f'Processing bundle: {bundle_path.name}')
    logger.info(f'Data path: {data_path}')
    logger.info(f'Output directory: {output_dir}')
    
    # Create dataset
    inference_dataset = InferenceDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=data_path,
        logger=logger
    )
    
    if len(inference_dataset) == 0:
        logger.warning(f'No point cloud files found in {data_path}. Skipping bundle {bundle_path.name}.')
        return False
    
    logger.info(f'Found {len(inference_dataset)} point cloud files')
    
    # Reset tracker for new bundle
    tracker.tracks = {}
    tracker.next_track_id = 0
    tracker.frame_count = 0
    
    # Run inference
    with torch.no_grad():
        for idx in range(len(inference_dataset)):
            data_dict = inference_dataset[idx]
            frame_id = data_dict['frame_id']
            
            logger.info(f'Processing frame {frame_id} ({idx+1}/{len(inference_dataset)})')
            
            # Prepare batch
            data_dict = inference_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            
            # Forward pass
            pred_dicts, _ = model.forward(data_dict)
            predictions = pred_dicts[0]
            
            # Convert to JSON format
            result_json = convert_predictions_to_json(
                predictions, frame_id, 
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
                tracker=tracker,
                cross_class_nms=args.cross_class_nms,
                use_mapped_classes=args.use_mapped_classes
            )
            
            # Save result
            output_file = output_dir / f'{frame_id}.json'
            with open(output_file, 'w') as f:
                json.dump(result_json, f, indent=2)
    
    logger.info(f'Completed processing bundle: {bundle_path.name}')
    return True


def main():
    parser = argparse.ArgumentParser(description='Run inference on point cloud data')
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='Path to config file (e.g., tools/cfgs/lion_models/lion_mamba_venti3d_8x_1f_1x_one_stride_64dim.yaml)')
    parser.add_argument('--ckpt', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory containing subdirectories with gta.yaml files')
    parser.add_argument('--score_threshold', type=float, default=0.50,
                        help='Score threshold for filtering detections (default: 0.50 - top 30% based on score distribution analysis)')
    parser.add_argument('--nms_threshold', type=float, default=0.6,
                        help='NMS IoU threshold for removing duplicate detections (default: 0.6 - average of evaluation thresholds)')
    parser.add_argument('--cross_class_nms', type=float, default=0.5,
                        help='Cross-class NMS IoU threshold for same object different classes (default: 0.5)')
    parser.add_argument('--track_distance', type=float, default=5.0,
                        help='Maximum distance for track association (default: 5.0)')
    parser.add_argument('--max_frames_lost', type=int, default=5,
                        help='Maximum frames a track can be lost before deletion (default: 5)')
    parser.add_argument('--use_mapped_classes', action='store_true', default=True,
                        help='Use mapped classes (Vehicle/Pedestrian/Static) instead of unmapped classes')
    
    args = parser.parse_args()
    
    # Change to tools directory to handle relative paths in config files
    original_cwd = os.getcwd()
    os.chdir(tools_dir)
    
    try:
        # Load config - cfg_file path should be relative to tools directory
        cfg_from_yaml_file(args.cfg_file, cfg)
    finally:
        # Change back to original directory
        os.chdir(original_cwd)

    # Setup logging
    logger = common_utils.create_logger()
    logger.info('Starting point cloud inference...')
    logger.info(f'Root directory: {args.root_dir}')
    logger.info(f'Model checkpoint: {args.ckpt}')
    
    # Find bundle directories
    bundle_dirs = find_bundle_directories(args.root_dir)
    if not bundle_dirs:
        logger.error(f"No subdirectories with gta.yaml files found in {args.root_dir}")
        return
    
    logger.info(f'Found {len(bundle_dirs)} bundle directories: {[d.name for d in bundle_dirs]}')
    
    # Build and load model (do this once for all bundles)
    # Create a dummy dataset for model building
    dummy_data_path = bundle_dirs[0] / 'pcd'
    if not dummy_data_path.exists():
        logger.error(f"First bundle {bundle_dirs[0].name} does not have pcd directory")
        return
    
    dummy_dataset = InferenceDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=dummy_data_path,
        logger=logger
    )
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dummy_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    
    logger.info('Model loaded successfully')
    
    # Create tracker
    tracker = SimpleTracker(max_distance=args.track_distance, max_frames_lost=args.max_frames_lost)
    
    # Process each bundle directory
    successful_bundles = 0
    for i, bundle_path in enumerate(bundle_dirs):
        logger.info(f'\n=== Processing bundle {i+1}/{len(bundle_dirs)}: {bundle_path.name} ===')
        
        success = process_bundle(bundle_path, model, cfg, tracker, args, logger)
        if success:
            successful_bundles += 1
    
    logger.info(f'\n=== Inference completed! ===')
    logger.info(f'Successfully processed {successful_bundles}/{len(bundle_dirs)} bundles')


if __name__ == '__main__':
    main()