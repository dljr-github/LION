#!/usr/bin/env python3

import pickle
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import os


def create_venti3d_infos(root_path, save_path, split='train', use_class_mapping=True):
    """
    Create dataset info for Venti3D dataset in OpenPCDet format with optional class mapping.
    
    Args:
        root_path: Path to the Venti3D dataset
        save_path: Path to save the info file
        split: train/val/test
        use_class_mapping: Whether to apply class mapping (default: True)
    """
    # Class mapping from original to final classes
    class_mapping = {
        'bicycle': 'Vehicle',
        'bus': 'Vehicle', 
        'car': 'Vehicle',
        'car.golfcar': 'Vehicle',
        'crane.armg': 'Static',
        'crane.quay': 'Static',
        'crane.rtg': 'Static',
        'eng_veh': 'Vehicle',
        'eng_veh.construct': 'Vehicle',
        'eng_veh.construct.ext': 'Vehicle',
        'eng_veh.forklift': 'Vehicle',
        'eng_veh.forklift.ext': 'Vehicle',
        'motorcycle': 'Vehicle',
        'pedestrian': 'Pedestrian',
        'tractor': 'Vehicle',
        'traffic_cone': 'Static',
        'trailer': 'Vehicle',
        'truck': 'Vehicle'
    }
    dataset_path = Path(root_path)
    
    # Read the split file
    split_file = dataset_path / 'ImageSets' / f'{split}.txt'
    with open(split_file, 'r') as f:
        frame_ids = [line.strip() for line in f.readlines()]
    
    print(f'Creating {split} infos for {len(frame_ids)} frames')
    
    venti3d_infos = []
    
    for frame_id in tqdm(frame_ids, desc=f'Processing {split} frames'):
        # Load point cloud
        points_file = dataset_path / 'points' / f'{frame_id}.npy'
        
        if not points_file.exists():
            print(f"Warning: Points file {points_file} does not exist, skipping")
            continue
            
        points = np.load(points_file)
        
        # Load labels
        labels_file = dataset_path / 'labels' / f'{frame_id}.txt'
        
        # Initialize annotations
        annotations = {
            'name': [],
            'gt_boxes_lidar': [],
            'num_points_in_gt': []
        }
        
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 8:
                            # Parse: x y z dx dy dz heading_angle category_name
                            x, y, z, dx, dy, dz, heading, category = parts[:8]
                            
                            # Map original class to final class if enabled
                            if use_class_mapping:
                                if category in class_mapping:
                                    mapped_category = class_mapping[category]
                                    annotations['name'].append(mapped_category)
                                else:
                                    print(f"Warning: Unknown class '{category}', skipping")
                                    continue
                            else:
                                # Use original class name
                                annotations['name'].append(category)
                            
                            # Create gt_boxes_lidar format: [x, y, z, dx, dy, dz, heading]
                            gt_box = [float(x), float(y), float(z), float(dx), float(dy), float(dz), float(heading)]
                            annotations['gt_boxes_lidar'].append(gt_box)
                            
                            # Count points in gt box (rough estimate)
                            center = np.array([float(x), float(y), float(z)])
                            dims = np.array([float(dx), float(dy), float(dz)])
                            
                            # Simple box filtering to count points
                            points_in_box = points[
                                (points[:, 0] >= center[0] - dims[0]/2) &
                                (points[:, 0] <= center[0] + dims[0]/2) &
                                (points[:, 1] >= center[1] - dims[1]/2) &
                                (points[:, 1] <= center[1] + dims[1]/2) &
                                (points[:, 2] >= center[2] - dims[2]/2) &
                                (points[:, 2] <= center[2] + dims[2]/2)
                            ]
                            annotations['num_points_in_gt'].append(len(points_in_box))
        
        # Convert to numpy arrays
        for key in annotations:
            if len(annotations[key]) > 0:
                if key in ['gt_boxes_lidar']:
                    annotations[key] = np.array(annotations[key], dtype=np.float32)
                elif key in ['num_points_in_gt']:
                    annotations[key] = np.array(annotations[key], dtype=np.int32)
                else:
                    annotations[key] = np.array(annotations[key])
            else:
                # Empty arrays
                if key in ['gt_boxes_lidar']:
                    annotations[key] = np.zeros((0, 7), dtype=np.float32)
                elif key in ['num_points_in_gt']:
                    annotations[key] = np.zeros(0, dtype=np.int32)
                else:
                    annotations[key] = np.array([])
        
        # Create info dict
        info = {
            'point_cloud': {
                'num_features': 4,
                'lidar_idx': frame_id,
                'velodyne_path': f'points/{frame_id}.npy'
            },
            'frame_id': frame_id,
            'annos': annotations
        }
        
        venti3d_infos.append(info)
    
    # Save the infos
    save_file = Path(save_path) / f'venti3d_infos_{split}.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(venti3d_infos, f)
    
    print(f'Saved {len(venti3d_infos)} infos to {save_file}')
    return venti3d_infos


def create_venti3d_gt_database(infos, root_path, save_path, split='train', used_classes=None, use_class_mapping=True):
    """
    Create GT database for data augmentation using mapped or original classes.
    """
    if used_classes is None:
        if use_class_mapping:
            used_classes = ['Vehicle', 'Pedestrian', 'Static']
        else:
            # Use all unique classes from infos
            used_classes = set()
            for info in infos:
                used_classes.update(info['annos']['name'])
            used_classes = list(used_classes)
    
    dataset_path = Path(root_path)
    save_path = Path(save_path)
    
    # Create database directory
    database_dir = save_path / 'gt_database'
    database_dir.mkdir(exist_ok=True)
    
    db_infos = {cls: [] for cls in used_classes}
    
    for info in tqdm(infos, desc=f'Creating GT database for {split}'):
        frame_id = info['frame_id']
        
        # Load points
        points_file = dataset_path / 'points' / f'{frame_id}.npy'
        points = np.load(points_file)
        
        annotations = info['annos']
        
        if len(annotations['name']) == 0:
            continue
            
        for i, name in enumerate(annotations['name']):
            if name not in used_classes:
                continue
                
            # Extract box info
            gt_box = annotations['gt_boxes_lidar'][i]  # [x, y, z, dx, dy, dz, heading]
            location = gt_box[:3]
            dimensions = gt_box[3:6]
            rotation_y = gt_box[6]
            num_points = annotations['num_points_in_gt'][i]
            
            if num_points < 5:  # Skip boxes with too few points
                continue
            
            # Extract points in box
            center = location
            dims = dimensions
            
            # Simple box filtering
            mask = (
                (points[:, 0] >= center[0] - dims[0]/2) &
                (points[:, 0] <= center[0] + dims[0]/2) &
                (points[:, 1] >= center[1] - dims[1]/2) &
                (points[:, 1] <= center[1] + dims[1]/2) &
                (points[:, 2] >= center[2] - dims[2]/2) &
                (points[:, 2] <= center[2] + dims[2]/2)
            )
            
            box_points = points[mask]
            
            if len(box_points) < 5:
                continue
            
            # Transform points to box coordinate system
            box_points[:, 0] -= center[0]
            box_points[:, 1] -= center[1] 
            box_points[:, 2] -= center[2]
            
            # Save to database
            db_filename = f'{frame_id}_{name}_{i}.bin'
            db_filepath = database_dir / db_filename
            box_points.astype(np.float32).tofile(db_filepath)
            
            # Add to db_infos
            db_info = {
                'name': name,
                'path': f'gt_database/{db_filename}',
                'gt_idx': i,
                'box3d_lidar': gt_box,
                'num_points_in_gt': len(box_points),
                'difficulty': 0
            }
            
            db_infos[name].append(db_info)
    
    # Save db_infos
    db_info_file = save_path / f'venti3d_dbinfos_{split}.pkl'
    with open(db_info_file, 'wb') as f:
        pickle.dump(db_infos, f)
    
    # Print statistics
    for cls, infos in db_infos.items():
        print(f'{cls}: {len(infos)} objects')
    
    print(f'GT database saved to {database_dir}')
    print(f'DB infos saved to {db_info_file}')
    
    return db_infos


def main():
    parser = argparse.ArgumentParser(description='Create Venti3D dataset info files')
    parser.add_argument('--root_path', required=True, help='Path to Venti3D dataset')
    parser.add_argument('--save_path', help='Path to save info files (default: same as root_path)')
    parser.add_argument('--disable_class_mapping', action='store_true', help='Disable class mapping and use original class names')
    
    args = parser.parse_args()
    
    root_path = args.root_path
    save_path = args.save_path if args.save_path else root_path
    use_class_mapping = not args.disable_class_mapping
    
    # Create train infos
    train_infos = create_venti3d_infos(root_path, save_path, 'train', use_class_mapping)
    
    # Create val infos  
    val_infos = create_venti3d_infos(root_path, save_path, 'val', use_class_mapping)
    
    # Create GT database for training
    create_venti3d_gt_database(train_infos, root_path, save_path, 'train', use_class_mapping=use_class_mapping)
    
    print('Dataset creation completed!')


if __name__ == '__main__':
    main()