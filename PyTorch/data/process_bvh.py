'''
Target: 
1. Remove the unused joints
2. Make the root position to be (0, 0, 0)
3. Align the root forwarding to x-axis
'''

import sys
sys.path.append('./')

import os
import utils.motion_modules as motion_modules
import style_helper as style100

from multiprocessing import Pool
from utils.bvh_motion import Motion

import argparse

NEEDED_STYLE = ['neutral']

parser = argparse.ArgumentParser(description='### Process BVH files')
parser.add_argument('-d', '--dir', default='data/100STYLE_mixamo/raw', type=str, help='The directory of BVH files')
parser.add_argument('-o', '--output', default='data/100STYLE_mixamo/simple', type=str, help='The output directory of processed BVH files')

def process_bvh(bvh_path, output_path):
    raw_motion = Motion.load_bvh(bvh_path)
    raw_motion.offsets[0].fill(0)
    motion = motion_modules.remove_joints(raw_motion, ['eye', 'index', 'middle', 'ring', 'pinky', 'thumb'])
    motion = motion_modules.root(motion)
    motion = motion_modules.temporal_scale(motion, 2)
    forward_angle = motion_modules.extract_forward(motion, 0, style100.left_shoulder_name, style100.right_shoulder_name, style100.left_hip_name, style100.right_hip_name)
    motion = motion_modules.rotate(motion, given_angle=forward_angle, axis='y')
    motion = motion_modules.on_ground(motion)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    motion.export(output_path)
    print('Finish processing %s' % bvh_path)


if __name__ == '__main__':
    cfg = parser.parse_args()
    process_folder = cfg.dir
    output_folder = cfg.output

    bvh_paths = []
    for root, dirs, files in os.walk(process_folder):
        for file in files:
            style_name = file.lower().split('_')[0]
            if file.endswith('.bvh') and style_name in NEEDED_STYLE:
                bvh_paths.append(os.path.join(root, file))
                
    output_paths = [f.replace(process_folder, output_folder) for f in bvh_paths]
    
    print('Total BVH files: %d' % len(bvh_paths))
    pool = Pool(processes=os.cpu_count()) 
        
    tasks = []
    for idx in range(len(bvh_paths)):
        bvh_path = bvh_paths[idx]
        output_path = output_paths[idx]
        pool.apply_async(process_bvh, args=(bvh_path, output_path))

    pool.close()  
    pool.join() 
