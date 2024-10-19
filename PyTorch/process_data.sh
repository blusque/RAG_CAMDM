

# process bvh
python process_data.py -d data/bvh -o data/processed/bvh
# process pose data
python make_pose_data.py -r data/processed/bvh -o data/processed/pose