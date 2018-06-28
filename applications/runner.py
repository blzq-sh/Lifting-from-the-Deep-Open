import os
from os import path
import sys
import subprocess

IN_DIR = '/home/ben/Downloads/input_pose_videos'
OUT_PARENT_DIR = '/home/ben/Documents/output_pose_batch'

if __name__ == '__main__':
    in_dir = path.abspath(IN_DIR)
    out_parent_dir = path.abspath(OUT_PARENT_DIR)
    if not path.isdir(out_parent_dir):
        print("Parent directory does not exist")
        sys.exit(1)

    for in_file in os.listdir(in_dir):
        in_file_path = os.path.join(in_dir, in_file)
        if path.exists(in_file_path):
            out_child_dir = path.join(out_parent_dir,
                                      "{}-out/".format(in_file))
            os.mkdir(out_child_dir)
            subprocess.run(['python3',
                            '/home/ben/Lifting-from-the-Deep-release/'
                            'applications/demo-args.py',
                            '-i', in_file_path,
                            '-o', out_child_dir,
                            '-m', 'openpose',
                            '--track_one'])
        else:
            print("File not found: {}".format(in_file))
