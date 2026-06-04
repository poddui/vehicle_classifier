import os
import random
import shutil

source_dir = 'data/train_all/articulated_truck'
already_moved_dir = 'data/train/truck_final'
target_dir = 'data/val/truck_val'

all_images = set(os.listdir(source_dir))
moved_images = set(os.listdir(already_moved_dir))

available_images = list(all_images - moved_images)

if len(available_images) < 100:
    print(f"Only {len(available_images)} images left. Cannot pick 100.")
else:
    selection = random.sample(available_images, 100)

    for file_name in selection:
        src_path = os.path.join(source_dir, file_name)
        dst_path = os.path.join(target_dir, file_name)
        shutil.copy2(src_path, dst_path)

    print(f"Successfully copied 100 unique images to {target_dir}")