import os
import shutil
from tqdm import tqdm

def create_imagenet100():
    '''
    Before running this code, be sure to first download the list files:
    https://github.com/Continvvm/continuum/releases/download/v0.1/train_100.txt
    https://github.com/Continvvm/continuum/releases/download/v0.1/val_100.txt

    Note that you should already downloaded and extracted the ImageNet dataset 
    before you run this code.
    '''

    base_dir = "/path/to/imagenet/containing/val/and/train"
    validation_label_txt = "path/to/val_100.txt"
    train_label_txt = "/path/to/train_100.txt"
    target_folder = "/path/to/target/imagenet100/folder"
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(os.path.join(target_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(target_folder, "val"), exist_ok=True)
    with open(validation_label_txt, "r") as f_in, open(train_label_txt, "r") as f_in_2:
        validation_lines = [item.split(" ")[0] for item in f_in.readlines()]
        train_lines = [item.split(" ")[0] for item in f_in_2.readlines()]
        print("validation set")
        for item in tqdm(validation_lines):
            tmp_folder = os.path.join(target_folder, os.path.dirname(item))
            os.makedirs(tmp_folder, exist_ok=True)
            shutil.copyfile(os.path.join(base_dir, item), os.path.join(target_folder, item))
        print("train set")
        for item in tqdm(train_lines):
            tmp_folder = os.path.join(target_folder, os.path.dirname(item))
            os.makedirs(tmp_folder, exist_ok=True)
            shutil.copyfile(os.path.join(base_dir, item), os.path.join(target_folder, item))

    print("completed")

create_imagenet100()