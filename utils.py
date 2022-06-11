import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import shutil
from glob import glob


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def train_test_val_split(images_clean, images_noise, split=(0.1, 0.1), shuffle=True, aim_path='/'):
    images_clean_list = sorted(glob(os.path.join(images_clean, '*.tif'), recursive=True))
    images_noise_list = sorted(glob(os.path.join(images_noise, '*.tif'), recursive=True))
        
    clean_train, clean_test_val, noise_train, noise_test_val = train_test_split(images_clean_list, images_noise_list, 
        test_size=(split[0]+split[1]), shuffle=shuffle)

    clean_test, clean_val, noise_test, noise_val = train_test_split(clean_test_val, noise_test_val, 
        test_size=split[0], shuffle=False)

    aim_train_clean = os.path.join(aim_path, "train", "clean")
    aim_train_noise = os.path.join(aim_path, "train", "noise")
    
    aim_test_clean = os.path.join(aim_path, "test", "clean")
    aim_test_noise = os.path.join(aim_path, "test", "noise")

    aim_val_clean = os.path.join(aim_path, "val", "clean")
    aim_val_noise = os.path.join(aim_path, "val", "noise")
    
    print("*"*50)
    print("INFO: train/test/val split dataset")
    print("train clean: ", len(clean_train))
    print("train noise: ", len(noise_train))
    print("test clean: ", len(clean_test))
    print("test noise: ", len(noise_test))
    print("val clean: ", len(clean_val))
    print("val noise: ", len(noise_val))
    print("*"*50)
    
    for im_cl, im_ns in zip(sorted(clean_train), sorted(noise_train)):
        shutil.move(im_cl, aim_train_clean)
        shutil.move(im_ns, aim_train_noise)
    
    for im_cl, im_ns in zip(sorted(clean_test), sorted(noise_test)):
        shutil.move(im_cl, aim_test_clean)
        shutil.move(im_ns, aim_test_noise)

    for im_cl, im_ns in zip(sorted(clean_val), sorted(noise_val)):
        shutil.move(im_cl, aim_val_clean)
        shutil.move(im_ns, aim_val_noise)
