import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_train_val_test_split(vga_data_path, vgr_data_path,
                                vga_save_path, vgr_save_path, 
                                train_size=0.7, random_state=42):
    """
    Create a train/val/test split from the given data paths.
    
    Args:
        vga_data_path (str): Path to the VGA data.
        vgr_data_path (str): Path to the VGR data.
        train_size (float): Proportion of the dataset to include in the train split.
    Returns:
        tuple: Paths for train, validation, and test datasets.
    """

    df_vga = pd.read_json(vga_data_path)
    df_vgr = pd.read_json(vgr_data_path)
    images = list(set(df_vga['image_id'].tolist() + df_vgr['image_id'].tolist()))

    images_train, images_val_test = train_test_split(images, test_size=1-train_size, random_state=random_state)
    images_val, images_test = train_test_split(images_val_test, test_size=0.5, random_state=random_state)

    print(f"Train/Val/Test split created with {len(images_train)} train, {len(images_val)} val, and {len(images_test)} test images.")

    train_df_vgr = df_vgr[df_vgr['image_id'].isin(images_train)]
    val_df_vgr = df_vgr[df_vgr['image_id'].isin(images_val)]
    test_df_vgr = df_vgr[df_vgr['image_id'].isin(images_test)]

    print (f"Train/Val/Test split created with {len(train_df_vgr)} train, {len(val_df_vgr)} val, and {len(test_df_vgr)} test samples in VGR dataset.")

    train_df_vga = df_vga[df_vga['image_id'].isin(images_train)]
    val_df_vga = df_vga[df_vga['image_id'].isin(images_val)]
    test_df_vga = df_vga[df_vga['image_id'].isin(images_test)]

    print (f"Train/Val/Test split created with {len(train_df_vga)} train, {len(val_df_vga)} val, and {len(test_df_vga)} test samples in VGA dataset.")

    train_df_vgr.to_json(os.path.join(vgr_save_path, 'train.json'))
    val_df_vgr.to_json(os.path.join(vgr_save_path, 'val.json'))
    test_df_vgr.to_json(os.path.join(vgr_save_path, 'test.json'))

    print(f"VGR datasets saved to {vgr_save_path}")

    train_df_vga.to_json(os.path.join(vga_save_path, 'train.json'))
    val_df_vga.to_json(os.path.join(vga_save_path, 'val.json'))
    test_df_vga.to_json(os.path.join(vga_save_path, 'test.json'))

    print(f"VGA datasets saved to {vga_save_path}")
