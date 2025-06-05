import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
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

    # make sure the save directories exist
    os.makedirs(vgr_save_path, exist_ok=True)
    os.makedirs(vga_save_path, exist_ok=True)

    train_df_vgr.to_json(os.path.join(vgr_save_path, 'train.json'), orient='records')
    val_df_vgr.to_json(os.path.join(vgr_save_path, 'val.json'), orient='records')
    test_df_vgr.to_json(os.path.join(vgr_save_path, 'test.json'), orient='records')

    print(f"VGR datasets saved to {vgr_save_path}")

    train_df_vga.to_json(os.path.join(vga_save_path, 'train.json'), orient='records')
    val_df_vga.to_json(os.path.join(vga_save_path, 'val.json'), orient='records')
    test_df_vga.to_json(os.path.join(vga_save_path, 'test.json'), orient='records')

    print(f"VGA datasets saved to {vga_save_path}")


class ARODataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 text_transform,
                 progress = True):
        self.dataset = pd.read_json(data_path)
        self.progress = progress

        required_columns = ['true_caption', 'false_caption', 'image_path']
        if not all(col in self.dataset.columns for col in required_columns):
            raise ValueError(f"ARODataset must contain the following columns: {required_columns}")

        self.precompute_text(text_transform)
        del text_transform

    def precompute_text(self, text_transform):
        # build a map from text to einsum inputs
        self.text_map = {}
        symbol_size_set = set()

        captions = list(set(self.dataset['true_caption'].tolist() + self.dataset['false_caption'].tolist()))
        batch_size = 6

        for i in tqdm(range(0, len(captions), batch_size), disable=not self.progress):
            batch_captions = captions[i:i + batch_size]

            einsum_inputs = text_transform(batch_captions, suppress_exceptions=True)['einsum_inputs']
            for caption, einsum_input in zip(batch_captions, einsum_inputs):
                self.text_map[caption] = einsum_input
                if einsum_input is not None:
                    _, symbol_size_list = einsum_input
                    for symbol, size in symbol_size_list:
                        symbol_size_set.add((symbol, size))
        del text_transform 
        self.symbols, self.sizes = zip(*symbol_size_set) if symbol_size_set else ([], [])
                    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset.iloc[idx]['image_path']

        true_caption = self.dataset.iloc[idx]['true_caption']
        false_caption = self.dataset.iloc[idx]['false_caption']

        true_caption = self.text_map.get(true_caption, None)
        false_caption = self.text_map.get(false_caption, None)

        def remove_shape(einsum_input):
            if einsum_input is None:
                return None
            einsum_expr, symbol_size_list = einsum_input
            return (einsum_expr, [sym for sym, _ in symbol_size_list])
        true_caption = remove_shape(true_caption)
        false_caption = remove_shape(false_caption)

        return {
            "image": image,
            "true_caption": true_caption,
            "false_caption": false_caption,
            "index": idx
        }

def aro_tn_collate_fn(batch):
    valid = [item['true_caption'] is not None and item['false_caption'] is not None for item in batch]
    images = [item['image'] for item, v in zip(batch, valid) if v]
    true_captions = [item['true_caption'] for item, v in zip(batch, valid) if v]
    false_captions = [item['false_caption'] for item, v in zip(batch, valid) if v]
    indices = [item['index'] for item, v in zip(batch, valid) if v]

    return {
        "images": images,
        "true_captions": true_captions,
        "false_captions": false_captions,
        "indices": indices
    }