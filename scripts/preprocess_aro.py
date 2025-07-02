import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

from discoclip.models import Tokenizer

def main(args):
    # Load data
    df_vgr = pd.read_json(os.path.join(args.input_dir, 'visual_genome_relation.json'))
    df_vga = pd.read_json(os.path.join(args.input_dir, 'visual_genome_attribution.json'))

    # Split images
    images = list(set(df_vga['image_id'].tolist() + df_vga['image_id'].tolist()))
    images_train, images_test = train_test_split(images, test_size=args.test_size, random_state=args.random_state)
    images_val, images_test = train_test_split(images_test, test_size=args.val_size, random_state=args.random_state)

    # Tokenize
    tokenizer = Tokenizer()
    df_vgr['true_caption_lemmas'] = df_vgr['true_caption'].apply(tokenizer)
    df_vga['true_caption_lemmas'] = df_vga['true_caption'].apply(tokenizer)
    df_vgr['false_caption_lemmas'] = df_vgr['false_caption'].apply(tokenizer)
    df_vga['false_caption_lemmas'] = df_vga['false_caption'].apply(tokenizer)

    # Create dataframes
    train_df_vgr = df_vgr[df_vgr['image_id'].isin(images_train)]
    val_df_vgr = df_vgr[df_vgr['image_id'].isin(images_val)]
    test_df_vgr = df_vgr[df_vgr['image_id'].isin(images_test)]

    train_df_vga = df_vga[df_vga['image_id'].isin(images_train)]
    val_df_vga = df_vga[df_vga['image_id'].isin(images_val)]
    test_df_vga = df_vga[df_vga['image_id'].isin(images_test)]

    # Print stats
    print(f'Train VGR size: {len(train_df_vgr)}')
    print(f'Validation VGR size: {len(val_df_vgr)}')
    print(f'Test VGR size: {len(test_df_vgr)}')

    print(f'Train VGA size: {len(train_df_vga)}')
    print(f'Validation VGA size: {len(val_df_vga)}')
    print(f'Test VGA size: {len(test_df_vga)}')

    # Save dataframes
    os.makedirs(args.output_dir, exist_ok=True)
    train_df_vgr.to_json(os.path.join(args.output_dir, 'train_vgr.json'), orient='records')
    val_df_vgr.to_json(os.path.join(args.output_dir, 'val_vgr.json'), orient='records')
    test_df_vgr.to_json(os.path.join(args.output_dir, 'test_vgr.json'), orient='records')
    train_df_vga.to_json(os.path.join(args.output_dir, 'train_vga.json'), orient='records')
    val_df_vga.to_json(os.path.join(args.output_dir, 'val_vga.json'), orient='records')
    test_df_vga.to_json(os.path.join(args.output_dir, 'test_vga.json'), orient='records')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='data/raw/aro')
    parser.add_argument('--output-dir', type=str, default='data/processed/aro')
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--val-size', type=float, default=0.5)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()
    main(args)
