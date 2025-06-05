import os
from argparse import ArgumentParser
from discoclip.data.aro_dataset import create_train_val_test_split

def parse_args():
    parser = ArgumentParser(description="Split ARO files into individual components.")
    parser.add_argument("--attribution-data-path", 
                        type=str, 
                        default="data/raw/aro/visual_genome_attribution.json",
                        help="Path to the input ARO file.")
    parser.add_argument("--relation-data-path",
                        type=str, 
                        default="data/raw/aro/visual_genome_relation.json",
                        help="Path to the input ARO file.")
    parser.add_argument("--attribution-output-dir",
                        type=str, 
                        default="data/processed/aro/visual_genome_attribution",
                        help="Directory to save the attribution data.")
    parser.add_argument("--relation-output-dir",
                        type=str, 
                        default="data/processed/aro/visual_genome_relation",
                        help="Directory to save the relation data.")
    return parser.parse_args()

def main():
    args = parse_args()
    create_train_val_test_split(vga_data_path=args.attribution_data_path,
                                vgr_data_path=args.relation_data_path,
                                vga_save_path=args.attribution_output_dir,
                                vgr_save_path=args.relation_output_dir)

if __name__ == "__main__":
    main()
