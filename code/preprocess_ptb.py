import argparse
import os
import pandas as pd
from utils import prepare_data_ptb_xl, ptb_splits # Import the splitter function

def main():
    parser = argparse.ArgumentParser(description="Preprocess and split the PTB-XL dataset.")
    parser.add_argument("raw_data_dir", type=str, help="Path to the root of the raw PTB-XL dataset (containing ptbxl_database.csv).")
    parser.add_argument("output_dir", type=str, help="Path to the directory to save .npy files and final CSV splits.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # --- Step 1: Process raw signals into .npy files and create master CSVs ---
    print("Starting PTB-XL data preparation (creating .npy files and master CSVs)...")
    prepare_data_ptb_xl(
        path_to_ptb_xl=args.raw_data_dir,
        path_to_out_folder=args.output_dir
    )
    print("Finished creating .npy files and master CSVs.")

    # --- Step 2: Split the master CSV into train, validation, and test sets ---
    print("\nSplitting the master CSV file into train, val, and test sets...")
    master_csv_path = os.path.join(args.output_dir, "ptb_all.csv")
    
    if not os.path.exists(master_csv_path):
        print(f"Error: Master CSV file not found at {master_csv_path}. Aborting split.")
        return

    df = pd.read_csv(master_csv_path)
    train_df, val_df, test_df = ptb_splits(df)

    # Save the split files
    train_df.to_csv(os.path.join(args.output_dir, "ptb_all_train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "ptb_all_val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "ptb_all_test.csv"), index=False)
    
    print("Finished splitting data.")
    print(f"\nPreprocessing complete. All files are saved in {args.output_dir}")

if __name__ == "__main__":
    main()