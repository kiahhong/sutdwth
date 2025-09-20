import os
import pandas as pd
from utils import ptb_splits

def main():
    print("Splitting ptb_all.csv using the predefined strat_fold column...")
    
    # Define the path to your main preprocessed CSV
    csv_path = os.path.join( "processed_data", "ptb_all.csv")
    output_dir = os.path.join( "processed_data")

    # Load the preprocessed data
    df = pd.read_csv(csv_path)

    # Use the built-in function to split the data
    train_df, val_df, test_df = ptb_splits(df)

    # Save the split files
    train_df.to_csv(os.path.join(output_dir, "ptb_all_train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "ptb_all_val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "ptb_all_test.csv"), index=False)
    
    print("Splitting complete. Train, val, and test CSVs are saved in 'processed_data'.")

if __name__ == "__main__":
    main()