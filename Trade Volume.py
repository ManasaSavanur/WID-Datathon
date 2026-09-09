from pathlib import Path
import pandas as pd


TARGET_YEAR = 2023
INPUT = "Trade Matrix_Full.csv"
OUTPUT_FILE = "Trade_Matrix_2023.csv"



def main() -> None:
	base_dir = Path(__file__).resolve().parent
	input_path = base_dir / INPUT
	output_path = base_dir / OUTPUT_FILE

	df = pd.read_csv(input_path, low_memory=False)
	df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
	df_2023 = df[df["Year"] == TARGET_YEAR].copy()
	df_2023 = df_2023[~df_2023["Element"].astype(str).str.contains("quantity", case=False, na=False)].copy()
	df_2023.to_csv(output_path, index=False)


if __name__ == "__main__":
	main()


