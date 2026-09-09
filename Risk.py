from pathlib import Path

import faostat as fao
import pandas as pd


YEARS = list(range(2019, 2024))
TOKEN_CANDIDATES = ["Token.txt", "token.txt"]
LOCAL_CSV = "Trade Indices.csv"
OUTPUT_CSV = "trade_indices_2019_2023_top5_bottom5_by_country.csv"


def load_token(base_dir: Path) -> str | None:
	"""Load FAOSTAT API token from Token.txt/token.txt, if available."""
	for name in TOKEN_CANDIDATES:
		token_path = base_dir / name
		if token_path.exists():
			token = token_path.read_text(encoding="utf-8").strip()
			if token:
				return token
	return None


def setup_faostat_auth(token: str | None) -> None:
	"""Configure faostat client, with token when available."""
	if token:
		fao.set_requests_args(token=token, lang="en", timeout=120)


def find_dataset_code_trade_indices() -> str:
	"""Find dataset code where metadata contains 'Trade Indices'."""
	datasets = fao.list_datasets_df()
	text_cols = [c for c in datasets.columns if datasets[c].dtype == object]
	if not text_cols:
		text_cols = list(datasets.columns)

	mask = pd.Series(False, index=datasets.index)
	for col in text_cols:
		mask = mask | datasets[col].astype(str).str.contains("Trade Indices", case=False, na=False)

	if not mask.any():
		raise RuntimeError("Could not locate 'Trade Indices' dataset in FAOSTAT metadata")

	hit = datasets.loc[mask].iloc[0]
	code_cols = [c for c in datasets.columns if "code" in str(c).lower()]
	if not code_cols:
		raise RuntimeError("Could not infer dataset code column from FAOSTAT metadata")
	return str(hit[code_cols[0]])


def fetch_trade_indices_df(dataset_code: str) -> pd.DataFrame:
	"""Download Trade Indices dataset from FAOSTAT."""
	api_filters = {
		"year": [str(y) for y in YEARS],
	}
	return fao.get_data_df(dataset_code, pars=api_filters, strval=True, limit=-1)


def load_local_trade_indices_csv(base_dir: Path) -> pd.DataFrame:
	"""Fallback loader for local Trade Indices CSV."""
	csv_path = base_dir / LOCAL_CSV
	if not csv_path.exists():
		raise FileNotFoundError(f"Local CSV not found: {csv_path}")
	return pd.read_csv(csv_path, dtype=str)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Normalize column names and fix common header typo."""
	out = df.copy()
	out.columns = [str(c).strip() for c in out.columns]
	if "omain Code" in out.columns and "Domain Code" not in out.columns:
		out = out.rename(columns={"omain Code": "Domain Code"})
	return out


def pick_required_columns(df: pd.DataFrame) -> tuple[str, str, str]:
	"""Pick year, value, and country columns from available schema."""
	year_col = "Year" if "Year" in df.columns else None
	value_col = "Value" if "Value" in df.columns else None

	if "Area" in df.columns:
		country_col = "Area"
	else:
		candidates = [c for c in df.columns if any(k in c.lower() for k in ["country", "area", "reporter"])]
		country_col = candidates[0] if candidates else None

	if not year_col or not value_col or not country_col:
		raise RuntimeError(
			"Could not detect required columns. Needed Year, Value, and country/area column"
		)

	return year_col, value_col, country_col


def top_bottom_by_country(df: pd.DataFrame) -> pd.DataFrame:
	"""Filter 2019-2023 then return top 5 and bottom 5 rows per country by Value."""
	year_col, value_col, country_col = pick_required_columns(df)

	work = df.copy()
	work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
	work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
	work = work[work[year_col].between(min(YEARS), max(YEARS), inclusive="both")]
	work = work.dropna(subset=[country_col, value_col])

	groups = []
	for country, g in work.groupby(country_col, dropna=True):
		g = g.sort_values(value_col)
		smallest = g.head(5).copy()
		smallest["RankType"] = "Smallest"
		smallest["RankWithinCountry"] = range(1, len(smallest) + 1)

		largest = g.tail(5).sort_values(value_col, ascending=False).copy()
		largest["RankType"] = "Largest"
		largest["RankWithinCountry"] = range(1, len(largest) + 1)

		groups.append(smallest)
		groups.append(largest)

	if not groups:
		return pd.DataFrame(columns=list(df.columns) + ["RankType", "RankWithinCountry"])

	result = pd.concat(groups, ignore_index=True)
	result = result.sort_values([country_col, "RankType", "RankWithinCountry"])
	return result


def main() -> None:
	base_dir = Path(__file__).resolve().parent

	token = load_token(base_dir)
	setup_faostat_auth(token)

	source = "FAOSTAT API"
	api_filters = {
		"year": [str(y) for y in YEARS],
	}
	try:
		dataset_code = find_dataset_code_trade_indices()
		raw = fetch_trade_indices_df(dataset_code)
	except Exception as exc:
		print(f"FAOSTAT API fetch failed ({exc}). Falling back to local CSV.")
		raw = load_local_trade_indices_csv(base_dir)
		source = "local CSV"

	raw = clean_columns(raw)
	result = top_bottom_by_country(raw)

	out_path = base_dir / OUTPUT_CSV
	result.to_csv(out_path, index=False)

	if token:
		print("Auth: token loaded from Token.txt/token.txt")
	else:
		print("Auth: no token found in Token.txt/token.txt; ran without explicit auth")

	print(f"Source: {source}")
	print(f"API filters requested: {api_filters}")
	print(f"Rows after filtering Year 2019-2023 and ranking by country: {len(result)}")
	print(f"Saved: {out_path}")
	if len(result) > 0:
		print(result.head(30).to_string(index=False))


if __name__ == "__main__":
	main()

