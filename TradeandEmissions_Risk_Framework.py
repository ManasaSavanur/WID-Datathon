from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TRADE_FILE_CANDIDATES = [
	"Trade_Matrix_2023_app.csv.gz",
	"Trade_Matrix_2023_app.csv",
	"Trade_Matrix_2023.csv",
	"Trade_Matrix_2023_ms.csv",
	"Trade Matrix_2023.csv",
	"Trade Matrix_Full.csv",
	"Trade Matrix_All.csv",
]
EMISSIONS_FILE_CANDIDATES = [
	"Emissions Intensities_2023.csv",
	"Emissions_Intensities_2023.csv",
	"emissions_intensities_2023.csv",
]
VULN_FILE_CANDIDATES = [
	"country_vulnerability_concentration_index.csv",
	"Country_Vulnerability_Concentration_Index.csv",
]

OUT_OVERLAP = BASE_DIR / "trade_emissions_overlap_2023.csv"
OUT_ACTIONS = BASE_DIR / "trade_emissions_priority_actions_2023.csv"


def normalize_filename(name: str) -> str:
	return "".join(ch for ch in name.lower() if ch.isalnum())


def resolve_data_file(base_dir: Path, candidates: list[str], label: str) -> Path:
	# Exact name match first.
	for candidate in candidates:
		path = base_dir / candidate
		if path.exists():
			return path

	# Relaxed match: ignore spacing/case/underscores.
	normalized_targets = {normalize_filename(c) for c in candidates}
	for path in base_dir.iterdir():
		if path.is_file() and normalize_filename(path.name) in normalized_targets:
			return path

	available_csv = sorted(
		[
			path.name
			for path in base_dir.iterdir()
			if path.is_file() and path.name.lower().endswith((".csv", ".csv.gz"))
		]
	)
	raise FileNotFoundError(
		f"Missing {label}. Tried: {candidates}. Available CSV files: {available_csv}"
	)


def normalize_m49(series: pd.Series) -> pd.Series:
	cleaned = series.astype(str).str.replace(r"[^0-9]", "", regex=True)
	cleaned = cleaned.replace("", np.nan)
	return cleaned.str.zfill(3)


def clean_numeric(series: pd.Series) -> pd.Series:
	return pd.to_numeric(
		series.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False).replace("-", np.nan),
		errors="coerce",
	)


def minmax(series: pd.Series) -> pd.Series:
	s = series.astype(float)
	mn, mx = s.min(skipna=True), s.max(skipna=True)
	if pd.isna(mn) or pd.isna(mx) or mx == mn:
		return pd.Series(0.0, index=s.index)
	return (s - mn) / (mx - mn)


def load_vulnerability(path: Path) -> pd.DataFrame:
	v = pd.read_csv(path, low_memory=False)
	v.columns = [str(c).strip() for c in v.columns]

	rename_map = {
		"reporter_code": "m49",
		"country": "country",
		"vulnerability_index": "vulnerability_index",
		"concentration_index": "concentration_index",
		"exposure_index": "trade_exposure_index",
		"import_dependence": "import_dependence",
		"net_import_exposure": "net_import_exposure",
		"hhi_partner": "hhi_partner",
		"hhi_item": "hhi_item",
	}

	for col in list(v.columns):
		trimmed = col.strip()
		if trimmed in rename_map:
			v = v.rename(columns={col: rename_map[trimmed]})

	# In case CSV had spaced column names, force-clean known numerics.
	numeric_cols = [
		"vulnerability_index",
		"concentration_index",
		"trade_exposure_index",
		"import_dependence",
		"net_import_exposure",
		"hhi_partner",
		"hhi_item",
	]
	for c in numeric_cols:
		if c in v.columns:
			v[c] = clean_numeric(v[c])

	v["m49"] = normalize_m49(v["m49"])
	v = v.dropna(subset=["m49"]).copy()
	v = v.sort_values("trade_exposure_index", ascending=False)
	return v


def load_emissions(path: Path) -> pd.DataFrame:
	e = pd.read_csv(path, low_memory=False)
	e.columns = [str(c).strip() for c in e.columns]
	e["Year"] = clean_numeric(e["Year"])
	e["Value"] = clean_numeric(e["Value"])

	e = e[e["Year"] == 2023].copy()
	e = e[e["Element"].astype(str).str.strip().eq("Emissions intensity")].copy()

	e["partner_m49"] = normalize_m49(e["Area Code (M49)"])
	e["item_cpc"] = e["Item Code (CPC)"].astype(str).str.replace("'", "", regex=False).str.strip()
	e = e.rename(columns={"Value": "emissions_intensity", "Area": "partner_country", "Item": "item_name"})

	e = e[["partner_m49", "item_cpc", "emissions_intensity", "partner_country", "item_name"]]
	e = e.dropna(subset=["partner_m49", "item_cpc", "emissions_intensity"])
	return e


def load_trade(path: Path) -> pd.DataFrame:
	t = pd.read_csv(path, low_memory=False)
	t.columns = [str(c).strip() for c in t.columns]

	t["Year"] = clean_numeric(t["Year"])
	t["Value"] = clean_numeric(t["Value"])

	t = t[t["Year"] == 2023].copy()
	t = t[t["Element"].astype(str).str.strip().eq("Import value")].copy()

	t["reporter_m49"] = normalize_m49(t["Reporter Country Code (M49)"])
	t["partner_m49"] = normalize_m49(t["Partner Country Code (M49)"])
	t["item_cpc"] = t["Item Code (CPC)"].astype(str).str.replace("'", "", regex=False).str.strip()
	t = t.rename(
		columns={
			"Reporter Countries": "reporter_country",
			"Partner Countries": "partner_country",
			"Item": "trade_item_name",
			"Value": "import_value",
		}
	)
	t = t.dropna(subset=["reporter_m49", "partner_m49", "item_cpc", "import_value"])
	return t


def emissions_exposure_metrics(trade: pd.DataFrame, emissions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	m = trade.merge(emissions, on=["partner_m49", "item_cpc"], how="left", suffixes=("", "_e"))
	m["has_emissions"] = m["emissions_intensity"].notna()
	m["weighted_emissions"] = m["import_value"] * m["emissions_intensity"]
	m["resolved_item_name"] = m["item_name"].fillna(m.get("trade_item_name"))

	coverage = m.groupby("reporter_m49", as_index=False).agg(
		import_value_total=("import_value", "sum"),
		import_value_with_emissions=("import_value", lambda x: x[m.loc[x.index, "has_emissions"]].sum()),
	)
	coverage["emissions_data_coverage"] = np.where(
		coverage["import_value_total"] > 0,
		coverage["import_value_with_emissions"] / coverage["import_value_total"],
		0.0,
	)

	stats = m.groupby("reporter_m49", as_index=False).agg(
		embodied_emissions_value=("weighted_emissions", "sum"),
		import_value_total=("import_value", "sum"),
		reporter_country=("reporter_country", "first"),
	)
	stats["trade_weighted_emissions_intensity"] = np.where(
		stats["import_value_total"] > 0,
		stats["embodied_emissions_value"] / stats["import_value_total"],
		np.nan,
	)

	# Share of import value sourced from high-emission supply links.
	threshold = m["emissions_intensity"].quantile(0.75)
	m["high_emissions_link"] = m["emissions_intensity"] >= threshold
	high_share = m.groupby("reporter_m49", as_index=False).agg(
		import_value_high_emissions=("import_value", lambda x: x[m.loc[x.index, "high_emissions_link"].fillna(False)].sum()),
		import_value_total=("import_value", "sum"),
	)
	high_share["high_emissions_import_share"] = np.where(
		high_share["import_value_total"] > 0,
		high_share["import_value_high_emissions"] / high_share["import_value_total"],
		0.0,
	)

	out = stats.merge(coverage[["reporter_m49", "emissions_data_coverage"]], on="reporter_m49", how="left")
	out = out.merge(high_share[["reporter_m49", "high_emissions_import_share"]], on="reporter_m49", how="left")
	out["emissions_data_coverage"] = out["emissions_data_coverage"].fillna(0.0)
	out["high_emissions_import_share"] = out["high_emissions_import_share"].fillna(0.0)
	return out, m


def _finalize_overlap(vulnerability: pd.DataFrame, emissions_metrics: pd.DataFrame) -> pd.DataFrame:
	emissions_metrics = emissions_metrics.rename(columns={"reporter_m49": "m49"})

	df = vulnerability.merge(emissions_metrics, on="m49", how="left")
	if "reporter_country" in df.columns:
		df["country"] = df["country"].fillna(df["reporter_country"])

	df["trade_weighted_emissions_intensity"] = df["trade_weighted_emissions_intensity"].fillna(df["trade_weighted_emissions_intensity"].median(skipna=True))
	df["high_emissions_import_share"] = df["high_emissions_import_share"].fillna(0.0)
	df["emissions_data_coverage"] = df["emissions_data_coverage"].fillna(0.0)

	df["emissions_intensity_n"] = minmax(df["trade_weighted_emissions_intensity"])
	df["high_emissions_share_n"] = minmax(df["high_emissions_import_share"])

	# Penalize low coverage by shrinking confidence on emissions metrics.
	df["emissions_exposure_index"] = (
		0.7 * df["emissions_intensity_n"] + 0.3 * df["high_emissions_share_n"]
	) * (0.5 + 0.5 * df["emissions_data_coverage"]) 

	df["overlap_index"] = 0.6 * df["vulnerability_index"] + 0.4 * df["emissions_exposure_index"]

	vuln_cut = df["vulnerability_index"].quantile(0.75)
	em_cut = df["emissions_exposure_index"].quantile(0.75)

	df["high_vulnerability"] = df["vulnerability_index"] >= vuln_cut
	df["high_emissions"] = df["emissions_exposure_index"] >= em_cut

	df["overlap_tier"] = np.select(
		[
			df["high_vulnerability"] & df["high_emissions"],
			df["high_vulnerability"] & ~df["high_emissions"],
			~df["high_vulnerability"] & df["high_emissions"],
		],
		["Critical overlap", "High vulnerability", "High emissions"],
		default="Moderate/Low",
	)

	df["priority_action"] = df.apply(assign_priority_action, axis=1)
	df = df.sort_values("overlap_index", ascending=False).reset_index(drop=True)
	return df


def assign_priority_action(row: pd.Series) -> str:
	if row["overlap_tier"] == "Critical overlap":
		if row["concentration_index"] >= 0.66:
			return "Diversify suppliers and launch low-emissions procurement standards immediately"
		return "Prioritize low-emissions substitution and strategic food buffers"
	if row["overlap_tier"] == "High vulnerability":
		return "Strengthen resilience: diversify import partners, stocks, and shock monitoring"
	if row["overlap_tier"] == "High emissions":
		return "Decarbonize food import basket through supplier standards and product switching"
	return "Maintain monitoring and incremental efficiency improvements"


def build_overlap_framework() -> pd.DataFrame:
	vulnerability = load_vulnerability(resolve_data_file(BASE_DIR, VULN_FILE_CANDIDATES, "vulnerability file"))
	emissions = load_emissions(resolve_data_file(BASE_DIR, EMISSIONS_FILE_CANDIDATES, "emissions file"))
	trade = load_trade(resolve_data_file(BASE_DIR, TRADE_FILE_CANDIDATES, "trade matrix file"))

	emissions_metrics, _ = emissions_exposure_metrics(trade, emissions)
	return _finalize_overlap(vulnerability, emissions_metrics)


def build_overlap_framework_with_details() -> Tuple[pd.DataFrame, pd.DataFrame]:
	vulnerability = load_vulnerability(resolve_data_file(BASE_DIR, VULN_FILE_CANDIDATES, "vulnerability file"))
	emissions = load_emissions(resolve_data_file(BASE_DIR, EMISSIONS_FILE_CANDIDATES, "emissions file"))
	trade = load_trade(resolve_data_file(BASE_DIR, TRADE_FILE_CANDIDATES, "trade matrix file"))
	emissions_metrics, merged_links = emissions_exposure_metrics(trade, emissions)
	df = _finalize_overlap(vulnerability, emissions_metrics)
	return df, merged_links


def export_outputs(df: pd.DataFrame) -> None:
	cols = [
		"m49",
		"country",
		"vulnerability_index",
		"concentration_index",
		"emissions_exposure_index",
		"trade_weighted_emissions_intensity",
		"high_emissions_import_share",
		"emissions_data_coverage",
		"overlap_index",
		"overlap_tier",
		"priority_action",
	]
	out = df[cols].copy()
	out.to_csv(OUT_OVERLAP, index=False)
	out[["m49", "country", "overlap_tier", "priority_action", "overlap_index"]].to_csv(OUT_ACTIONS, index=False)


def build_country_flag_lookup() -> dict[str, str]:
	path = BASE_DIR / "FAOcountryProfile.csv"
	lookup: dict[str, str] = {}
	if not path.exists():
		return lookup
	country_df = pd.read_csv(path, low_memory=False)
	country_df.columns = [str(c).strip() for c in country_df.columns]
	for _, row in country_df.iterrows():
		for col in ["ISO3_CODE", "ISO3_WB_CODE", "SHORT_NAME", "OFFICIAL_FAO_NAME", "UNOFFICIAL1_NAME", "UNOFFICIAL2_NAME", "UNOFFICIAL3_NAME"]:
			val = row.get(col)
			if pd.isna(val) or str(val).strip() == "":
				continue
			name = str(val).strip()
			lookup.setdefault(name.lower(), str(row.get("ISO3_CODE", "")).strip().upper())
			lookup.setdefault(str(row.get("SHORT_NAME", "")).strip().lower(), str(row.get("ISO3_CODE", "")).strip().upper())
			lookup.setdefault(str(row.get("OFFICIAL_FAO_NAME", "")).strip().lower(), str(row.get("ISO3_CODE", "")).strip().upper())
			lookup.setdefault(str(row.get("UNOFFICIAL1_NAME", "")).strip().lower(), str(row.get("ISO3_CODE", "")).strip().upper())
			lookup.setdefault(str(row.get("UNOFFICIAL2_NAME", "")).strip().lower(), str(row.get("ISO3_CODE", "")).strip().upper())
			lookup.setdefault(str(row.get("UNOFFICIAL3_NAME", "")).strip().lower(), str(row.get("ISO3_CODE", "")).strip().upper())
	return {k: v for k, v in lookup.items() if v}


def get_country_flag_url(country_name: str) -> str | None:
	lookup = build_country_flag_lookup()
	key = str(country_name or "").strip().lower()
	iso3 = lookup.get(key)
	if not iso3:
		return None
	return f"https://raw.githubusercontent.com/ManasaSavanur/CountryFlags/master/flags/{iso3.lower()}.png"


def display_df_with_headers(df: pd.DataFrame) -> pd.DataFrame:
	return df.rename(columns={
		"country": "Country",
		"overlap_index": "Overlap Index",
		"vulnerability_index": "Vulnerability Index",
		"emissions_exposure_index": "Emissions Exposure",
		"concentration_index": "Concentration Index",
		"trade_weighted_emissions_intensity": "Trade-weighted Emissions Intensity",
		"high_emissions_import_share": "High-Emission Import Share",
		"emissions_data_coverage": "Emissions Data Coverage",
		"overlap_tier": "Overlap Tier",
		"priority_action": "Priority Action",
		"partner_country": "Partner Country",
		"import_value": "Import Value",
		"import_share": "Import Share",
		"avg_emissions_intensity": "Average Emissions Intensity",
		"risk_contribution": "Risk Contribution",
		"resolved_item_name": "Food Item",
		"trade_item_name": "Food Item",
		"partner_m49": "Partner M49",
		"item_cpc": "Item CPC",
	})


def run_cli() -> None:
	df = build_overlap_framework()
	export_outputs(df)

	print("Trade-Emissions overlap framework built successfully.")
	print(f"Saved overlap table: {OUT_OVERLAP}")
	print(f"Saved action table: {OUT_ACTIONS}")
	print("\nTop 15 critical overlap candidates:")
	top = df[df["overlap_tier"] == "Critical overlap"].head(15)
	if len(top) == 0:
		top = df.head(15)
	print(
		top[
			[
				"m49",
				"country",
				"overlap_index",
				"vulnerability_index",
				"emissions_exposure_index",
				"overlap_tier",
				"priority_action",
			]
		].to_string(index=False)
	)


def run_streamlit() -> None:
	import streamlit as st

	st.set_page_config(page_title="Food Trade x Emissions Risk", layout="wide")
	st.title("Food-Trade Vulnerability and Emissions Overlap")
	st.caption("Competition prototype: country ranking, overlap diagnosis, and prioritized actions")

	df, links = build_overlap_framework_with_details()
	export_outputs(df)

	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Countries scored", int(len(df)))
	c2.metric("Critical overlap countries", int((df["overlap_tier"] == "Critical overlap").sum()))
	c3.metric("Median overlap index", f"{df['overlap_index'].median():.3f}")
	c4.metric("Avg emissions coverage", f"{df['emissions_data_coverage'].mean():.1%}")

	st.subheader("Priority Ranking")
	priority_df = df[["country", "overlap_index", "overlap_tier", "vulnerability_index", "emissions_exposure_index", "concentration_index", "priority_action"]].copy()
	priority_df.insert(0, "Priority Rank", np.arange(1, len(priority_df) + 1))
	priority_df = display_df_with_headers(priority_df)
	priority_df = priority_df.rename(columns={"Priority Rank": "Rank"})
	st.dataframe(priority_df, use_container_width=True, hide_index=True)

	st.subheader("Overlap Map")
	scatter_df = df[["country", "vulnerability_index", "emissions_exposure_index", "overlap_tier"]].copy()
	st.scatter_chart(scatter_df, x="vulnerability_index", y="emissions_exposure_index", color="overlap_tier")

	st.subheader("Action Playbook")
	tiers = ["Critical overlap", "High vulnerability", "High emissions", "Moderate/Low"]
	tabs = st.tabs(tiers)
	for tab, tier in zip(tabs, tiers):
		with tab:
			s = df[df["overlap_tier"] == tier][["country", "priority_action", "overlap_index"]].head(10).copy()
			if len(s) == 0:
				st.write("No countries in this tier.")
			else:
				playbook_df = display_df_with_headers(s)
				st.dataframe(playbook_df, use_container_width=True, hide_index=True)
				if len(playbook_df) > 0:
					bar_df = playbook_df.set_index("Country")[["Overlap Index"]].sort_values("Overlap Index", ascending=False)
					st.bar_chart(bar_df)

	st.download_button(
		"Download overlap results CSV",
		data=df.to_csv(index=False).encode("utf-8"),
		file_name="trade_emissions_overlap_2023.csv",
		mime="text/csv",
	)

	st.subheader("Country Drill-Down")
	countries = df["country"].dropna().astype(str).tolist()
	default_country = countries[0] if countries else None
	selected_country = st.selectbox("Select a country", countries, index=0) if default_country else None

	if selected_country:
		row = df[df["country"] == selected_country].iloc[0]
		selected_m49 = row["m49"]
		sub = links[links["reporter_m49"] == selected_m49].copy()
		sub = sub[sub["import_value"] > 0].copy()

		if len(sub) == 0:
			st.warning("No link-level trade records found for the selected country.")
			return

		flag_url = get_country_flag_url(selected_country)
		flag_col, text_col = st.columns([1, 5])
		if flag_url:
			flag_col.image(flag_url, width=80)
		text_col.markdown(f"### {selected_country}")
		text_col.caption(f"Overlap tier: {row['overlap_tier']} | Priority action: {row['priority_action']}")

		sub["emissions_intensity_filled"] = sub["emissions_intensity"].fillna(sub["emissions_intensity"].median(skipna=True))
		sub["weighted_emissions"] = sub["import_value"] * sub["emissions_intensity_filled"]
		den = sub["weighted_emissions"].max() if sub["weighted_emissions"].max() > 0 else 1.0
		sub["risk_score"] = sub["weighted_emissions"] / den

		c1, c2, c3, c4 = st.columns(4)
		c1.metric("Overlap index", f"{row['overlap_index']:.3f}")
		c2.metric("Vulnerability", f"{row['vulnerability_index']:.3f}")
		c3.metric("Emissions exposure", f"{row['emissions_exposure_index']:.3f}")
		c4.metric("Tier", str(row["overlap_tier"]))

		total_import = sub["import_value"].sum()
		total_weighted = sub["weighted_emissions"].sum()

		partner_tbl = (
			sub.groupby(["partner_m49", "partner_country"], as_index=False)
			.agg(import_value=("import_value", "sum"), weighted_emissions=("weighted_emissions", "sum"))
		)
		partner_tbl["Import Share"] = np.where(total_import > 0, partner_tbl["import_value"] / total_import, 0.0)
		partner_tbl["Average Emissions Intensity"] = np.where(
			partner_tbl["import_value"] > 0,
			partner_tbl["weighted_emissions"] / partner_tbl["import_value"],
			0.0,
		)
		partner_tbl["Risk Contribution"] = np.where(
			total_weighted > 0,
			partner_tbl["weighted_emissions"] / total_weighted,
			0.0,
		)
		partner_tbl = partner_tbl.sort_values("Risk Contribution", ascending=False)
		partner_tbl = partner_tbl[["partner_country", "import_value", "Import Share", "Average Emissions Intensity", "Risk Contribution"]].head(10).copy()
		partner_tbl = display_df_with_headers(partner_tbl)

		item_tbl = (
			sub.groupby(["item_cpc", "resolved_item_name"], as_index=False)
			.agg(import_value=("import_value", "sum"), weighted_emissions=("weighted_emissions", "sum"))
		)
		item_tbl["Import Share"] = np.where(total_import > 0, item_tbl["import_value"] / total_import, 0.0)
		item_tbl["Average Emissions Intensity"] = np.where(
			item_tbl["import_value"] > 0,
			item_tbl["weighted_emissions"] / item_tbl["import_value"],
			0.0,
		)
		item_tbl["Risk Contribution"] = np.where(
			total_weighted > 0,
			item_tbl["weighted_emissions"] / total_weighted,
			0.0,
		)
		item_tbl = item_tbl.sort_values("Risk Contribution", ascending=False)
		item_tbl = item_tbl[["resolved_item_name", "import_value", "Import Share", "Average Emissions Intensity", "Risk Contribution"]].head(10).copy()
		item_tbl = display_df_with_headers(item_tbl)

		partner_chart = partner_tbl.set_index("Partner Country")[["Import Share", "Average Emissions Intensity"]].copy()
		item_chart = item_tbl.set_index("Food Item")[["Import Share", "Average Emissions Intensity"]].copy()

		partner_col, item_col = st.columns(2)
		with partner_col:
			st.markdown("### Top Risky Partner Countries")
			st.dataframe(partner_tbl, use_container_width=True, hide_index=True)
			st.bar_chart(partner_chart)
		with item_col:
			st.markdown("### Top Risky Food Items")
			st.dataframe(item_tbl, use_container_width=True, hide_index=True)
			st.bar_chart(item_chart)

		fig, ax = __import__('matplotlib').pyplot.subplots(figsize=(4, 4))
		pie_df = item_tbl.head(5).copy()
		if len(pie_df) > 0:
			labels = pie_df["Food Item"].astype(str)
			values = pie_df["Import Share"].astype(float)
			ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
			ax.set_title("Top item contribution")
			st.pyplot(fig)
			__import__('matplotlib').pyplot.close(fig)

		st.markdown("### What-If: Supplier Disruption Simulation")
		partner_options = partner_tbl["Partner Country"].fillna("Unknown").astype(str).tolist()
		default_partner = partner_options[0] if partner_options else None
		shock_partner = st.selectbox("Supplier to disrupt", partner_options, index=0) if default_partner else None
		shock_pct = st.slider("Disruption severity (%)", min_value=5, max_value=100, value=30, step=5)

		if shock_partner:
			loss_factor = shock_pct / 100.0
			base_import = total_import
			base_emissions = total_weighted

			shock = sub.copy()
			mask = shock["partner_country"].fillna("Unknown").astype(str).eq(shock_partner)
			shock_import = pd.to_numeric(shock["import_value"], errors="coerce").fillna(0.0).astype(float)
			shock_emissions = pd.to_numeric(shock["weighted_emissions"], errors="coerce").fillna(0.0).astype(float)
			multiplier = np.where(mask.to_numpy(), 1.0 - loss_factor, 1.0)
			shock["import_value"] = shock_import.to_numpy() * multiplier
			shock["weighted_emissions"] = shock_emissions.to_numpy() * multiplier

			post_import = shock["import_value"].sum()
			post_emissions = shock["weighted_emissions"].sum()
			import_loss_pct = (base_import - post_import) / base_import if base_import > 0 else 0.0
			emissions_change_pct = (base_emissions - post_emissions) / base_emissions if base_emissions > 0 else 0.0

			m1, m2, m3 = st.columns(3)
			m1.metric("Import value shock", f"{import_loss_pct:.1%}")
			m2.metric("Embodied emissions change", f"{emissions_change_pct:.1%}")
			m3.metric("Partner share (pre-shock)", f"{partner_tbl.loc[partner_tbl['Partner Country'].eq(shock_partner), 'Import Share'].sum():.1%}")

			st.info(
				"Priority response: replace disrupted flows with lower-emissions suppliers first for high-risk items, "
				"then activate short-term food buffers where substitution is constrained."
			)


if __name__ == "__main__":
	run_cli()
