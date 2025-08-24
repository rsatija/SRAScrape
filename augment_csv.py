import os
import csv
import argparse
from typing import Dict, Any

from sra_tissue_classifier import (
	fetch_sra_xml_for_srx,
	extract_metadata_from_sra_xml,
	try_fetch_biosample_json,
	build_prompt,
	call_gemini,
	srx_link,
)


def classify_row(srx: str, api_key: str) -> Dict[str, Any]:
	"""Classify a single SRX and return outputs, raising on network errors."""
	sra_xml = fetch_sra_xml_for_srx(srx)
	meta = extract_metadata_from_sra_xml(sra_xml)
	if not meta:
		raise RuntimeError("Missing metadata")

	biosample_json = try_fetch_biosample_json(meta.get("biosample"))
	if biosample_json:
		try:
			bs = biosample_json
			parts = []
			for k in ("organism", "isolation_source", "tissue", "cell_type", "cell_line"):
				v = bs.get(k)
				if isinstance(v, str) and v:
					parts.append(f"{k}: {v}")
			if parts:
				meta["metadata_text"] += "\n" + "\n".join(parts)
		except Exception:
			pass

	prompt = build_prompt(meta)
	resp = call_gemini(prompt, api_key)
	return {
		"srx_link": srx_link(meta.get("srx") or srx),
		"summary_5_words": resp.get("summary_5_words", ""),
		"tissue_guess": resp.get("tissue_guess", ""),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Augment first N SRX rows in CSV with classifier outputs")
	parser.add_argument("--input", default="other_samples_predictions.csv", help="Input CSV path")
	parser.add_argument("--output", default="other_samples_predictions_augmented_first20.csv", help="Output CSV path")
	parser.add_argument("--limit", type=int, default=20, help="Number of rows to process from start")
	parser.add_argument("--api-key", dest="api_key", default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"), help="Gemini API key")
	args = parser.parse_args()

	if not args.api_key:
		raise SystemExit("Gemini API key is required via --api-key or GEMINI_API_KEY env var")

	with open(args.input, newline="") as infile:
		reader = csv.DictReader(infile)
		fieldnames = list(reader.fieldnames or [])
		first_col = fieldnames[0] if fieldnames else None
		new_cols = ["srx_link", "summary_5_words", "tissue_guess"]
		augmented_fields = fieldnames + [c for c in new_cols if c not in fieldnames]

		rows_out = []
		for i, row in enumerate(reader, start=1):
			if i > args.limit:
				break
			srx = (row.get(first_col) or "").strip() if first_col else ""
			out = {**row}
			try:
				if srx:
					res = classify_row(srx, args.api_key)
					out.update(res)
			except Exception:
				# On any error, leave new columns empty and continue
				for c in new_cols:
					out.setdefault(c, "")
			rows_out.append(out)

	with open(args.output, "w", newline="") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=augmented_fields)
		writer.writeheader()
		for r in rows_out:
			writer.writerow(r)

	print(f"Wrote {len(rows_out)} rows to {args.output}")


if __name__ == "__main__":
	main()
