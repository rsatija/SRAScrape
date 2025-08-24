import os
import sys
import argparse
import json
from typing import Any, Dict, Optional

import requests
import xmltodict

# Optional import of Gemini; we handle ImportError gracefully with a helpful message
try:
	import google.generativeai as genai
except ImportError:  # pragma: no cover
	genai = None


NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch_sra_xml_for_srx(srx_accession: str) -> Dict[str, Any]:
	"""Fetch SRA XML metadata for a given SRX accession and return as dict."""
	url = f"{NCBI_EUTILS_BASE}/efetch.fcgi"
	params = {"db": "sra", "id": srx_accession, "retmode": "xml"}
	resp = requests.get(url, params=params, timeout=30)
	resp.raise_for_status()
	return xmltodict.parse(resp.text)


def get_first(item_or_list):
	if isinstance(item_or_list, list):
		return item_or_list[0] if item_or_list else None
	return item_or_list


def extract_metadata_from_sra_xml(sra_xml: Dict[str, Any]) -> Dict[str, Any]:
	"""Extract useful metadata fields from SRA efetch XML structure.

	Returns a dict containing keys like: srx, srs, biosample, sample_title, study_title,
	tissue, cell_type, organism, attributes, and a compact text blob of metadata_text.
	"""
	pkg = sra_xml.get("EXPERIMENT_PACKAGE_SET", {}).get("EXPERIMENT_PACKAGE")
	if isinstance(pkg, list):
		pkg = pkg[0]
	if not pkg:
		return {}

	experiment = pkg.get("EXPERIMENT", {})
	sample = pkg.get("SAMPLE", {})
	study = pkg.get("STUDY", {})

	srx = experiment.get("@accession")
	srs = sample.get("@accession")

	# BioSample from sample identifiers
	biosample = None
	identifiers = sample.get("IDENTIFIERS", {})
	external_ids = identifiers.get("EXTERNAL_ID")
	if external_ids:
		external_ids_list = external_ids if isinstance(external_ids, list) else [external_ids]
		for ex in external_ids_list:
			if ex.get("@namespace") == "BioSample":
				biosample = ex.get("#text")
				break

	sample_title = sample.get("TITLE")
	study_title = study.get("DESCRIPTOR", {}).get("STUDY_TITLE")
	study_abstract = study.get("DESCRIPTOR", {}).get("STUDY_ABSTRACT")

	# Organism
	sample_name = sample.get("SAMPLE_NAME", {})
	organism = sample_name.get("SCIENTIFIC_NAME")

	# Attributes
	attrs = {}
	for attr in (sample.get("SAMPLE_ATTRIBUTES", {}) or {}).get("SAMPLE_ATTRIBUTE", []) or []:
		tag = attr.get("TAG")
		val = attr.get("VALUE")
		if tag:
			attrs[tag.lower()] = val

	tissue = attrs.get("tissue") or attrs.get("tissue_type")
	cell_type = attrs.get("cell_type") or attrs.get("celltype")
	cell_line = attrs.get("cell_line") or attrs.get("cell-line")

	# Compose a compact metadata text for LLM
	parts = []
	if study_title:
		parts.append(f"Study title: {study_title}")
	if study_abstract:
		parts.append(f"Study abstract: {study_abstract}")
	if sample_title:
		parts.append(f"Sample title: {sample_title}")
	if organism:
		parts.append(f"Organism: {organism}")
	for k, v in attrs.items():
		parts.append(f"{k}: {v}")
	metadata_text = "\n".join(parts)

	return {
		"srx": srx,
		"srs": srs,
		"biosample": biosample,
		"sample_title": sample_title,
		"study_title": study_title,
		"study_abstract": study_abstract,
		"organism": organism,
		"tissue": tissue,
		"cell_type": cell_type,
		"cell_line": cell_line,
		"attributes": attrs,
		"metadata_text": metadata_text,
	}


def try_fetch_biosample_json(biosample_accession: Optional[str]) -> Optional[Dict[str, Any]]:
	"""Fetch BioSample JSON if available to enrich metadata."""
	if not biosample_accession:
		return None
	# BioSample E-utilities esummary is not perfect; use NCBI BioSample API JSON if available
	url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/biosample/accession/{biosample_accession}"
	try:
		resp = requests.get(url, timeout=30)
		if resp.status_code == 200:
			return resp.json()
	except Exception:
		return None
	return None


def build_prompt(metadata: Dict[str, Any]) -> str:
	"""Build a concise prompt for Gemini summarization."""
	context = metadata.get("metadata_text", "")
	prompt = (
		"You are analyzing scRNA-seq sample metadata to determine tissue/cell-type of origin.\n"
		"From the metadata below, produce:\n"
		"1) a five-word summary of the sample's tissue or cell type of origin;\n"
		"2) the best-guess tissue type as a single concise noun phrase.\n"
		"Return strict JSON with keys: summary_5_words, tissue_guess.\n\n"
		f"Metadata:\n{context}\n"
	)
	return prompt


def call_gemini(prompt: str, api_key: str) -> Dict[str, str]:
	if genai is None:
		raise RuntimeError(
			"google-generativeai is not installed. Please install dependencies from requirements.txt"
		)
	genai.configure(api_key=api_key)
	model = genai.GenerativeModel("gemini-1.5-flash")
	resp = model.generate_content(prompt)
	text = resp.text or ""
	# Try to parse JSON; if the model responded with prose, attempt to extract braces
	data: Dict[str, str]
	try:
		data = json.loads(text)
	except Exception:
		start = text.find("{")
		end = text.rfind("}")
		if start != -1 and end != -1 and end > start:
			try:
				data = json.loads(text[start:end + 1])
			except Exception:
				data = {"summary_5_words": "", "tissue_guess": ""}
		else:
			data = {"summary_5_words": "", "tissue_guess": ""}
	# Normalize fields and trim to 5 words max for summary
	summary = (data.get("summary_5_words") or "").strip()
	if summary:
		words = summary.split()
		if len(words) > 5:
			summary = " ".join(words[:5])
	tissue_guess = (data.get("tissue_guess") or "").strip()
	return {"summary_5_words": summary, "tissue_guess": tissue_guess}


def srx_link(srx: str) -> str:
	return f"https://www.ncbi.nlm.nih.gov/sra/?term={srx}"


def main() -> None:
	parser = argparse.ArgumentParser(description="Classify tissue for scRNA-seq SRX accessions")
	parser.add_argument("srx", help="SRX accession, e.g., SRX22288182")
	parser.add_argument(
		"--api-key",
		dest="api_key",
		default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
		help="Gemini API key (or set GEMINI_API_KEY/GOOGLE_API_KEY)",
	)
	args = parser.parse_args()

	if not args.api_key:
		print("Error: Gemini API key required. Pass --api-key or set GEMINI_API_KEY.", file=sys.stderr)
		sys.exit(2)

	# Fetch SRA metadata
	sra_xml = fetch_sra_xml_for_srx(args.srx)
	meta = extract_metadata_from_sra_xml(sra_xml)
	if not meta:
		print("Error: Could not retrieve SRA metadata for the given SRX.", file=sys.stderr)
		sys.exit(1)

	# Optionally enrich from BioSample
	biosample_json = try_fetch_biosample_json(meta.get("biosample"))
	if biosample_json:
		# Try to pull extra attributes if present
		try:
			# This structure can vary; we will not rely on exact shape but try a few common fields
			bs = biosample_json
			text_parts = []
			for k in ("organism", "isolation_source", "tissue", "cell_type", "cell_line"):
				v = bs.get(k)
				if isinstance(v, str) and v:
					text_parts.append(f"{k}: {v}")
			if text_parts:
				meta["metadata_text"] += "\n" + "\n".join(text_parts)
		except Exception:
			pass

	prompt = build_prompt(meta)
	gemini = call_gemini(prompt, args.api_key)

	print(json.dumps({
		"srx_link": srx_link(meta.get("srx") or args.srx),
		"summary_5_words": gemini.get("summary_5_words"),
		"tissue_guess": gemini.get("tissue_guess"),
	}, indent=2))


if __name__ == "__main__":
	main()
