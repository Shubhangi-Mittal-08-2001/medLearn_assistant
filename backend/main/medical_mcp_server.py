"""Local MCP gateway for authoritative NLM PubMed and RxNorm APIs."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
REQUEST_TIMEOUT = float(os.getenv("MEDICAL_API_TIMEOUT", "20"))

mcp = FastMCP(
    "MedLearn Medical APIs",
    instructions=(
        "Use PubMed for medical literature/evidence and RxNorm for normalized drug "
        "identifiers and terminology. RxNorm is not an adverse-effect or interaction source."
    ),
    host=os.getenv("MEDICAL_MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MEDICAL_MCP_PORT", "8010")),
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


def _ncbi_identity() -> dict[str, str]:
    values = {"tool": "medlearn-assistant"}
    if email := os.getenv("NCBI_EMAIL"):
        values["email"] = email
    if api_key := os.getenv("NCBI_API_KEY"):
        values["api_key"] = api_key
    return values


async def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(url, params=params, headers={"User-Agent": "MedLearn-Assistant/0.1"})
        response.raise_for_status()
        return response.json()


async def _get_text(url: str, params: dict[str, Any]) -> str:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(url, params=params, headers={"User-Agent": "MedLearn-Assistant/0.1"})
        response.raise_for_status()
        return response.text


@mcp.tool()
async def pubmed_search(query: str, max_results: int = 5, sort: str = "relevance") -> dict[str, Any]:
    """Search PubMed literature and return cited article metadata and PubMed links."""
    if not query.strip():
        raise ValueError("query is required")
    limit = min(max(max_results, 1), 20)
    search = await _get_json(
        f"{NCBI_BASE_URL}/esearch.fcgi",
        {"db": "pubmed", "term": query.strip(), "retmode": "json", "retmax": limit,
         "sort": sort, **_ncbi_identity()},
    )
    pmids = search.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return {"query": query, "count": 0, "articles": []}
    summary = await _get_json(
        f"{NCBI_BASE_URL}/esummary.fcgi",
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "json", **_ncbi_identity()},
    )
    records = summary.get("result", {})
    articles = []
    for pmid in pmids:
        item = records.get(pmid, {})
        articles.append({
            "pmid": pmid,
            "title": item.get("title"),
            "authors": [author.get("name") for author in item.get("authors", [])],
            "journal": item.get("fulljournalname") or item.get("source"),
            "publication_date": item.get("pubdate"),
            "doi": next((entry.get("value") for entry in item.get("articleids", [])
                         if entry.get("idtype") == "doi"), None),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return {"query": query, "count": len(articles), "articles": articles}


@mcp.tool()
async def pubmed_article_details(pmids: list[str]) -> dict[str, Any]:
    """Fetch PubMed titles and abstracts for up to ten PMIDs."""
    clean = [str(pmid).strip() for pmid in pmids if str(pmid).strip().isdigit()][:10]
    if not clean:
        raise ValueError("pmids must contain at least one numeric PubMed ID")
    xml_text = await _get_text(
        f"{NCBI_BASE_URL}/efetch.fcgi",
        {"db": "pubmed", "id": ",".join(clean), "retmode": "xml", **_ncbi_identity()},
    )
    root = ET.fromstring(xml_text)
    articles = []
    for citation in root.findall(".//PubmedArticle"):
        pmid = citation.findtext(".//MedlineCitation/PMID")
        title_node = citation.find(".//ArticleTitle")
        abstract_nodes = citation.findall(".//Abstract/AbstractText")
        articles.append({
            "pmid": pmid,
            "title": "".join(title_node.itertext()) if title_node is not None else None,
            "abstract": "\n".join("".join(node.itertext()) for node in abstract_nodes),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
        })
    return {"articles": articles}


@mcp.tool()
async def rxnorm_lookup(drug_name: str, max_results: int = 5) -> dict[str, Any]:
    """Resolve a drug name to active RxNorm concept identifiers (RxCUIs)."""
    if not drug_name.strip():
        raise ValueError("drug_name is required")
    payload = await _get_json(
        f"{RXNORM_BASE_URL}/approximateTerm.json",
        {"term": drug_name.strip(), "maxEntries": min(max(max_results, 1), 20), "option": 1},
    )
    candidates = payload.get("approximateGroup", {}).get("candidate", []) or []
    return {
        "query": drug_name,
        "concepts": [{key: item.get(key) for key in ("rxcui", "name", "score", "rank", "source")}
                     for item in candidates],
        "source": "RxNorm, U.S. National Library of Medicine",
    }


@mcp.tool()
async def rxnorm_concept_details(rxcui: str) -> dict[str, Any]:
    """Return the normalized name and type for one RxNorm concept identifier."""
    if not rxcui.strip().isdigit():
        raise ValueError("rxcui must be numeric")
    payload = await _get_json(f"{RXNORM_BASE_URL}/rxcui/{rxcui.strip()}/properties.json", {})
    return {
        "rxcui": rxcui.strip(),
        "properties": payload.get("properties"),
        "source": "RxNorm, U.S. National Library of Medicine",
        "documentation": "https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html",
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
