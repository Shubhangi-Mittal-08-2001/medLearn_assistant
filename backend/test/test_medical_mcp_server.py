from unittest.mock import AsyncMock, patch

import pytest

from main.medical_mcp_server import pubmed_article_details, pubmed_search, rxnorm_lookup


@pytest.mark.asyncio
async def test_pubmed_search_returns_citable_metadata():
    responses = [
        {"esearchresult": {"idlist": ["123"]}},
        {"result": {"123": {"title": "GLP-1 safety", "authors": [{"name": "A Author"}],
                             "source": "Medical Journal", "pubdate": "2025",
                             "articleids": [{"idtype": "doi", "value": "10.1/example"}]}}},
    ]
    with patch("main.medical_mcp_server._get_json", AsyncMock(side_effect=responses)):
        result = await pubmed_search("GLP-1 long-term adverse effects", max_results=5)

    assert result["articles"][0]["pmid"] == "123"
    assert result["articles"][0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/123/"


@pytest.mark.asyncio
async def test_pubmed_details_parses_abstract():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>
    <Article><ArticleTitle>Safety study</ArticleTitle><Abstract>
    <AbstractText>Long-term evidence.</AbstractText></Abstract></Article>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    with patch("main.medical_mcp_server._get_text", AsyncMock(return_value=xml)):
        result = await pubmed_article_details(["123"])
    assert result["articles"][0]["abstract"] == "Long-term evidence."


@pytest.mark.asyncio
async def test_rxnorm_lookup_returns_normalized_concepts():
    payload = {"approximateGroup": {"candidate": [
        {"rxcui": "11289", "name": "warfarin", "score": "100", "rank": "1", "source": "RXNORM"}
    ]}}
    with patch("main.medical_mcp_server._get_json", AsyncMock(return_value=payload)):
        result = await rxnorm_lookup("warfarin")
    assert result["concepts"][0]["rxcui"] == "11289"
    assert "National Library of Medicine" in result["source"]
