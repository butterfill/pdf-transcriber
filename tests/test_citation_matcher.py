"""Regression tests for the provider-independent citation matcher."""
from pdf_transcriber_gemini.citation_matcher import CitationResolver, canonical_author


def ref(n, authors, year, label=None):
    return {"id": f"src-ref-{n:03d}", "authors": authors, "year": year, "label": label}


def test_name_normalization():
    assert canonical_author("J. M. Zacks") == "Zacks, J. M."
    assert canonical_author("D. Y. von Cramon") == "von Cramon, D. Y."
    assert canonical_author("et al.") is None


def test_author_year_and_coauthors():
    r = CitationResolver([
        ref(1, ["J. Zacks", "N. Speer", "K. Swallow"], "2007"),
        ref(2, ["Zacks, J.", "Swallow, K."], "2007"),
    ])
    assert r.resolve("Zacks et al., 2007").code == "ambiguous_author_year"
    assert r.resolve("Zacks & Swallow, 2007").reference_id == "src-ref-002"
    assert r.resolve("Zacks, Speer, & Swallow, 2007").reference_id == "src-ref-001"


def test_suffix_numeric_and_missing_evidence():
    r = CitationResolver([
        ref(1, ["A. Smith"], "2020a", "[1]"),
        ref(2, ["A. Smith"], "2020b", "[2]"),
    ])
    assert r.resolve("Smith (2020a)").reference_id == "src-ref-001"
    assert r.resolve("Smith (2020)").reference_id is None
    assert r.resolve("[2]").reference_id == "src-ref-002"
    assert r.resolve("2020a").code == "missing_cited_author"
    assert r.resolve("ibid.").code == "unsupported_citation_form"


def test_author_count_must_agree_with_citation():
    single = CitationResolver([ref(1, ["Smith, A."], "2020")])
    multi = CitationResolver([ref(1, ["Smith, A.", "Brown, B."], "2020")])
    assert single.resolve("Smith et al., 2020").reference_id is None
    assert multi.resolve("Smith (2020)").reference_id is None
    assert multi.resolve("Smith et al., 2020").reference_id == "src-ref-001"


def test_unknown_authors_never_shift_or_disambiguate_candidates():
    for authors in (["Alice Smith", "Brown, B."], ["Smith, A.", "et al."],
                    ["Smith, A.", "Beth Brown"]):
        resolver = CitationResolver([ref(1, authors, "2020")])
        assert resolver.resolve("Brown (2020)").reference_id is None
        assert resolver.resolve("Smith et al., 2020").reference_id is None
    resolver = CitationResolver([
        ref(1, ["Smith, A.", "Brown, B."], "2020"),
        ref(2, ["Smith, A.", "Beth Brown"], "2020"),
    ])
    assert resolver.resolve("Smith & Brown, 2020").code == "incomplete_bibliography_metadata"


def test_numeric_labels_require_balanced_brackets_and_years_trim_spaces():
    for label in ("[2", "2]"):
        assert CitationResolver([ref(1, ["Smith"], "2020", label)]).resolve("[2]").reference_id is None
    resolver = CitationResolver([ref(1, ["Smith"], " 2020A ", "[02]")])
    assert resolver.resolve("Smith (2020a)").reference_id == "src-ref-001"
    assert resolver.resolve("[2]").reference_id == "src-ref-001"


def test_corporate_diacritics_and_family_particles():
    resolver = CitationResolver([
        ref(1, ["World Health Organization"], "2020"),
        ref(2, ["J. García"], "2021"),
        ref(3, ["D. Y. von Cramon"], "2007"),
    ])
    assert resolver.resolve("World Health Organization (2020)").reference_id == "src-ref-001"
    assert resolver.resolve("Garcia (2021)").reference_id == "src-ref-002"
    assert resolver.resolve("von Cramon (2007)").reference_id == "src-ref-003"
