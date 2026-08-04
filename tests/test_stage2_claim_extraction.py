from stage2_review_gate import extract_claims, normalize_claim_text, validate_claim_rows


def test_claims_are_atomic_and_source_reconstructable():
    caption = (
        "Water appeared on previously dry streets, and several buildings are surrounded by water. "
        "Some rooftops appear damaged. The event was a hurricane, highlighting catastrophic disruption."
    )
    claims, stats = extract_claims(caption, "review-001")
    assert stats["rhetorical_fragments_removed"] == 1
    assert claims
    assert len({normalize_claim_text(claim["claim_text"]) for claim in claims}) == len(claims)
    assert all(claim["primary_type"] for claim in claims)
    assert all(c["source_span"]["text"] == caption[c["source_span"]["start"]:c["source_span"]["end"]] for c in claims)
    audit = validate_claim_rows([{"review_id": "review-001", "candidate_caption": caption, "claims": claims}])
    assert audit["duplicate_gate_pass"] is True
    assert audit["duplicate_claim_records"] == 0


def test_duplicate_claim_text_fails_the_bundle_gate():
    caption = "A damaged building appeared."
    claims, _ = extract_claims(caption, "review-002")
    duplicate = dict(claims[0])
    duplicate["claim_id"] = "review-002:claim:2"
    duplicate["primary_type"] = "changed_object"
    try:
        validate_claim_rows([{"review_id": "review-002", "candidate_caption": caption, "claims": [claims[0], duplicate]}])
    except ValueError as error:
        assert "duplicate claim text" in str(error)
    else:
        raise AssertionError("duplicate claim text must fail the bundle gate")
