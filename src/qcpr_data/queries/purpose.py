"""Deterministic retrieval-purpose and visual-attribute helpers.

The repair pipeline deliberately keeps three different signals separate:

* source/provenance evidence (for example a human caption or an official
  relation field),
* lexical/semantic evidence (normalised collisions and conservative
  attribute neighbours), and
* physical-pair evidence (the pair exists and can be inspected).

The generic no-change phrase list is only a weak feature.  A row is generic
only when the source status/scope or the absence of any specific visual
attribute agrees with that feature; the phrase list is never the classifier
on its own.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Mapping

QUERY_PURPOSES = (
    "exact_discriminative",
    "semantic_multi_positive",
    "localized",
    "direction_sensitive",
    "stable_scene_specific",
    "generic_no_change",
    "long_series",
    "unsupported_or_reject",
)

_GENERIC_PHRASES = {
    "there is no difference",
    "there are no differences",
    "no visible differences exist",
    "the two images are the same",
    "the two scenes seem identical",
    "the scene is the same as before",
    "no change has occurred",
    "no change is occurred",
    "almost nothing has changed",
    "there is no change",
    "there is no alteration",
    "there are no alterations",
    "no alteration",
    "no alterations",
    "nothing changed",
    "nothing has changed",
    "no visible change",
    "no visible changes",
    "unchanged",
}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "in", "is", "it", "of", "on", "or",
    "shows", "the", "there", "to", "was", "were", "with",
}

_OBJECTS = {
    "building": {"building", "buildings", "structure", "structures", "house", "houses", "roof", "roofs", "villa", "villas", "mansion", "mansions", "mall", "malls"},
    "road": {"road", "roads", "lane", "lanes", "street", "streets", "path", "paths", "highway", "highways", "crossroad", "crossroads", "roadside", "junction", "junctions"},
    "vegetation": {"tree", "trees", "forest", "forests", "vegetation", "canopy", "woodland", "woods", "grass", "grasses", "plant", "plants", "shrub", "shrubs"},
    "field": {"field", "fields", "farmland", "farm", "farms", "crop", "crops", "agriculture", "cultivated"},
    "water": {"water", "river", "rivers", "lake", "lakes", "pond", "ponds", "shore", "shoreline"},
    "soil": {"soil", "ground", "land", "bare", "bareland", "wasteland", "desert", "hole", "holes", "clearing", "clearings", "sand", "earth"},
    "vehicle": {"vehicle", "vehicles", "car", "cars", "truck", "trucks"},
    "industrial": {"industrial", "factory", "factories", "warehouse", "warehouses", "facility", "facilities", "container", "containers"},
}

_DIRECTIONS = {
    "appearance": {"appears", "appear", "appeared", "new", "built", "build", "constructed", "construction", "added", "emerged", "emergence", "expanded", "expansion", "increase", "increased", "growth", "grow", "grew", "grown", "placed", "put", "filled", "fill"},
    "disappearance": {"disappears", "disappear", "disappeared", "demolished", "demolition", "removed", "removal", "lost", "loss", "cleared", "clearance", "decreased", "decrease", "destroyed", "destruction", "shrank", "shrinkage", "collapsed", "collapse"},
    "conversion": {"converted", "conversion", "changed", "change", "turned", "became", "replaced", "replacement", "transition"},
}

_SPATIAL = {
    "upper": {"top", "upper", "north", "northern"},
    "lower": {"bottom", "lower", "south", "southern"},
    "left": {"left", "western", "west"},
    "right": {"right", "eastern", "east"},
    "center": {"center", "centre", "central", "middle"},
    "corner": {"corner", "corners"},
    "adjacent": {"adjacent", "near", "nearby", "beside", "alongside", "next"},
}

_SURFACES = {
    "built": {"building", "buildings", "structure", "structures", "house", "houses", "villa", "villas", "mansion", "mansions", "mall", "malls", "container", "containers", "parking", "road", "roads", "lane", "lanes", "urban", "construction"},
    "vegetated": {"tree", "trees", "forest", "forests", "vegetation", "canopy", "field", "fields", "crop", "crops", "farmland", "grass", "grasses", "plant", "plants", "green"},
    "water": {"water", "river", "rivers", "lake", "lakes", "pond", "ponds"},
    "bare": {"bare", "bareland", "wasteland", "desert", "hole", "holes", "soil", "ground", "land", "clearing", "clearings"},
}

_UNSUPPORTED_TERMS = {
    "because", "due", "caused", "cause", "likely", "probably", "suggesting",
    "severe", "severity", "moderate", "considerable", "significant", "negligible",
    "minor", "slight", "minimal", "disaster", "flood", "fire", "earthquake",
}

_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "several", "many", "few", "numerous", "single", "multiple",
}


def normalize_query_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_query_text(text))


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def extract_visual_attributes(text: Any) -> dict[str, Any]:
    tokens = [_stem(token) for token in _tokens(text)]
    token_set = set(tokens)
    objects = sorted(name for name, words in _OBJECTS.items() if token_set & {_stem(w) for w in words})
    directions = sorted(name for name, words in _DIRECTIONS.items() if token_set & {_stem(w) for w in words})
    spatial = sorted(name for name, words in _SPATIAL.items() if token_set & {_stem(w) for w in words})
    surfaces = sorted(name for name, words in _SURFACES.items() if token_set & {_stem(w) for w in words})
    count_tokens = [token for token in tokens if token.isdigit() or token in _NUMBER_WORDS]
    specific_tokens = sorted(
        token for token in token_set
        if token not in _STOPWORDS and token not in _NUMBER_WORDS and len(token) >= 3
    )
    return {
        "objects": objects,
        "directions": directions,
        "change_types": directions,
        "spatial_relations": spatial,
        "surfaces": surfaces,
        "count_tokens": sorted(set(count_tokens)),
        "specific_tokens": specific_tokens,
        "has_specific_visual_claim": bool(objects or directions or spatial or surfaces),
    }


def unsupported_claims(text: Any) -> list[str]:
    tokens = set(_tokens(text))
    found = sorted(tokens & _UNSUPPORTED_TERMS)
    # Counts are not automatically rejected: source-human review may support
    # them.  The flag is retained so the review gate can require evidence.
    if any(token.isdigit() or token in _NUMBER_WORDS for token in tokens):
        found.append("unverified_count_candidate")
    return sorted(set(found))


def semantic_signature(attributes: Mapping[str, Any]) -> str:
    """Return an attribute-only signature; no event/source identity enters it."""

    fields = (
        ("objects", attributes.get("objects", [])),
        ("directions", attributes.get("directions", [])),
        ("change_types", attributes.get("change_types", [])),
        ("spatial_relations", attributes.get("spatial_relations", [])),
        ("surfaces", attributes.get("surfaces", [])),
    )
    value = "|".join(f"{name}={','.join(sorted(map(str, values)))}" for name, values in fields)
    return value


def semantic_group_id(attributes: Mapping[str, Any]) -> str:
    return "semantic:" + hashlib.sha256(semantic_signature(attributes).encode("utf-8")).hexdigest()[:20]


def attribute_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("objects", "directions", "change_types", "spatial_relations", "surfaces")
    matches = {field: bool(set(left.get(field, [])) & set(right.get(field, []))) for field in fields}
    object_match = matches["objects"]
    direction_match = matches["directions"]
    change_match = matches["change_types"]
    if object_match and direction_match and change_match:
        grade = 3
    elif object_match and (direction_match or change_match):
        grade = 2
    elif any(matches.values()):
        grade = 1
    else:
        grade = 0
    return {"grade": grade, "matches": matches}


def generic_evidence(row: Mapping[str, Any], attributes: Mapping[str, Any]) -> dict[str, Any]:
    text_norm = normalize_query_text(row.get("text") or row.get("normalized_text"))
    source_status = str(row.get("change_status") or row.get("source_change_status") or "").casefold()
    scope = str(row.get("query_scope") or row.get("original_query_scope") or "").casefold()
    phrase_signal = text_norm in _GENERIC_PHRASES
    reasons = []
    if source_status in {"no_change", "unchanged", "stable"}:
        reasons.append("source_change_status_no_change")
    if scope in {"generic_no_change", "generic", "no_change"}:
        reasons.append("source_scope_generic_no_change")
    if phrase_signal:
        reasons.append("normalized_generic_phrase_match")
    if not attributes.get("has_specific_visual_claim"):
        reasons.append("no_specific_visual_attributes")
    is_generic = bool(
        source_status in {"no_change", "unchanged", "stable"}
        or scope in {"generic_no_change", "generic", "no_change"}
        or (phrase_signal and not attributes.get("has_specific_visual_claim"))
    )
    return {"is_generic": is_generic, "phrase_signal": phrase_signal, "reasons": reasons}


def _has_sequence(row: Mapping[str, Any]) -> bool:
    scope = str(row.get("query_scope") or row.get("original_query_scope") or "").casefold()
    item_type = str(row.get("item_type") or "").casefold()
    source = str(row.get("source_dataset") or row.get("source") or "").casefold()
    return scope in {"long_series", "long_series_change_candidate", "stable_scene_candidate"} or item_type in {"sequence", "long_series"} or source == "tamms"


def classify_query(
    row: Mapping[str, Any],
    *,
    collision_count: int = 1,
    neighbour_item_count: int = 1,
    positive_set_size: int = 1,
    stable_anchor_count: int = 0,
    stable_identifiability: float = 0.0,
    has_physical_pair: bool = True,
) -> dict[str, Any]:
    """Classify a row into exactly one retrieval purpose.

    Priority follows the operational purpose of the view.  A source row may
    have several evidence flags, but the returned ``purpose`` is singular.
    """

    text = str(row.get("text") or row.get("caption") or row.get("normalized_text") or "").strip()
    attributes = extract_visual_attributes(text)
    generic = generic_evidence({**row, "text": text}, attributes)
    scope = str(row.get("query_scope") or row.get("original_query_scope") or "").casefold()
    verification = str(row.get("verification") or row.get("verification_status") or "").casefold()
    source = str(row.get("source_dataset") or row.get("dataset_name") or row.get("source") or "").casefold()
    unsupported = unsupported_claims(text)
    reasons: list[str] = []
    stable_scope = scope in {"stable", "stable_scene_specific", "stable_scene_candidate"} or stable_anchor_count >= 2

    # A real stable candidate is allowed to originate from a physical
    # no-change pair, but its independently supported anchors take precedence
    # over the source's old no_change status.  This prevents the old generic
    # label from swallowing a factual stable-scene description.
    if stable_scope and stable_anchor_count >= 2:
        if stable_identifiability >= 0.5 and positive_set_size == 1:
            purpose = "stable_scene_specific"
            reasons.append("two_or_more_common_visual_anchors_and_unique_signature")
        else:
            purpose = "semantic_multi_positive"
            reasons.append("stable_scene_not_pair_discriminative")
    elif stable_scope:
        purpose = "unsupported_or_reject"
        reasons.append("stable_candidate_has_fewer_than_two_common_visual_anchors")
    elif generic["is_generic"]:
        purpose = "generic_no_change"
        reasons.extend(generic["reasons"])
    elif _has_sequence({**row, "source_dataset": source}):
        purpose = "long_series"
        reasons.append("sequence_or_long_series_provenance")
    elif scope in {"localized", "localized_candidate", "localized_eval"} or row.get("localized_relation") is not None or row.get("dense_sidecar"):
        purpose = "localized"
        reasons.append("localized_scope_or_region_sidecar")
    elif scope in {"direction", "direction_sensitive", "direction_candidate"}:
        purpose = "direction_sensitive"
        reasons.append("direction_view_requested")
    elif not has_physical_pair or not text or (verification in {"source_unverified", "generated_unverified", "unverified", ""} and scope not in {"semantic", "semantic_group", "official_relation"}):
        purpose = "unsupported_or_reject"
        reasons.append("missing_physical_pair_or_unverified_text")
    elif scope in {"semantic", "semantic_group", "official_relation"} or positive_set_size > 1 or collision_count > 1 or neighbour_item_count > 1:
        purpose = "semantic_multi_positive"
        reasons.append("verified_attribute_collision_or_semantic_neighbours")
    elif scope in {"exact", "exact_pair", "exact_discriminative"} and attributes.get("has_specific_visual_claim") and positive_set_size == 1 and collision_count == 1 and neighbour_item_count == 1:
        purpose = "exact_discriminative"
        reasons.append("singleton_verified_text_with_no_attribute_neighbours")
    elif unsupported and verification not in {"human", "human_rewritten", "independently_source_verified"}:
        purpose = "unsupported_or_reject"
        reasons.append("unsupported_claim_terms")
    else:
        purpose = "unsupported_or_reject"
        reasons.append("no_safe_single_pair_purpose")

    return {
        "purpose": purpose,
        "reasons": sorted(set(reasons)),
        "generic_evidence": generic,
        "attributes": attributes,
        "unsupported_claims": unsupported,
        "normalized_text": normalize_query_text(text),
        "semantic_signature": semantic_signature(attributes),
        "semantic_group_id": semantic_group_id(attributes),
        "collision_count": int(collision_count),
        "neighbour_item_count": int(neighbour_item_count),
        "positive_set_size": int(positive_set_size),
        "stable_anchor_count": int(stable_anchor_count),
        "stable_identifiability": float(stable_identifiability),
    }
