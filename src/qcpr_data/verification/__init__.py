"""Human-verification gates; generated text cannot be promoted implicitly."""

from .promotion import can_promote_text
from .reviewers import validate_review_pair

__all__ = ["can_promote_text", "validate_review_pair"]
