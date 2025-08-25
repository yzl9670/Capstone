# feedback.py
# Single-pass rubric grading with robust JSON parsing. No LEED, no vector DB.

import os
import json
import re
import logging
from typing import Any, Dict, List, Optional

import openai

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -----------------------------------------------------------------------------
# OpenAI setup
# -----------------------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logging.error("OpenAI API key not found. Please set OPENAI_API_KEY.")
    raise EnvironmentError("OpenAI API key not found. Please set OPENAI_API_KEY.")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -----------------------------------------------------------------------------
# Public API
#   - grade_with_rubric(text, rubric, model)
#   - render_feedback_text(rubric, items, total, overall_label)
# -----------------------------------------------------------------------------

def grade_with_rubric(
    text: str,
    rubric: Dict[str, Any],
    model: Optional[str] = None,
    *,
    temperature: float = 0.0,
    max_input_chars: int = 12000,
    request_timeout: int = 30
) -> Dict[str, Any]:
    """
    Run a single LLM call to grade 'text' against 'rubric'.

    Expected rubric shape (from app.py):
      {
        "course_name": str,
        "total_points": int,
        "scale": [3,2,1],             # allowed scores per criterion
        "labels": ["Excellent", ...], # overall labels from best to worst
        "criteria": [
            {"name": "Correctness", "desc": "", "levels": {"3":"...", "2":"...", "1":"..."}},
            ...
        ]
      }

    Return shape (must match app.py usage):
      {
        "items": [ {"name": str, "score": number, "reasons": str}, ... ],
        "total": number,
        "overall_label": str,
        "raw_model_json": dict
      }
    """
    assert isinstance(text, str) and text.strip(), "text must be a non-empty string"
    assert isinstance(rubric, dict) and rubric.get("criteria"), "rubric must contain 'criteria'"

    model = model or DEFAULT_MODEL

    # 1) Prepare input: trim overly long text to keep latency manageable
    text = _normalize_whitespace(text)
    if len(text) > max_input_chars:
        logging.info(f"Input exceeds {max_input_chars} chars; truncating.")
        text = text[:max_input_chars]

    # 2) Build a compact rubric spec for the model
    criteria = rubric.get("criteria", [])
    scale = rubric.get("scale", [3, 2, 1])
    labels = rubric.get("labels", ["Excellent", "Satisfactory", "Needs Improvement"])
    max_points_per_item = max(scale) if scale else 3
    total_points = rubric.get("total_points", len(criteria) * max_points_per_item)

    # 3) Compose prompt: force strict JSON output
    system_msg = (
        "You are a strict writing TA. "
        "Grade the given student text against the rubric. "
        "Only respond with a valid JSON object. No markdown, no commentary."
    )

    user_msg = _build_prompt_json_only(text, rubric)

    # 4) Call the model
    try:
        logging.debug(f"Calling OpenAI model: {model}")
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=800,
            timeout=request_timeout,
        )
        content = resp["choices"][0]["message"]["content"]
    except Exception as e:
        logging.exception("OpenAI call failed")
        # Hard fallback: minimal neutral grading
        fallback = _fallback_grading(criteria, scale, labels, total_points)
        return fallback

    # 5) Parse/repair JSON
    data = _parse_json_strict(content)
    if data is None:
        logging.warning("Strict JSON parse failed. Attempting to extract/repair JSON.")
        data = _parse_json_relaxed(content)
    if data is None:
        logging.error("Failed to parse model JSON. Using fallback grading.")
        return _fallback_grading(criteria, scale, labels, total_points)

    # 6) Validate & coerce JSON into expected shape
    items = _coerce_items(data.get("items", []), criteria, scale, max_points_per_item)
    total = _safe_number(data.get("total", sum(i["score"] for i in items)))
    # If the model didn't provide a sensible label, compute one
    overall_label = data.get("overall_label") or _compute_overall_label(
        items, total_points, labels
    )

    result = {
        "items": items,
        "total": total,
        "overall_label": overall_label,
        "raw_model_json": data,
    }
    logging.debug("Grading completed.")
    return result


def render_feedback_text(
    rubric: Dict[str, Any],
    items: List[Dict[str, Any]],
    total: float,
    overall_label: str
) -> str:
    """
    Render a concise, student-facing feedback block.

    This string is what the frontend shows inside the chat bubble.
    (Frontend currently hides numeric scores on the right panel, but we keep them here for transparency.)
    """
    course = rubric.get("course_name", "Course")
    scale = rubric.get("scale", [3, 2, 1])
    max_points_per_item = max(scale) if scale else 3
    total_points = rubric.get("total_points", len(items) * max_points_per_item)

    lines = []
    header = f"{course} – Overall: {overall_label}  |  Total: {round(total, 2)}/{total_points}"
    lines.append(header)
    lines.append("")  # blank line

    for it in items:
        nm = it.get("name", "Unnamed Criterion")
        sc = _safe_number(it.get("score", 0))
        rs = (it.get("reasons") or "").strip()
        # Keep reasons short and readable
        rs = _truncate(rs, 400)
        lines.append(f"- {nm}: {sc}/{max_points_per_item} — {rs}")

    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------------

def _build_prompt_json_only(text: str, rubric: Dict[str, Any]) -> str:
    """
    Build a single message that instructs the model to return JSON only.
    """
    scale = rubric.get("scale", [3, 2, 1])
    labels = rubric.get("labels", ["Excellent", "Satisfactory", "Needs Improvement"])
    criteria = rubric.get("criteria", [])
    total_points = rubric.get("total_points")  # may be None

    # Keep the rubric block compact but explicit
    rubric_block = {
        "scale": scale,
        "labels": labels,
        "total_points": total_points,
        "criteria": [
            {
                "name": c.get("name", ""),
                "desc": c.get("desc", ""),
                "levels": c.get("levels", {}),
            }
            for c in criteria
        ],
    }

    # JSON response contract
    contract = {
        "items": [
            {
                "name": "<criterion name copied exactly from rubric>",
                "score": "<one of scale values, numeric>",
                "reasons": "<1–2 sentences explaining the score with concrete evidence from the text>",
            }
        ],
        "total": "<sum of item scores as a number>",
        "overall_label": f"<one of {labels}>",
    }

    return (
        "Grade the STUDENT_TEXT using the RUBRIC. "
        "Follow these rules strictly:\n"
        "1) Score each criterion with one of the allowed 'scale' values only.\n"
        "2) Provide 1–2 concise sentences of reasons per item (no bullet lists).\n"
        "3) Sum scores into 'total'.\n"
        "4) Choose 'overall_label' from the given labels only.\n"
        "5) Respond with a single valid JSON object, no markdown, no commentary.\n\n"
        f"RUBRIC = {json.dumps(rubric_block, ensure_ascii=False)}\n\n"
        f"STUDENT_TEXT = {json.dumps(text, ensure_ascii=False)}\n\n"
        f"RESPONSE_SHAPE = {json.dumps(contract, ensure_ascii=False)}"
    )

# -----------------------------------------------------------------------------
# JSON parsing & coercion helpers
# -----------------------------------------------------------------------------

def _parse_json_strict(s: str) -> Optional[Dict[str, Any]]:
    """Try a strict JSON parse."""
    try:
        return json.loads(s)
    except Exception:
        return None


JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    flags=re.DOTALL | re.IGNORECASE,
)

BRACE_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _parse_json_relaxed(s: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from common patterns:
    - ```json ... ```
    - the largest {...} block
    Then attempt to minimally repair and parse.
    """
    match = JSON_BLOCK_RE.search(s)
    candidate = match.group(1) if match else None

    if not candidate:
        # Fall back to the largest {...} block
        match2 = BRACE_RE.search(s)
        if match2:
            candidate = match2.group(0)

    if not candidate:
        return None

    # Very light repair: remove trailing commas in objects/arrays
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        return json.loads(candidate)
    except Exception:
        return None


def _coerce_items(
    items: Any,
    criteria: List[Dict[str, Any]],
    scale: List[float],
    max_points_per_item: float
) -> List[Dict[str, Any]]:
    """
    Ensure we have one item per rubric criterion in order.
    Clamp/normalize scores to allowed scale. Provide safe reasons.
    """
    scale_set = set(float(x) for x in scale)
    # Build quick lookup from model output (if present)
    model_map = {}
    if isinstance(items, list):
        for obj in items:
            if isinstance(obj, dict) and "name" in obj:
                model_map[str(obj["name"]).strip()] = obj

    out: List[Dict[str, Any]] = []
    for c in criteria:
        name = c.get("name", "Unnamed Criterion")
        obj = model_map.get(name, {})
        raw_score = _safe_number(obj.get("score", min(scale)))
        # snap to nearest allowed score
        score = _nearest_allowed(raw_score, scale_set)
        reasons = str(obj.get("reasons", "")).strip() or "No reasons provided."

        out.append({
            "name": name,
            "score": float(score),
            "reasons": _truncate(reasons, 600),
        })
    return out


def _nearest_allowed(x: float, allowed: set) -> float:
    """Snap numeric x to the nearest value in 'allowed'."""
    try:
        return min(allowed, key=lambda a: abs(a - float(x)))
    except Exception:
        return min(allowed) if allowed else float(x)


def _safe_number(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _compute_overall_label(
    items: List[Dict[str, Any]],
    total_points: Optional[float],
    labels: List[str]
) -> str:
    """
    Compute a coarse overall label if the model didn't provide one.
    Uses percentage thresholds mapped across the given labels.
    """
    if not items:
        return labels[-1] if labels else "Overall"

    earned = sum(_safe_number(it.get("score", 0.0)) for it in items)
    if not total_points or total_points <= 0:
        # fallback if rubric omitted total_points
        max_item = max((it.get("score", 0.0) for it in items), default=3.0)
        total_points = len(items) * (max_item if max_item > 0 else 3.0)

    pct = earned / float(total_points)

    # Map percentages to ordered labels (best -> worst)
    # e.g., 0.85+ -> labels[0], 0.6+ -> labels[1], else labels[-1]
    if len(labels) >= 3:
        if pct >= 0.85:
            return labels[0]
        elif pct >= 0.60:
            return labels[1]
        else:
            return labels[-1]
    elif len(labels) == 2:
        return labels[0] if pct >= 0.75 else labels[1]
    elif len(labels) == 1:
        return labels[0]
    return "Overall"


def _fallback_grading(
    criteria: List[Dict[str, Any]],
    scale: List[float],
    labels: List[str],
    total_points: Optional[float]
) -> Dict[str, Any]:
    """
    Used when the OpenAI call fails or JSON cannot be parsed.
    Assign the lowest allowed score with a generic reason.
    """
    min_score = min(scale) if scale else 1
    items = [{
        "name": c.get("name", "Unnamed Criterion"),
        "score": float(min_score),
        "reasons": "Fallback grading used due to a parsing or API error.",
    } for c in criteria]

    total = sum(it["score"] for it in items)
    overall = _compute_overall_label(items, total_points, labels)
    return {
        "items": items,
        "total": total,
        "overall_label": overall,
        "raw_model_json": {"error": "fallback"},
    }

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _normalize_whitespace(s: str) -> str:
    """Collapse excessive whitespace to keep the prompt tight."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # compress 3+ newlines to 2
    s = re.sub(r"\n{3,}", "\n\n", s)
    # trim super long spaces
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _truncate(s: str, max_len: int) -> str:
    """Truncate long strings with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"
