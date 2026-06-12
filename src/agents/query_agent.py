"""
src/agents/query_agent.py — ADVANCED QUERY UNDERSTANDING AGENT

Upgrades from rag_pipeline.py:
  ✔ Multi-intent detection (not just single winner)
  ✔ Question decomposition for compound/complex queries
  ✔ Ambiguity scoring
  ✔ Entity extraction (Kannada + English topics/concepts)
  ✔ Query complexity classification → guides retrieval strategy
  ✔ Smart query rewriting with synonym expansion (moved from rag_pipeline.py)

Replaces:
  - classify_question()   → now multi-intent
  - rewrite_query()       → now entity-aware
"""

import re
import unicodedata
from typing import List, Dict, Tuple

# ── Intent patterns ──────────────────────────────────────────

_INTENT_PATTERNS: List[Tuple[str, List[str], float]] = [
    ("definition",  [r"ಎಂದರೇನು", r"ವ್ಯಾಖ್ಯಾನ", r"ಅರ್ಥ", r"what is", r"define", r"definition", r"meaning"], 1.0),
    ("procedure",   [r"ಹೇಗೆ", r"ಯಾವ ರೀತಿ", r"steps", r"how to", r"process", r"ಪ್ರಕ್ರಿಯೆ"], 1.0),
    ("benefits",    [r"ಪ್ರಯೋಜನ", r"ಲಾಭ", r"benefit", r"uses", r"advantages", r"importance"], 1.0),
    ("comparison",  [r"ವ್ಯತ್ಯಾಸ", r"difference", r"compare", r"\bvs\b", r"ಹೋಲಿಸಿ"], 1.0),
    ("list",        [r"ಪ್ರಕಾರಗಳು", r"ಪಟ್ಟಿ", r"types", r"list", r"kinds", r"examples of"], 0.8),
    ("example",     [r"ಉದಾಹರಣೆ", r"example", r"instance", r"such as"], 0.8),
    ("formula",     [r"ಸೂತ್ರ", r"formula", r"equation", r"calculate", r"value of"], 1.0),
    ("cause",       [r"ಏಕೆ", r"ಕಾರಣ", r"why", r"reason", r"because", r"leads to"], 1.0),
    ("story",       [r"ಕಥೆ", r"what happened", r"history of", r"ಇತಿಹಾಸ"], 0.8),
    ("explanation", [r"ವಿವರಣೆ", r"explain", r"describe", r"ವಿಶ್ಲೇಷಣೆ"], 0.9),
    ("negation",    [r"ಅಲ್ಲ", r"not", r"except", r"without", r"ಹೊರತು"], 0.7),
    ("quantity",    [r"ಎಷ್ಟು", r"how many", r"how much", r"count", r"number of"], 1.0),
]

_QUERY_EXPANSION: Dict[str, str] = {
    "definition":  "what is definition meaning ಎಂದರೇನು ಅರ್ಥ ವ್ಯಾಖ್ಯಾನ",
    "procedure":   "how steps process ಹೇಗೆ ಯಾವ ರೀತಿ ಪ್ರಕ್ರಿಯೆ",
    "benefits":    "advantages uses benefits ಪ್ರಯೋಜನ ಲಾಭ ಉಪಯೋಗ",
    "comparison":  "difference compare vs ವ್ಯತ್ಯಾಸ ಹೋಲಿಕೆ",
    "list":        "types list components ಪ್ರಕಾರಗಳು ವಿಧಗಳು",
    "example":     "example instance ಉದಾಹರಣೆ",
    "formula":     "formula calculate equation ಸೂತ್ರ ಲೆಕ್ಕ",
    "cause":       "why reason cause ಏಕೆ ಕಾರಣ",
    "story":       "what happened story history ಕಥೆ ಇತಿಹಾಸ",
    "explanation": "explain describe elaborate ವಿವರಣೆ ವಿಶ್ಲೇಷಣೆ",
    "concept":     "what explain overview ಏನು ವಿವರ",
    "negation":    "except not without difference",
    "quantity":    "how many count number total ಎಷ್ಟು ಸಂಖ್ಯೆ",
}

_SYNONYMS: Dict[str, List[str]] = {
    "ಹೇಗೆ":    ["ಯಾವ ರೀತಿ", "ಪ್ರಕ್ರಿಯೆ", "ವಿಧಾನ"],
    "ಏಕೆ":     ["ಕಾರಣ", "ಯಾಕೆ", "ಹೇತು"],
    "ಎಂದರೇನು": ["ವ್ಯಾಖ್ಯಾನ", "ಅರ್ಥ", "ಪರಿಕಲ್ಪನೆ"],
    "ಎಷ್ಟು":   ["ಸಂಖ್ಯೆ", "ಪ್ರಮಾಣ", "count"],
}

# Ambiguity signals
_AMBIGUITY_SIGNALS = [
    r"\bಅದು\b", r"\bಇದು\b", r"\bit\b", r"\bthis\b", r"\bthat\b",
    r"\bthe thing\b", r"\bಅವರು\b", r"\bಆ\b",   # pronouns without referent
]

# Compound query splitters
_COMPOUND_SPLITS = [
    r"\bಮತ್ತು\b", r"\band\b", r"\balso\b", r"\bಮಾತ್ರವಲ್ಲ\b",
    r"\bas well as\b", r"\bಜೊತೆಗೆ\b", r"[,;]\s+",
]


class QueryAgent:

    def run(self, state) -> object:
        q = (state.original_question or "").strip()
        if not q:
            return state

        q_nfc = unicodedata.normalize("NFC", q)
        q_lower = q_nfc.lower()

        # ── 1. Multi-intent detection ────────────────────────
        intent_scores: Dict[str, float] = {}
        for intent, patterns, weight in _INTENT_PATTERNS:
            score = sum(weight for p in patterns if re.search(p, q_lower))
            if score > 0:
                intent_scores[intent] = intent_scores.get(intent, 0) + score

        if intent_scores:
            sorted_intents = sorted(intent_scores, key=lambda x: -intent_scores[x])
            state.intent     = sorted_intents[0]
            state.sub_intents = sorted_intents[1:3]   # up to 2 secondary intents
        else:
            state.intent     = "concept"
            state.sub_intents = []

        # ── 2. Complexity classification ─────────────────────
        compound_matches = sum(
            1 for pat in _COMPOUND_SPLITS if re.search(pat, q_lower)
        )
        word_count = len(q_nfc.split())

        if compound_matches >= 2 or word_count > 25:
            state.query_complexity = "complex"
        elif compound_matches >= 1 or word_count > 12:
            state.query_complexity = "compound"
        else:
            state.query_complexity = "simple"

        # ── 3. Question decomposition (for compound/complex) ─
        state.decomposed_queries = self._decompose(q_nfc, state.query_complexity)

        # ── 4. Entity extraction ─────────────────────────────
        state.entities = self._extract_entities(q_nfc)

        # ── 5. Ambiguity scoring ─────────────────────────────
        amb_hits = sum(1 for p in _AMBIGUITY_SIGNALS if re.search(p, q_lower))
        state.ambiguity_score = min(1.0, amb_hits * 0.3)

        # ── 6. Query rewriting ───────────────────────────────
        state.rewritten_query = self._rewrite(
            q_nfc,
            state.intent,
            state.entities,
            state.sub_intents,
        )

        return state

    # ── Helpers ──────────────────────────────────────────────

    def _decompose(self, question: str, complexity: str) -> List[str]:
        """Split compound questions into atomic sub-queries."""
        if complexity == "simple":
            return [question]

        sub_qs = [question]  # always include original
        for pat in _COMPOUND_SPLITS:
            parts = re.split(pat, question)
            parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]
            if len(parts) > 1:
                sub_qs = parts
                break

        # Deduplicate while preserving order
        seen = set()
        result = []
        for q in sub_qs:
            norm = " ".join(q.lower().split())
            if norm not in seen:
                seen.add(norm)
                result.append(q)

        return result[:4]  # max 4 sub-queries

    def _extract_entities(self, question: str) -> List[str]:
        """
        Extract topic/concept entities from the question.
        Strategy: take meaningful Kannada words (3+ chars) and
        capitalized English words (likely proper nouns / terms).
        """
        entities = []

        # Kannada tokens of length >= 3 that are NOT stop words
        _KN_STOPS = {
            "ಮತ್ತು", "ಇದು", "ಅದು", "ಇದರ", "ಅದರ", "ಇವು", "ಅವು",
            "ಹೇಗೆ", "ಏಕೆ", "ಎಂದರೇನು", "ಯಾವ", "ಏನು", "ಎಷ್ಟು",
        }
        kn_tokens = re.findall(r'[\u0C80-\u0CFF]{3,}', question)
        for t in kn_tokens:
            if t not in _KN_STOPS:
                entities.append(t)

        # Capitalized English terms (potential technical terms)
        en_caps = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', question)
        entities.extend(en_caps)

        # Deduplicate
        seen = set()
        result = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                result.append(e)

        return result[:8]

    def _rewrite(self, question: str, intent: str,
                 entities: List[str], sub_intents: List[str]) -> str:
        """
        Build a rich embed-text for semantic retrieval.
        Includes: intent tag, synonym expansion, entity tags, sub-intent.
        """
        q = question
        # Append synonyms for known Kannada keywords
        for kw, syns in _SYNONYMS.items():
            if kw in q:
                q += " " + " ".join(syns[:2])

        # Primary intent expansion
        expansion = _QUERY_EXPANSION.get(intent, "")

        # Secondary intent expansions (partial)
        sub_expansion = " ".join(
            _QUERY_EXPANSION.get(si, "").split()[:3]
            for si in sub_intents
        )

        # Entity emphasis
        entity_str = " ".join(entities[:5]) if entities else ""

        parts = [
            f"intent: {intent}",
            f"expansion: {expansion}",
        ]
        if sub_expansion:
            parts.append(f"sub_intent: {sub_expansion}")
        if entity_str:
            parts.append(f"entities: {entity_str}")
        parts.append(q)

        return " | ".join(parts)