"""
context_features.py
Context-aware urgency features for Sinhala and Tamil transcripts.

Eight features per language (16 total: 8 sin_ + 8 tam_):
  danger_count          — fire, smoke, flood, trapped, breathing difficulty
  immediacy_count       — now, quickly, hurry
  help_request_count    — help, save us, please come, distress exclamation
  safety_state_count    — explicitly safe (true safety only, not location)
  location_context_count— neighbour's house, nearby, upstairs (NOT subtracted)
  high_risk_phrase_count— high-value two-word patterns
  person_at_risk_count  — child, elderly
  context_score         — danger + immediacy + high_risk_phrase
                          + person_at_risk + help_request - safety_state

Word lists built from:
  - Sinhala fine-tune transcripts of all 129 Sinhala calls (TF-IDF ranked)
  - Whisper-base transcripts of all 66 Tamil calls (TF-IDF ranked)
  - Manual audit of 30 Sinhala + 10 Tamil calls
"""

from __future__ import annotations

PHRASE_WINDOW = 6   # max token distance for a two-word phrase to count


# ═══════════════════════════════════════════════════════════════════════════════
# SINHALA
# ═══════════════════════════════════════════════════════════════════════════════

SINHALA_DANGER = [
    # Fire — all surface forms found in fine-tune output
    "ගිනිගන්නව", "ගින්නක්", "ගිනිඅරගෙන", "ගිනිගෙ", "ගිනිගන්",
    "ගින්න", "ඇවිල්ල", "පැතිරෙනානෙ",
    # Smoke
    "දුම", "දුමක්",
    # Water / flood — only when inside/submerged, not water alone
    "ගංවතුර", "ගංවතුරට", "යටවෙලා", "ගලා", "ගං",
    # Trapped / inside
    "හිරවෙලා", "සිරවෙලා", "ඉරවෙලා",
    "ඇතුළට", "ඇතුට", "ඇතුරට",
    # Breathing difficulty
    "හුස්ම ගන්නත් අමාරු", "හුස්ම", "ඔස්මගන්න", "හෘස්ම", "උස්ම",
    "අමාරුයි", "අමාරුයි.", "බැහැ",
]

SINHALA_IMMEDIACY = [
    "ඉක්මනට", "ඉක්මනින්", "ඉහ්මනින්", "ඉග්මනින්", "ඉක්මනකින්",
    "දැන්ම", "වේගෙන්", "කරුණාකර",
]

SINHALA_HELP_REQUEST = [
    "උදව්", "උදව්කරන්න", "උදව් කරන්න",
    "බේරගන්න", "හදිස්සිය",
    "අනේ",        # distress exclamation — appears in 13 High calls
    "එන්න",       # come! (imperative)
    "කරන්න",      # do it / help (imperative context)
]

SINHALA_SAFETY_STATE = [
    # True safety — caller/victim confirmed safe
    "ආරක්ෂිතයි", "ආරක්ෂිතනයි", "ආරක්ෂිත",
]

SINHALA_LOCATION_CONTEXT = [
    # Caller is NOT at the scene — reduces but does not eliminate urgency
    # The RF learns this signal; we do NOT subtract it from context_score
    "අල්ලපු", "අල්ලපුගෙදර",   # neighbour's house
    "ළඟ", "ළඟට", "ළඟටම",       # near but not at location
    "උඩ තට්ටු", "උඩ මහළ",      # upstairs (temporary safety)
    "සමහර විට",                 # maybe / possibly
]

SINHALA_PERSON_AT_RISK = [
    "ළමයා", "ළමයිය", "ළමේ", "ළහමයා",   # child / children
    "අම්මා", "තාත්තා",                   # mother / father
    "අප්පච්චි", "ආච්චි", "සීයා",         # grandparents
]

# Two-word phrase patterns within PHRASE_WINDOW tokens.
# Single-element tuple → single word is high-risk on its own.
SINHALA_HIGH_RISK_PHRASES = [
    # House on fire
    ("ගෙදර",   "ගිනිගන්නව"),
    ("ගෙදර",   "ගින්නක්"),
    ("ගෙදර",   "ගිනිඅරගෙන"),
    ("ගෙදර",   "ඇවිල්ල"),
    ("ගෙතරම",  "ගිනිගන්නව"),
    ("ගෙතරම",  "ගින්නක්"),
    ("සාල",    "ගිනිගන්නව"),
    # Water entering house
    ("වතුර",   "ඇතුළට"),
    ("වතුර",   "ඇතුට"),
    ("ගෙදර",   "යටවෙලා"),
    # Trapped / cannot escape
    ("හිරවෙලා", "ඉන්නේ"),
    ("හිරවෙලා", "ඉන්නෙ"),
    ("හිරවෙලා", "ඉන්නවා"),
    ("ඉරවෙලා",  "ඉන්නේ"),
    # Cannot breathe
    ("හුස්ම",  "අමාරු"),
    ("හුස්ම",  "බැහැ"),
    ("ඔස්මගන්න", "බැහැ"),
    ("උස්ම",   "අමාරු"),
    # Child in immediate danger
    ("ළමයා",  "ගං"),
    ("ළමයිය", "ගිනි"),
    ("ළමේ",   "ගිනි"),
    # Urgent help request
    ("උදව්",  "ඉක්මනට"),
    ("උදව්",  "දැන්ම"),
    ("කරුණාකර", "උදව්"),
    ("අනේ",   "උදව්"),
    # Single-word — always high risk regardless of context
    ("බේරගන්න",),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TAMIL
# ═══════════════════════════════════════════════════════════════════════════════

TAMIL_DANGER = [
    # Fire — canonical + inflected forms
    "தீ", "தீயில்", "தீப்பிடித்", "தீபராவது",
    # Smoke
    "புகை",
    # Water / flood
    "வெள்ளம்", "வெள்ளனிர்", "தண்ணீர்",
    # Breathing
    "மூச்சு",
    # Trapped
    "சிக்கி", "சிக்கிக்கொண்",
    # Injury / accident
    "இரத்தம்", "காயம்", "விபத்து", "விபத்தைப்",
]

TAMIL_IMMEDIACY = [
    "இப்போதே", "உடனே",
    "சீக்கிரம்", "சேக்கிரம்", "சிக்கிரம்",   # spelling variants
    "விரைவாக", "தயவுசெய்து",
]

TAMIL_HELP_REQUEST = [
    "உதவி", "ஓ்தவி",                           # canonical + variant
    "உதவுங்கள்", "உதவி செய்யுங்கள்",
    "செய்யுங்கள்",    # imperative — do it / come help
    "வாருங்கள்",      # come (imperative)
    "ஐயோ",            # distress exclamation (appears in 2 High calls)
]

TAMIL_SAFETY_STATE = [
    "பாதுகாப்பாக", "பாதுகாப்பு",
]

TAMIL_LOCATION_CONTEXT = [
    "அருகில்",            # nearby
    "அக்கம்பக்கத்து",    # neighbour's
    "மேல் தளம்",         # upper floor
]

TAMIL_PERSON_AT_RISK = [
    "குழந்தை", "குழந்தைகள்", "குளந்தைகளுடன்",   # child / children + variant
]

TAMIL_HIGH_RISK_PHRASES = [
    # House on fire
    ("வீட்டில்",    "தீ"),
    ("வீட்டினுள்",  "தீ"),
    ("விட்டுக்குள்", "தீ"),
    # Water inside house
    ("தண்ணீர்",    "உள்ளே"),
    ("விட்டுக்குள்", "தண்ணீர்"),
    ("உள்ளே",      "தண்ணீர்"),
    # Child in danger
    ("குழந்தை",    "தீ"),
    ("குழந்தை",    "வெள்ளம்"),
    ("குளந்தைகளுடன்", "தீ"),
    # Cannot breathe
    ("மூச்சு",     "முடியவில்லை"),
    # Trapped
    ("சிக்கி",     "மாட்டிக்கொண்"),
    # Urgent help
    ("உதவி",       "இப்போதே"),
    ("உதவி",       "உடனே"),
    ("ஓ்தவி",      "இப்போதே"),
    # Single-word high risk
    ("உதவுங்கள்",),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Matching helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _word_in_text(text: str, word: str) -> bool:
    return word in text


def _phrase_match(text: str, phrase: tuple) -> bool:
    if len(phrase) == 1:
        return _word_in_text(text, phrase[0])
    w1, w2 = phrase[0], phrase[1]
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if w1 in tok:
            lo = max(0, i - PHRASE_WINDOW)
            hi = min(len(tokens), i + PHRASE_WINDOW + 1)
            if any(w2 in tokens[j] for j in range(lo, hi)):
                return True
    return False


def _count_words(text: str, word_list: list) -> int:
    return sum(1 for w in word_list if _word_in_text(text, w))


def _count_phrases(text: str, phrase_list: list) -> int:
    return sum(1 for p in phrase_list if _phrase_match(text, p))


def _zero(prefix: str) -> dict:
    return {
        f"{prefix}_danger_count":           0,
        f"{prefix}_immediacy_count":        0,
        f"{prefix}_help_request_count":     0,
        f"{prefix}_safety_state_count":     0,
        f"{prefix}_location_context_count": 0,
        f"{prefix}_high_risk_phrase_count": 0,
        f"{prefix}_person_at_risk_count":   0,
        f"{prefix}_context_score":          0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_sinhala_context_features(text: str) -> dict:
    if not text or not text.strip():
        return _zero("sin")
    danger   = _count_words(text, SINHALA_DANGER)
    immediacy= _count_words(text, SINHALA_IMMEDIACY)
    help_req = _count_words(text, SINHALA_HELP_REQUEST)
    safety   = _count_words(text, SINHALA_SAFETY_STATE)
    location = _count_words(text, SINHALA_LOCATION_CONTEXT)
    phrase   = _count_phrases(text, SINHALA_HIGH_RISK_PHRASES)
    person   = _count_words(text, SINHALA_PERSON_AT_RISK)
    # context_score counts unique concept categories that fired (max 5)
    # using min(count,1) prevents double-counting overlapping matches
    score = (min(danger,1) + min(immediacy,1) + min(help_req,1)
             + min(phrase,1) + min(person,1) - min(safety,1))
    return {
        "sin_danger_count":           danger,
        "sin_immediacy_count":        immediacy,
        "sin_help_request_count":     help_req,
        "sin_safety_state_count":     safety,
        "sin_location_context_count": location,
        "sin_high_risk_phrase_count": phrase,
        "sin_person_at_risk_count":   person,
        "sin_context_score":          score,
    }


def extract_tamil_context_features(text: str) -> dict:
    if not text or not text.strip():
        return _zero("tam")
    danger   = _count_words(text, TAMIL_DANGER)
    immediacy= _count_words(text, TAMIL_IMMEDIACY)
    help_req = _count_words(text, TAMIL_HELP_REQUEST)
    safety   = _count_words(text, TAMIL_SAFETY_STATE)
    location = _count_words(text, TAMIL_LOCATION_CONTEXT)
    phrase   = _count_phrases(text, TAMIL_HIGH_RISK_PHRASES)
    person   = _count_words(text, TAMIL_PERSON_AT_RISK)
    score = (min(danger,1) + min(immediacy,1) + min(help_req,1)
             + min(phrase,1) + min(person,1) - min(safety,1))
    return {
        "tam_danger_count":           danger,
        "tam_immediacy_count":        immediacy,
        "tam_help_request_count":     help_req,
        "tam_safety_state_count":     safety,
        "tam_location_context_count": location,
        "tam_high_risk_phrase_count": phrase,
        "tam_person_at_risk_count":   person,
        "tam_context_score":          score,
    }


def extract_context_features(text: str, language: str) -> dict:
    """
    Language-aware dispatcher. Returns 16 features total (8 sin_ + 8 tam_).
    Non-relevant language features are always zero.

    language == 'sinhala' → sin_* filled, tam_* = 0
    language == 'tamil'   → tam_* filled, sin_* = 0
    otherwise             → all zero (English/Singlish/Tanglish use keyword_count)
    """
    lang = language.lower().strip()
    if lang == "sinhala":
        return {**extract_sinhala_context_features(text), **_zero("tam")}
    if lang == "tamil":
        return {**_zero("sin"), **extract_tamil_context_features(text)}
    return {**_zero("sin"), **_zero("tam")}


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic helper
# ═══════════════════════════════════════════════════════════════════════════════

def explain(text: str, language: str) -> None:
    lang = language.lower().strip()
    print(f"\n{'─'*60}")
    print(f"Language  : {language}")
    print(f"Transcript: {text[:120]}")
    print()

    if lang == "sinhala":
        d = [w for w in SINHALA_DANGER          if w in text]
        i = [w for w in SINHALA_IMMEDIACY       if w in text]
        h = [w for w in SINHALA_HELP_REQUEST    if w in text]
        s = [w for w in SINHALA_SAFETY_STATE    if w in text]
        l = [w for w in SINHALA_LOCATION_CONTEXT if w in text]
        p = [w for w in SINHALA_PERSON_AT_RISK  if w in text]
        ph= [x for x in SINHALA_HIGH_RISK_PHRASES if _phrase_match(text, x)]
    elif lang == "tamil":
        d = [w for w in TAMIL_DANGER            if w in text]
        i = [w for w in TAMIL_IMMEDIACY         if w in text]
        h = [w for w in TAMIL_HELP_REQUEST      if w in text]
        s = [w for w in TAMIL_SAFETY_STATE      if w in text]
        l = [w for w in TAMIL_LOCATION_CONTEXT  if w in text]
        p = [w for w in TAMIL_PERSON_AT_RISK    if w in text]
        ph= [x for x in TAMIL_HIGH_RISK_PHRASES if _phrase_match(text, x)]
    else:
        print("  (no context features for this language)")
        return

    print(f"  Danger        : {d  or '—'}")
    print(f"  Immediacy     : {i  or '—'}")
    print(f"  Help request  : {h  or '—'}")
    print(f"  Safety state  : {s  or '—'}")
    print(f"  Location      : {l  or '—'}")
    print(f"  Person at risk: {p  or '—'}")
    print(f"  High-risk phrs: {ph or '—'}")

    feats  = extract_context_features(text, language)
    prefix = "sin" if lang == "sinhala" else "tam"
    print(f"\n  danger={feats[f'{prefix}_danger_count']}  "
          f"immediacy={feats[f'{prefix}_immediacy_count']}  "
          f"help={feats[f'{prefix}_help_request_count']}  "
          f"safety={feats[f'{prefix}_safety_state_count']}  "
          f"location={feats[f'{prefix}_location_context_count']}  "
          f"phrase={feats[f'{prefix}_high_risk_phrase_count']}  "
          f"person={feats[f'{prefix}_person_at_risk_count']}  "
          f"score={feats[f'{prefix}_context_score']}")
