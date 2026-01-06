import sys
import json
import os
import time
import re
from datetime import datetime

# Optional: lightweight NLP snapshots or mock source check
TRUSTED_SOURCES_MOCK = [
    "earthquake.gov", "weather.service", "official.news", "confirmed.report"
]

def extract_claims(text):
    """
    Step 1: Claim Extraction (NLP/Rule-based)
    Extracts factual assertions.
    """
    # Simple regex for breaking news patterns
    breaking_patterns = [
        r"(?i)breaking", r"(?i)just in", r"(?i)happening now", r"(?i)confirmed",
        r"(?i)massive", r"(?i)hits", r"(?i)killed", r"(?i)dead"
    ]
    
    claims = []
    for p in breaking_patterns:
        if re.search(p, text):
            claims.append(f"High-urgency claim: {re.findall(p, text)[0]}")
            
    # Look for location/event patterns
    # (In prod, use spaCy or small BERT on ONNX)
    return claims

def classify_claim_type(text):
    """
    Step 2: Claim Type Classification
    """
    categories = {
        "disaster": [r"earthquake", r"flood", r"fire", r"storm", r"hurricane"],
        "politics": [r"election", r"president", r"protest", r"senate", r"vote"],
        "health": [r"virus", r"outbreak", r"vaccine", r"pandemic"],
        "breaking": [r"breaking", r"live now", r"alert"]
    }
    
    for cat, patterns in categories.items():
        for p in patterns:
            if re.search(p, text.lower()):
                return cat
    return "general"

def check_temporal_consistency(caption, media_path):
    """
    Step 3: Temporal & Context Check (CRITICAL)
    Compares media creation age with caption 'now' claims.
    """
    is_breaking_claim = any(x in caption.lower() for x in ["today", "now", "breaking", "just in", "happening"])
    
    try:
        # Get file creation time
        stat = os.stat(media_path)
        # Using modification time as creation time (OS dependent, but good proxy)
        creation_time = datetime.fromtimestamp(stat.st_mtime)
        now = datetime.now()
        
        # Age in hours
        age_hours = (now - creation_time).total_seconds() / 3600
        
        if is_breaking_claim and age_hours > 24:
            return 1.0, f"Context Mismatch: Media is {int(age_hours)}h old, but caption claims it is current/breaking."
        
        return 0.0, None
    except Exception as e:
        return 0.5, f"Temporal Check Error: {str(e)}"

def check_patterns(text):
    """
    Step 5: Known Misinfo Patterns
    """
    score = 0.0
    reasons = []
    
    # Emotional urgency
    if re.search(r"!!+|MUST WATCH|UNBELIEVABLE|SHOCKING", text):
        score += 0.3
        reasons.append("Sensationalist language patterns")
        
    # Absolute claims
    if re.search(r"(?i)everyone", r"(?i)confirmed by all", r"(?i)100% true"):
        score += 0.2
        reasons.append("Suspicious absolute claim markers")
        
    return score, reasons

def verify_context(caption, media_path):
    """
    Scoring & Decision
    """
    start_time = time.time()
    reasons = []
    
    # 1. Claim Check
    claims = extract_claims(caption)
    if not claims and len(caption.split()) < 10:
        # Too short/no claims = allow
        return {"verdict": "ALLOW", "score": 0.0, "reasons": []}
    
    # 2. Temporal Consistency (High Weight)
    temp_score, temp_reason = check_temporal_consistency(caption, media_path)
    if temp_reason:
        reasons.append(temp_reason)
        
    # 3. Pattern Check
    pattern_score, pattern_reasons = check_patterns(caption)
    reasons.extend(pattern_reasons)
    
    # 4. Source Check (Mocked for Demo)
    # If it's a disaster claim but we don't 'know' about it
    source_score = 0.0
    claim_type = classify_claim_type(caption)
    if claim_type in ["disaster", "breaking"] and temp_score > 0.5:
        source_score = 0.6
        reasons.append(f"Unverified {claim_type} claim: No matching reports from trusted bulletins.")

    # Weighted Scoring
    # fake_news_score = 0.4*Temporal + 0.4*Source + 0.2*Pattern
    final_score = (0.4 * temp_score) + (0.4 * source_score) + (0.2 * pattern_score)
    final_score = min(1.0, round(final_score, 2))
    
    # Decisions (Strict)
    verdict = "ALLOW"
    if final_score >= 0.60:
        verdict = "BLOCK_FAKE"
    elif final_score >= 0.40:
        verdict = "BLOCK_UNVERIFIED"

    return {
        "verdict": verdict,
        "score": final_score,
        "reasons": reasons,
        "metadata": {
            "claim_type": claim_type,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"verdict": "ERROR", "score": 1.0, "reasons": ["Missing caption or file path"]}))
        sys.exit(1)
        
    caption = sys.argv[1]
    file_path = sys.argv[2]
    
    try:
        result = verify_context(caption, file_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"verdict": "ERROR", "score": 1.0, "reasons": [str(e)]}))
        sys.exit(1)
