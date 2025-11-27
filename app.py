# app.py
from flask import Flask, request, render_template, redirect, url_for
import os
import re
import json
from collections import Counter

import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords

# Try to download NLTK data (safe if already present)
nltk_data_needed = ["punkt", "stopwords"]
for pkg in nltk_data_needed:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except Exception:
        try:
            nltk.download(pkg)
        except Exception:
            pass

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Simple curated skills dictionary for common roles ===
JOB_PROFILES = {
    "software_engineer": {
        "title": "Software Engineer",
        "skills": ["python","java","c++","data structures","algorithms","git","sql","rest","api","linux","docker","aws","html","css","javascript"]
    },
    "data_scientist": {
        "title": "Data Scientist",
        "skills": ["python","pandas","numpy","scikit-learn","machine learning","deep learning","tensorflow","pytorch","statistics","sql","visualization","matplotlib","seaborn"]
    },
    "frontend_engineer": {
        "title": "Frontend Engineer",
        "skills": ["javascript","react","vue","angular","html","css","typescript","webpack","responsive","sass"]
    },
    "devops_engineer": {
        "title": "DevOps / SRE",
        "skills": ["docker","kubernetes","ci/cd","aws","gcp","azure","terraform","bash","monitoring","prometheus","ansible"]
    }
}

# A small extended skill synonyms mapping (normalize common variants)
SKILL_SYNONYMS = {
    "ml": "machine learning",
    "dl": "deep learning",
    "tv": "tensorflow",
    "tf": "tensorflow",
    "js": "javascript",
    "py": "python",
    "db": "database",
    "sql": "sql"
}

STOP = set(stopwords.words("english"))

def clean_and_tokenize(text):
    # lowercase, remove non-alphanumeric (keep + # for things like c++)
    text = text.lower()
    text = re.sub(r"[^a-z0-9+\#\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP and len(t) > 1]
    return tokens

def extract_text_from_pdf(path):
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
                text.append(page_text)
            except Exception:
                pass
    return "\n".join(text)

def extract_text_from_docx(path):
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def extract_text(path):
    lower = path.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif lower.endswith(".docx"):
        return extract_text_from_docx(path)
    else:
        # try reading as text fallback
        try:
            with open(path, "r", encoding="utf8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

def normalize_skill(token):
    token = token.strip()
    if token in SKILL_SYNONYMS:
        return SKILL_SYNONYMS[token]
    return token

def find_skills(tokens):
    tokens_set = set(tokens)
    found = set()

    # Check multi-word skills first from JOB_PROFILES
    multi_skills = set()
    for profile in JOB_PROFILES.values():
        for s in profile["skills"]:
            if " " in s:
                multi_skills.add(s)

    joined = " ".join(tokens)
    for ms in multi_skills:
        if ms in joined:
            found.add(ms)

    # Check single-word skills
    for t in tokens:
        nt = normalize_skill(t)
        if nt in tokens_set:
            # later we will match against profile skills
            found.add(nt)

    # additionally return frequent tokens as keywords
    freq = Counter(tokens)
    top = [w for w, c in freq.most_common(40)]
    return list(found), top

def score_for_profile(found_skills, profile_skills):
    # count matches (case-insensitive)
    profile_set = set([s.lower() for s in profile_skills])
    found_set = set([s.lower() for s in found_skills])
    matches = profile_set.intersection(found_set)
    score = int(100 * len(matches) / max(1, len(profile_set)))
    missing = sorted(profile_set - matches)
    return score, sorted(list(matches)), missing

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    resume_text = ""
    if request.method == "POST":
        f = request.files.get("resume")
        if not f:
            return redirect(url_for("home"))

        filename = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(filename)

        text = extract_text(filename)
        resume_text = text[:5000]  # preview

        tokens = clean_and_tokenize(text)
        found_multi, top_keywords = find_skills(tokens)

        # Build a found_skills list: include top keywords + multi-word found
        found_skills = set(found_multi)
        for kw in top_keywords:
            found_skills.add(normalize_skill(kw))

        # Score against each job profile
        profile_scores = {}
        for key, profile in JOB_PROFILES.items():
            score, matches, missing = score_for_profile(found_skills, profile["skills"])
            profile_scores[key] = {
                "title": profile["title"],
                "score": score,
                "matches": matches,
                "missing": missing
            }

        # Best match
        best = max(profile_scores.items(), key=lambda x: x[1]["score"])[1]

        # Basic readability: average words per sentence
        sentences = nltk.sent_tokenize(text)
        words = len(tokens)
        avg_words_per_sentence = round(words / max(1, len(sentences)), 1)

        # Simple ATS friendly suggestions
        suggestions = []
        if best["score"] < 60:
            suggestions.append("Include more role-specific keywords (e.g., tools, libraries, languages).")
        if avg_words_per_sentence > 25:
            suggestions.append("Shorten long sentences — keep bullet points for accomplishments.")
        if len(tokens) < 200:
            suggestions.append("Add more concrete project details and metrics (e.g., reduced latency by 30%).")

        result = {
            "profile_scores": profile_scores,
            "best_match": best,
            "top_keywords": top_keywords[:30],
            "preview": resume_text,
            "avg_words_per_sentence": avg_words_per_sentence,
            "suggestions": suggestions
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
