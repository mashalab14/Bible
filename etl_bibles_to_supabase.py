import os, re, sys, json, hashlib, argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional, List
import numpy as np
import psycopg2
from lxml import etree

# ========= Config =========
THEME_DESCRIPTIONS = {
    "comfort":   "assurance of God's care and relief from worry and burdens",
    "hope":      "expectation of God's future goodness and promises",
    "trust":     "placing confidence in God rather than self or fear",
    "wisdom":    "guidance, understanding, prudence, counsel for living",
    "forgiveness":"pardoning sin, mercy, letting go of offense",
    "love":      "love of God and neighbor, compassion, kindness",
    "joy":       "gladness, rejoicing, praise, thanksgiving",
    "strength":  "courage, perseverance, endurance through trials",
    "guidance":  "direction, path, lamp, shepherding and leading",
    "peace":     "rest, calm, stillness, absence of turmoil",
    "repentance":"turning from sin, confession, renewal",
    "healing":   "restoration, recovery, God's care in sickness",
    "generosity":"giving, sharing, kindness to others in need",
    "patience":  "waiting, longsuffering, self-control, restraint",
    "perseverance":"steadfastness under pressure, not giving up"
}
MOODS = ["anxious","tired","grateful","hopeful","sad","lonely","guilty","angry","bereaved"]
TONE_LABELS = ["calming","encouraging","corrective","celebratory","contemplative"]
DAYPARTS = ["morning","day","evening","night"]

# Safety keywords as backstop (semantic will handle most cases)
RE_VIOLENCE = re.compile(r"\b(slay|sword|blood|war|kill|stone|smite|spear|battle|strike)\b", re.I)
RE_SEXUAL   = re.compile(r"\b(adulter|fornication|prostitut|lust|naked|whore|harlot)\b", re.I)
RE_REBUKE   = re.compile(r"\b(woe|hypocrite|wrath|abomination|rebuke|condemn)\b", re.I)
RE_SLAVERY  = re.compile(r"\b(slave|bondservant|slave-master|slaves|bondservants)\b", re.I)

# ========= Embeddings (optional) =========
# If sentence-transformers is installed, we’ll use MiniLM for semantic tagging.
EMBED_DIM = 384
_model = None
def get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _model = None
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    m = get_model()
    if m is None:
        # Fallback: deterministic pseudo-embeddings (fine for schema tests; not for production tagging)
        rng = np.random.default_rng(42)
        vecs = rng.random((len(texts), EMBED_DIM)).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs
    vecs = m.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")

# Build tag centroids once
_tag_centroids = None
def build_tag_centroids() -> Dict[str, np.ndarray]:
    global _tag_centroids
    if _tag_centroids is not None:
        return _tag_centroids
    labels = []
    for tag, desc in THEME_DESCRIPTIONS.items():
        labels.append(f"{tag}: {desc}")
    theme_vecs = embed_texts(labels)
    _tag_centroids = {tag: theme_vecs[i] for i, tag in enumerate(THEME_DESCRIPTIONS.keys())}
    return _tag_centroids

def softmax(x):
    x = np.asarray(x, dtype="float32")
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).tolist()

# ========= OSIS parsing =========
BOOK_NAME_MAP = {  # minimal mapping; OSIS often already has nice names
    "Gen": "Genesis","Exod":"Exodus","Lev":"Leviticus","Num":"Numbers","Deut":"Deuteronomy",
    "Josh":"Joshua","Judg":"Judges","Ruth":"Ruth","1Sam":"1 Samuel","2Sam":"2 Samuel",
    "1Kgs":"1 Kings","2Kgs":"2 Kings","1Chr":"1 Chronicles","2Chr":"2 Chronicles",
    "Ezra":"Ezra","Neh":"Nehemiah","Esth":"Esther","Job":"Job","Ps":"Psalms","Prov":"Proverbs",
    "Eccl":"Ecclesiastes","Song":"Song of Solomon","Isa":"Isaiah","Jer":"Jeremiah","Lam":"Lamentations",
    "Ezek":"Ezekiel","Dan":"Daniel","Hos":"Hosea","Joel":"Joel","Amos":"Amos","Obad":"Obadiah",
    "Jonah":"Jonah","Mic":"Micah","Nah":"Nahum","Hab":"Habakkuk","Zeph":"Zephaniah","Hag":"Haggai",
    "Zech":"Zechariah","Mal":"Malachi",
    "Matt":"Matthew","Mark":"Mark","Luke":"Luke","John":"John","Acts":"Acts",
    "Rom":"Romans","1Cor":"1 Corinthians","2Cor":"2 Corinthians","Gal":"Galatians","Eph":"Ephesians",
    "Phil":"Philippians","Col":"Colossians","1Thess":"1 Thessalonians","2Thess":"2 Thessalonians",
    "1Tim":"1 Timothy","2Tim":"2 Timothy","Titus":"Titus","Phlm":"Philemon","Heb":"Hebrews",
    "Jas":"James","1Pet":"1 Peter","2Pet":"2 Peter","1John":"1 John","2John":"2 John","3John":"3 John",
    "Jude":"Jude","Rev":"Revelation"
}

def extract_plain_text(node) -> str:
    # Join text in mixed-content <verse> nodes, stripping extra whitespace
    text = "".join(node.itertext())
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_osis(path: Path) -> Iterable[Dict]:
    tree = etree.parse(str(path))
    # Verses typically use //verse[@osisID]
    for v in tree.xpath("//verse[@osisID]"):
        osis_id = v.attrib["osisID"]  # e.g., "1Pet.5.7"
        text = extract_plain_text(v)
        # Derive book code, chapter, verse from osisID
        parts = osis_id.split(".")
        if len(parts) < 3:
            continue
        book_code = parts[0]
        try:
            chapter = int(parts[1])
        except ValueError:
            # Some OSIS use book.chX.vY. Do a more defensive parse.
            chapter = int(re.sub(r"\D", "", parts[1]) or "0")
        verse_num = int(re.sub(r"\D", "", parts[2]) or "0")
        book = BOOK_NAME_MAP.get(book_code, book_code)
        ref_display = f"{book} {chapter}:{verse_num}"
        yield {
            "osis_id": osis_id,
            "book": book,
            "chapter": chapter,
            "verse": verse_num,
            "ref_display": ref_display,
            "text": text
        }

# ========= Tagging (semantic-first, keyword backstop) =========
def readability_grade(text: str) -> float:
    words = re.findall(r"\w+", text)
    sents = re.findall(r"[.!?]", text)
    syll = sum(len(re.findall(r"[aeiouy]+", w.lower())) for w in words) or 1
    return round(0.39*(len(words)/max(1,len(sents))) + 11.8*(syll/max(1,len(words))) - 15.59, 1)

def safety_flags(text: str) -> Dict[str, bool]:
    t = text
    violence = bool(RE_VIOLENCE.search(t))
    sexual = bool(RE_SEXUAL.search(t))
    rebuke = bool(RE_REBUKE.search(t))
    slavery = bool(RE_SLAVERY.search(t))
    kid_safe = not (violence or sexual)
    return {"violence": violence, "sexual": sexual, "slavery": slavery, "harsh_rebuke": rebuke, "kid_safe": kid_safe}

def daypart_probs_from_semantics(v_embed: Optional[np.ndarray]) -> List[float]:
    if v_embed is None:
        return [0.3,0.4,0.2,0.1]
    labels = [
        "morning: dawn, new mercies, light, beginning",
        "day: labor, walking, sunshine, activity",
        "evening: rest from labor, sunset, reflection",
        "night: darkness, fear, protection, rest, peace in the night"
    ]
    cents = embed_texts(labels)
    sims = (cents @ v_embed).tolist()
    return softmax(sims)

def tone_probs_from_semantics(v_embed: Optional[np.ndarray]) -> List[float]:
    if v_embed is None:
        return [0.4,0.3,0.1,0.1,0.1]
    labels = [
        "calming: gentle reassurance, peace, rest, comfort",
        "encouraging: hope, promise, joy, uplift",
        "corrective: rebuke, warning, command to change",
        "celebratory: praise, thanksgiving, rejoicing",
        "contemplative: wisdom, meditate, ponder, understanding"
    ]
    cents = embed_texts(labels)
    sims = (cents @ v_embed).tolist()
    return softmax(sims)

def theme_tags_from_semantics(v_embed: Optional[np.ndarray], text: str, top_k=3) -> List[str]:
    if v_embed is None:
        # fallback keyword-ish guesses
        t = text.lower()
        picks = []
        if any(w in t for w in ["peace","rest","refuge","care","burden","anxiety"]): picks.append("comfort")
        if "hope" in t or "promise" in t: picks.append("hope")
        if "trust" in t or "refuge" in t or "shield" in t: picks.append("trust")
        if "wisdom" in t or "understanding" in t: picks.append("wisdom")
        if not picks: picks = ["comfort"]
        return picks[:top_k]
    cents = build_tag_centroids()
    names = list(cents.keys())
    M = np.stack([cents[n] for n in names])  # [T, D]
    sims = (M @ v_embed).tolist()
    order = np.argsort(sims)[::-1]
    out = []
    for idx in order[:top_k]:
        if sims[idx] < 0.25:  # sanity threshold; tweak as you like
            continue
        out.append(names[idx])
    return out or ["comfort"]

def mood_tags_from_semantics(v_embed: Optional[np.ndarray], text: str, top_k=2) -> List[str]:
    # We approximate moods by reusing short descriptions
    labels = [
        "anxious: worry, fear, burdened, need reassurance",
        "tired: weary, exhausted, need strength and rest",
        "grateful: thankful, praise, appreciation",
        "hopeful: expectation of good, trust in future",
        "sad: sorrowful, downcast, lament",
        "lonely: alone, abandoned, need presence",
        "guilty: ashamed, confession, repentance",
        "angry: wrath, frustration, need gentleness",
        "bereaved: grief, loss, mourning"
    ]
    names = [l.split(":")[0] for l in labels]
    if v_embed is None:
        # crude fallback
        t = text.lower()
        picks = []
        if any(w in t for w in ["fear","anxiety","care","trouble"]): picks.append("anxious")
        if "weary" in t or "rest" in t: picks.append("tired")
        if "praise" in t or "thanks" in t: picks.append("grateful")
        if "hope" in t or "joy" in t: picks.append("hopeful")
        return (picks or ["hopeful"])[:top_k]
    cents = embed_texts(labels)
    sims = (cents @ v_embed).tolist()
    order = np.argsort(sims)[::-1]
    return [names[i] for i in order[:top_k]]

def familiarity_score(text: str) -> float:
    # Shorter and simpler verses get a small familiarity bump.
    L = len(text)
    base = 0.5 + max(0, (140 - L)) / 400.0
    return max(0.0, min(1.0, round(base, 3)))

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ========= DB I/O =========
def upsert_verse(cur, tr: str, row: Dict):
    cur.execute("""
      insert into verses(osis_id, translation, book, chapter, verse, ref_display, text,
                         char_count, word_count, reading_grade, text_hash)
      values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
      on conflict (osis_id, translation) do update
      set text=excluded.text,
          char_count=excluded.char_count,
          word_count=excluded.word_count,
          reading_grade=excluded.reading_grade,
          text_hash=excluded.text_hash
    """, (
        row["osis_id"], tr, row["book"], row["chapter"], row["verse"], row["ref_display"], row["text"],
        len(row["text"]), len(re.findall(r"\w+", row["text"])), readability_grade(row["text"]), sha1(row["text"])
    ))

def upsert_annotation(cur, tr: str, osis_id: str, ann: Dict, embed_vec: Optional[np.ndarray]):
    embed_param = None
    if embed_vec is not None:
        embed_param = list(map(float, embed_vec.tolist()))
    cur.execute("""
      insert into verse_annotations(osis_id, translation, themes, moods, daypart_probs, tone_probs,
                                    safety, familiarity, embedding)
      values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
      on conflict (osis_id, translation) do update
      set themes=excluded.themes,
          moods=excluded.moods,
          daypart_probs=excluded.daypart_probs,
          tone_probs=excluded.tone_probs,
          safety=excluded.safety,
          familiarity=excluded.familiarity,
          embedding=excluded.embedding
    """, (
        osis_id, tr, ann["themes"], ann["moods"], ann["daypart_probs"], ann["tone_probs"],
        json.dumps(ann["safety"]), ann["familiarity"], embed_param
    ))

# ========= Main =========
def main():
    ap = argparse.ArgumentParser("ETL: KJV/WEB/ASV -> Supabase")
    ap.add_argument("--dsn", default=os.getenv("SUPABASE_DB_URL"), help="Supabase Postgres connection URI")
    ap.add_argument("--kjv", default="data/kjv.osis.xml")
    ap.add_argument("--web", default="data/web.osis.xml")
    ap.add_argument("--asv", default="data/asv.osis.xml")
    ap.add_argument("--no-embeddings", action="store_true", help="Skip semantic embeddings (uses keyword-ish fallbacks)")
    ap.add_argument("--commit-size", type=int, default=2000, help="Rows per transaction commit")
    args = ap.parse_args()

    if not args.dsn:
        print("Missing DSN. Set SUPABASE_DB_URL or pass --dsn", file=sys.stderr)
        sys.exit(1)

    # Preload embedding model unless disabled
    use_embeddings = not args.no_embeddings and get_model() is not None
    if not use_embeddings:
        print("Embeddings disabled (no model found or --no-embeddings). Falling back to keyword-ish tagging.")

    files = [("KJV", Path(args.kjv)), ("WEB", Path(args.web)), ("ASV", Path(args.asv))]
    conn = psycopg2.connect(args.dsn)
    cur = conn.cursor()

    batch_texts: List[str] = []
    batch_meta:  List[Tuple[str,str,Dict]] = []  # (translation, osis_id, row)
    count = 0

    for tr, path in files:
        if not path.exists():
            print(f"[WARN] Missing file for {tr}: {path}", file=sys.stderr)
            continue
        print(f"Parsing {tr} from {path} ...")
        for row in parse_osis(path):
            # Write verse now
            upsert_verse(cur, tr, row)

            # Queue for annotation
            batch_texts.append(row["text"])
            batch_meta.append((tr, row["osis_id"], row))
            count += 1

            # Commit by chunks (and annotate)
            if len(batch_texts) >= args.commit_size:
                process_annotations(cur, batch_texts, batch_meta, use_embeddings)
                conn.commit()
                batch_texts.clear()
                batch_meta.clear()
                print(f"Committed {count} rows...")

    # Flush tail
    if batch_texts:
        process_annotations(cur, batch_texts, batch_meta, use_embeddings)
        conn.commit()
        print(f"Committed final {count} rows.")

    cur.close()
    conn.close()
    print("Done.")

def process_annotations(cur, texts: List[str], metas: List[Tuple[str,str,Dict]], use_embeddings: bool):
    vecs = None
    if use_embeddings:
        vecs = embed_texts(texts)
    for i, (tr, osis_id, row) in enumerate(metas):
        v_embed = vecs[i] if vecs is not None else None
        ann = {
            "themes": theme_tags_from_semantics(v_embed, row["text"]),
            "moods":  mood_tags_from_semantics(v_embed, row["text"]),
            "daypart_probs": daypart_probs_from_semantics(v_embed),
            "tone_probs":    tone_probs_from_semantics(v_embed),
            "safety":        safety_flags(row["text"]),
            "familiarity":   familiarity_score(row["text"])
        }
        upsert_annotation(cur, tr, osis_id, ann, v_embed)

if __name__ == "__main__":
    main()