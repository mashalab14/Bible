# import_kjv_usfx.py
# Ingest KJV (USFX XML) into Supabase/Postgres `public.verses`
# Requires: SUPABASE_DB_URL env var (keyword DSN or URL with sslmode=require)

import os, re, sys, hashlib
from pathlib import Path
import psycopg2
from lxml import etree

# -------- Config --------
DB_DSN = os.getenv("SUPABASE_DB_URL")
if not DB_DSN:
    print("Set SUPABASE_DB_URL. Example:", file=sys.stderr)
    print("  export SUPABASE_DB_URL=\"host=db.<REF>.supabase.co port=5432 dbname=postgres user=postgres password='YOUR_PASS' sslmode=require\"", file=sys.stderr)
    sys.exit(1)

USFX_PATH = Path("data/kjv.usfx.xml")  # produced by: unzip -p data/kjv_usfx.zip eng-kjv_usfx.xml > data/kjv.usfx.xml

# USFX book id -> OSIS short code
USFX_TO_OSIS = {
    "GEN":"Gen","EXO":"Exod","LEV":"Lev","NUM":"Num","DEU":"Deut","JOS":"Josh","JDG":"Judg","RUT":"Ruth",
    "1SA":"1Sam","2SA":"2Sam","1KI":"1Kgs","2KI":"2Kgs","1CH":"1Chr","2CH":"2Chr","EZR":"Ezra","NEH":"Neh",
    "EST":"Esth","JOB":"Job","PSA":"Ps","PRO":"Prov","ECC":"Eccl","SNG":"Song","ISA":"Isa","JER":"Jer",
    "LAM":"Lam","EZE":"Ezek","DAN":"Dan","HOS":"Hos","JOL":"Joel","AMO":"Amos","OBA":"Obad","JON":"Jonah",
    "MIC":"Mic","NAM":"Nah","HAB":"Hab","ZEP":"Zeph","HAG":"Hag","ZEC":"Zech","MAL":"Mal",
    "MAT":"Matt","MRK":"Mark","LUK":"Luke","JHN":"John","ACT":"Acts","ROM":"Rom","1CO":"1Cor","2CO":"2Cor",
    "GAL":"Gal","EPH":"Eph","PHP":"Phil","COL":"Col","1TH":"1Thess","2TH":"2Thess","1TI":"1Tim","2TI":"2Tim",
    "TIT":"Titus","PHM":"Phlm","HEB":"Heb","JAS":"Jas","1PE":"1Pet","2PE":"2Pet","1JN":"1John","2JN":"2John",
    "3JN":"3John","JUD":"Jude","REV":"Rev"
}

# OSIS short -> human-friendly book name
OSIS_TO_NAME = {
    "Gen":"Genesis","Exod":"Exodus","Lev":"Leviticus","Num":"Numbers","Deut":"Deuteronomy",
    "Josh":"Joshua","Judg":"Judges","Ruth":"Ruth","1Sam":"1 Samuel","2Sam":"2 Samuel",
    "1Kgs":"1 Kings","2Kgs":"2 Kings","1Chr":"1 Chronicles","2Chr":"2 Chronicles",
    "Ezra":"Ezra","Neh":"Nehemiah","Esth":"Esther","Job":"Job","Ps":"Psalms","Prov":"Proverbs",
    "Eccl":"Ecclesiastes","Song":"Song of Solomon","Isa":"Isaiah","Jer":"Jeremiah","Lam":"Lamentations",
    "Ezek":"Ezekiel","Dan":"Daniel","Hos":"Hosea","Joel":"Joel","Amos":"Amos","Obad":"Obadiah",
    "Jonah":"Jonah","Mic":"Micah","Nah":"Nahum","Hab":"Habakkuk","Zeph":"Zephaniah","Hag":"Haggai",
    "Zech":"Zechariah","Mal":"Malachi","Matt":"Matthew","Mark":"Mark","Luke":"Luke","John":"John","Acts":"Acts",
    "Rom":"Romans","1Cor":"1 Corinthians","2Cor":"2 Corinthians","Gal":"Galatians","Eph":"Ephesians","Phil":"Philippians",
    "Col":"Colossians","1Thess":"1 Thessalonians","2Thess":"2 Thessalonians","1Tim":"1 Timothy","2Tim":"2 Timothy",
    "Titus":"Titus","Phlm":"Philemon","Heb":"Hebrews","Jas":"James","1Pet":"1 Peter","2Pet":"2 Peter",
    "1John":"1 John","2John":"2 John","3John":"3 John","Jude":"Jude","Rev":"Revelation"
}

# -------- Helpers --------
def readability_grade(text: str) -> float:
    words = re.findall(r"\w+", text)
    sents = re.findall(r"[.!?]", text)
    syll  = sum(len(re.findall(r"[aeiouy]+", w.lower())) for w in words) or 1
    return round(0.39*(len(words)/max(1,len(sents))) + 11.8*(syll/max(1,len(words))) - 15.59, 1)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

SKIP_NAMES = {"note","xref","f","footnote","x","rf","fn"}  # ignore footnotes/xrefs

def localname(tag: str) -> str:
    # strip namespace: {ns}name -> name
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def parse_usfx_stream(path: Path):
    """
    Proper USFX streaming parser:
    - Handles <v id="..."/> milestones with text in following siblings
    - Tracks current book and chapter
    - Accumulates text until the next verse/chapter
    - Ignores notes/xrefs
    """
    if not path.exists():
        raise FileNotFoundError(path)

    current_book_code = None
    current_book_name = None
    current_osis_book = None
    current_chapter = None
    current_verse = None
    buffer = []

    # iterparse is memory-friendly and lets us follow tails
    for event, elem in etree.iterparse(str(path), events=("start","end")):
        name = localname(elem.tag)

        if event == "start":
            # Entering a book
            if name == "book" and current_osis_book is None:
                # Start of a book context
                current_book_code = elem.attrib.get("id")
                current_osis_book = USFX_TO_OSIS.get(current_book_code)
                current_book_name = OSIS_TO_NAME.get(current_osis_book, current_osis_book) if current_osis_book else None

            # Chapter marker
            elif name in ("c","chapter"):
                chap_str = elem.attrib.get("id") or elem.attrib.get("number") or ""
                chap = int(re.sub(r"\D","", chap_str) or "0")
                if chap > 0:
                    # flush any open verse before chapter changes
                    if current_verse is not None and buffer and current_osis_book and current_chapter:
                        text = " ".join(s.strip() for s in buffer if s and s.strip())
                        yield make_row(current_osis_book, current_book_name, current_chapter, current_verse, text)
                    current_chapter = chap
                    current_verse = None
                    buffer = []

            # Verse marker (milestone or container)
            elif name in ("v","verse"):
                vnum_str = elem.attrib.get("id") or elem.attrib.get("number") or ""
                vnum = int(re.sub(r"\D","", vnum_str) or "0")
                if vnum > 0:
                    # flush previous verse
                    if current_verse is not None and buffer and current_osis_book and current_chapter:
                        text = " ".join(s.strip() for s in buffer if s and s.strip())
                        yield make_row(current_osis_book, current_book_name, current_chapter, current_verse, text)
                    current_verse = vnum
                    buffer = []
                # If this <verse> has inline text, capture it too (rare)
                if current_verse and current_chapter and name not in SKIP_NAMES:
                    if elem.text and elem.text.strip():
                        buffer.append(elem.text)

            else:
                # Accumulate text while inside an active verse
                if current_verse and current_chapter and name not in SKIP_NAMES:
                    if elem.text and elem.text.strip():
                        buffer.append(elem.text)

        else:  # event == "end"
            # Collect tails while inside a verse
            if current_verse and current_chapter and name not in SKIP_NAMES:
                if elem.tail and elem.tail.strip():
                    buffer.append(elem.tail)

            # Closing a book: flush any open verse, then reset book context
            if name == "book":
                if current_verse is not None and buffer and current_osis_book and current_chapter:
                    text = " ".join(s.strip() for s in buffer if s and s.strip())
                    yield make_row(current_osis_book, current_book_name, current_chapter, current_verse, text)
                current_book_code = None
                current_osis_book = None
                current_book_name = None
                current_chapter = None
                current_verse = None
                buffer = []
            # Free memory
            elem.clear()

def make_row(osis_book, book_name, chapter, verse, text):
    ref_display = f"{book_name} {chapter}:{verse}"
    clean = re.sub(r"\s+", " ", text).strip()
    return {
        "osis_id": f"{osis_book}.{chapter}.{verse}",
        "translation": "KJV",
        "book": book_name,
        "chapter": chapter,
        "verse": verse,
        "ref_display": ref_display,
        "text": clean,
        "char_count": len(clean),
        "word_count": len(re.findall(r"\w+", clean)),
        "reading_grade": readability_grade(clean),
        "text_hash": sha1(clean),
    }

# -------- DB upsert --------
UPSERT_SQL = """
insert into public.verses
  (osis_id, translation, book, chapter, verse, ref_display, text,
   char_count, word_count, reading_grade, text_hash)
values (%(osis_id)s, %(translation)s, %(book)s, %(chapter)s, %(verse)s, %(ref_display)s, %(text)s,
        %(char_count)s, %(word_count)s, %(reading_grade)s, %(text_hash)s)
on conflict (osis_id, translation) do update
set text = excluded.text,
    char_count = excluded.char_count,
    word_count = excluded.word_count,
    reading_grade = excluded.reading_grade,
    text_hash = excluded.text_hash;
"""

def main():
    if not USFX_PATH.exists():
        print(f"Missing file: {USFX_PATH}. Extract it with:", file=sys.stderr)
        print("  unzip -p data/kjv_usfx.zip eng-kjv_usfx.xml > data/kjv.usfx.xml", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    rows = 0
    try:
        for row in parse_usfx_stream(USFX_PATH):
            # Skip empty verses (some files have blank milestones)
            if not row["text"]:
                continue
            cur.execute(UPSERT_SQL, row)
            rows += 1
            if rows % 2000 == 0:
                conn.commit()
                print(f"... {rows} rows committed")
        conn.commit()
    finally:
        cur.close()
        conn.close()

    print(f"Inserted/updated {rows} KJV verses from USFX.")

if __name__ == "__main__":
    main()