# import_kjv.py
import os, re, sys, hashlib
from pathlib import Path
import psycopg2
from lxml import etree

DB_DSN = os.getenv("SUPABASE_DB_URL")
if not DB_DSN:
    print("Set SUPABASE_DB_URL (with sslmode=require).", file=sys.stderr); sys.exit(1)

KJV_PATH = Path("data/kjv.osis.xml")

BOOK_NAME_MAP = {
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

def readability_grade(text: str) -> float:
    words = re.findall(r"\w+", text)
    sents = re.findall(r"[.!?]", text)
    syll  = sum(len(re.findall(r"[aeiouy]+", w.lower())) for w in words) or 1
    return round(0.39*(len(words)/max(1,len(sents))) + 11.8*(syll/max(1,len(words))) - 15.59, 1)

def extract_plain_text(node):
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()

def parse_osis(path: Path):
    tree = etree.parse(str(path))
    for v in tree.xpath("//verse[@osisID]"):
        osis_id = v.attrib["osisID"]              # e.g., 1Pet.5.7
        parts = osis_id.split(".")
        if len(parts) < 3:
            continue
        book_code = parts[0]
        chapter = int(re.sub(r"\D","",parts[1]) or "0")
        verse = int(re.sub(r"\D","",parts[2]) or "0")
        book = BOOK_NAME_MAP.get(book_code, book_code)
        ref_display = f"{book} {chapter}:{verse}"
        text = extract_plain_text(v)
        yield {
            "osis_id": osis_id,
            "translation": "KJV",
            "book": book,
            "chapter": chapter,
            "verse": verse,
            "ref_display": ref_display,
            "text": text,
            "char_count": len(text),
            "word_count": len(re.findall(r"\w+", text)),
            "reading_grade": readability_grade(text),
            "text_hash": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        }

def main():
    if not KJV_PATH.exists():
        print(f"Missing file: {KJV_PATH}.", file=sys.stderr); sys.exit(1)

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    rows = 0
    for row in parse_osis(KJV_PATH):
        cur.execute("""
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
        """, row)
        rows += 1
        if rows % 2000 == 0:
            conn.commit()

    conn.commit()
    cur.close(); conn.close()
    print(f"Inserted/updated {rows} KJV verses.")

if __name__ == "__main__":
    main()