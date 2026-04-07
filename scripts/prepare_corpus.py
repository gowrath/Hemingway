from __future__ import annotations

import re
from pathlib import Path
from urllib.request import Request, urlopen

SOURCES = [
    {
        "title": "In Our Time",
        "url": "https://www.gutenberg.org/ebooks/61085.txt.utf-8",
    },
    {
        "title": "The Sun Also Rises",
        "url": "https://www.gutenberg.org/ebooks/67138.txt.utf-8",
    },
    {
        "title": "Men Without Women",
        "url": "https://www.gutenberg.org/ebooks/69683.txt.utf-8",
    },
    {
        "title": "A Farewell to Arms",
        "url": "https://www.gutenberg.org/ebooks/75201.txt.utf-8",
    },
    {
        "title": "Three Stories & Ten Poems",
        "url": "https://www.gutenberg.org/ebooks/59603.txt.utf-8",
    },
]

START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)
MULTIBLANK_RE = re.compile(r"\n{3,}")


def download_text(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; HemingwayCorpusBuilder/1.0)"
        },
    )
    with urlopen(request) as response:
        raw = response.read()
    return raw.decode("utf-8-sig", errors="replace")



def strip_gutenberg_boilerplate(text: str) -> str:
    start_match = START_RE.search(text)
    if start_match:
        text = text[start_match.end():]

    end_match = END_RE.search(text)
    if end_match:
        text = text[:end_match.start()]

    return text.strip()



def drop_obvious_front_matter_lines(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        lower = stripped.lower()
        if lower.startswith("produced by"):
            continue
        if lower.startswith("credits:"):
            continue
        if lower.startswith("release date:"):
            continue
        if lower.startswith("language:"):
            continue
        if lower.startswith("character set encoding:"):
            continue
        if lower.startswith("most recently updated:"):
            continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = MULTIBLANK_RE.sub("\n\n", text)
    return text.strip()



def prepare_book(title: str, url: str) -> str:
    print(f"Downloading: {title}")
    text = download_text(url)
    text = strip_gutenberg_boilerplate(text)
    text = drop_obvious_front_matter_lines(text)
    return f"\n\n### {title} ###\n\n{text}\n"



def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "hemingway.txt"

    books = [prepare_book(item["title"], item["url"]) for item in SOURCES]
    merged = "\n".join(books).strip() + "\n"

    output_path.write_text(merged, encoding="utf-8")
    print(f"\nWrote cleaned corpus to: {output_path.resolve()}")
    print(f"Books included: {len(SOURCES)}")


if __name__ == "__main__":
    main()
