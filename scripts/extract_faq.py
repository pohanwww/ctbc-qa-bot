"""
CTBC FAQ Extractor.

Extracts Q&A pairs from manually collected CTBC Bank FAQ HTML.

Usage:
    python -m scripts.extract_faq

    Or with options:
    python -m scripts.extract_faq --input qa_manual_collect.txt --output data/processed/ctbc_faq
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# Category mapping based on data-qaid prefix
CATEGORY_MAP = {
    "cc_product": "信用卡 - 產品介紹",
    "cc_finances": "信用卡 - 信用卡理財",
    "cc_feedback": "信用卡 - 客戶回饋計劃",
    "cc_additional": "信用卡 - 附加功能/服務",
    "cc_ctbc_security": "信用卡 - 網路刷卡安全",
    "cc_acquirer": "信用卡 - 特約商店服務",
    "cc_loss": "信用卡 - 卡片遺失與毀損",
    "cc_mobilepayment": "信用卡 - 行動支付",
    "cc_activtate": "信用卡 - 開卡",
    "cc_feepayment": "信用卡 - 代繳各項費用",
    "cc_dispute": "信用卡 - 刷卡疑慮",
    "cc_limit": "信用卡 - 信用卡額度",
    "cc_statement": "信用卡 - 帳務與使用",
    "cc_notice": "信用卡 - 消費通知與使用",
    "cc_taxes": "信用卡 - 繳稅相關",
    "cc_pay": "信用卡 - 繳款與入帳",
    "cc_tca": "信用卡 - 分期靈活金",
    "loan_personal": "貸款 - 信用貸款",
    "loan_mortgage": "貸款 - 房屋貸款",
    "loan_car": "貸款 - 汽車貸款",
    "loan_other": "貸款 - 其他貸款",
    "deposit": "存款服務",
    "transfer": "轉帳服務",
    "foreign": "外匯服務",
    "atm": "ATM服務",
    "online": "網路銀行",
    "mobile": "行動銀行",
    "general": "一般服務",
}


def extract_category(qaid: str) -> str:
    """Extract category from data-qaid attribute."""
    if not qaid:
        return "一般服務"

    # Try to match known prefixes
    for prefix, category in CATEGORY_MAP.items():
        if qaid.startswith(prefix):
            return category

    # Extract category from qaid pattern (e.g., cc_dispute_qa_006 -> 信用卡)
    parts = qaid.split("_")
    if len(parts) >= 2:
        main_cat = parts[0]
        if main_cat == "cc":
            return "信用卡"
        elif main_cat == "loan":
            return "貸款服務"
        elif main_cat == "deposit":
            return "存款服務"
        elif main_cat == "transfer":
            return "轉帳服務"
        elif main_cat in ("foreign", "fx"):
            return "外匯服務"

    return "一般服務"


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Clean up common patterns
    text = text.strip()

    return text


def clean_html_content(html_content: str) -> str:
    """Clean HTML content while preserving some structure."""
    if not html_content:
        return ""

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, "lxml")

    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()

    # Replace <br> with newlines
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Get text, using separator for block elements
    text = soup.get_text(separator=" ")

    # Clean up whitespace
    lines = []
    for line in text.split("\n"):
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)

    return " ".join(lines)


def extract_faq_from_html(html_content: str) -> list[dict]:
    """
    Extract FAQ entries from HTML content.

    Args:
        html_content: Raw HTML string

    Returns:
        List of FAQ entry dictionaries
    """
    soup = BeautifulSoup(html_content, "lxml")
    faq_entries = []

    # Find all FAQ toggle elements
    faq_elements = soup.find_all("div", class_="twrbo-c-toggle--qna")

    logger.info(f"Found {len(faq_elements)} FAQ elements")

    for element in faq_elements:
        # Extract data-qaid for ID and category
        qaid = element.get("data-qaid", "")

        # Find question (title)
        title_elem = element.find("div", class_="twrbo-c-toggle__title")
        if not title_elem:
            continue

        question = clean_text(title_elem.get_text())
        if not question or len(question) < 5:
            continue

        # Find answer (content)
        content_elem = element.find("div", class_="twrbo-c-toggle__content")
        if not content_elem:
            continue

        # Get answer as cleaned HTML then text
        answer = clean_html_content(str(content_elem))
        if not answer or len(answer) < 10:
            continue

        # Generate unique ID
        content_hash = hashlib.md5(f"{question}{answer}".encode()).hexdigest()[:8]
        entry_id = f"ctbc_{qaid}" if qaid else f"ctbc_faq_{content_hash}"

        # Extract category
        category = extract_category(qaid)

        faq_entries.append(
            {
                "id": entry_id,
                "qaid": qaid,
                "category": category,
                "question_zh": question,
                "answer_zh": answer,
                "question_en": "",  # Can be filled with translation later
                "answer_en": "",
                "source_url": "https://www.ctbcbank.com/faq",
            }
        )

    return faq_entries


def save_faq_entries(entries: list[dict], output_dir: Path) -> Path:
    """Save FAQ entries to JSONL file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "ctbc_faq.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(entries)} FAQ entries to {output_file}")

    # Save metadata
    categories = {}
    for entry in entries:
        cat = entry.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "CTBC Bank FAQ (manually collected)",
                "total_entries": len(entries),
                "categories": categories,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"Saved metadata to {metadata_file}")
    return output_file


def main() -> None:
    """Main entry point for FAQ extraction."""
    parser = argparse.ArgumentParser(description="Extract FAQ from CTBC HTML")

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("qa_manual_collect.txt"),
        help="Input HTML file (default: qa_manual_collect.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ctbc_faq"),
        help="Output directory (default: data/processed/ctbc_faq)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Check input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    # Read HTML content
    logger.info(f"Reading HTML from {args.input}")
    with open(args.input, encoding="utf-8") as f:
        html_content = f.read()

    logger.info(f"Read {len(html_content)} characters")

    # Extract FAQ entries
    entries = extract_faq_from_html(html_content)

    if not entries:
        logger.warning("No FAQ entries extracted!")
        return

    # Print category statistics
    logger.info("\n=== Category Statistics ===")
    categories = {}
    for entry in entries:
        cat = entry.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count}")

    # Save entries
    save_faq_entries(entries, args.output)

    logger.info(f"\n✅ Successfully extracted {len(entries)} FAQ entries!")


if __name__ == "__main__":
    main()
