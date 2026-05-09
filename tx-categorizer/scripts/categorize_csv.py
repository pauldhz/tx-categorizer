import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import predict_transaction


def detect_delimiter(file_path: Path) -> str:
    with file_path.open(encoding="utf-8") as f:
        first_line = f.readline()
    return "," if first_line.count(",") > first_line.count(";") else ";"


def categorize_csv(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)

    delimiter = detect_delimiter(input_file)
    print(f"ℹ️  Délimiteur détecté : '{delimiter}'")

    out_fieldnames = ["Date", "Type", "Description", "Montant signé", "Catégorie", "Sous catégorie"]

    rows = []
    with input_file.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in, delimiter=delimiter)

        for row in reader:
            description = row.get("description", "") or row.get("name", "")
            montant = float(row.get("amount", 0) or 0)

            result = predict_transaction({
                "date": row.get("date", ""),
                "type": row.get("type", ""),
                "description": description,
                "montant": montant,
            })

            rows.append({
                "Date": row.get("date", ""),
                "Type": row.get("type", ""),
                "Description": description,
                "Montant signé": montant,
                "Catégorie": result.get("category", ""),
                "Sous catégorie": result.get("subcategory", ""),
            })

    with output_file.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ {len(rows)} transactions traitées → {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/categorize_csv.py <input.csv> <output.csv>")
        sys.exit(1)

    categorize_csv(sys.argv[1], sys.argv[2])