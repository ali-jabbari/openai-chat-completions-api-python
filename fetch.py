from __future__ import annotations

import os
import sys
from typing import Iterable

from openai import OpenAI


product_name = "potato starch"
product_hs_code = "1108.13"

market_stats = [
    {"year": 2024, "value": 168965030, "qty": 160993476.66, "avg": 1.0495},
    {"year": 2023, "value": 168639450, "qty": 140369402.97, "avg": 1.2014},
    {"year": 2022, "value": 161627550, "qty": 157935832, "avg": 1.0234},
    {"year": 2021, "value": 122944090, "qty": 156156357, "avg": 0.7873},
    {"year": 2020, "value": 113388610, "qty": 132741849, "avg": 0.8542},
]
unit = "Kg"


def format_market_stats(rows: Iterable[dict]) -> str:
    """Return the newline-separated stats block used in the user prompt."""
    return "\n".join(
        f"{row['year']} – value: {row['value']}, qty: {row['qty']}, avg: {row['avg']}"
        for row in rows
    )


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set the OPENAI_API_KEY environment variable.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)
    stats_text = format_market_stats(market_stats)

    completion = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        top_p=1,
        presence_penalty=0,
        frequency_penalty=0,
        messages=[
            {
                "role": "system",
                "content": "You are a concise business analyst. Write short, high-quality market insights under 100 words.",
            },
            {
                "role": "user",
                "content": (
                    f"Short, business-style insight. Summarize the US import market potential for "
                    f"{product_name} (HS {product_hs_code}) using the following data. Write under 100 words, "
                    "concise, analytical, and decision-focused for a Chilean exporter.\n\n"
                    f"{stats_text}"
                    f" the unit is {unit}"
                ),
            },
        ],
    )

    print(completion.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
