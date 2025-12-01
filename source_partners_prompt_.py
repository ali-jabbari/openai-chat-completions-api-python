from __future__ import annotations

import os
import sys
from openai import OpenAI
from typing import Literal


max_words = 70

SAMPLE_DATA = {
    "country": "CHINA",
    "HS": "4015.12",
    "product": "Surgical Gloves",
    "top_importer": [
        {
            "Importer": "USA",
            "total_value": 962122000,
            "total_qty": 76655200,
            "average": 12.55129462841399,
            "unit": "Kg",
            "c_name": "United States",
        },
        {
            "Importer": "JPN",
            "total_value": 311500000,
            "total_qty": 94255200,
            "average": 3.3048574508356037,
            "unit": "Kg",
            "c_name": "Japan",
        },
        {
            "Importer": "CAN",
            "total_value": 75038400,
            "total_qty": 5978540,
            "average": 12.55129178695802,
            "unit": "Kg",
            "c_name": "Canada",
        },
        {
            "Importer": "FRA",
            "total_value": 74877900,
            "total_qty": 16584300,
            "average": 4.51498706608057,
            "unit": "Kg",
            "c_name": "France",
        },
        {
            "Importer": "DEU",
            "total_value": 74695500,
            "total_qty": 13883800,
            "average": 5.380047249312148,
            "unit": "Kg",
            "c_name": "Germany",
        },
    ],
}


def build_market_prompt(data: object):
    return [
        {
            "role": "system",
            "content": (
                "Act as a market research expert. "
                "Using ONLY the JSON importer data provided by the user (no external data), "
                f"generate a short insight under {max_words} words. "
                "Output must use simple HTML tags like <p>, <b>, <br>. "
            ),
        },
        {"role": "user", "content": f"Here is the data in JSON:\n{data}"},
    ]


def generate_market_insight(data: object) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("ERROR: Set the OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-5-mini",
        top_p=1,
        presence_penalty=0,
        frequency_penalty=0,
        messages=build_market_prompt(data),
    )
    return completion.choices[0].message.content


def main() -> int:
    try:
        result = generate_market_insight(SAMPLE_DATA)
        print(result)
        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
