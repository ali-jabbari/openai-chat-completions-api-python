from __future__ import annotations

import os
import sys
from openai import OpenAI
from typing import Literal

max_words = 70;

SAMPLE_DATA = {
    "country": "US",
    "HS": "4015.12",
    "product": "Surgical Gloves",
    "years": [
        {
            "year": 2024,
            "total_value": 1320446810,
            "total_qty": 105204077.787,
            "average": 12.551289244,
        },
        {
            "year": 2023,
            "total_value": 1163561990,
            "total_qty": 80305952.363,
            "average": 14.489112647,
        },
        {
            "year": 2022,
            "total_value": 1512841920,
            "total_qty": 117122206.825,
            "average": 12.916781207,
        },
        {
            "year": 2021,
            "total_value": 3763354160,
            "total_qty": 282750357.556,
            "average": 13.309812205,
        },
        {
            "year": 2020,
            "total_value": 3292970450,
            "total_qty": 271402849.106,
            "average": 12.133146210,
        },
    ]
}

def build_market_prompt(data: object):
    return [
        {
            "role": "system",
            "content": (
                "Act as a market research expert. "
                "Using ONLY the JSON import data provided by the user (no external data), "
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
