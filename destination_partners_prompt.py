from __future__ import annotations

import os
import sys
from openai import OpenAI
from typing import Literal


max_words = 70

SAMPLE_DATA = {
    "country": "US",
    "HS": "4015.12",
    "product": "Surgical Gloves",
    "top_exporter": [
        {
            "Exporter": "CHN",
            "total_value": 962122000,
            "total_qty": 76655200,
            "average": 12.55129462841399,
            "unit": "Kg",
            "c_name": "China",
        },
        {
            "Exporter": "CAN",
            "total_value": 188747000,
            "total_qty": 15038100,
            "average": 12.551253150331492,
            "unit": "Kg",
            "c_name": "Canada",
        },
        {
            "Exporter": "VNM",
            "total_value": 51933300,
            "total_qty": 4137680,
            "average": 12.551308946076063,
            "unit": "Kg",
            "c_name": "Vietnam",
        },
        {
            "Exporter": "KHM",
            "total_value": 33940900,
            "total_qty": 2704180,
            "average": 12.551272474465458,
            "unit": "Kg",
            "c_name": "Cambodia",
        },
        {
            "Exporter": "GTM",
            "total_value": 11709700,
            "total_qty": 932946,
            "average": 12.551315938971817,
            "unit": "Kg",
            "c_name": "Guatemala",
        },
    ],
}


def build_market_prompt(data: object):
    return [
        {
            "role": "system",
            "content": (
                "Act as a market research expert. "
                "Using ONLY the JSON exporter data provided by the user (no external data), "
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
