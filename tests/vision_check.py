#!/usr/bin/env python3
"""Vision smoke test against a model served by LM Studio (OpenAI-compatible API).

Sends a small synthetic image with a known ground truth
(tests/fixtures/test-vision.png: red circle "7", blue square "4",
green triangle "2") and asks 12 easy questions about it, then prints a score.

    lms load <model-identifier> --gpu max -c 8192 -y
    python3 tests/vision_check.py <model-identifier>

A vision-capable model should score 12/12; a score near 4/12 means the model is
mostly guessing (the image was not really understood). Run the same command on
the source model and on the converted model to compare.
"""

import base64
import json
import sys
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:1234/v1/chat/completions"
IMAGE = Path(__file__).parent / "fixtures" / "test-vision.png"

QUESTIONS = [
    (
        "count",
        "How many separate shapes are in the image? Answer with just a number.",
        ["3"],
    ),
    ("tl_color", "What color is the shape in the top-left? One word.", ["red"]),
    ("tr_color", "What color is the shape in the top-right? One word.", ["blue"]),
    ("bt_color", "What color is the shape at the bottom? One word.", ["green"]),
    ("tl_digit", "What digit is inside the top-left shape? Just the digit.", ["7"]),
    ("tr_digit", "What digit is inside the top-right shape? Just the digit.", ["4"]),
    ("bt_digit", "What digit is inside the bottom shape? Just the digit.", ["2"]),
    (
        "tl_shape",
        "What shape is in the top-left: circle, square or triangle?",
        ["circle"],
    ),
    (
        "tr_shape",
        "What shape is in the top-right: circle, square or triangle?",
        ["square"],
    ),
    (
        "bt_shape",
        "What shape is at the bottom: circle, square or triangle?",
        ["triangle"],
    ),
    ("bt_lower", "Is the green shape below the red shape? yes or no.", ["yes"]),
    ("has_digits", "Are there any digits printed in this image? yes or no.", ["yes"]),
]


def ask(model: str, question: str, image_b64: str, max_tokens: int = 900) -> str:
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as resp:
        data = json.load(resp)
    return (data["choices"][0]["message"].get("content") or "").strip()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    model = sys.argv[1]
    image_b64 = base64.b64encode(IMAGE.read_bytes()).decode()

    score = 0
    for name, question, expected in QUESTIONS:
        answer = ask(model, question, image_b64)
        hit = any(w in answer.lower() for w in expected)
        score += hit
        print(
            f"[{name:11s}] expect~{expected[0]:8s} got={answer[:40]!r:44s} "
            f"{'OK' if hit else 'MISS'}"
        )
    print(f"SCORE {score}/{len(QUESTIONS)}  model={model}")
    return 0 if score >= len(QUESTIONS) else 1


if __name__ == "__main__":
    sys.exit(main())
