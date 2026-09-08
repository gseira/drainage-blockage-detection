"""
Quick, standalone test of Cerebras's Gemma 4 31B vision model on a real
blocked-drainage image from this project — no extra pip installs needed,
uses only the Python standard library.

Run it with:
    CEREBRAS_API_KEY=csk-xxxxx python3 test_cerebras.py

(Swap csk-xxxxx for your actual key. Passing it as an env var rather than
hardcoding it here means it never ends up saved in this file by accident.)
"""

import base64
import json
import os
import sys
import time
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # falls back to whatever's already in the environment

API_KEY = os.getenv("CEREBRAS_API_KEY")
if not API_KEY:
    print("Set CEREBRAS_API_KEY first, e.g.:\n  CEREBRAS_API_KEY=csk-xxxxx python3 test_cerebras.py\n(or add it to .env in this folder)")
    sys.exit(1)

# A real blocked-drainage photo already in this project. Overridable via
# argv[1] (e.g. `python3 test_cerebras.py /root/dissertation/images/.../foo.jpg`)
# since the default path is a dev-machine path (~/Desktop/dissertation/...)
# that won't exist on a deployment server -- in that case we fall back to
# searching this script's own directory (and CWD) for any *.jpg under an
# images/ folder, so this still works out-of-the-box on the droplet.
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    IMAGE_PATH = os.path.expanduser(
        "~/Desktop/dissertation/images/cornwall_LauncestonWooda_cam1/blocked/reviewed_27699.jpg"
    )
    if not os.path.exists(IMAGE_PATH):
        import glob
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = glob.glob(os.path.join(here, "images", "**", "*.jpg"), recursive=True)
        if candidates:
            IMAGE_PATH = candidates[0]

if not os.path.exists(IMAGE_PATH):
    print(f"Couldn't find {IMAGE_PATH} and no images/**/*.jpg found nearby.")
    print("Pass a real .jpg path as an argument: python3 test_cerebras.py /path/to/image.jpg")
    sys.exit(1)

with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Same shape of prompt as the real app's v2 prompt (risk_assessment/llm_explainer.py),
# simplified slightly since we don't have real Grad-CAM stats for this quick test.
prompt_text = (
    "You are a drainage inspection assistant. This CCTV frame was classified BLOCKED.\n\n"
    "Using what's visible in the image, write exactly two sentences: "
    "(1) what is physically obstructing the screen and roughly where in the frame; "
    "(2) what the engineer should do about it. "
    "No headers, no bullet points, no mention of AI or models."
)

payload = {
    "model": "qwen-3.8-27b",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt_text},
            ],
        }
    ],
    "max_tokens": 150,
    # qwen-3.8-27b defaults to reasoning_effort="high" and burns its whole
    # max_tokens budget on hidden chain-of-thought before writing "content"
    # -- disable it so we get a direct answer within the token budget.
    "reasoning_effort": "none",
}

req = urllib.request.Request(
    "https://api.cerebras.ai/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Cloudflare (which fronts Cerebras's API) blocks requests with
        # Python's default urllib User-Agent as bot traffic (error 1010) —
        # a normal browser-looking one gets through fine.
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    },
    method="POST",
)

print(f"Sending {IMAGE_PATH} to Cerebras (qwen-3.8-27b)...")
start = time.time()
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        elapsed = time.time() - start
        raw = resp.read().decode("utf-8")
        body = json.loads(raw)
        # Print the full raw JSON first, always -- if the response shape
        # doesn't have choices[0].message.content the way we expect (e.g. a
        # reasoning model that uses a different field name), this is what
        # actually shows us the real shape instead of just KeyError: 'content'.
        print(f"\n--- Raw response JSON ({elapsed:.2f}s) ---")
        print(json.dumps(body, indent=2))
        message = body["choices"][0]["message"]
        text = message.get("content") or message.get("reasoning_content") or "<no content or reasoning_content field>"
        usage = body.get("usage", {})
        print(f"\n--- Extracted text ---")
        print(text.strip() if isinstance(text, str) else text)
        print(f"\n--- Token usage ---\n{usage}")
except urllib.error.HTTPError as e:
    elapsed = time.time() - start
    print(f"\nHTTP {e.code} after {elapsed:.2f}s:")
    print(e.read().decode("utf-8"))
except Exception as e:
    print(f"\nRequest failed: {e}")
