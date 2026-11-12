# Client examples for Muse Spark, run as checks. Needs OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (source endpoint.txt).
# LONG_CTX="131072 524288" adds needle-in-haystack probes of about that many prompt tokens.
import base64
import io
import os
import time

from openai import OpenAI
from PIL import Image

client = OpenAI(timeout=2400)
MODEL = os.environ["OPENAI_MODEL"]


def chat(content, max_tokens=1024, **kw):
    return client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
        extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}}, **kw)


def check(name, fn):
    t = time.time()
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, repr(e)[:300]
    print(f"{name} {'PASS' if ok else 'FAIL'} {time.time() - t:.0f}s {detail}", flush=True)


def text():
    m = chat("What is the capital of France? Answer in one word.").choices[0].message
    reasoning = getattr(m, "reasoning", None) or getattr(m, "reasoning_content", None)
    # The reasoning parser should move the thinking out of content.
    return "paris" in (m.content or "").lower() and bool(reasoning), f"content={m.content!r} reasoning={str(reasoning)[:80]!r}"


def tool():
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    m = chat("What is the weather in Chicago right now?", tools=tools, tool_choice="auto").choices[0].message
    if not m.tool_calls:
        return False, f"no tool call; content={m.content!r}"
    f = m.tool_calls[0].function
    return f.name == "get_weather" and "chicago" in f.arguments.lower(), f"{f.name}({f.arguments})"


def image():
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (220, 20, 20)).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    m = chat([{"type": "image_url", "image_url": {"url": url}},
              {"type": "text", "text": "What single colour fills this image? Answer with one word."}]).choices[0].message
    return "red" in (m.content or "").lower(), f"content={m.content!r}"


def long_ctx(n):
    # ~10 tokens per filler sentence; n // 12 keeps the prompt under n.
    filler = "The sky is blue and the grass is green. " * (n // 12)
    r = chat(f"The secret number is 4417.\n{filler}\nWhat is the secret number? Answer with the number only.")
    return "4417" in (r.choices[0].message.content or ""), f"prompt_tokens={r.usage.prompt_tokens}"


check("TEXT", text)
check("TOOL", tool)
check("IMAGE", image)
for n in os.environ.get("LONG_CTX", "").split():
    check(f"LONG{n}", lambda: long_ctx(int(n)))
