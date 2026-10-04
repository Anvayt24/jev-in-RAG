"""One-call smoke test: confirm the OpenRouter key, endpoint and Jev response shape."""
import os
import time

from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient

load_dotenv()

MODEL = "typesafe/jev-1.13"
BASE_URL = "https://openrouter.ai/api"

client = TypeSafeClient(api_key=os.environ["OPENROUTER_API_KEY"], base_url=BASE_URL)

t0 = time.perf_counter()
result = client.system_one(
    "I was charged twice for my subscription.",
    {"refund": Noul(instructions="Is the customer asking for money back?")},
    model=MODEL,
)
dt = time.perf_counter() - t0

print("model reported :", result.model)
print("refund noul    :", result.answers["refund"].noul)
print("usage          :", result.usage)
print("latency (s)    :", round(dt, 3))
print("raw usage json :", result.raw_http_response.json().get("usage"))
