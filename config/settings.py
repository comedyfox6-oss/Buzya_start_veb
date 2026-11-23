import os

MODEL_ID = os.getenv(
    "BUZYA_MODEL_ID",
    "RWKV/RWKV7-G1l-1.5B-20261031",
)

# The official checkpoint is stored in bfloat16.
# CPU inference uses the portable eager WKV implementation.
MODEL_DTYPE = os.getenv("BUZYA_DTYPE", "bfloat16")
MAX_NEW_TOKENS = int(os.getenv("BUZYA_MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.getenv("BUZYA_TEMPERATURE", "0.8"))
TOP_P = float(os.getenv("BUZYA_TOP_P", "0.7"))
