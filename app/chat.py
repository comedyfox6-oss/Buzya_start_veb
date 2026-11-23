from app.model import RWKVModel
from config.settings import MAX_NEW_TOKENS


def main() -> None:
    print("=== BUZYA / RWKV-7 CPU ===")
    print("Type /exit to stop. The model runs locally inside this Codespace.")
    print("The first launch downloads the model weights from Hugging Face.\n")

    runtime = RWKVModel()
    runtime.load()

    messages: list[dict[str, str]] = []

    while True:
        try:
            user_text = input("\nТы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopping Buzya.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if user_text == "/reset":
            messages.clear()
            print("Conversation context cleared.")
            continue

        messages.append({"role": "user", "content": user_text})

        try:
            answer = runtime.generate(messages, max_new_tokens=MAX_NEW_TOKENS)
        except Exception:
            messages.pop()
            raise

        print(f"Бузя: {answer}")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
