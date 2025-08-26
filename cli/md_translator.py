import os
import argparse
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================
# LOAD ENV
# ==========================
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")  # stored in .env: DEEPSEEK_API_KEY=xxxxx
API_URL = "https://api.deepseek.com/v1/chat/completions"


def translate_text(text, source_lang, target_lang):
    """Send request to DeepSeek API for translation"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": f"You are a translation engine. Translate {source_lang} to {target_lang}. Keep formatting intact."
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.3
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def process_file(file_path, output_path, source_lang, target_lang, is_file_output=False):
    """Translate a single .md file"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    translated = translate_text(content, source_lang, target_lang)

    if is_file_output:
        # output_path is a full file path
        out_path = output_path
    else:
        # output_path is a folder
        filename = os.path.basename(file_path)
        out_path = os.path.join(output_path, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(translated)

    print(f"✅ Translated: {file_path} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Translate Markdown files using DeepSeek API")
    parser.add_argument("--input", "-i", required=True, help="Markdown file or folder containing .md files to translate")
    parser.add_argument("--output", "-o", help="Output folder (or file). If omitted, a new path will be generated based on the input")
    parser.add_argument("--source", "-s", default="vi", help="Source language (default: vi)")
    parser.add_argument("--target", "-t", default="en", help="Target language (default: en)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    input_path = args.input
    source_lang = args.source
    target_lang = args.target
    workers = args.workers

    # Handle default output path
    if args.output:
        output_path = args.output
    else:
        if os.path.isfile(input_path):
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_{target_lang}{ext}"  # full file path
        elif os.path.isdir(input_path):
            output_path = f"{input_path.rstrip(os.sep)}"  # folder
        else:
            print("❌ Invalid INPUT_PATH.")
            return

    # If input is a single file
    if os.path.isfile(input_path) and input_path.endswith(".md"):
        if not args.output:  # no --output → full file path
            process_file(input_path, output_path, source_lang, target_lang, is_file_output=True)
        else:  # --output provided
            if os.path.isdir(output_path) or output_path.endswith(os.sep):
                os.makedirs(output_path, exist_ok=True)
                process_file(input_path, output_path, source_lang, target_lang)
            else:
                # If --output is given as a file path
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                process_file(input_path, output_path, source_lang, target_lang, is_file_output=True)

    # If input is a folder → parallel processing
    elif os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".md")]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_file, f, output_path, source_lang, target_lang)
                for f in files
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Error: {e}")
    else:
        print("❌ INPUT_PATH must be a .md file or a folder containing .md files.")


if __name__ == "__main__":
    main()
