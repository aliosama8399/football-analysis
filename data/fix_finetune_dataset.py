"""
Fix Fine-Tuning Dataset (Ollama 429 Errors)
=============================================
Reads football_tactical.jsonl, finds records with Ollama 429 errors,
resubmits their prompts one-by-one, and SAVES after every fix so you
can safely stop (Ctrl+C) and resume without losing progress.

Usage:
    python data/fix_finetune_dataset.py --provider ollama
    python data/fix_finetune_dataset.py --provider gemini
"""

import json
import time
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.llm_providers import get_llm_provider

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "finetune"
MASTER_FILE = DATA_DIR / "football_tactical.jsonl"


def save_records(records, path):
    """Overwrite the JSONL file with current records."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def is_broken(rec):
    """Check if a record has the 429 error in its assistant response."""
    for m in rec.get("messages", []):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if "429 Client Error" in content or "[OLLAMA] API error" in content:
                return True
    return False


def get_user_prompt(rec):
    """Extract the user message from a record."""
    for m in rec.get("messages", []):
        if m.get("role") == "user":
            return m["content"]
    return ""


def set_assistant_content(rec, new_content):
    """Update the assistant message in a record."""
    for m in rec.get("messages", []):
        if m.get("role") == "assistant":
            m["content"] = new_content
            return


def rebuild_splits(records):
    """Regenerate train/val JSON files from clean records only."""
    clean = [r for r in records if not is_broken(r)]
    # Extra filter: assistant response should be substantial (>100 chars)
    valid = []
    for r in clean:
        for m in r.get("messages", []):
            if m.get("role") == "assistant" and len(m.get("content", "")) > 100:
                valid.append(r)
                break

    if len(valid) < 10:
        print(f"  ⚠ Only {len(valid)} valid records, skipping split.")
        return

    train, val = train_test_split(valid, test_size=0.15, random_state=42)
    train_path = DATA_DIR / "football_train.json"
    val_path = DATA_DIR / "football_val.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    print(f"  💾 football_train.json: {len(train)} records")
    print(f"  💾 football_val.json:   {len(val)} records")


def fix_dataset(provider_name, model_name):
    if not MASTER_FILE.exists():
        print(f"❌ Cannot find {MASTER_FILE}")
        return

    # Load LLM
    print(f"🔧 Teacher LLM: {provider_name} ({model_name or 'default'})")
    teacher = get_llm_provider(provider_type=provider_name, model_name=model_name)

    # Load all records
    records = []
    with open(MASTER_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Find broken ones (resume-safe: only records still broken)
    bad_indices = [i for i, r in enumerate(records) if is_broken(r)]

    if not bad_indices:
        print("✅ No broken records found! Dataset is clean.")
        rebuild_splits(records)
        return

    total = len(records)
    good = total - len(bad_indices)
    print(f"📊 Total: {total} | ✅ Good: {good} | ❌ Broken: {len(bad_indices)}")
    print(f"⏳ Fixing one-by-one with live saves (safe to Ctrl+C anytime)...\n")

    fixed_count = 0
    for num, idx in enumerate(bad_indices, 1):
        rec = records[idx]
        match_id = rec.get("match_id", "Unknown")
        prompt = get_user_prompt(rec)
        if not prompt:
            print(f"  [{num}/{len(bad_indices)}] ⏭ No user prompt, skipping {match_id}")
            continue

        try:
            print(f"  [{num}/{len(bad_indices)}] 🔄 {match_id}...", end=" ", flush=True)
            response = teacher._call_api(prompt)

            if "API error" in response or "429" in response:
                print(f"❌ Still failing, will retry next run")
                time.sleep(5)
                continue

            set_assistant_content(rec, response)
            fixed_count += 1

            # SAVE immediately so progress is never lost
            save_records(records, MASTER_FILE)
            print(f"✅ saved")

            time.sleep(1.5)

        except KeyboardInterrupt:
            print(f"\n\n🛑 Interrupted! {fixed_count} records fixed so far.")
            print(f"   Run the same command again to resume.\n")
            save_records(records, MASTER_FILE)
            return

        except Exception as e:
            print(f"❌ {e}")
            time.sleep(5)

    remaining = sum(1 for r in records if is_broken(r))
    print(f"\n{'='*60}")
    print(f"  ✅ Fixed {fixed_count} records this run")
    print(f"  📊 Remaining broken: {remaining}")
    print(f"{'='*60}")

    if remaining == 0:
        print("\n🎉 ALL records are now clean! Rebuilding train/val splits...")
        rebuild_splits(records)
        print("🚀 DONE!")
    else:
        print(f"\n⚠ {remaining} records still broken. Run again to retry them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix 429 errors in finetune JSONL")
    parser.add_argument("--provider", type=str, default="ollama")
    parser.add_argument("--model-name", type=str, default="")
    args = parser.parse_args()
    fix_dataset(args.provider, args.model_name)
