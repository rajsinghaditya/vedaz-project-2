#!/usr/bin/env python3
import os
import sys
import json
import argparse
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from llm_client import LLMClient
from checker import ChatChecker

def load_dataset(file_path: str) -> list:
    """
    Loads dataset from either standard JSON list or JSONL file.
    """
    if not os.path.exists(file_path):
        print(f"{Fore.RED}Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)
        
    chats = []
    # Check extension
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    chats.append(json.loads(line))
                except Exception as e:
                    print(f"{Fore.YELLOW}Warning: Skipping invalid JSON line {idx + 1}: {e}", file=sys.stderr)
    else:
        # Fallback to standard JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    chats = data
                else:
                    chats = [data]
            except Exception as e:
                print(f"{Fore.RED}Error: Failed to parse JSON file: {e}", file=sys.stderr)
                sys.exit(1)
                
    return chats

def main():
    parser = argparse.ArgumentParser(description="Vedaz Chat Checker - Stage 2 (Task 1)")
    parser.add_argument("input_file", help="Path to the .json or .jsonl chats file")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM-as-a-judge safety checks (requires api key)")
    parser.add_argument("--dup-threshold", type=float, default=0.8, help="Near-duplicate Jaccard threshold (default: 0.8)")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--train-out", default="train.jsonl", help="Output train jsonl file")
    parser.add_argument("--test-out", default="test.jsonl", help="Output test jsonl file")
    
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}=== Vedaz Chat Checker CLI ===")
    print(f"Loading chats from: {args.input_file}")
    
    chats = load_dataset(args.input_file)
    print(f"Loaded {len(chats)} chats.")
    
    # Initialize LLM Client if requested
    llm_client = None
    if args.use_llm:
        llm_client = LLMClient()
        if not llm_client.is_configured():
            print(f"{Fore.YELLOW}Warning: --use-llm set but no API key configured. Running with keyword checks only.")
            
    checker = ChatChecker(llm_client=llm_client)
    
    results = []
    clean_chats = []
    flagged_chats = []
    invalid_chats = []
    
    total_words = 0
    total_tokens = 0
    lengths = []
    
    for idx, chat in enumerate(chats):
        chat_id = chat.get("id", f"chat_{idx + 1}")
        messages = chat.get("messages", [])
        
        # Check chat
        check_res = checker.check_chat(messages)
        
        is_valid = check_res["is_valid"]
        is_safe = check_res["is_safe"]
        length = check_res["length"]
        errors = check_res["errors"]
        
        lengths.append(length["words"])
        total_words += length["words"]
        total_tokens += length["estimated_tokens"]
        
        status_str = f"{Fore.GREEN}PASS"
        if not is_valid:
            status_str = f"{Fore.RED}INVALID"
            invalid_chats.append((chat_id, errors))
        elif not is_safe:
            status_str = f"{Fore.YELLOW}FLAGGED"
            flagged_chats.append((chat_id, errors))
            clean_chats.append(chat) # Still a valid shape, but fails safety checks
        else:
            clean_chats.append(chat)
            
        results.append([
            chat_id,
            len(messages),
            length["words"],
            length["estimated_tokens"],
            status_str
        ])
        
    # Print results summary table
    print(f"\n{Fore.CYAN}--- Chats Validation Summary ---")
    headers = ["Chat ID", "Turns", "Words", "Est. Tokens", "Status"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Print statistics
    num_chats = len(chats)
    num_invalid = len(invalid_chats)
    num_flagged = len(flagged_chats)
    num_safe = num_chats - num_invalid - num_flagged
    
    avg_words = sum(lengths) / num_chats if num_chats > 0 else 0
    max_words = max(lengths) if num_chats > 0 else 0
    min_words = min(lengths) if num_chats > 0 else 0
    
    stats_data = [
        ["Total Chats", num_chats],
        ["Valid & Safe Chats", f"{Fore.GREEN}{num_safe}"],
        ["Flagged (Safety Violation)", f"{Fore.YELLOW}{num_flagged}"],
        ["Invalid Shape", f"{Fore.RED}{num_invalid}"],
        ["Average Word Count", f"{avg_words:.1f}"],
        ["Min / Max Word Count", f"{min_words} / {max_words}"],
        ["Total Est. Tokens", total_tokens]
    ]
    print(f"\n{Fore.CYAN}--- Dataset Metrics ---")
    print(tabulate(stats_data, tablefmt="simple"))
    
    # Report violations
    if invalid_chats:
        print(f"\n{Fore.RED}=== Invalid Shape Violations ===")
        for cid, errs in invalid_chats:
            print(f"- {Fore.RED}{cid}: {errs}")
            
    if flagged_chats:
        print(f"\n{Fore.YELLOW}=== Safety Rule Violations ===")
        for cid, errs in flagged_chats:
            print(f"- {Fore.YELLOW}{cid}:")
            for err in errs:
                print(f"  * {err}")
                
    # Near-duplicates
    print(f"\n{Fore.CYAN}--- Running Near-Duplicate Detection (Jaccard similarity >= {args.dup_threshold}) ---")
    dups = checker.find_near_duplicates(chats, threshold=args.dup_threshold)
    if dups:
        dup_headers = ["Chat ID 1", "Chat ID 2", "Similarity"]
        dup_table = [[d["chat_id_1"], d["chat_id_2"], f"{d['similarity']:.2%}"] for d in dups]
        print(f"{Fore.YELLOW}Found {len(dups)} near-duplicate chat pair(s):")
        print(tabulate(dup_table, headers=dup_headers, tablefmt="grid"))
    else:
        print(f"{Fore.GREEN}No near-duplicate chats found!")
        
    # Split training / test sets (only from clean, valid & safe chats)
    print(f"\n{Fore.CYAN}--- Dataset Splitting ---")
    
    # We filter out invalid and unsafe chats from the split
    safe_clean_chats = [c for c in clean_chats if c not in [chats[i] for i, r in enumerate(results) if "PASS" not in r[4]]]
    
    if not safe_clean_chats:
        print(f"{Fore.RED}Error: No valid & safe chats to split!", file=sys.stderr)
        return
        
    split_idx = int(len(safe_clean_chats) * args.split_ratio)
    train_set = safe_clean_chats[:split_idx]
    test_set = safe_clean_chats[split_idx:]
    
    # Handle edge case where test set is empty due to small size
    if not test_set and len(safe_clean_chats) > 1:
        train_set = safe_clean_chats[:-1]
        test_set = safe_clean_chats[-1:]
        
    print(f"Total safe & clean chats: {len(safe_clean_chats)}")
    print(f"Splitting into {args.split_ratio:.0%} train and {1-args.split_ratio:.0%} test.")
    print(f"Writing {len(train_set)} chats to: {args.train_out}")
    print(f"Writing {len(test_set)} chats to: {args.test_out}")
    
    # Write to jsonl
    with open(args.train_out, 'w', encoding='utf-8') as f:
        for c in train_set:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
            
    with open(args.test_out, 'w', encoding='utf-8') as f:
        for c in test_set:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
            
    print(f"{Fore.GREEN}Success! Splitting completed.")

if __name__ == "__main__":
    main()
