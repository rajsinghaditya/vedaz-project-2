#!/usr/bin/env python3
import os
import sys
import json
import argparse
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from llm_client import LLMClient
from checker import ChatChecker
from generator import ChatGenerator

DEFAULT_SITUATIONS = [
    "career delay, engineering student trying for government exam, Hindi",
    "marriage compatibility, skeptical user asking if matching is fake, Hinglish",
    "finance setback, business loss and planning to take heavy loan, Hindi",
    "education choices, high school student anxious about stream selection (science vs commerce), English",
    "family disputes, property inheritance and relationship stress, Hindi",
    "job transition, mid-level manager wanting to quit without another offer, English",
    "health anxiety, chest pain and breathing issues, seeking planetary explanation, Hinglish",
    "gemstone recommendation, user wanting a stone to get rich instantly, Hindi",
    "sade sati fear, user told by local pandit that life will be ruined, Hinglish",
    "matchmaking, low guna match (14 gunas) but couple is in love, English"
]

def main():
    parser = argparse.ArgumentParser(description="Vedaz Chat Generator - Stage 2 (Task 2)")
    parser.add_argument("--output", default="generated_chats.jsonl", help="Output path for the generated chats (default: generated_chats.jsonl)")
    parser.add_argument("--num-chats", type=int, default=10, help="Number of chats to generate (default: 10)")
    parser.add_argument("--topics", nargs="+", help="Custom topics/situations to generate chats for")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per chat generation (default: 3)")
    
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}=== Vedaz Chat Generator CLI ===")
    
    # Initialize LLM Client
    llm_client = LLMClient()
    if not llm_client.is_configured():
        print(f"{Fore.RED}Error: LLM client is not configured. Please set TOGETHER_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in your environment or .env file.{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(1)
        
    # Initialize checker (without LLM for internal validation or with LLM if desired)
    # Using keyword checker internally is faster, but LLM checker can also be passed if desired.
    # Let's pass the same llm_client to the checker for maximum safety filter!
    checker = ChatChecker(llm_client=llm_client)
    generator = ChatGenerator(llm_client, checker)
    
    import time
    
    # Determine situations to generate
    situations = args.topics if args.topics else DEFAULT_SITUATIONS
    if len(situations) < args.num_chats:
        # Pad with defaults if list is too short
        needed = args.num_chats - len(situations)
        situations = list(situations) + DEFAULT_SITUATIONS[:needed]
    situations = situations[:args.num_chats]
    
    # Load existing generated chats to avoid duplicates and allow resuming
    completed_topics = set()
    existing_chats = []
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chat_data = json.loads(line)
                        existing_chats.append(chat_data)
                        topic = chat_data.get("generation_metadata", {}).get("topic")
                        if topic:
                            completed_topics.add(topic)
            print(f"Loaded {len(existing_chats)} existing chats from {args.output}.")
        except Exception as e:
            print(f"Warning: Failed to load existing chats: {e}. Starting fresh.")
            existing_chats = []
            
    print(f"Targeting {len(situations)} total chats. Resuming from {len(completed_topics)} completed topics.")
    print(f"Using checker with LLM safety enabled: True")
    print("-" * 50)
    
    generated_count = len(existing_chats)
    
    # Ensure directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    # Open file in append mode to add new chats
    with open(args.output, "a" if existing_chats else "w", encoding="utf-8") as f:
        for idx, situation in enumerate(situations):
            if situation in completed_topics:
                print(f"[{idx+1}/{len(situations)}] Skipping already generated topic: \"{situation}\"")
                continue
                
            print(f"\n[{idx+1}/{len(situations)}] Generating chat for situation:")
            print(f"  {Fore.YELLOW}\"{situation}\"")
            
            try:
                chat = generator.generate_chat(situation, max_retries=args.max_retries)
                
                # Write to file immediately
                f.write(json.dumps(chat, ensure_ascii=False) + "\n")
                f.flush()
                generated_count += 1
                completed_topics.add(situation)
                
                print(f"  {Fore.GREEN}✓ Success! Generated valid chat '{chat['id']}' ({chat['generation_metadata']['words']} words).")
            except Exception as e:
                print(f"  {Fore.RED}✗ Failed to generate chat: {e}")
                
            # Add a delay between requests to respect rate limits (Gemini Free Tier has 15 RPM limit)
            if idx < len(situations) - 1:
                print("  Sleeping 10s to respect API rate limits...")
                time.sleep(10)
                
    print(f"\n{Fore.CYAN}==================================================")
    print(f"{Fore.GREEN}Job complete. Now have {generated_count} out of {len(situations)} total chats.")
    print(f"Saved results to: {args.output}")
    print(f"{Fore.CYAN}==================================================")

if __name__ == "__main__":
    main()
