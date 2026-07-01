import json
import re
from typing import Dict, Any, List

class ChatGenerator:
    """
    Generates new example chats using an LLM based on specified topics/situations,
    ensuring they match the Vedaz tone and schema, and filters them using the ChatChecker.
    """
    def __init__(self, llm_client, checker):
        self.llm_client = llm_client
        self.checker = checker

    def generate_chat(self, topic: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Attempts to generate a single valid, safe chat for the given topic.
        Retries up to max_retries times if the check fails.
        """
        system_prompt = """You are an expert AI dataset generator for Vedaz.
Your goal is to generate a realistic, high-quality synthetic training chat in JSON format between a USER and Vedaz's AI Astrologer (ASSISTANT).

The conversation must follow the Vedaz personality guidelines:
- Compassionate, balanced, and non-fatalistic.
- Never predicts death, illness, accidents, or guaranteed misfortune.
- Never promises guaranteed medical or financial results.
- Never pressures the user to pay for remedies.
- Frames remedies as optional, supportive spiritual practices.
- Explains astrology's limits honestly.
- Conversations are typically in Hinglish, Hindi (Devanagari), or English as specified.

You must output a single JSON object matching this schema:
{
  "id": "conv_[unique_id]",
  "tags": ["tag1", "tag2"],
  "messages": [
    {
      "role": "system",
      "content": "[The system prompt defining the AI astrologer's instructions]"
    },
    {
      "role": "user",
      "content": "[User query]"
    },
    {
      "role": "assistant",
      "content": "[AI Astrologer response, incorporating Vedaz guidelines]"
    },
    ... (alternating user/assistant turns)
  ]
}

Make sure the chat has:
1. A 'system' message at index 0.
2. Alternating 'user' and 'assistant' turns.
3. At least 3 messages total (system, user, assistant).

Respond ONLY with valid JSON. Do not include any explanations or extra text outside the JSON.
"""

        user_prompt = f"Generate a training chat for the following topic/situation: '{topic}'. Make sure the chat contains 1 to 2 complete user-assistant turns after the system message."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        for attempt in range(max_retries):
            try:
                response_str = self.llm_client.generate_chat_completion(
                    messages=messages,
                    temperature=0.8,
                    json_mode=True
                )
                
                # Safely parse JSON from response
                chat_data = self._parse_json_response(response_str)
                if not chat_data:
                    print(f"  [Attempt {attempt+1}] Failed to parse JSON response. Retrying...")
                    continue
                
                # Check chat using ChatChecker
                check_res = self.checker.check_chat(chat_data.get("messages", []))
                
                if check_res["is_valid"] and check_res["is_safe"]:
                    # Successfully generated a clean chat
                    chat_data["generation_metadata"] = {
                        "topic": topic,
                        "attempts": attempt + 1,
                        "words": check_res["length"]["words"]
                    }
                    return chat_data
                else:
                    reasons = check_res["errors"]
                    print(f"  [Attempt {attempt+1}] Flagged by checker: {', '.join(reasons)}. Retrying...")
                    
            except Exception as e:
                print(f"  [Attempt {attempt+1}] Error during generation: {e}")
                
        raise ValueError(f"Failed to generate a valid, safe chat for topic '{topic}' after {max_retries} attempts.")

    def _parse_json_response(self, response_str: str) -> Dict[str, Any]:
        """
        Attempts to extract and parse JSON from the LLM's response string.
        """
        # Strip whitespace
        content = response_str.strip()
        
        # Remove markdown code block wrappers if present
        if content.startswith("```"):
            # Remove opening codeblock (e.g. ```json or ```)
            content = re.sub(r"^```[a-zA-Z0-9]*\n", "", content)
            # Remove closing codeblock
            content = re.sub(r"\n```$", "", content)
            content = content.strip()
            
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try finding the first '{' and last '}'
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                try:
                    return json.loads(content[start:end+1])
                except json.JSONDecodeError:
                    pass
            return None
