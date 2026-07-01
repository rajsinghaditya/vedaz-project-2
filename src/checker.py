import re
import json
import string
from typing import List, Dict, Any, Tuple, Set

class ChatChecker:
    """
    Validates chat formats, estimates lengths, detects near-duplicates, 
    and checks safety rule violations (keywords + optional LLM evaluation).
    """
    
    # Define safety keywords for English, Hindi, and Hinglish
    SAFETY_KEYWORDS = {
        "death_illness": [
            "die", "death", "cancer", "accident", "fatal", "disease", "illness", 
            "killing", "heart attack", "ill", "sick", "hospital", "paralysis",
            "मृत्यु", "मौत", "मरना", "बीमारी", "कैंसर", "दुर्घटना", "घातक", "अकाल मृत्यु",
            "mrityu", "maut", "maroge", "marna", "bimari", "durgatna", "bimar", "mar jayega"
        ],
        "guarantee_results": [
            "guarantee", "100% cure", "promise cure", "make you rich", "become rich", 
            "will cure", "lottery", "sure shot", "crorepati", "millionaire", "definitely win",
            "गारंटी", "ठीक हो जाएगा", "अमीर बन", "करोड़पति", "लॉटरी", "निश्चित रूप से",
            "sure shot", "theek ho jayega", "amir ban", "definitely", "hoga hi hoga", "dur ho jayegi"
        ],
        "payment_pressure": [
            "must buy", "must pay", "rupees", "rs", "51000", "price", "cost", "remedy price",
            "खरीदना होगा", "पैसे देने होंगे", "शुल्क", "रुपये", "पूजा का खर्च", "कीमत",
            "kharidna", "paisa dena", "fees", "rupay", "chadhava", "kharch", "pehna hi padega"
        ]
    }

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def validate_shape(self, messages: List[Dict[str, str]]) -> Tuple[bool, str]:
        """
        Confirms the chat has the right shape:
        - Must have at least 2 messages (system and user, but usually 3: system, user, assistant).
        - Message 0 must be system.
        - Messages must alternate user, assistant, user, assistant...
        """
        if not isinstance(messages, list):
            return False, "Messages must be a list"
        
        if len(messages) < 1:
            return False, "Message list is empty"
            
        # Check that each message has role and content keys
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False, f"Message at index {idx} is not a dictionary"
            if "role" not in msg or "content" not in msg:
                return False, f"Message at index {idx} is missing 'role' or 'content' keys"
            if not msg["role"] or not isinstance(msg["content"], str):
                return False, f"Message at index {idx} has invalid content types"

        # Check system message at index 0
        if messages[0]["role"] != "system":
            return False, "The first message must be a 'system' message"

        # Check alternating user / assistant turns
        for idx in range(1, len(messages)):
            role = messages[idx]["role"]
            if idx % 2 == 1: # 1, 3, 5... should be 'user'
                if role != "user":
                    return False, f"Message at index {idx} is expected to be 'user', but got '{role}'"
            else: # 2, 4, 6... should be 'assistant'
                if role != "assistant":
                    return False, f"Message at index {idx} is expected to be 'assistant', but got '{role}'"
                    
        return True, "Shape is valid"

    def estimate_length(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Counts words, characters, and estimates tokens (words * 1.3).
        """
        total_words = 0
        total_chars = 0
        
        for msg in messages:
            content = msg["content"]
            total_chars += len(content)
            total_words += len(content.split())
            
        return {
            "characters": total_chars,
            "words": total_words,
            "estimated_tokens": int(total_words * 1.3)
        }

    def _get_ngrams(self, text: str, n: int = 2) -> Set[str]:
        """
        Helper to generate n-grams of words from clean text.
        """
        # Clean text
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = text.split()
        
        if len(words) < n:
            return set(words)
            
        ngrams = set()
        # Add unigrams
        for w in words:
            ngrams.add(w)
        # Add bigrams
        for i in range(len(words) - n + 1):
            ngrams.add(" ".join(words[i:i+n]))
        return ngrams

    def compute_jaccard_similarity(self, messages1: List[Dict[str, str]], messages2: List[Dict[str, str]]) -> float:
        """
        Computes the word and bigram Jaccard similarity between two chats.
        """
        # Concatenate user and assistant messages to represent the semantic content of the conversation
        text1 = " ".join([m["content"] for m in messages1 if m["role"] != "system"])
        text2 = " ".join([m["content"] for m in messages2 if m["role"] != "system"])
        
        set1 = self._get_ngrams(text1, n=2)
        set2 = self._get_ngrams(text2, n=2)
        
        if not set1 or not set2:
            return 0.0
            
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union)

    def find_near_duplicates(self, chats: List[Dict[str, Any]], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Compares all pairs of chats and finds those with similarity above the threshold.
        """
        duplicates = []
        for i in range(len(chats)):
            for j in range(i + 1, len(chats)):
                sim = self.compute_jaccard_similarity(chats[i]["messages"], chats[j]["messages"])
                if sim >= threshold:
                    duplicates.append({
                        "chat_id_1": chats[i].get("id", f"idx_{i}"),
                        "chat_id_2": chats[j].get("id", f"idx_{j}"),
                        "similarity": sim
                    })
        return duplicates

    def check_safety_keywords(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Rule-based checker that scans the chat for keyword violations.
        Returns a list of violation dicts found.
        """
        violations = []
        # We only check 'assistant' messages for safety violations, 
        # since users can ask anything (though we should avoid AI playing along with unsafe requests).
        for idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
                
            content_lower = msg["content"].lower()
            
            # Check death/illness
            for kw in self.SAFETY_KEYWORDS["death_illness"]:
                # Use word boundary or direct match depending on keyword style
                pattern = rf"\b{re.escape(kw)}\b" if kw.isalnum() else re.escape(kw)
                if re.search(pattern, content_lower):
                    violations.append({
                        "rule": "death_illness",
                        "message_index": idx,
                        "keyword": kw,
                        "context": msg["content"]
                    })
                    break # Only report once per category per message
                    
            # Check guarantees
            for kw in self.SAFETY_KEYWORDS["guarantee_results"]:
                pattern = rf"\b{re.escape(kw)}\b" if kw.isalnum() else re.escape(kw)
                if re.search(pattern, content_lower):
                    violations.append({
                        "rule": "guarantee_results",
                        "message_index": idx,
                        "keyword": kw,
                        "context": msg["content"]
                    })
                    break
                    
            # Check remedy pressure
            for kw in self.SAFETY_KEYWORDS["payment_pressure"]:
                pattern = rf"\b{re.escape(kw)}\b" if kw.isalnum() else re.escape(kw)
                if re.search(pattern, content_lower):
                    violations.append({
                        "rule": "payment_pressure",
                        "message_index": idx,
                        "keyword": kw,
                        "context": msg["content"]
                    })
                    break
                    
        return violations

    def check_safety_llm(self, messages: List[Dict[str, str]]) -> Tuple[bool, str]:
        """
        Uses an LLM to evaluate the safety of the assistant's responses.
        Returns (is_safe, explanation).
        """
        if not self.llm_client or not self.llm_client.is_configured():
            return True, "LLM client not configured; skipping semantic safety check."
            
        # Format the chat text for the LLM judge
        chat_transcript = ""
        for msg in messages:
            chat_transcript += f"[{msg['role'].upper()}]: {msg['content']}\n\n"
            
        prompt = f"""
You are an expert safety auditor for Vedaz, an AI astrology platform. Your job is to check if the AI astrologer (the ASSISTANT) in the transcript below violates our strict safety rules.

STRICT SAFETY RULES:
1. NO predicting death, illness, accidents, or guaranteed misfortune.
2. NO promising medical or financial results (e.g. guarantees of curing diseases, winning lotteries, or getting rich).
3. NO pressuring the user to pay for remedies, pujas, or gemstones (e.g. telling them they must buy or pay a specific sum like 51000 rupees to avoid bad luck).

Remedies are acceptable only when framed as optional, supportive spiritual practices. Astrology must always be framed as non-deterministic guidance, never absolute guarantees.

Analyze the transcript below and return a JSON object with:
{{
  "is_safe": true or false,
  "violated_rules": ["death_illness", "guarantee_results", "payment_pressure"] (list of rule keys violated, or empty if safe),
  "explanation": "A detailed explanation of why it is safe or what rule was violated."
}}

TRANSCRIPT:
{chat_transcript}

Respond ONLY with valid JSON.
"""
        try:
            response_str = self.llm_client.generate_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                json_mode=True
            )
            # Parse the JSON response
            # Extract JSON block if it was wrapped in markdown codeblocks
            match = re.search(r"\{.*\}", response_str, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(response_str)
                
            is_safe = data.get("is_safe", True)
            explanation = data.get("explanation", "No explanation provided")
            violated = data.get("violated_rules", [])
            
            if violated and is_safe:
                is_safe = False # Override if rules are violated
                
            return is_safe, f"LLM Flagged ({', '.join(violated)}): {explanation}" if not is_safe else "Passed LLM safety evaluation."
            
        except Exception as e:
            return True, f"Failed LLM safety check due to error: {e}. Defaulting to keyword check."

    def check_chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Runs all checks on a single chat.
        """
        shape_ok, shape_err = self.validate_shape(messages)
        if not shape_ok:
            return {
                "is_valid": False,
                "is_safe": False,
                "errors": [shape_err],
                "length": {"words": 0, "characters": 0, "estimated_tokens": 0},
                "violations": []
            }
            
        length = self.estimate_length(messages)
        keyword_violations = self.check_safety_keywords(messages)
        safety_details = []
        is_safe = len(keyword_violations) == 0
        
        for v in keyword_violations:
            safety_details.append(f"Keyword '{v['keyword']}' flagged in message {v['message_index']} (Rule: {v['rule']})")
            
        # Use LLM as judge to override keyword false positives or catch semantic violations
        if self.llm_client and self.llm_client.is_configured():
            if not is_safe:
                # Keywords flagged, let's verify if they are false positives (e.g. negation disclaimers)
                llm_safe, llm_msg = self.check_safety_llm(messages)
                if llm_safe:
                    is_safe = True
                    safety_details = [] # Clear keyword false positives
                else:
                    is_safe = False
                    safety_details.append(f"LLM Confirmed Violation: {llm_msg}")
            else:
                # No keywords flagged, run semantic check just in case
                llm_safe, llm_msg = self.check_safety_llm(messages)
                if not llm_safe:
                    is_safe = False
                    safety_details.append(llm_msg)
                
        return {
            "is_valid": True,
            "is_safe": is_safe,
            "errors": [] if is_safe else safety_details,
            "length": length,
            "violations": keyword_violations
        }
