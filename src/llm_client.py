import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env if present
load_dotenv()

class LLMClient:
    """
    A unified wrapper around OpenAI, Together AI, and Google Gemini API.
    Uses OpenAI SDK for Together/OpenAI, and the new google-genai SDK for native Gemini.
    """
    def __init__(self):
        self.provider = None
        self.model = None
        self.client = None
        self.gemini_configured = False
        
        # Check environment variables to determine the provider
        together_key = os.getenv("TOGETHER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        # Allow override via environment variables
        env_model = os.getenv("VEDAZ_MODEL")
        
        if together_key:
            self.provider = "together"
            self.client = OpenAI(
                api_key=together_key,
                base_url="https://api.together.xyz/v1"
            )
            self.model = env_model or "deepseek-ai/DeepSeek-V3"
            print(f"[LLM] Configured Together AI client using model: {self.model}")
            
        elif openai_key:
            self.provider = "openai"
            self.client = OpenAI(api_key=openai_key)
            self.model = env_model or "gpt-4o-mini"
            print(f"[LLM] Configured OpenAI client using model: {self.model}")
            
        elif gemini_key:
            self.provider = "gemini"
            from google import genai
            self.client = genai.Client(api_key=gemini_key)
            self.model = env_model or "gemini-2.5-flash"
            self.gemini_configured = True
            print(f"[LLM] Configured Gemini native client using model: {self.model}")
            
        else:
            # Fallback/Warning: print instructions to user
            print("[WARNING] No API keys found! Please set TOGETHER_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in your environment or .env file.", file=sys.stderr)
            self.provider = "none"
            self.client = None
            self.model = None

    def is_configured(self) -> bool:
        return (self.client is not None) or self.gemini_configured

    def generate_chat_completion(self, messages, temperature=0.7, json_mode=False) -> str:
        """
        Generates a text or JSON completion given a list of messages.
        """
        if not self.is_configured():
            raise ValueError("LLMClient is not configured. Please set an API key in the environment or a .env file.")
            
        import time
        max_attempts = 5
        backoff_seconds = 8
        
        for attempt in range(max_attempts):
            try:
                if self.provider == "gemini":
                    from google.genai import types
                    
                    # Extract system instruction
                    system_instruction = None
                    for m in messages:
                        if m["role"] == "system":
                            system_instruction = m["content"]
                            break
                            
                    # Convert user/assistant messages to google-genai content formats
                    contents = []
                    for m in messages:
                        if m["role"] == "system":
                            continue
                        role = "user" if m["role"] == "user" else "model"
                        contents.append(
                            types.Content(
                                role=role,
                                parts=[types.Part.from_text(text=m["content"])]
                            )
                        )
                        
                    # Create config
                    config = types.GenerateContentConfig(
                        temperature=temperature,
                        system_instruction=system_instruction
                    )
                    if json_mode:
                        config.response_mime_type = "application/json"
                        
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config
                    )
                    return response.text
                else:
                    # Together or OpenAI
                    extra_args = {}
                    if json_mode:
                        extra_args["response_format"] = {"type": "json_object"}
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        **extra_args
                    )
                    return response.choices[0].message.content
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = "429" in err_msg or "resource_exhausted" in err_msg or "rate limit" in err_msg
                
                if is_rate_limit and attempt < max_attempts - 1:
                    print(f"  [Rate Limit] Hit 429 quota error. Sleeping {backoff_seconds}s before retry (Attempt {attempt+1}/{max_attempts})...", file=sys.stderr)
                    time.sleep(backoff_seconds)
                    backoff_seconds *= 2.0 # Exponential backoff
                    continue
                
                print(f"[LLM Error] Failed to generate completion (Attempt {attempt+1}/{max_attempts}): {e}", file=sys.stderr)
                raise e
