import logging
import math
import time
from typing import Dict, Any, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        # 使用 OpenAI 官方客户端，兼容 OpenAI 风格的 /v1/chat/completions 接口
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
        max_retries: int = 9,
    ) -> Dict[str, Any]:
        """
        Generate completion from the LLM (OpenAI-compatible client).
        Returns a dict with 'content', and structured 'usage'.

        带有指数退避的重试机制，防止短暂网络问题或超时直接导致任务失败。
        """
        base_delay = 15
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                # Dynamically construct kwargs, omitting None values to avoid API errors (like local vLLM failing to return logprobs)
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": 120,
                }
                if stop is not None:
                    kwargs["stop"] = stop

                response = self.client.chat.completions.create(**kwargs)

                if not response.choices:
                    logger.error(f"No choices in response: {response}")
                    return {"content": "", "error": "No choices"}

                choice = response.choices[0]
                content = choice.message.content or ""

                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                total_tokens = getattr(usage, "total_tokens", 0) or 0

                return {
                    "content": content,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                    "raw_response": response,
                }

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Generate failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait:.1f}s: {e}"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"Generate failed after {max_retries} attempts: {e}")

        return {
            "content": "",
            "error": str(last_error) if last_error is not None else "Unknown error",
        }
