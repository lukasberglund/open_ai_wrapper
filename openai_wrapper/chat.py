import concurrent.futures
import os
import time
from dataclasses import dataclass
from typing import Callable, List

import diskcache as dc
import openai
from tenacity import retry

from openai_wrapper.complete import get_cost_per_token
from openai_wrapper.utils import BASE_CACHE_DIR, RETRY_KWARGS

CACHE_DIR = os.path.join(BASE_CACHE_DIR, "chat")
cache = dc.Cache(CACHE_DIR, size_limit=10 * 1e9)



@cache.memoize()
def complete_memoized(*args, **kwargs):
    return openai.ChatCompletion.create(*args, **kwargs)

@retry(**RETRY_KWARGS)
def retry_with_exp_backoff(fn, *args, **kwargs):
    return fn(*args, **kwargs)


@retry(**RETRY_KWARGS)
def complete_conditional_memoize_with_retrying(nocache=False, *args, **kwargs):
    temperature = kwargs.get("temperature", None)
    should_cache = (temperature == 0 and not nocache)
    if should_cache:
        return complete_memoized(**kwargs)
    else:
        return openai.ChatCompletion.create(*args, **kwargs)


@dataclass
class ChatMessage:
    role: str
    content: str

class OpenAIChat:
    def __init__(self, model="gpt-3.5-turbo", log_requests=True):
        self.queries = []
        self.model = model
        self.log_requests = log_requests
        os.makedirs(CACHE_DIR, exist_ok=True)

    def generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.0,
        nocache=False,
        **kwargs,
    ) -> str:
        dict_messages = [message.__dict__ for message in messages]
        response = self._complete(
            messages=dict_messages,
            temperature=temperature,
            nocache=nocache,
            **kwargs,
        )

        return response.choices[0].message.content  # type: ignore

    def _complete(self, messages: list[ChatMessage], **kwargs):
        """Request OpenAI API ChatCompletion with:
        - request throttling
        - request splitting
        - persistent caching
        """

        nocache = kwargs.pop("nocache", False)
        response = complete_conditional_memoize_with_retrying(nocache=nocache,model=self.model, messages=messages, **kwargs)
       
        if self.log_requests:
            self.log_request(
                messages,
                response,
            )
            
        return response

    def log_request(
        self,
        messages: list[ChatMessage],
        response,
    ):
        n_tokens_sent = response.usage.prompt_tokens  # type: ignore
        n_tokens_received = response.usage.completion_tokens  # type: ignore
        n_tokens_total = n_tokens_sent + n_tokens_received
        cost = (n_tokens_total) * get_cost_per_token(self.model)

        timestamp_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"
        with open(os.path.join(CACHE_DIR, f"{timestamp_str}-{self.model}.txt"), "a") as f:
            f.write("<REQUEST METADATA AFTER NEWLINE>\n")
            f.write(
                f"Chat request @ {timestamp_str}. Tokens sent: {n_tokens_sent}. Tokens received: {n_tokens_received}. Cost: ${cost:.4f}\n"
            )
            for i, choice in enumerate(response.choices):
                f.write(f"\n<PROMPT AFTER NEWLINE>\n")
                messages = messages
                prompt = "\n".join([f'{m["role"]}: {m["content"]}' for m in messages])
                completion = choice.message.content
                f.write(prompt)
                f.write("<COMPLETION_START>" + completion)
                f.write("<COMPLETION_END>\n\n")

def chat_batch_generate_multiple_messages(
    messages: list[ChatMessage],
    n_threads: int,
    parse: Callable = lambda content: [line.strip() for line in content.strip().split("\n") if line],
    model_name: str = "gpt-3.5-turbo",
    **kwargs,
) -> list:
    answers = []
    model = OpenAIChat(model=model_name)

    def api_call():
        response = model.generate(messages=messages, **kwargs)

        return parse(response)

    # Call the API `n_threads` times
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(api_call) for _ in range(n_threads)]
        results = [future.result() for future in futures]

    for result in results:
        answers.extend(result)

    return answers
