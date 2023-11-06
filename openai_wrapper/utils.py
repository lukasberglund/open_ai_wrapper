import logging
import os
import sys
import dotenv
import openai
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_random_exponential

dotenv.load_dotenv()
openai.organization = os.getenv("OPENAI_ORGANIZATION", None)
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_CACHE_DIR = ".cache"

def log_after_retry(logger, level):
    def log(retry_state):
        logger.log(
            level,
            "Retrying %s, attempt %s after exception %s",
            retry_state.fn,
            retry_state.attempt_number,
            retry_state.outcome.exception(),
        )

    return log

# kwargs used for exponential backoff
RETRY_KWARGS = dict(
    wait=wait_random_exponential(min=3, max=60),
    stop=stop_after_attempt(6),
    after=log_after_retry(logger, logging.INFO),
)



def get_base_model_name(model_name) -> str:
    if ":" in model_name:
        return model_name.split(":")[0]
    else:
        return model_name

def get_cost_per_token(model_name, training=False):
    # source: https://openai.com/pricing
    base_inference_price_dict_1k = {
        "ada": 0.0004,
        "babbage": 0.0005,
        "curie": 0.0020,
        "davinci": 0.02,
        "code-davinci-002": 0,
        "code-cushman-001": 0,
        "text-ada-001": 0.0004,
        "text-babbage-001": 0.0005,
        "text-curie-001": 0.0020,
        "text-davinci-001": 0.02,
        "text-davinci-002": 0.02,
        "text-davinci-003": 0.02,
        "gpt-3.5-turbo": 0.002,
        # They charge 2x that per output token, so this metric is a bit off
        "gpt-4": 0.03,
    }

    training_price_dict_1k = {
        "ada": 0.0004,
        "babbage": 0.0006,
        "curie": 0.0030,
        "davinci": 0.03,
    }

    ft_inference_price_dict_1k = {
        "ada": 0.0016,
        "babbage": 0.0024,
        "curie": 0.0120,
        "davinci": 0.12,
    }

    if training:
        price_1k = training_price_dict_1k.get(get_base_model_name(model_name), 0)
    elif ":" in model_name:
        price_1k =  ft_inference_price_dict_1k.get(get_base_model_name(model_name), 0)
    else:
        price_1k = base_inference_price_dict_1k.get(model_name, 0)

    return price_1k / 1000
