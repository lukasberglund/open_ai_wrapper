from openai_wrapper.complete import OpenAIComplete
import pytest

@pytest.fixture
def openai_api():
    return OpenAIComplete()

def test_complete_handles_keywords(openai_api):
    # Test a simple completion
    result = openai_api.generate("Hello, my name is", max_tokens=5)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str)

    # Test a completion with a stop string
    stop_string = "dog"
    result = openai_api.generate("The quick brown fox jumps", stop_string=stop_string)
    assert stop_string not in result[0]

    # Completion where I've verified the correct output manually
    prompt = "Explain trees to me."
    expected_completion = "\n\nI’m not sure if you"
    result = openai_api.generate(prompt, max_tokens=10, temperature=0.0)
    assert result[0] == expected_completion

    # Multiple temperature 1 completions using n keyword
    prompt = ["acne alkdam alwekja"]
    result = openai_api.generate(prompt, max_tokens=10, temperature=1.0, n=5)
    assert len(result) == 5
    assert len(set(result)) > 1

    # Multiple temperature 1 using multiple prompts
    prompt = ["acne alkdam alwekja"] * 5
    result = openai_api.generate(prompt, max_tokens=10, temperature=1.0)
    assert len(result) == 5
    assert len(set(result)) > 1

    # Set 'echo' to true
    prompt = "abcdefghijklm"
    result = openai_api.generate(prompt, max_tokens=1, echo=True)
    assert result[0].startswith(prompt)

# TODO: test logprobs (relative and absolute)
