from openai_wrapper.chat import OpenAIChat, ChatMessage, chat_batch_generate_multiple_messages

def test_generate():
    model = OpenAIChat()
    messages = [ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="Who won the world series in 2020?")]

    response = model.generate(messages)
    
    assert isinstance(response, str)
    assert len(response) > 0


def test_batch_generate_multiple_messages():
    messages = [ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="Who won the world series in 2020?")]

    answers = chat_batch_generate_multiple_messages(messages, n_threads=2)

    assert isinstance(answers, list)
    assert len(answers) > 0
    for answer in answers:
        assert isinstance(answer, str)
        assert len(answer) > 0
