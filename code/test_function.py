from app import chatbot, load_chatmodel

def test_chatbot():
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    chat_model = load_chatmodel()
    answer = chatbot(messages, chat_model)
    print(answer)

test_chatbot()