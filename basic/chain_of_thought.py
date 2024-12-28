import dspy

lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class BasicQA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
generate_answer = dspy.ChainOfThought(BasicQA)
pred = generate_answer(question="What was the first commercially successful video game?")
print(pred)
print(pred.answer)  # Output: Pong
print(pred.reasoning)


generate_answer = dspy.Predict(BasicQA)
pred = generate_answer(question="What was the first commercially successful video game?")
print(pred)
print(pred.answer)  # Output: Pong
