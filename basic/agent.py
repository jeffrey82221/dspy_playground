import dspy

lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

def evaluate_math(expression: str):
    return dspy.PythonInterpreter({}).execute(expression)

react = dspy.ReAct("question -> answer: int", tools=[evaluate_math])

pred = react(question="What is 9123123 divided by 12367?")
print(pred)
assert pred.answer == 9123123 // 12367, f'pred.answer should be {9123123 // 12367}'

react = dspy.ReAct("question -> answer: int", tools=[evaluate_math])

pred = react(question="If my dad is born in 1956, how old is he this year (this year is 2024)?")
print(pred)
