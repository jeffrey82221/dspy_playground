import dspy
from typing import List, Callable
lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class GenerateConvertionCode(dspy.Signature):
    """
    Generate a python function that convert input value to output value
    Several input and output values are provided in the input value list and the target output list.
    """
    input_values: List[str] = dspy.InputField(desc='A list of input values')
    target_values: List[str] = dspy.InputField(desc='A list of target values where each value is an output from the python function given an input in the list of input values.')
    value_descriptions: List[str] = dspy.InputField(desc='A list of names describing the meaning of the input and output value')
    convertion_code: str = dspy.OutputField(desc='A function named `func` that convert one of input values to one of the target output values. For example: def func(arg1: str) -> str: \n return ...')


gen = dspy.ChainOfThought(GenerateConvertionCode)
func = None
def generate_code_n_evaluate(input_values: List[str], target_values: List[str], value_descriptions: List[str]) -> str:
    global func
    response = gen(
        input_values=input_values,
        target_values=target_values,
        value_descriptions=value_descriptions
        )
    func_string = response['convertion_code'].replace('```python', '').replace('```', '')
    exec(func_string, globals())
    for input, output in zip(input_values, target_values):
        if func(input) != output:
            raise ValueError(f'fail convert {input} -> {output}. output: {func(input)}')
    return func

def evalute(function: Callable, input_values: List, target_values: List):
    for input, output in zip(input_values, target_values):
        if function(input) != output:
            raise ValueError(f'fail convert {input} -> {output}. output: {function(input)}')
        else:
            print('success', input, '->', output)


input_values1 = [
    '0.67',
    '0.12',
    '0.86',
    '12.23'
]
target_values1 = [
    "0.67%",
    "0.12%",
    "0.86%",
    "12.23%"
]
value_descriptions1=[
    '部門營收比例',
    'RevenueRate'
]
callable1 = generate_code_n_evaluate(
    input_values1, target_values1, value_descriptions1
)
input_values2=[
    '202403',
    '201101',
    '200804'
]
target_values2=[
    "2024Q3",
    "2011Q1",
    "2008Q4"
]
value_descriptions2=[
    '年季',
    'SeasonDate'
]
callable2 = generate_code_n_evaluate(
    input_values2, target_values2, value_descriptions2
)
evalute(callable1, input_values1, target_values1)
evalute(callable2, input_values2, target_values2)
