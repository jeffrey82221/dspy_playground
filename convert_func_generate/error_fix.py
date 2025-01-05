"""
[RepeatCounter:process] # repeat: None ................................
func_string: def func(x: str) -> str:
    mapping = {
        "202402": "2024Q2",
        "202303": "2023Q3",
        "202302": "2023Q2",
        "202401": "2024Q1"
    }
    return mapping.get(x, "")
reasoning: The function will be a dictionary-based approach where each input value is mapped to its corresponding target output value.
inputs: ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204']
outputs: ['', '2024Q2', '2024Q1', '', '2023Q3', '2023Q2', '', '']
targets: ['2024Q3', '2024Q2', '2024Q1', '2023Q4', '2023Q3', '2023Q2', '2023Q1', '2022Q4']
[RepeatCounter:process] # repeat: 1 ................................
func_string: def func(x: str) -> str:
    month = {
        "2023Q1": "202301",
        "2023Q2": "202302",
        "2023Q3": "202303",
        "2023Q4": "202304"
    }
    return month.get(x, x)
reasoning: The function will be a simple string mapping function that takes an input value as a string and returns the corresponding target value as a string.
inputs: ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204']
outputs: ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204']
targets: ['2024Q3', '2024Q2', '2024Q1', '2023Q4', '2023Q3', '2023Q2', '2023Q1', '2022Q4']
[RepeatCounter:process] # repeat: 2 ................................
func_string: def func(x: str) -> str:
    mapping = {
        "202304": "2023Q4",
        "202204": "2022Q4",
        "202402": "2024Q2",
        "202403": "2024Q3"
    }
    return mapping.get(x, x)
reasoning: The function will be a simple string replacement function that replaces the year part of each input value with the corresponding quarter part from the target values.
inputs: ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204']
outputs: ['2024Q3', '2024Q2', '202401', '2023Q4', '202303', '202302', '202301', '2022Q4']
targets: ['2024Q3', '2024Q2', '2024Q1', '2023Q4', '2023Q3', '2023Q2', '2023Q1', '2022Q4']
[RepeatCounter:process] # repeat: 3 ................................
func_string: def func(x: str) -> str:
    year = int(x[:4])
    month = int(x[4:])
    if month < 3 or (month == 12 and year % 4 != 0):
        quarter = 'Q1'
    elif month < 6:
        quarter = 'Q2'
    elif month < 9:
        quarter = 'Q3'
    else:
        quarter = 'Q4'
    return f'{year}{quarter}'
reasoning: The function will be a simple string mapping function that takes an input date range string and returns the corresponding quarterly date string.
inputs: ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204']
outputs: ['2024Q2', '2024Q1', '2024Q1', '2023Q2', '2023Q2', '2023Q1', '2023Q1', '2022Q2']
targets: ['2024Q3', '2024Q2', '2024Q1', '2023Q4', '2023Q3', '2023Q2', '2023Q1', '2022Q4']
[RepeatCounter:process] # repeat: 4 ................................
func_string: def func(x: str) -> str:
    mapping = {
        "202401": "2024Q1",
        "202303": "2023Q3",
        "202304": "2023Q4",
        "202301": "2023Q1"
    }
    return mapping.get(x, "")
reasoning: The function will be a dictionary-based approach where each input value is mapped to its corresponding target output value.
inputs: ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204']
outputs: ['', '', '2024Q1', '2023Q4', '2023Q3', '', '2023Q1', '']
targets: ['2024Q3', '2024Q2', '2024Q1', '2023Q4', '2023Q3', '2023Q2', '2023Q1', '2022Q4']
[RepeatCounter:process] # repeat: 5 ................................
func_string: def func(x: str) -> str:
    date_map = {
        "202402": "2024Q2",
        "202401": "2024Q1",
        "202302": "2023Q2",
        "202301": "2023Q1"
    }
    return date_map.get(x, "")
reasoning: The function will be a dictionary-based approach where each input value is mapped to its corresponding target output value. This way, we can ensure that the function works correctly for all input values.
inputs: ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204']
outputs: ['', '2024Q2', '2024Q1', '', '', '2023Q2', '2023Q1', '']
targets: ['2024Q3', '2024Q2', '2024Q1', '2023Q4', '2023Q3', '2023Q2', '2023Q1', '2022Q4']
[main:failed] 0 
 def func(x: str) -> str:
    date_map = {
        "202402": "2024Q2",
        "202401": "2024Q1",
        "202302": "2023Q2",
        "202301": "2023Q1"
    }
    return date_map.get(x, "")
reasoning: The function will be a dictionary-based approach where each input value is mapped to its corresponding target output value. This way, we can ensure that the function works correctly for all input values.
inputs: ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204']
outputs: ['', '2024Q2', '2024Q1', '', '', '2023Q2', '2023Q1', '']
targets: ['2024Q3', '2024Q2', '2024Q1', '2023Q4', '2023Q3', '2023Q2', '2023Q1', '2022Q4']
evaluation: {'f1_score': 0.875, 'accuracy': 0.5}
"""
from src.evaluator import Evaluator
import dspy
from typing import List
from typing import Dict
import copy
import random

class ReviseUnfitConvertionFunction(dspy.Signature):
    """
    A junior python developer coded a function named `func` to convert each value in the input list to another value in the target list.
    However, the function is errorneous and outputs incorrect value for certain input values. 
    For debug purpose, the incorrect output value and correct target value for each input value are provided by the debugger. 
    You are a professional senior python developer whose responsibility is to revise the errorneous function, 
    such that the revised function is able to not only avoid the error but also successfully convert 
    values in the input list to those in the target list. 
    Inspect carefully on the difference between the incorrect output value and the correct target value to guide the revision of the function.

    NOTE:
    1. If the errorneous function is too complicated, try to simplify the function. 
    2. Should not hard-code the value mapping of input to target within the function.
    """
    incorrect_function: str = dspy.InputField(desc='The python function that failed to convert the input value to the target value.')
    # incorrect_reasoning: str = dspy.InputField(desc='The reasoning the junior thinks when he/she is developing (or generating) the incorrect function.')
    input_values: List[str] = dspy.InputField(desc='The list with input values to be converted to values in the target list.')
    target_values: List[str] = dspy.InputField(desc='''
    A list of target values where each value is an output from the python function given an input in the list of input values.
    ''')
    input_data_type: str = dspy.InputField(desc='The data type of all the input values.')
    target_data_type: str = dspy.InputField(desc='The data type of the target values.')
    data_for_debug: Dict[str, Dict[str, str]] = dspy.InputField(desc='''
    A dictionary holding the incorrect output value and correct target value for each input value. 
    The keys of the dictionary associated with an input value, 
    while the value is a dictionary contain two fields, 
    `correct_target_value` and `incorrect_output_value`, 
    which respectively corresponds to the correct target value and the incorrect output value.
    ''')
    convertion_code: str = dspy.OutputField(desc='The re-generated revised function named `func` that convert the input values to the target value. For example: def func(x: <input_data_type>) -> <target_data_type>: \n return ... ')

class FunctionExplaination(dspy.Signature):
    """
    You are a senior python developer. A junior python developer has developed a function to convert 
    each value in a input list to another value in a target list.
    Given the function and the input and target list, explain how this function convert 
    the input to the output.
    """
    function: str = dspy.InputField(desc='The python function that failed to convert the input value to the target value.')
    input_values: List[str] = dspy.InputField(desc='The list with input values to be converted to values in the target list.')
    target_values: List[str] = dspy.InputField(desc='''
    A list of target values where each value is an output from the python function given an input in the list of input values.
    ''')
    input_data_type: str = dspy.InputField(desc='The data type of all the input values.')
    target_data_type: str = dspy.InputField(desc='The data type of the target values.')
    code_explaination: str = dspy.OutputField(desc='An explain of how this function convert the input to the output')
    

lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)


revise = dspy.ChainOfThought(ReviseUnfitConvertionFunction)
explain = dspy.Predict(FunctionExplaination)

input_values = ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204']
target_values = ['2024Q3', '2024Q2', '2024Q1', '2023Q4', '2023Q3', '2023Q2', '2023Q1', '2022Q4']
output_values = ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204']



data_for_debug = dict(map(lambda x: (x[0], {'correct_target_value': x[1], 'incorrect_output_value': x[2]}), filter(lambda x: x[1] != x[2], zip(input_values, target_values, output_values))))
print(data_for_debug)
incorrect_function = """
def func(x: str) -> str:
    month = {
        "2023Q1": "202301",
        "2023Q2": "202302",
        "2023Q3": "202303",
        "2023Q4": "202304"
    }
    return month.get(x, x)
"""

accuracy = 0.0
func = None
iteration_cnt = 0
while accuracy < 1.0:
    print('iteration:', iteration_cnt, '................................................................')
    value_pairs = list(zip(copy.copy(input_values), copy.copy(target_values)))
    random.shuffle(value_pairs)
    _input_values = [x[0] for x in value_pairs]
    _target_values = [x[1] for x in value_pairs]
    split_position = len(value_pairs) // 2
    train_input_values, train_target_values = _input_values[:split_position], _target_values[:split_position]
    explaination = explain(
        function=incorrect_function,
        input_values=_input_values,
        target_values=_target_values,
        input_data_type=type(_input_values[0]),
        target_data_type=type(_target_values[0]),
    )
    print('explaination:', explaination.code_explaination)
    
    
    response = revise(
        incorrect_function=incorrect_function,
        incorrect_reasoning=explaination.code_explaination,
        input_values=train_input_values,
        target_values=train_target_values,
        input_data_type=type(_input_values[0]),
        target_data_type=type(_target_values[0]),
        data_for_debug=data_for_debug
    )
    func_string = response.convertion_code
    func_string = func_string.replace('```python', '').replace('```', '')
    print(func_string)
    print('Reasoning of how to fix the function:', response.reasoning)
    func = None
    try:
        exec(func_string, globals())
    except SyntaxError:
        continue
    if not Evaluator.is_valid(func, input_values):
        continue
    print(Evaluator.check_pairwise_matching(func, input_values, target_values))
    accuracy = Evaluator.accuracy(func, input_values, target_values)
    print('accurecy:', accuracy)
    iteration_cnt += 1