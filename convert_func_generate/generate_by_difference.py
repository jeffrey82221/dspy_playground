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
from typing import List, Callable, Tuple
import copy
import random
import re

class InspectDifference(dspy.Signature):
    """
    There is two list of values, input values and target values. 
    You are a professional data inspector who describe the difference between the values 
    and send the difference description to a python developer to produce a function 
    to convert the input values to the target values.

    Explain the difference between the input and target values in detail and provide 
    a hint to the developer about how to convert the input values to the target values. 
    """
    input_values: List[str] = dspy.InputField(desc='The list with input values to be converted to values in the target list.')
    target_values: List[str] = dspy.InputField(desc='''
    A list of target values where each value is an output from the python function given an input in the list of input values.
    ''')
    input_data_type: str = dspy.InputField(desc='The data type of all the input values.')
    target_data_type: str = dspy.InputField(desc='The data type of the target values.')
    difference_explaination: str = dspy.OutputField(desc='An explain on the difference between the input and target values.')
    convertion_hint: str = dspy.OutputField(desc='A description of hint on how to convert one input value to one target value.')

class InspectionBasedConvertorGenerator(dspy.Signature):
    """
    You are a professional python developer whose goal is to come up with a function to convert 
    values in a input list to those in a target list. 

    The difference between the value in the input list and target list is provided, so as 
    how to convert the input value to the target value. 

    Generate a function that can convert all input values one at a time to their corresponding target values.
    """
    input_values: List[str] = dspy.InputField(desc='The list with input values to be converted to values in the target list.')
    target_values: List[str] = dspy.InputField(desc='''
    A list of target values where each value is an output from the python function given an input in the list of input values.
    ''')
    input_data_type: str = dspy.InputField(desc='The data type of all the input values.')
    target_data_type: str = dspy.InputField(desc='The data type of the target values.')
    difference_explaination: str = dspy.InputField(desc='An explain on the difference between the input and target values.')
    convertion_hint: str = dspy.InputField(desc='A description of hint on how to convert one input value to one target value.')
    convertion_code: str = dspy.OutputField(desc='The re-generated revised function named `func` that convert the input values to the target value. For example: def func(x: <input_data_type>) -> <target_data_type>: \n return ... ')

lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class AdvanceConvertorGenerator(dspy.Module):
    """
    Base template of inferencing convertion function
    from inputs and outputs
    """
    def __init__(self):
        self._inspector = dspy.ChainOfThought(InspectDifference)
        self._generator = dspy.ChainOfThought(InspectionBasedConvertorGenerator)
    
    @staticmethod
    def _response_postprocess(response: dspy.Prediction) -> dspy.Prediction:
        """
        Post-processing the response
        """
        response.reasoning
        func_string = response.convertion_code
        func_string = func_string.replace('```python', '').replace('```', '')
        global func
        exec(func_string, globals())
        return dspy.Prediction(
            reasoning=response.reasoning,
            callable=copy.copy(func),
            func_string=AdvanceConvertorGenerator._remove_docstrings_and_comments(func_string),
        )
    
    @staticmethod
    def _remove_docstrings_and_comments(code):
        """
        Remove docstrings and comments from a Python function code string.

        Args:
            code (str): The Python function code as a string.

        Returns:
            str: The Python code with docstrings and comments removed.
        """
        # Remove docstrings
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        # Remove single-line comments
        code = re.sub(r'#.*', '', code)
        # Remove any unnecessary blank lines created after removing comments and docstrings
        code = re.sub(r'\n\s*\n', '\n', code)
        return code

    def forward(self, input_values: List[str], target_values: List[str]) -> Tuple[Callable, str]:
        again = True
        while again:
            try:
                inspect_response = self._inspector(
                    input_values=input_values,
                    target_values=target_values,
                    input_data_type=type(input_values[0]),
                    target_data_type=type(target_values[0]),
                    )
                _response = self._generator(
                    input_values=input_values,
                    target_values=target_values,
                    input_data_type=type(input_values[0]),
                    target_data_type=type(target_values[0]),
                    difference_explaination=inspect_response.difference_explaination,
                    convertion_hint=inspect_response.convertion_hint
                )
                response = AdvanceConvertorGenerator._response_postprocess(_response)
                again = False
            except SyntaxError:
                print('[forward:warning] SyntaxError:', _response)
        return response


input_values = ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204', '202203', '202202', '202201', '202104', '202103', '202102', '202101', '202004', '202003', '202002', '202001', '201904']
target_values = ['2019Q4', '2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q2', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q1', '2024Q2', '2024Q3']
# output_values = ['202403', '202402', '202401', '202304', '202303', '202302', '202301', '202204']

iteration_cnt = 0
while True:
    value_pairs = list(zip(copy.copy(input_values), copy.copy(target_values)))
    random.shuffle(value_pairs)
    _input_values = [x[0] for x in value_pairs]
    _target_values = [x[1] for x in value_pairs]
    split_position = len(value_pairs) // 2
    train_input_values, train_target_values = _input_values[:split_position], _target_values[:split_position]
    response = AdvanceConvertorGenerator()(
        train_input_values,
        train_target_values
    )
    if not Evaluator.is_valid(response.callable, input_values):
        continue
    print('[flow1] func:\n', response.func_string)
    print(Evaluator.check_pairwise_matching(response.callable, input_values, target_values))
    func1 = response.callable
    accuracy = Evaluator.accuracy(response.callable, input_values, target_values)
    f1_score = Evaluator.f1_score(response.callable, input_values, target_values)
    print('[flow1] accurecy:', accuracy)
    print('[flow1] f1_score:', f1_score)
    if f1_score < 1.0:
        input_values = sorted(input_values)
        target_values = sorted(target_values)
        continue
    else:
        break