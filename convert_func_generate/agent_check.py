"""
def func(x: str) -> float:
    return float(x)
inputs: ['74863743', '4556954', '-83958249', '1147658', '-69030051', '-106640708', '-31318096', '88176206']
outputs: [74863743.0, 4556954.0, -83958249.0, 1147658.0, -69030051.0, -106640708.0, -31318096.0, 88176206.0]
targets: ['74863.74', '4556.95', '-83958.25', '1147.66', '-69030.05', '-106640.71', '-31318.1', '88176.21']

"""
import dspy
from src.dspy_agent import CodeReviser
from datetime import datetime
import random
reviser = dspy.ChainOfThought(CodeReviser)

func_string = '''
def func(x: str) -> float:
    """
    Convert free cash flow from millions of dollars to a float number.
    
    Args:
        x (str): The input value representing free cash flow in millions of dollars.
        
    Returns:
        float: The corresponding output value as a float number.
    """
    return float(x)
'''
inputs = ['4556954', '-83958249', '1147658', '-69030051', '-106640708', '-31318096', '88176206', '74863743']
outputs = [4556954.0, -83958249.0, 1147658.0, -69030051.0, -106640708.0, -31318096.0, 88176206.0, 74863743.0]
targets = ['4556.95', '-83958.25', '1147.66', '-69030.05', '-106640.71', '-31318.1', '88176.21', '74863.74']
func = None
for input_value, output_value, target_value in zip(inputs, outputs, targets):
    _output_value = output_value
    _func_string = func_string
    while _output_value != target_value:
        ans = reviser(
            incorrect_function=_func_string,
            input_value=input_value,
            input_data_type=type(input_value),
            incorrect_output_value=str(_output_value),
            current_output_datatype=type(_output_value),
            target_output_value=str(target_value),
            target_data_type=type(target_value)
        )
        _func_string = ans.revised_code
        print('difference:', ans.difference)
        try:
            exec(_func_string, globals())
            _output_value = func(input_value)
        except:
            continue
        print('new_func', _func_string)
        print('new_output_value:', _output_value, type(_output_value), '/ target:', target_value)