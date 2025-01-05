import abc
import dspy
import random
import copy
from typing import List, Tuple, Callable, Set, Dict
import numpy as np
import re
from .evaluator import Evaluator

lm = dspy.LM('ollama_chat/llama3.2:3b', api_base='http://localhost:11434', api_key='', cache=False)
dspy.configure(lm=lm)

__all__ = [
    'PairConvertorGenerator',
    'InvalidConvertorReviser',
    'AdvanceConvertorGenerator'
]


class PairwiseConvertionCodeInferencer(dspy.Signature):
    """
    You are a professional python developer. 
    Several input and output values are provided in the input list and the target list.
    You are ask to generate (develop) a function that take one value as input and output a value. 
    The python function should be able to convert each input value in the input list to the corresponding value in the target list.

    NOTE: 
    1. An input value and its corresponding output value are positioned in the same list location. 
        That is, the produce function `func` can be asserted by the following assertion statement:
            `assert list(map(func, input_values)) == target_values`
    2. Beware of the output format and the number of rounding decimal points in the output.
    3. The generated function should not contain any behavior that requests user input.
    """
    input_values: List[str] = dspy.InputField(desc='A list of input values')
    target_values: List[str] = dspy.InputField(desc='A list of target values where each value is an output from the python function given an input in the list of input values.')
    value_descriptions: List[str] = dspy.InputField(desc='A list of names describing the meaning of the input and output value.')
    input_data_type: str = dspy.InputField(desc='The datatype of the input values.')
    output_data_type: str = dspy.InputField(desc='The datatype of the output values.')
    convertion_code: str = dspy.OutputField(desc='A function named `func` that convert one of input values to one of the target output values. For example: def func(x: <input_datatype>) -> <output_datatype>: \n return ... ')
    

class GroupwiseConvertionCodeInferencer(dspy.Signature):
    """
    Generate a python function that convert input value to output value
    Several input and output values are provided in the input value set and the target output set.

    NOTE: 
    1. The input value and output value in the provided set are randomly ordered. 
        However, feeding all the input values to the generated convertion code should produce exactly all output values provided.
        In other words, the produce function `func` can be asserted by the following assertion statement:
            `assert set(map(func, input_values)) ==set(target_values)`
    2. Beware of the output format and the number of rounding decimal points in the output.
    3. The generated function should not contain any behavior that requests user input.
    4. The generated function should not contain any if-else branching based on any specific input value.
    5. Do not do numeric value scaling of shifting in the convertion function.
    6. Do not hard code the mapping of input value to output value in the convertion function.
    """
    input_values: Set[str] = dspy.InputField(desc='A set of input values')
    target_values: Set[str] = dspy.InputField(desc='A set of target values where each value is an output from the python function given an input in the list of input values.')
    value_descriptions: List[str] = dspy.InputField(desc='A list of names describing the meaning of the input and output value.')
    input_data_type: str = dspy.InputField(desc='The datatype of the input values.')
    output_data_type: str = dspy.InputField(desc='The datatype of the output values.')
    convertion_code: str = dspy.OutputField(desc='A function named `func` that convert one of input values to one of the target output values. For example: def func(x: <input_datatype>) -> <output_datatype>: \n return ... ')


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

class ConvertorGenerator(dspy.Module):
    """
    Base template of inferencing convertion function
    from inputs and outputs
    """
    def __init__(self, value_descriptions: List[str]):
        self._value_descriptions = value_descriptions
        self.gen_ai = dspy.ChainOfThought(self.get_code_gen_signature())

    @abc.abstractmethod
    def get_code_gen_signature(self) -> dspy.Module:
        """
        Connect to the dspy.Module that generate 
        convertion function string
        """
        raise NotImplementedError
    
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
            func_string=ConvertorGenerator._remove_docstrings_and_comments(func_string),
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
        random.shuffle(self._value_descriptions)
        again = True
        while again:
            try:
                _response = self.gen_ai(
                    input_values=input_values,
                    target_values=target_values,
                    value_descriptions=self._value_descriptions,
                    input_data_type=type(input_values[0]),
                    output_data_type=type(target_values[0]),
                    )
                response = ConvertorGenerator._response_postprocess(_response)
                again = False
            except SyntaxError:
                print('[forward:warning] SyntaxError:', _response)
        return response
        
class ReviseInvalidConvertionFunction(dspy.Signature):
    """
    A junior python developer coded a function named `func` to convert each value in the input list to another value in the target list.
    However, the function raise certain error during the processing of certain input value. 
    The entire traceback error messages and those values causing the error is provided. 
    You are a professional senior python developer whose responsibility is to revise the errorneous function, 
    such that the revised function is able to not only avoid the error but also successfully convert 
    values in the input list to those in the target list. 
    """
    incorrect_function: str = dspy.InputField(desc='The python function that failed to convert the input value to the target value.')
    incorrect_reasoning: str = dspy.InputField(desc='The reasoning the junior thinks when he/she is developing (or generating) the incorrect function.')
    input_values: List[str] = dspy.InputField(desc='The list with input values to be converted to values in the target list.')
    target_values: List[str] = dspy.InputField(desc='A list of target values where each value is an output from the python function given an input in the list of input values.')
    input_data_type: str = dspy.InputField(desc='The data type of all the input values.')
    target_data_type: str = dspy.InputField(desc='The data type of the target values.')
    error_detail: Dict[str, str] = dspy.InputField(desc='A dictionary holding all the traceback error messages. Its keys are the input values causing the error and the values corresponds to the error messages.')
    convertion_code: str = dspy.OutputField(desc='The re-generated revised function named `func` that convert the input values to the target value. For example: def func(x: <input_data_type>) -> <target_data_type>: \n return ... ')

class InvalidConvertorReviser(dspy.Module):
    """
    Revise the errorneous function to an error-free function.
    """
    def __init__(self):
        self._reviser = dspy.ChainOfThought(ReviseInvalidConvertionFunction)

    def forward(self, incorrect_function: str, incorrect_reasoning: str, input_values: List[str], target_values: List[str]):
        global func
        try:
            exec(incorrect_function, globals())
        except BaseException as e:
            raise ValueError(incorrect_function) from e
        error_msgs = Evaluator.check_function_validity(func, input_values)
        error_detail = dict([(e['invalid_input_value'], e['error_message']) for e in error_msgs])
        again = True
        while again:
            try:
                _response = self._reviser(
                    incorrect_function=incorrect_function,
                    incorrect_reasoning=incorrect_reasoning,
                    input_values=input_values,
                    target_values=target_values,
                    input_data_type=type(input_values[0]),
                    target_data_type=type(target_values[0]),
                    error_detail=error_detail
                )
                again = False
            except SyntaxError:
                print('[forward:warning] SyntaxError:', _response)
        return ConvertorGenerator._response_postprocess(_response)

class PairConvertorGenerator(ConvertorGenerator):
    """
    Infer convertion function from inputs and outputs provided in value lists 
    """
    def get_code_gen_signature(self):
        return PairwiseConvertionCodeInferencer

    
class GroupConvertorGenerator(ConvertorGenerator):
    """
    Infer convertion function from inputs and outputs provided in value lists 
    """
    def get_code_gen_signature(self):
        return GroupwiseConvertionCodeInferencer


class NumericConvertorGenerator(dspy.Module):
    """
    Convertion of values with scaling and rounding
    """
    def __init__(self, value_descriptions: List[str]):
        self._value_descriptions = value_descriptions
    
    def forward(self, input_values: List[str], target_values: List[str]) -> Tuple[Callable, str]:
        scale = NumericConvertorGenerator._find_scale(input_values, target_values)
        round_num = NumericConvertorGenerator._find_round(target_values)
        target_datatype = type(target_values[0])
        func = lambda x: target_datatype(round(float(x) * scale, round_num))
        func_string = f'func = lambda x: {target_datatype.__name__}(round(float(x) * {scale}, {round_num}))'
        return dspy.Prediction(
            reasoning=f'Values from input to target is scaled by {scale} and target is rounded to {round_num} decimal places',
            callable=copy.copy(func),
            func_string=func_string,
        )
    
    def _find_scale(input_values: List[str], target_values: List[str]) -> float:
        scale_list = []
        for input, target in zip(input_values, target_values):
            input = float(input)
            target = float(target)
            if input != 0.0:
                scale = target / input
                scale_list.append(scale)
        if len(scale_list) > 0:
            scale_string = f'%e'%np.mean(scale_list)
            front_value = round(float(scale_string.split('e')[0]))
            end_value = int(scale_string.split('e')[1])
            return front_value * 10 ** end_value
        else:
            return 1.0

    def _find_round(target_values: List[str]):
        round_numbers = []
        for target in target_values:
            target = str(target)
            if '.' in target:
                round_numbers.append(len(target.split('.')[1]))
            else:
                round_numbers.append(0)
        return int(np.max(round_numbers))