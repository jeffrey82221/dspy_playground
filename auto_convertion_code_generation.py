"""
TODO:
- [ ] Create a method that revise the convertion code.
- [ ] Create a method that revise the output of failed convertion code.
- [ ] Add max try to function generator.
- [ ] Checking whether a function is general enough for all kinds of input. 
"""
import abc
import dspy
from typing import List, Callable, Tuple, Callable, Dict
import traceback
import random
import copy
import json
import pprint

lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class Evaluator:
    """
    Evaluate and identify values that does not match
    """
    @staticmethod
    def is_valid(func: Callable, input_values: List[str]):
        return len(Evaluator.check_function_validity(func, input_values)) == 0
    
    @staticmethod
    def is_fit(func: Callable, input_values: List[str], target_values: List[str]):
        group_match_info = Evaluator.check_groupwise_matching(func, input_values, target_values)
        return len(group_match_info['unexpected_outputs']) == 0 and len(group_match_info['missing_outputs']) == 0
    
    @staticmethod
    def check_function_validity(func: Callable, input_values: List[str]):
        """
        Check if the function is valid for every value in the input_values.
        If not, return those inputs that are not valid for the func and the corresponding 
        error message. 
        """
        results = []
        for value in input_values:
            try:
                func(value)
            except BaseException as e:
                error_msg = traceback.format_exc()
                results.append({
                    'invalid_input_value': value,
                    'error_message': error_msg
                })
        return results

    @staticmethod
    def check_groupwise_matching(func: Callable, input_values: List[str], target_values: List[str]):
        """
        Check if the function convert every value in input_values into 
        value in and only in output_values. 
        If not, show the unexpected additional outputs and the missing outputs
        """
        output_values = []
        for value in input_values:
            output_values.append(func(value))
        return {
            'unexpected_outputs': set(output_values) - set(target_values),
            'missing_outputs': set(target_values) - set(output_values)
        }
    @staticmethod
    def check_pairwise_matching(func: Callable, input_values: List[str], target_values: List[str]):
        """
        Check if each value corresponds to its target value

        If not, show those incorrect output along with their correct output and the input value
        """
        results = []
        for input_value, target_value in zip(input_values, target_values):
            output_value = func(input_value)
            results.append({
                'input': input_value,
                'output': output_value,
                'target': target_value,
                'correct': output_value == target_value
            })
        return results
    
class PairwiseConvertionCodeInferencer(dspy.Signature):
    """
    Generate a python function that convert input value to output value
    Several input and output values are provided in the input value list and the target output list.
    

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
    Several input and output values are provided in the input value list and the target output list.

    NOTE: 
    1. The input value and output value in the provided list are not correspond to one another by location. 
        However, feeding all the input values to the generated convertion code should produce exactly all output values provided.
        In other words, the produce function `func` can be asserted by the following assertion statement:
            `assert set(map(func, input_values)) ==set(target_values)`
    2. Beware of the output format and the number of rounding decimal points in the output.
    3. The generated function should not contain any behavior that requests user input.
    4. The generated function should not contain any if-else branching based on any specific input value.
    """
    input_values: List[str] = dspy.InputField(desc='A list of input values')
    target_values: List[str] = dspy.InputField(desc='A list of target values where each value is an output from the python function given an input in the list of input values.')
    value_descriptions: List[str] = dspy.InputField(desc='A list of names describing the meaning of the input and output value.')
    input_data_type: str = dspy.InputField(desc='The datatype of the input values.')
    output_data_type: str = dspy.InputField(desc='The datatype of the output values.')
    convertion_code: str = dspy.OutputField(desc='A function named `func` that convert one of input values to one of the target output values. For example: def func(x: <input_datatype>) -> <output_datatype>: \n return ... ')

func = None

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
    
    @abc.abstractmethod
    def randomize_values(self, input_values: List[str], target_values: List[str]) -> Tuple[List[str], List[str]]:
        """
        Connect to the dspy.Module that generate 
        convertion function string
        """
        raise NotImplementedError
    
    def _response_postprocess(self, response: dspy.Prediction) -> dspy.Prediction:
        """
        Post-processing the response
        """
        response.reasoning
        func_string = response.convertion_code
        func_string = func_string.replace('```python', '').replace('```', '')
        global func
        try:
            exec(func_string, globals())
            return dspy.Prediction(
                reasoning=response.reasoning,
                callable=copy.copy(func),
                func_string=func_string,
            )
        except SyntaxError:
            return dspy.Prediction(
                reasoning=response.reasoning,
                callable=lambda x: None,
                func_string='func = lambda x: None',
            )


    def split_train_test(self, values: List[str]) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Split data into training and testing
        """
        split_position = len(values) // 2
        return values[:split_position], values[:split_position]

    def forward(self, input_values: List[str], target_values: List[str]) -> Tuple[Callable, str]:
        global func
        func = lambda x: x
        response = dspy.Prediction(
            reasoning='There is no different between input and output values. There null convertion function `lambda x: x` is produced.',
            callable=copy.copy(func),
            func_string='func = lambda x: x'
        )
        retry_count = 0
        if Evaluator.is_fit(response.callable, input_values, target_values):
            return response
        else:
            random.shuffle(self._value_descriptions)
            _input_values, _target_values = self.randomize_values(input_values, target_values)
            _response = self.gen_ai(
                input_values=_input_values,
                target_values=_target_values,
                value_descriptions=self._value_descriptions,
                input_data_type=type(input_values[0]),
                output_data_type=type(target_values[0]),
                )
            response = self._response_postprocess(_response)
            print('retry:', retry_count)
            print('function:', response.func_string)
        while not Evaluator.is_valid(
                response.callable, _input_values
                ):
            retry_count += 1
            random.shuffle(self._value_descriptions)
            _input_values, _target_values = self.randomize_values(input_values, target_values)
            _response = self.gen_ai(
                input_values=_input_values,
                target_values=_target_values,
                value_descriptions=self._value_descriptions,
                input_data_type=type(input_values[0]),
                output_data_type=type(target_values[0]),
                )
            response = self._response_postprocess(_response)
            print('retry:', retry_count)
            print('function:', response.func_string)
        return response
        
class PairConvertorGenerator(ConvertorGenerator):
    """
    Infer convertion function from inputs and outputs provided in value lists 
    """
    def get_code_gen_signature(self):
        return PairwiseConvertionCodeInferencer
    
    

    def randomize_values(self, input_values: List[str], target_values: List[str]) -> Tuple[List[str], List[str]]:
        """
        Randomize input values so that LLM can produce 
        different results.
        """
        value_pairs = list(zip(copy.copy(input_values), copy.copy(target_values)))
        _value_pairs = copy.copy(value_pairs * random.randint(0, 3))
        
        random.shuffle(_value_pairs)
        _value_pairs = random.sample(_value_pairs, random.randint(0, len(_value_pairs)))
        _input_values = [x[0] for x in _value_pairs]
        _target_values = [x[1] for x in _value_pairs]
        return _input_values, _target_values
    
class GroupConvertorGenerator(ConvertorGenerator):
    """
    Infer convertion function from inputs and outputs provided in value lists 
    """
    def get_code_gen_signature(self):
        return GroupwiseConvertionCodeInferencer
    
    def randomize_values(self, input_values: List[str], target_values: List[str]) -> Tuple[List[str], List[str]]:
        """
        Randomize input values so that LLM can produce 
        different results.
        """
        _input_values = copy.copy(input_values)
        _target_values = copy.copy(target_values)
        random.shuffle(_input_values)
        random.shuffle(_target_values)
        return _input_values, _target_values



"""
TODO:
- [ ] Add function failure problem and the failure function to another code generator 
    to refine the code. 
"""

class EvaluateDataGenerator:
    """
    Generate training data to evaluate the LLM-based
    Column selector.
    """
    def __init__(self, paths: List[str]):
        self._paths = paths

    def generate(self):
        for path in self._paths:
            columns = json.loads(open(f'training_data/{path}/columns.json', 'r').read())
            rows = json.loads(open(f'training_data/{path}/rows.json', 'r').read())
            for i, (col1, col2) in enumerate(zip(columns['ground_truth'], columns['input'])):
                target_values = [row['ground_truth'][i] for row in rows]
                input_values = [row['input'][i] for row in rows]
                yield {
                    'value_descriptions': [col1, col2],
                    'target_values': target_values,
                    'input_values': input_values
                }




if __name__ == '__main__':
    for i, instance in enumerate(EvaluateDataGenerator(['f00027']).generate()):
        # pprint.pprint(instance)
        generator = PairConvertorGenerator(instance['value_descriptions'])
        response = generator(instance['input_values'], instance['target_values'])
        assert Evaluator.is_valid(response.callable, instance['input_values'])
        print(i)
        print(response.func_string)
        if not Evaluator.is_fit(response.callable, 
            instance['input_values'], 
            instance['target_values']):
            print('Incorrect on', i, 'th inference')
            print('column description:', instance['value_descriptions'])
            pairwise_validity = Evaluator.check_pairwise_matching(
                response.callable, 
                instance['input_values'], 
                instance['target_values']
            )
            pprint.pprint(pairwise_validity)
            
        
        