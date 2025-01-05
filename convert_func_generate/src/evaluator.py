import traceback
from typing import Callable, List
from jarowinkler import jarowinkler_similarity
from scipy.stats import gmean



class Evaluator:
    """
    Evaluate and identify values that does not match
    """
    @staticmethod
    def is_valid(func: Callable, input_values: List[str]):
        return len(Evaluator.check_function_validity(func, input_values)) == 0
    
    @staticmethod
    def is_fit(func: Callable, input_values: List[str], target_values: List[str]):
        return Evaluator.f1_score(func, input_values, target_values) == 1.0
    
    @staticmethod
    def f1_score(func: Callable, input_values: List[str], target_values: List[str]):
        group_match_info = Evaluator.check_groupwise_matching(func, input_values, target_values)
        hit = len(target_values) - len(group_match_info['unexpected_outputs'])
        return hit / len(target_values)
    
    @staticmethod
    def accuracy(func: Callable, input_values: List[str], target_values: List[str]):
        match_info = Evaluator.check_pairwise_matching(func, input_values, target_values)
        return sum([row['correct'] for row in match_info]) / len(match_info)

    @staticmethod
    def rate_similarity(values1: List[str], values2: List[str]):
        """
        Compare the match-ability between two value list
        """
        sims = []
        for value1, value2 in zip(values1, values2):
            sims.append(jarowinkler_similarity(str(value1), str(value2)))
        target_sim = gmean(sims)
        return target_sim

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