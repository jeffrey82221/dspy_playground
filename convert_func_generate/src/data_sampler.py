import abc
from typing import List, Tuple, Dict
import json
import copy
import random
from scipy.stats import gmean
from .evaluator import Evaluator


__all__ = [
    'EvaluateDataGenerator',
    'PairTrainTestDataSampler',
    'GroupTrainTestDataSampler',
    'ReorderInputAndTarget'
]

class EvaluateDataGenerator:
    """
    Generate training data to evaluate the LLM-based
    Column selector.
    """
    def __init__(self, paths: List[str]):
        self._paths = paths

    def generate(self):
        for path in self._paths:
            columns = json.loads(open(f'../training_data/{path}/columns.json', 'r').read())
            rows = json.loads(open(f'../training_data/{path}/rows.json', 'r').read())
            for i, (col1, col2) in enumerate(zip(columns['ground_truth'], columns['input'])):
                target_values = [row['ground_truth'][i] for row in rows]
                input_values = [row['input'][i] for row in rows]
                yield {
                    'value_descriptions': [col1, col2],
                    'target_values': target_values,
                    'input_values': input_values
                }

class TrainTestDataSampler:
    """
    Sampling training and testing data 
    for LLM inferencing
    """
    @abc.abstractmethod
    def randomize_values(self, input_values: List[str], target_values: List[str]) -> Tuple[List[str], List[str]]:
        """
        Connect to the dspy.Module that generate 
        convertion function string
        """
        raise NotImplementedError
    
    def split(self, input_values: List[str], target_values: List[str]) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
        """
        Get training input/target values
        and testing input/target values
        """
        _input_values, _target_values = (
            copy.deepcopy(input_values), 
            copy.deepcopy(target_values)
        )
        _input_values, _target_values = self.randomize_values(
            _input_values, _target_values
        )
        train_input_values, test_input_values = self._split_train_test(_input_values)
        train_target_values, test_target_values = self._split_train_test(_target_values)
        return (train_input_values, train_target_values), (test_input_values, test_target_values)


    def _reorder_values(self, input_values: List[str], target_values: List[str]) -> Tuple[List[str], List[str]]:
        """
        Randomize input values so that LLM can produce 
        different results.
        """
        random_similarities = []
        _input_values, _target_values = copy.copy(input_values), copy.copy(target_values)
        for i in range(30):
            random.shuffle(_input_values)
            random.shuffle(_target_values)
            random_similarities.append(Evaluator.rate_similarity(_input_values, _target_values))
        baseline_similariy = gmean(random_similarities)
        if Evaluator.rate_similarity(sorted(input_values), sorted(target_values)) > baseline_similariy:
            return sorted(input_values), sorted(target_values)
        else:
            return input_values, target_values
        
    def _split_train_test(self, values: List[str]) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Split data into training and testing
        """
        split_position = len(values) // 2
        return values[:split_position], values[:split_position]



class PairTrainTestDataSampler(TrainTestDataSampler):
    """
    Sampling training and testing data 
    for pair-wise data LLM inferencing
    """
    def randomize_values(self, input_values: List[str], target_values: List[str]) -> Tuple[List[str], List[str]]:
        """
        Randomize input values so that LLM can produce 
        different results.
        """
        value_pairs = list(zip(copy.copy(input_values), copy.copy(target_values)))
        random.shuffle(value_pairs)
        _input_values = [x[0] for x in value_pairs]
        _target_values = [x[1] for x in value_pairs]
        return _input_values, _target_values
    
    

    

class GroupTrainTestDataSampler(TrainTestDataSampler):
    """
    Sampling training and testing data 
    for group-wise data LLM inferencing
    """
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