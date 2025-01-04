import dspy
from typing import List
import random
from collections import Counter
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import pprint
lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class SelectBackendColumn(dspy.Signature):
    """
    Given columns of a frontend table and a backend table and a target frontend column, 
    select the backend column corresponds to the target frontend column from 
    two candidate backend columns. 

    Note that the backend_columns and frontend_columns are matched to one another. 
    """
    frontend_columns: List[str] = dspy.InputField(desc='The list of columns is the frontend table')
    backend_columns: List[str] = dspy.InputField(desc='The list of columns is the backend table')
    candidate_backend_columns: str = dspy.InputField(desc='The two candidate backend columns')
    target_frontend_column: str = dspy.InputField(desc='The target frontend column')
    selected_backend_column: str = dspy.OutputField(desc='The backend column select from the two candidates. It is not wrapped with «»')
    confidence: float = dspy.OutputField()


select = dspy.ChainOfThought(SelectBackendColumn)


frontend_columns = [
        "部門別",
        "部門營業收入",
        "部門營收比例",
        "部門損益",
        "部門損益比例",
        "部門稅前純益率"
    ]

backend_columns = [
        "Name",
        "Revenue",
        "RevenueRate",
        "ProfitAndLoss",
        "ProfitAndLossRate",
        "PreTexNetProfitRate"
    ]

class EvaluateDataGenerator:
    """
    Generate training data to evaluate the LLM-based
    Column selector.
    """
    def __init__(self, frontend_columns: List[str], backend_columns: List[str]):
        self._frontend_columns = copy.copy(frontend_columns)
        self._backend_columns = copy.copy(backend_columns)
        self._paired_columns = list(zip(self._frontend_columns, self._backend_columns))

    def generate(self, n: int=2):
        while True:
            candidate_sample = random.sample(self._paired_columns, n)
            answer_sample = random.sample(candidate_sample, 1)
            yield {
                'candiate_backend_columns': [s[1] for s in candidate_sample],
                'target_frontend_column': answer_sample[0][0],
                'ground_truth': answer_sample[0][1]
            }


class LLMBasedColumnSelector:
    """
    Select the backend columns match to a target frontend columns
    from a list of candidates
    """
    def __init__(self, frontend_columns: List[str], backend_columns: List[str], verbose: bool=False):
        self._frontend_columns = copy.copy(frontend_columns)
        self._backend_columns = copy.copy(backend_columns)
        self._verbose = verbose

    def select_from_candidate(self, candiate_backend_columns: List[str], target_frontend_column: str, max_iteration: int=50):
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self._make_one_call, i, candiate_backend_columns, target_frontend_column) for i in range(max_iteration)]
        
        for future in as_completed(futures):
            results.append(future.result())

        result_summary = Counter(results).most_common()
        if self._verbose:
            print(result_summary)
        return result_summary[0][0]

    
    def _make_one_call(self, i: int, candiate_backend_columns: List[str], target_frontend_column: str):
        """
        Calling LLM one time
        """
        retry_cnt = 0
        while True:
            random.shuffle(self._frontend_columns)
            random.shuffle(self._backend_columns)
            random.shuffle(candiate_backend_columns)
            if self._verbose:
                print(f'[_make_one_call ({i})] context:', self._frontend_columns, self._backend_columns)
                print(f'[_make_one_call ({i})] candidates:', candiate_backend_columns)
                print(f'[_make_one_call ({i})] target:', target_frontend_column)
            response = select(
                frontend_columns=self._frontend_columns,
                backend_columns=self._backend_columns,
                candidate_backend_columns=candiate_backend_columns,
                target_frontend_column=target_frontend_column
                )
            answer = response.selected_backend_column
            answer = answer.replace('«', '').replace('»', '').strip()
            if answer not in candiate_backend_columns:
                retry_cnt += 1
                print(f'[Warning] retry {retry_cnt} answer {answer} not in candiate_backend_columns: {candiate_backend_columns}')
                continue
            if self._verbose:
                print(f'[_make_one_call ({i})] ANSWER:', response.selected_backend_column)
            return answer

if __name__ == '__main__':
    trainer = EvaluateDataGenerator(frontend_columns, backend_columns)
    matcher = LLMBasedColumnSelector(frontend_columns, backend_columns, verbose=False)
    for i, eval_instance in enumerate(trainer.generate(n=2)):
        assert eval_instance['ground_truth'] in eval_instance['candiate_backend_columns'], f"{eval_instance['target_frontend_column']} not in {eval_instance['candiate_backend_columns']}"
        pprint.pprint(eval_instance)
        predict = matcher.select_from_candidate(
            eval_instance['candiate_backend_columns'], eval_instance['target_frontend_column'], max_iteration=30)
        assert predict == eval_instance['ground_truth'], f"predict: {predict}; candiates: {eval_instance['candiate_backend_columns']}; target: {eval_instance['target_frontend_column']}; ground_truth: {eval_instance['ground_truth']}"
        print('i', i, 'success')