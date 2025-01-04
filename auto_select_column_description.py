import dspy
from typing import List
import random

lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class MapFrontend2BackendColumn(dspy.Signature):
    """
    There are two related frontend and backend tables where the data are consistent
     but the columns have different names and the data in the tables may have different representation. 
    The goal is to find the column name in the backend table that associated with 
    a particular column in the frontend table. 
    As guidence, all columns in the frontend and backend tables 
    In addition, a row of the frontend table and the associated row in the backend table are provided.

    According to the provided information, infer the index of the frontend column associated with the target backend column.
    """
    frontend_columns: List[str] = dspy.InputField(desc='Columns in the frontend table.')
    backend_columns: List[str] = dspy.InputField(desc='Columns in the backend table.')
    target_backend_column: str = dspy.InputField(desc='Target columns to map to a frontend column')
    one_frontend_row: List[str] = dspy.InputField(desc='Values in a row of frontend table. The ordering is the same as that of the frontend columns.')
    one_backend_row: List[str] = dspy.InputField(desc='Values in a row of backend table correspond to the provided frontend row. The ordering is the same as that of the backend columns.')
    associated_frontend_column_index: int = dspy.OutputField(desc='The index to the frontend column corresponds to the target backend column.')

def equal_checker(a: str, b: str):
    # simple tool that check whether a and b is the same.
    return a == b

def select_element_from_a_list(a_list: List[str], index: int):
    return a_list[index]

backend_data = list(zip([
    "Name",
    "Revenue",
    "RevenueRate",
    "ProfitAndLoss",
    "ProfitAndLossRate",
    "PreTaxNetProfitRate"], [
        "其他營運部門",
        "0",
        "0.0",
        "50999",
        "0.67",
        "0.0"
    ]))

target_column = 'Revenue'
random.shuffle(backend_data)
frontend_columns = ["部門別",
        "部門營業收入",
        "部門營收比例",
        "部門損益",
        "部門損益比例",
        "部門稅前純益率"]
backend_columns = [x[0] for x in backend_data]
one_backend_row = [x[1] for x in backend_data]
select = dspy.ReAct(MapFrontend2BackendColumn, tools=[equal_checker, select_element_from_a_list])
response = select(
    frontend_columns=frontend_columns,
    backend_columns=backend_columns,
    target_backend_column=target_column,
    one_frontend_row=["其他營運部門",
            "0",
            "0.00%",
            "50999",
            "0.67%",
            "0.00%"],
    one_backend_row=one_backend_row,
    )
print('response:', response)
select_col = frontend_columns[response.associated_frontend_column_index]
print(target_column, '->', select_col)
# lm.inspect_history(n=4)
