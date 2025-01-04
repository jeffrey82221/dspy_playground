import dspy
from typing import List
lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class ReorderColumn(dspy.Signature):
    """
    Re-order the input field names to match the target field names.
    The target field names and the input field names
    have different ways of representation (e.g., different language)
    but having same sementic meaning.
    Additionally,
    value instances of both target fields and input fields (input_field_values and target_field_values, respectively) are provided to help the understanding 
    of the field names.
    Also, the value instances listed are in the same ordering to their associeted list of field names. 

    When re-ordering the input field names, use both the value instances 
    and the meaning of field names as guidence. 
    """
    target_field_names: List[str] = dspy.InputField(desc='The target field names in Chinese')
    target_field_values: List[str] = dspy.InputField(desc='The values associated with the target field names to help understanding the target fields. They are in the same ordering to the target fields.')
    input_field_names: List[str] = dspy.InputField(desc='The input field names in English')
    input_field_values: List[str] = dspy.InputField(desc='The values associated with the input field names to help understanding the target fields. They are in the same ordering to the input fields.')
    ordering_indices: List[int] = dspy.OutputField(desc='The indices to reorder input_field_names to match it to target_field_names sementically. Its number of int should be the same as the number of string of input_field_names.')
    confidence: float = dspy.OutputField()

def reorder_columns(
    target_field_names: List[str], 
    target_field_values: List[str], 
    input_field_names: List[str],
    input_field_values: List[str]
    ):
    assert len(target_field_names) == len(target_field_values)
    assert len(input_field_names) == len(input_field_values)
    assert len(input_field_names) == len(target_field_names)
    order = dspy.ChainOfThought(ReorderColumn)
    response = order(
        target_field_names=target_field_names,
        target_field_values=target_field_values,
        input_field_names=input_field_names,
        input_field_values=input_field_values
        )
    assert len(response.ordering_indices) == len(input_field_names)
    return response.ordering_indices

class MapFrontend2BackendColumn(dspy.Signature):
    """
    There are two related frontend and backend tables where the data are consistent
     but the columns have different names and the data in the tables may have different representation. 
    The goal is to find the column name in the backend table that associated with 
    a particular column in the frontend table. 
    As guidence, all columns in the frontend and backend tables 
    In addition, a row of the frontend table and the associated row in the backend table are provided.

    According to the provided information, list the backend columns associated with each of the frontend columns.
    List in the same order as the list of the frontend columns.
    """
    frontend_columns: List[str] = dspy.InputField(desc='Columns in the frontend table.')
    backend_columns: List[str] = dspy.InputField(desc='Columns in the backend table.')
    one_frontend_row: List[str] = dspy.InputField(desc='Values in a row of frontend table. The ordering is the same as that of the frontend columns.')
    one_backend_row: List[str] = dspy.InputField(desc='Values in a row of backend table correspond to the provided frontend row. The ordering is the same as that of the backend columns.')
    backend_columns_associated_with_frontend_columns: List[str] = dspy.OutputField(desc='A list of backend columns corresponds to the frontend columns.')
    confidence: float = dspy.OutputField()

def reorder_columns_v2(
    frontend_columns,
    backend_columns,
    one_frontend_row,
    one_backend_row
):
    assert len(frontend_columns) == len(one_frontend_row)
    assert len(backend_columns) == len(one_backend_row)
    select = dspy.ChainOfThought(MapFrontend2BackendColumn)
    response = select(
        frontend_columns=frontend_columns,
        backend_columns=backend_columns,
        one_frontend_row=one_frontend_row,
        one_backend_row=one_backend_row,
        )
    assert len(response.backend_columns_associated_with_frontend_columns) == len(frontend_columns), f'length of backend result: {len(response.backend_columns_associated_with_frontend_columns)} != {len(frontend_columns)}'
    if len(backend_columns) == len(frontend_columns):
        assert set(response.backend_columns_associated_with_frontend_columns) == set(backend_columns)
    return response.backend_columns_associated_with_frontend_columns

if __name__ == '__main__':
    ordered_field_names = reorder_columns_v2(
        frontend_columns=[
            "年月",
        "單月營收",
        "單月月增率",
        "單月年增率",
        "累計營收",
        "累計年增率",
        "盈餘",
        "每股盈餘（元）"
        ],
        backend_columns=[
            "Date",
        "ClosePr",
        "MonthlyRevenue",
        "MonthlyRevenueMonthlyChange",
        "MonthlyRevenueYearGrowth",
        "AccumulatedRevenue",
        "AccumulatedRevenueGrowth",
        "MonthlyConsolidatedRevenue",
        "MonthlyConsolidatedRevenueMonthlyChange",
        "MonthlyConsolidatedRevenueYearGrowth",
        "AccumulatedConsolidatedRevenue",
        "AccumulatedConsolidatedRevenueGrowth",
        "AccumulatedSurplus",
        "AccumulatedEPS",
        "MonthlyRevenueM",
        "MonthlyConsolidatedRevenueM",
        "AccumulatedRevenueM",
        "AccumulatedConsolidatedRevenueM"
        ],
        one_frontend_row=[
            "202411",
            "4444.2",
            "-13.1",
            "-6.9",
            "59602.38",
            "20.2",
            "--",
            "--"
        ],
        one_backend_row=[
            "202411",
            "23.25",
            "4444233",
            "-13.06",
            "-6.93",
            "59602379",
            "20.19",
            "4444233",
            "-13.06",
            "-6.93",
            "59602379",
            "20.19",
            "",
            "",
            "4444.23",
            "4444.23",
            "59602.38",
            "59602.38"
        ]
        )
    print('reorder response:', ordered_field_names)
