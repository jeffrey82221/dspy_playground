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
    ordered_field_names: List[str] = dspy.OutputField(desc='The ordered input field names that matched with the target field names sementically')
    confidence: float = dspy.OutputField()

classify = dspy.ChainOfThought(ReorderColumn)
response = classify(
    target_field_names=[
        "年季",
        "收盤價",
        "法人預估本益比",
        "本益比(近4季)",
        "本益比(季高)",
        "本益比(季低)",
        "EPS"
    ],
    target_field_values=[
        "2019Q4",
        "13.0",
        "12.7",
        "13.3",
        "13.0",
        "12.3",
        "0.27"
    ],
    input_field_names=[
        "SeasonDate",
        "ClosePr",
        "EPS",
        "PERSeasonHigh",
        "PERSeasonLow",
        "PER",
        "PERByTWSE"
    ],
    input_field_values=[
        "202403",
        "24.25",
        "0.49",
        "15.6",
        "13.0",
        "13.7",
        "13.7"
    ]
    )
print('reorder response:', response)