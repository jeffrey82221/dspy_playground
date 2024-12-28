import dspy
from typing import List
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

    According to the provided information, list the backend columns associated with each of the frontend columns.
    List in the same order as the list of the frontend columns.
    """
    frontend_columns: List[str] = dspy.InputField(desc='Columns in the frontend table.')
    backend_columns: List[str] = dspy.InputField(desc='Columns in the backend table.')
    one_frontend_row: List[str] = dspy.InputField(desc='Values in a row of frontend table. The ordering is the same as that of the frontend columns.')
    one_backend_row: List[str] = dspy.InputField(desc='Values in a row of backend table correspond to the provided frontend row. The ordering is the same as that of the backend columns.')
    associated_frontend_columns: List[str] = dspy.OutputField(desc='A list of frontend columns corresponds to the backend columns.')
    confidence: float = dspy.OutputField()

select = dspy.ChainOfThought(MapFrontend2BackendColumn)
response = select(
    frontend_columns=['年季', '毛利率', '營業利益率', 
                      '稅前純益率', '稅後純益率', '稅後股東權益報酬率', '稅後資產報酬率', 
                      '每股營業額', '公告每股淨值', '每股稅後盈餘'],
    backend_columns=['DateRange', 'GrossProfitMargin', 'OperatingProfitMargin', 
                     'EarningsBeforeTaxesRatio', 'NetProfitMargin', 'ROE', 'ROA', 
                     'RevenuePerShare', 'BookValuePerShare', 'EPS', 
                     'RevenueGrowthRate', 'OperatingProfitGrowthRate', 
                     'EarningsBeforeTaxesGrowthRate', 'NetProfitGrowthRatio', 
                     'ReceivablesTurnoverRatio', 'PayablesTurnoverRatio', 
                     'InventoryTurnoverRatio', 'FixedAssetsTurnoverRatio', 
                     'TotalAssetsTurnoverRatio', 'CurrentRatio', 
                     'LiquidityRatio', 'InterestCover', 'DebtRatio', 'GearingRatio', 
                     'TotalAssets', 'ShareholdersEquity', 'EquityMultiplier'],
    one_frontend_row=["2024Q3",
            "49.62",
            "23.24",
            "23.24",
            "19.57",
            "3.15",
            "0.21",
            "2.52",
            "16.06",
            "0.49"],
    one_backend_row=["202403",
            "49.62",
            "23.24",
            "23.24",
            "19.57",
            "3.15",
            "0.21",
            "2.52",
            "16.06",
            "0.49",
            "15.79",
            "14.07",
            "14.07",
            "11.6",
            "",
            "",
            "",
            "2.38",
            "0.01",
            "1092.74",
            "1092.74",
            "",
            "93.39",
            "1412.39",
            "3080324525",
            "203673042",
            "15.12386958407583464"],
    )
print('response:', response)

