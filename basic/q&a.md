# What is dspy.Signature?

In DSPy, a Signature is a declarative specification that defines the input and output behavior of a module. It acts as a blueprint for tasks assigned to large language models (LLMs), focusing on what needs to be done rather than how it should be achieved. This approach eliminates the need for manually crafting detailed prompts, enabling developers to build modular, efficient, and reproducible LLM applications.

```python
class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""

    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

module = dspy.Predict(ExtractInfo)

text = "Apple Inc. announced its latest iPhone 14 today." \
    "The CEO, Tim Cook, highlighted its new features in a press release."
response = module(text=text)

print(response.title)
print(response.headings)
print(response.entities)
```

# What is dspy.ChainOfThought? 

The prediction results can have a reasoning explaination: 

Input: 

```python
math = dspy.ChainOfThought("question -> answer: float")
math(question="Two dice are tossed. What is the probability that the sum equals two?")
```

Output: 
```python
Prediction(
    reasoning='When two dice are tossed, each die has 6 faces, resulting in a total of 6 x 6 = 36 possible outcomes. The sum of the numbers on the two dice equals two only when both dice show a 1. This is just one specific outcome: (1, 1). Therefore, there is only 1 favorable outcome. The probability of the sum being two is the number of favorable outcomes divided by the total number of possible outcomes, which is 1/36.',
    answer=0.0277776
)
```

NOTE: 
###  dspy.Predict v.s. dspy.ChainOfThought

=> dspy.Predict has no reasoning in the response prediction. 

# What is dspy.ColBERTv2?

Use ColBERTv2 to load the extracts from the Wikipedia 2017 dataset. ColBERT is a fast and accurate retrieval model, enabling scalable BERT-based search over large text collections in tens of milliseconds. ColBERT is simply one of the many options that can be used to retrieve information from a vector database. 


```python
results: List[Dict] = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
```

# What is dspy.ReAct?

The `dspy.ReAct` module in DSPy is used to implement ReAct (Reasoning and Acting) agents, which combine reasoning capabilities with tool usage to solve tasks effectively. It is particularly useful for creating agents that can perform multi-step reasoning while interacting with external tools.