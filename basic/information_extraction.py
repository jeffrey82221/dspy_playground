import dspy
lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""

    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

print('Information Extraction DEMO:')
module = dspy.Predict(ExtractInfo)

text = "Apple Inc. announced its latest iPhone 14 today." \
    "The CEO, Tim Cook, highlighted its new features in a press release."
response = module(text=text)


print('response:', response)
print('title:', response.title)
print('heading:', response.headings)
print('entities:', response.entities)

print('Chain of Thought Version:')
cot_module = dspy.ChainOfThought(ExtractInfo)

text = "Apple Inc. announced its latest iPhone 14 today." \
    "The CEO, Tim Cook, highlighted its new features in a press release."
response = cot_module(text=text)
print('reasoning:', response.reasoning)
print('title:', response.title)
print('heading:', response.headings)
print('entities:', response.entities)