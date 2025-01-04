"""
Goal of this code space is to understand how to use 
`haystack` to build a streaming pipeline. 

REF:
https://dev.to/arya_minus/async-haystack-streaming-over-fastapi-endpoint-2kj0

https://docs.haystack.deepset.ai/docs/serialization

Steps:

pip3 install haystack-ai
"""
from haystack import Pipeline
from haystack import component

@component
class RepeatWordComponent:
    def __init__(self, times: int):
        self.times = times

    @component.output_types(result=str)
    def run(self, word: str):
        return {
            'ans': word * self.times}


pipe = Pipeline()
pipe.add_component(instance=RepeatWordComponent(3), name='convert')

ans = pipe.run({'word': 'hello'})
print(ans)