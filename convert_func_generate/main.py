from src.data_sampler import EvaluateDataGenerator, PairTrainTestDataSampler
from src.workflow_tool import WorkflowController, WorkflowNode
from src.dspy_agent import AdvanceConvertorGenerator, NumericConvertorGenerator, InvalidConvertorReviser
from src.evaluator import Evaluator
import pprint

MAX_ITERATION = 5

class IsNullConvertion(WorkflowNode):
    def determine_downstream(self, payload):
        if list(payload['input_values']) == list(payload['target_values']):
            return 'do_null_convertion'
        else:
            return 'continue'
        
        
class NullConvertorProducer(WorkflowNode):
    def process(self, payload):
        payload['convertor'] = {
            'callable': lambda x: x,
            'reasoning': 'The input values are exactly the same the the target values.',
            'func_string': 'func = lambda x: x'
        }
        payload['is_fit'] = Evaluator.is_fit(lambda x: x, payload['input_values'], payload['target_values'])
        return payload

class IsNumericConvertion(WorkflowNode):
    def determine_downstream(self, payload):
        try:
            list(map(float, payload['input_values']))
            list(map(float, payload['target_values']))
            return 'do_numeric_convertion'
        except ValueError:
            return 'continue'

class NumericConvertorProducer(WorkflowNode):    
    def process(self, payload):
        generator = NumericConvertorGenerator(payload['value_descriptions'])
        response = generator(
            input_values=payload['input_values'],
            target_values=payload['target_values']
        )
        payload['convertor'] = {
            'callable': response.callable,
            'reasoning': response.reasoning,
            'func_string': response.func_string
        }
        payload['is_fit'] = Evaluator.is_fit(response.callable, payload['input_values'], payload['target_values'])
        return payload

class PairwiseDataSampler(WorkflowNode):
    def process(self, payload):
        (
            train_input_values, train_target_values
        ), (
            test_input_values, test_target_values
        ) = PairTrainTestDataSampler().split(
            payload['input_values'],
            payload['target_values']
        )
        payload.update(
            {
                'train_input_values': train_input_values,
                'train_target_values': train_target_values,
                'test_input_values': test_input_values,
                'test_target_values': test_target_values,
            }
        )
        return payload

class PairConvertorInference(WorkflowNode):
    def determine_downstream(self, payload):
        if Evaluator.is_valid(payload['convertor']['callable'], payload['input_values']):
            return 'next'
        else:
            return 'again'
        
    def process(self, payload):
        generator = AdvanceConvertorGenerator()
        response = generator(payload['train_input_values'], payload['train_target_values'])
        payload.update(
            {
                'convertor': {
                    'callable': response.callable,
                    'reasoning': response.reasoning,
                    'func_string': response.func_string,
                }
            }
        )
        return payload
    
class InvalidConvertorRevise(WorkflowNode):
    def determine_downstream(self, payload):
        if Evaluator.is_valid(payload['convertor']['callable'], payload['input_values']):
            return 'next'
        else:
            return 'again'
        
    def process(self, payload):
        payload['train_target_values']
        reviser = InvalidConvertorReviser()
        response = reviser(
            payload['convertor']['func_string'],
            payload['convertor']['reasoning'],
            payload['input_values'],
            payload['target_values']
        )
        payload.update(
            {
                'convertor': {
                    'callable': response.callable,
                    'reasoning': response.reasoning,
                    'func_string': response.func_string,
                }
            }
        )
        return payload
    
class FitEvaluator(WorkflowNode):
    def determine_downstream(self, payload):
        if Evaluator.is_fit(payload['convertor']['callable'], payload['input_values'], payload['target_values']):
            return 'end'
        else:
            return 'again'
        
    def process(self, payload):
        payload.update(
            {
                'is_fit': Evaluator.is_fit(payload['convertor']['callable'], payload['input_values'], payload['target_values'])
            }
        )
        return payload
    
class RepeatCounter(WorkflowNode):
    def determine_downstream(self, payload):
        if payload['repeat_count'] > MAX_ITERATION:
            return 'end'
        elif payload['repeat_count'] >  2 and (payload['evaluation']['f1_score'] > payload['evaluation']['accuracy'] + 0.1):
            return 'sort_values'
        else:
            return 'feedback'
        
    def process(self, payload):
        print('[RepeatCounter:process] # repeat:', payload.get('repeat_count'), ''.join(['.'] * 50))
        print('func_string:', payload['convertor']['func_string'])
        print('reasoning:', payload['convertor']['reasoning'])
        print('inputs:', payload['input_values'])
        print('outputs:', list(map(payload['convertor']['callable'], payload['input_values'])))
        print('targets:', payload['target_values'])
        payload.update({
                'evaluation': {
                    'f1_score': Evaluator.f1_score(
                        payload['convertor']['callable'], 
                        payload['input_values'], 
                        payload['target_values']
                    ),
                    'accuracy': Evaluator.accuracy(
                        payload['convertor']['callable'], 
                        payload['input_values'], 
                        payload['target_values']
                    )
                }
            })
        pprint.pprint(payload['evaluation'])
        if 'repeat_count' in payload:
            payload['repeat_count'] += 1
        else:
            payload['repeat_count'] = 1
        return payload

class ValueSort(WorkflowNode):
    """
    Sorting Values for better inferencing on 
    Convertion
    """
    def process(self, payload):
        payload['input_values'] = sorted(payload['input_values'])
        payload['target_values'] = sorted(payload['target_values'])
        return payload


class CompareOutputAndGroundTruth(WorkflowNode):
    """
    Add the output for debugging
    """
    def process(self, payload):
        payload.update(
            {
                'evaluation': {
                    'f1_score': Evaluator.f1_score(
                        payload['convertor']['callable'], 
                        payload['input_values'], 
                        payload['target_values']
                    ),
                    'accuracy': Evaluator.accuracy(
                        payload['convertor']['callable'], 
                        payload['input_values'], 
                        payload['target_values']
                    )
                }
            }
        )
        return payload
        

# Define Operator Nodes
is_null_convertion = IsNullConvertion()
null_convertion_producer = NullConvertorProducer()
is_numeric_convertion = IsNumericConvertion()
numeric_convertion_producer = NumericConvertorProducer()
pairwise_data_generator = PairwiseDataSampler()
pairwise_data_generator_for_invalid_reviser = PairwiseDataSampler()
invalid_reviser = InvalidConvertorRevise()
pairwise_convertor_inferencer = PairConvertorInference()
fit_evaluator = FitEvaluator()
repeat_counter = RepeatCounter()
value_sortor = ValueSort()
final_debug = CompareOutputAndGroundTruth()

# Connecting the Operator Nodes
is_null_convertion.attach_downstream('do_null_convertion', null_convertion_producer)
null_convertion_producer.attach_downstream('end', final_debug)

is_null_convertion.attach_downstream('continue', is_numeric_convertion)
is_numeric_convertion.attach_downstream('continue', pairwise_data_generator)
is_numeric_convertion.attach_downstream('do_numeric_convertion', numeric_convertion_producer)
numeric_convertion_producer.attach_downstream('end', final_debug)

pairwise_data_generator.attach_downstream('next', pairwise_convertor_inferencer)
pairwise_convertor_inferencer.attach_downstream('next', fit_evaluator)
pairwise_convertor_inferencer.attach_downstream('again', pairwise_data_generator_for_invalid_reviser)
pairwise_data_generator_for_invalid_reviser.attach_downstream('next', invalid_reviser)
invalid_reviser.attach_downstream('next', fit_evaluator)
invalid_reviser.attach_downstream('again', pairwise_data_generator_for_invalid_reviser)

fit_evaluator.attach_downstream('end', final_debug)
fit_evaluator.attach_downstream('again', repeat_counter)
repeat_counter.attach_downstream('end', final_debug)
repeat_counter.attach_downstream('feedback', pairwise_data_generator)
repeat_counter.attach_downstream('sort_values', value_sortor)
value_sortor.attach_downstream('next', pairwise_data_generator)

controller = WorkflowController(is_null_convertion, verbose=False, verbose_callback=lambda x: (x.get('convertor'), x['input_values']))



if __name__ == '__main__':
    for i, instance in enumerate(EvaluateDataGenerator(['f00032q']).generate()):
        # instance['input_values'] = list(map(float, instance['input_values']))
        # print(set(instance['input_values']) == set(instance['target_values']))
        print('====================================================================')
        payload = controller.run(instance)
        if not payload['is_fit']:
            print('[main:failed]', i, '\n', payload['convertor']['func_string'])
            print('reasoning:', payload['convertor']['reasoning'])
            print('inputs:', payload['input_values'])
            print('outputs:', list(map(payload['convertor']['callable'], payload['input_values'])))
            print('targets:', payload['target_values'])
            print('evaluation:', payload['evaluation'])
        else:
            print('[main:success] no.', i, '\n', 
                  'accuracy:', payload['evaluation']['accuracy'], 
                  'f1_score:', payload['evaluation']['f1_score'], 
                  '\nfunc:\n', 
                  payload['convertor']['func_string'], [r.__class__ for r in payload['workflow_records']])