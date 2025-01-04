import abc
from typing import Any, Dict



class WorkflowNode:
    """
    A branching node in the workflow
    """
    def __init__(self):
        self._next_nodes: Dict[str, 'WorkflowNode'] = dict()

    def attach_downstream(self, node_name: str, node: 'WorkflowNode'):
        self._next_nodes.update({node_name: node})

    @property
    def is_end(self) -> bool:
        """
        check whether the workflow is in ending state
        """
        return len(self._next_nodes) == 0
    
    @abc.abstractmethod
    def do_branch(self, payload: Any) -> str:
        """
        Define the downstream node 
        where the state should switch upon
        given the output. 
        """
        if not self.is_end:
            assert len(self._next_nodes) == 1
            return list(self._next_nodes.keys())[0]

    @abc.abstractmethod
    def process(self, payload: Any) -> Any:
        """
        The operation that convert input to output in the branching nodes
        """
        return payload

class WorkflowController:
    """
    Controlling the operation of the workflow.
    """
    def __init__(self, start_node: WorkflowNode):
        self._start_node = start_node
        self._current_node = start_node

    def start(self, payload):
        """
        Go from one step to another
        """
        self._current_node = self._start_node
        self._node_records = []
        self._input_payload_records = []
        self._output_payload_records = []
        while not self._current_node.is_end:
            self._node_records.append(self._current_node)
            self._input_payload_records.append(payload)
            payload = self._current_node.process(payload)
            self._output_payload_records.append(payload)
            downstreams = self._current_node._next_nodes
            name = self._current_node.do_branch(payload)
            self._current_node = downstreams[name]
            
        return payload
    
    @property
    def records(self):
        return list(map(lambda x: f'<{x[0].__class__.__name__}: {x[1]}=>{x[2]}>', zip(self._node_records, self._input_payload_records, self._output_payload_records)))

class EndNode(WorkflowNode):
    """
    Ending Node
    """
    
class IsValid(WorkflowNode):
    """
    Invalid nodes
    """
    def do_branch(self, payload):
        if payload == 0:
            return 'Y'
        else:
            return 'N'
    

class PlusOne(WorkflowNode):
    """
    Simulate the working of GenAI
    """
    def process(self, input) -> int:
        return input + 1
    
class TenReach(WorkflowNode):
    """
    Ten reach and stop
    """
    def do_branch(self, payload):
        if payload == 10:
            return 'Y'
        else:
            return 'N'




if __name__ == '__main__':
    end = EndNode()
    valid_1 = IsValid()
    plus_one = PlusOne()
    ten_validate = TenReach()
    valid_1.attach_downstream('Y', end)
    valid_1.attach_downstream('N', plus_one)
    ten_validate.attach_downstream('Y', end)
    ten_validate.attach_downstream('N', plus_one)
    plus_one.attach_downstream('next', ten_validate)
    c = WorkflowController(valid_1)
    ans = c.start(0)
    print(ans)
    print(c.records)
    ans = c.start(1)
    print(ans)
    print(c.records)