"""
For building up agent interaction workflow
"""
import abc
from typing import Dict
from typing import Any, Callable

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
    def determine_downstream(self, payload: Any) -> str:
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
    def __init__(self, start_node: WorkflowNode, verbose: bool=False, verbose_callback: Callable=lambda x: x):
        self._verbose = verbose
        self._start_node = start_node
        self._current_node = start_node
        self._verbose_callback = verbose_callback

    def run(self, payload):
        """
        Go from one step to another
        """
        self._current_node = self._start_node
        self._node_records = []
        self._input_payload_records = []
        self._output_payload_records = []
        while not self._current_node.is_end:
            self._operate_node(self._current_node, payload)
            downstreams = self._current_node._next_nodes
            name = self._current_node.determine_downstream(payload)
            self._current_node = downstreams[name]
        self._operate_node(self._current_node, payload)
        return payload
    
    def _operate_node(self, current_node: WorkflowNode, payload: Dict):
        if self._verbose:
            print('[_operate_node] Start', current_node)
        self._node_records.append(current_node)
        self._input_payload_records.append(payload)
        payload = current_node.process(payload)
        self._record_workflow_in_payload(payload)
        self._output_payload_records.append(payload)
        if self._verbose:
            print('[_operate_node] Show payload data:', self._verbose_callback(payload))
        if self._verbose:
            print('[_operate_node] End', current_node)

    
    def _record_workflow_in_payload(self, payload: Dict):
        if 'workflow_records' in payload:
            payload['workflow_records'].append(self._current_node)
        else:
            payload['workflow_records'] = [self._current_node]

    @property
    def records(self):
        return list(map(lambda x: f'<{x[0].__class__.__name__}: {x[1]}=>{x[2]}>', zip(self._node_records, self._input_payload_records, self._output_payload_records)))

class EndNode(WorkflowNode):
    """
    Ending Node
    """

