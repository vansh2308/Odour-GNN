import warnings
from math import sqrt
from typing import Any, Dict, Optional, Union

from torch.nn.parameter import Parameter
import torch
from torch import Tensor

from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import ExplainerConfig, ModelConfig


class GNNExplainer(ExplainerAlgorithm):
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None


    def forward( self, model: torch.nn.Module, x: Tensor, edge_index: Tensor, target: Tensor) -> Explanation:
        self._train(model, x, edge_index, target=target)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        self._clean_model(model)
        return Explanation(node_mask=node_mask, edge_mask=edge_mask)
    

    def supports(self) -> bool:
        return True


    def _train( self, model: torch.nn.Module, x: Tensor, edge_index: Tensor, target: Tensor):
        self._initialize_masks(x, edge_index)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            y_hat, y = model(h, edge_index), target

            loss = self._loss(y_hat, y)
            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        device = x.device
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = Parameter(torch.randn(E, device=device) * std)



    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        loss = self._loss_multiclass_classification(y_hat, y)
        
        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None
            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
            loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None






class Explainer:
    def __init__( self, model: torch.nn.Module, explanation_type: str, model_config: Union[ModelConfig, Dict[str, Any]]):

        explainer_config = ExplainerConfig(
            explanation_type=explanation_type,
            node_mask_type='object',
            edge_mask_type='object',
        )
        self.model = model
        self.algorithm = GNNExplainer()
        self.model_config = ModelConfig.cast(model_config)
        self.algorithm.connect(explainer_config, self.model_config)

    @torch.no_grad()
    def get_prediction(self, *args, **kwargs) -> Tensor:
        training = self.model.training
        self.model.eval()

        with torch.no_grad():
            out = self.model(*args, **kwargs)

        self.model.train(training)
        return out


    def __call__( self, x: Union[Tensor, Dict[NodeType, Tensor]], edge_index: Union[Tensor, Dict[EdgeType, Tensor]]) -> Union[Explanation]:
        
        prediction: Optional[Tensor] = None
        prediction = self.get_prediction(x, edge_index)
        target = self.get_target(prediction)

        training = self.model.training
        self.model.eval()

        explanation = self.algorithm( self.model, x, edge_index, target=target)
        self.model.train(training)

        # Add explainer objectives to the `Explanation` object:
        explanation._model_config = self.model_config
        explanation.prediction = prediction
        explanation.target = target

        # Add model inputs to the `Explanation` object:
        explanation.x = x
        explanation.edge_index = edge_index
        explanation.validate_masks()
        return explanation.threshold(None)


    def get_target(self, prediction: Tensor) -> Tensor:
        return prediction.argmax(dim=-1)
