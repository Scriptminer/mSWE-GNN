# Libraries
import torch
import torch.nn as nn
from models.models import BaseFloodModel, make_mlp, activation_functions
from torch_geometric.nn import ChebConv, TAGConv, GATConv
from torch import Tensor
from torch_geometric.utils import scatter
from torch.linalg import vector_norm
from typing import Optional

from utils.dataset import create_scale_mask

import numpy as np
import datetime

class GNN(BaseFloodModel):
    '''
    GNN encoder-processor-decoder
    ------
    num_node_features: int, number of features per node
    num_edge_features: int, number of features per edge
    hid_features: int, number of features per node (and edge) in the GNN layers
    K: int, K-hop neighbourhood
    n_GNN_layers: int, number of GNN layers
    dropout: float, add dropout layer in decoder
    type_GNN: str (default='SWEGNN'), specifies the type of GNN model
        options: 
            "GNN_A" : Adjacency as graph shift operator 
            "GNN_L" : Laplacian as graph shift operator
            "GAT"   : Graph Attention, i.e., learned shift operator
            "SWEGNN": learned graph shift operator
    edge_mlp: bool, adds MLP as edge encoder (valid only for 'SWEGNN')
    mlp_layers: int (default=2), number of MLP layers in the GNN processor
    mlp_activation: str (default='prelu'), activation function for the MLP layers
    gnn_activation: str (default='tanh'), activation function for the GNN layers
    with_WL: bool (default=False), adds water level as static input
    normalize: bool (default=True), normalize learned fluxes in SWE-GNN
    with_filter_matrix: bool (default=True), adds filter matrix to the GNN processor (i.e., adds the H in the graph convolution S*X*H)
    with_gradient: bool (default=True), adds the gradient of the water variables in the GNN processor
    base_model_kwargs: dict, additional arguments for the BaseFloodModel, e.g., learned_residuals, seed, residuals_base, etc.
    '''
    def __init__(self, num_node_features, num_edge_features, hid_features=32, K=2, n_GNN_layers=2, type_GNN="SWEGNN", 
                 mlp_layers=1, mlp_activation='prelu', gnn_activation='prelu', dropout=0, 
                 with_WL=True, normalize=True, with_filter_matrix=True, edge_mlp=True,
                 with_gradient=True, **base_model_kwargs):
        super(GNN, self).__init__(**base_model_kwargs)
        self.type_model = "GNN"
        self.hid_features = hid_features
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.type_GNN = type_GNN
        self.edge_mlp = edge_mlp
        self.with_WL = with_WL
        self.gnn_activation = gnn_activation
        self.dynamic_node_features = self.previous_t*self.out_dim
        self.static_node_features = num_node_features - self.dynamic_node_features + self.with_WL
        
        # Edge encoder
        if type_GNN == "SWEGNN" and edge_mlp:
            self.num_edge_features = hid_features
            self.edge_encoder = make_mlp(num_edge_features, hid_features, hid_features, n_layers=mlp_layers, bias=True,
                                         activation=mlp_activation, device=self.device)
        
        # Node encoder
        if type_GNN == "SWEGNN":
            self.dynamic_node_encoder = make_mlp(self.dynamic_node_features, hid_features, hid_features, n_layers=mlp_layers,
                                        activation=mlp_activation, device=self.device)
    
            self.static_node_encoder = make_mlp(
                self.static_node_features, hid_features, hid_features, n_layers=2, bias=True,
                activation=mlp_activation, device=self.device)
        else:
            self.node_encoder = make_mlp(num_node_features + self.with_WL, hid_features, hid_features, n_layers=mlp_layers, bias=True,
                                        activation=mlp_activation, device=self.device)
        
        # GNN
        self.gnn_processor = self._make_gnn(hid_features, K_hops=K, n_GNN_layers=n_GNN_layers, n_layers=mlp_layers, 
                                            activation=mlp_activation, bias=True, type_GNN=type_GNN, 
                                            normalize=normalize, with_filter_matrix=with_filter_matrix,
                                            with_gradient=with_gradient)

        self.gnn_activation = activation_functions(gnn_activation, device=self.device)
        
        # Decoder
        self.node_decoder = make_mlp(hid_features, self.out_dim, hid_features, n_layers=mlp_layers, dropout=dropout,
                                     activation=mlp_activation, device=self.device)

    def _make_gnn(self, hidden_size, K_hops=1, n_GNN_layers=1, type_GNN='SWEGNN', **swegnn_kwargs):
        """Builds GNN module"""
        convs = nn.ModuleList()
        for l in range(n_GNN_layers):
            if type_GNN == "GNN_L":
                convs.append(ChebConv(hidden_size, hidden_size, K=K_hops))
            elif type_GNN == "GNN_A":
                convs.append(TAGConv(hidden_size, hidden_size, K=K_hops))
            elif type_GNN == "GAT":
                convs.append(GATConv(hidden_size, hidden_size, heads=1))
            elif type_GNN == "SWEGNN":
                convs.append(SWEGNN(hidden_size, hidden_size, self.num_edge_features, K=K_hops, 
                            device=self.device, **swegnn_kwargs))
            else:
                raise("Only 'GNN_A', 'GNN_L', 'GAT', and 'SWEGNN' are valid for now")
        return convs
    
    def forward(self, graph):
        """Build encoder-decoder block"""    
        x = graph.x.clone()
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        # 1. Node and edge encoder
        if self.type_GNN == "SWEGNN" and self.edge_mlp:
            edge_attr = self.edge_encoder(edge_attr)
        
        x0 = x
        x_s = x[:,:self.static_node_features-self.with_WL]
        x_d = x[:,self.static_node_features-self.with_WL:]

        if self.with_WL:
            # Add water level as static input
            WL = x_s[:,-1] + x_d[:,-self.out_dim]
            x_s = torch.cat((x_s, WL.unsqueeze(-1)), 1)
        
        if self.type_GNN == "SWEGNN":
            x_s = self.static_node_encoder(x_s)
            x = x_d = self.dynamic_node_encoder(x_d)
        else:
            x = self.node_encoder(torch.cat((x_s, x_d), 1))

        # 2. Processor 
        for i, conv in enumerate(self.gnn_processor):
            if self.type_GNN == "SWEGNN":
                x = conv(x_s, x_d, edge_index, edge_attr)
            else:
                x = conv(x=x, edge_index=edge_index)

            # Add non-linearity
            if self.gnn_activation is not None:
                x = self.gnn_activation(x)

            x_d = x

        # 3. Decoder
        x = self.node_decoder(x)
                    
        # Add residual connections
        x = x + self._add_residual_connection(x0)
        
        # ReLU because of negative water depth or discharge
        x = torch.relu(x)

        # Mask very small water depth
        x = self._mask_small_WD(x, epsilon=0.0001)

        return x
    
class MSGNN(BaseFloodModel):
    '''
    Multi-Scale GNN encoder-processor-decoder
    Each scale is processed by a separate GNN and separate scales are connected by intra-scale edges

    ------
    node_features: int, number of features per node
    num_edge_features: int, number of features per edge
    num_scales: int, number of multi-scale graphs

    hid_features: int (default=32), number of features per node (and edge) in the GNN layers
    K: int or list, K-hop neighbourhood per scale (if int, same number of layers for all scales)
    n_GNN_layers: int or list, number of GNN layers per scale (if int, same number of layers for all scales)
    edge_mlp: bool (default=True), adds MLP as edge encoder (valid only for 'SWEGNN')
    skip_connections: bool (default=True), adds skip connections across scales
    learned_pooling: bool (default=False), adds learnable pooling

    mlp_layers: int (default=2), number of MLP layers in the GNN processor
    mlp_activation: str (default='prelu'), activation function for the MLP layers
    gnn_activation: str (default='tanh'), activation function for the GNN layers
    with_WL: bool (default=False), adds water level as static input
    normalize: bool (default=True), normalize learned fluxes in SWE-GNN
    with_filter_matrix: bool (default=True), adds filter matrix to the GNN processor (i.e., adds the H in the graph convolution S*X*H)
    with_gradient: bool (default=True), adds the gradient of the water variables in the GNN processor

    base_model_kwargs: dict, additional arguments for the BaseFloodModel, e.g., learned_residuals, seed, residuals_base, etc.
    '''
    def __init__(self, num_node_features, num_edge_features, num_scales, hid_features=32, K=2, 
                 mlp_layers=2, mlp_activation='prelu', gnn_activation='tanh', 
                 learned_pooling=False, skip_connections=True,
                 with_WL=False, normalize=True, with_filter_matrix=True, edge_mlp=True,
                 with_gradient=True, **base_model_kwargs):
        super(MSGNN, self).__init__(**base_model_kwargs)
        self.type_model = "MSGNN"
        self.hid_features = hid_features
        self.num_node_features = num_node_features
        self.edge_mlp = edge_mlp
        self.with_WL = with_WL
        self.num_scales = num_scales        
        self.gnn_activation = gnn_activation
        self.dynamic_node_features = self.previous_t*self.NUM_WATER_VARS
        self.static_node_features = num_node_features - self.dynamic_node_features + self.with_WL
        self.learned_pooling = learned_pooling
        self.skip_connections = skip_connections
        self.K = [K]*num_scales if isinstance(K, int) else K
        self.K = self.K + self.K[::-1][1:] # add reverse K_hops for the coarse to fine
        assert len(self.K) == num_scales*2-1, "K must be an int or a list of length num_scales or num_scales*2-1"
        
        # Edge encoder
        if edge_mlp:
            self.edge_encoder = make_mlp(num_edge_features, hid_features, hid_features, n_layers=mlp_layers, bias=True,
                                         activation=mlp_activation, device=self.device)
            num_edge_features = hid_features
        
        # Node encoders
        self.dynamic_node_encoder = make_mlp(self.dynamic_node_features, hid_features, hid_features, n_layers=mlp_layers,
                                    activation=mlp_activation, device=self.device)

        self.static_node_encoder = make_mlp(self.static_node_features, hid_features, hid_features, 
                                            n_layers=mlp_layers, bias=True, activation=mlp_activation, device=self.device)
                
        # Intra-scale GNN
        self.intra_scale_gnn = nn.ModuleList([
            SWEGNN(hid_features, hid_features, 0, K=1, n_layers=mlp_layers, 
                    activation=mlp_activation, bias=True, normalize=True, 
                    with_filter_matrix=False, with_gradient=False,
                    device=self.device) for _ in range(num_scales-1)])

        # learnable pooling
        if learned_pooling:
            self.pooling_mlp = make_mlp(hid_features*2, hid_features, hid_features, 
                                        n_layers=mlp_layers, activation=mlp_activation, device=self.device)
        
        # Processor  
        # GNN (1 per each scale)
        self.gnn_processor = nn.ModuleList([
            SWEGNN(hid_features, hid_features, num_edge_features, K=K, 
                    n_layers=mlp_layers, activation=mlp_activation, bias=True, 
                    normalize=normalize, with_filter_matrix=with_filter_matrix, 
                    with_gradient=with_gradient)
                    for K in self.K])

        self.gnn_activation = activation_functions(gnn_activation, device=self.device)
        
        # Decoder
        self.node_decoder = make_mlp(hid_features, self.out_dim, hid_features, n_layers=mlp_layers, dropout=0,
                                     activation=mlp_activation, device=self.device)
        
    def _pooling(self, x, row_fine, col_coarse, reduce='mean', learnable=False):
        """Pool multiscale attributes from finest to coarsest scale
        
        Args:
            x (Tensor): node features
            row_fine (Tensor): row indices of the finest scale
            col_coarse (Tensor): column indices of the coarsest scale
            reduce (str, optional): reduce operation. Defaults to 'mean'.
            learnable (bool, optional): learnable pooling. Defaults to False.
        """
        if learnable:
            e_ij = self.pooling_mlp(torch.cat((x[row_fine], x[col_coarse]), -1))
            x = scatter(src=e_ij, index=col_coarse, dim=0, dim_size=x.shape[0], reduce=reduce)
        else:
            x = scatter(src=x[row_fine], index=col_coarse, dim=0, dim_size=x.shape[0], reduce=reduce)
        return x
    
    def _create_scale_mask(self, data):
        """Creates a mask of shape N with entry i for each scale i
        
        mask = e.g., [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, ...]
        """
        num_nodes = data.x.size(0)
        return create_scale_mask(num_nodes, self.num_scales, data.node_ptr, data, device=self.device)
    
    def forward(self, graph):
        """Multiscale encoder-processor-decoder"""
        x = graph.x.clone()
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        edge_ptr = graph.edge_ptr
        intra_mesh_edge_index = graph.intra_mesh_edge_index
        intra_edge_ptr = graph.intra_edge_ptr
        
        # Create scale mask
        mask = self._create_scale_mask(graph)
        
        # 1. Node and edge encoder
        if self.edge_mlp:
            edge_attr = self.edge_encoder(edge_attr)
        
        x0 = x
        x_s = x[:,:self.static_node_features-self.with_WL]
        x_d = x[:,self.static_node_features-self.with_WL:]

        if self.with_WL:
            # Add water level as static input
            WL = x_s[:,-1] + x_d[:,-self.out_dim]
            x_s = torch.cat((x_s, WL.unsqueeze(-1)), 1)
        
        x_s = self.static_node_encoder(x_s)
        x_d = self.dynamic_node_encoder(x_d)

        # Create temporary variables to save the features after down and upgoing GNNs
        x_down = torch.zeros_like(x_d, device=self.device)
        x_up = torch.zeros_like(x_d, device=self.device)

        # 2. Processor
        # fine to coarse but skipping the coarsest scale (which is processed in the next loop)
        for i in range(self.num_scales-1):
            # downgoing GNN pass
            print(f"++++ Fine to Coarse {i}") #x_d = self.gnn_processor[i](x_s, x_d, edge_index[:,edge_ptr[i]:edge_ptr[i+1]], edge_attr[edge_ptr[i]:edge_ptr[i+1]])
            x_d = torch.utils.checkpoint.checkpoint(self.gnn_processor[i], x_s, x_d, edge_index[:,edge_ptr[i]:edge_ptr[i+1]], edge_attr[edge_ptr[i]:edge_ptr[i+1]], use_reentrant=False, debug=False, context_fn=pass_tracker_context_fn)

            # keep in memory the last operation before pooling (which would be overwritten otherwise)
            x_down = x_down + x_d * (mask==i)[:,None]
                
            # Pool multiscale attributes from finest to coarsest scale
            col_coarse, row_fine = intra_mesh_edge_index[:,intra_edge_ptr[i]:intra_edge_ptr[i+1]]
            x_d = self._pooling(x_d, row_fine, col_coarse, reduce='mean', learnable=self.learned_pooling)

        x_down = x_down + x_d

        # coarse to fine
        for i in range(self.num_scales):
            gnn_id = self.num_scales-1+i
            # upgoing GNN pass
            print(f"++++ Coarse to Fine {gnn_id}") #x_d = self.gnn_processor[gnn_id](x_s, x_d, edge_index[:,edge_ptr[-i-2]:edge_ptr[-i-1]], edge_attr[edge_ptr[-i-2]:edge_ptr[-i-1]])
            x_d = torch.utils.checkpoint.checkpoint(self.gnn_processor[gnn_id], x_s, x_d, edge_index[:,edge_ptr[-i-2]:edge_ptr[-i-1]], edge_attr[edge_ptr[-i-2]:edge_ptr[-i-1]], use_reentrant=False, debug=False, context_fn=pass_tracker_context_fn)

            # save GNN output before pooling
            x_up = x_up + x_d * (mask==self.num_scales-i-1)[:,None]

            # Un-pool multiscale attributes from coarsest to finest scale
            if i < self.num_scales-1:
                intra_scale_edges = intra_mesh_edge_index[:,intra_edge_ptr[-i-2]:intra_edge_ptr[-i-1]]
                x_d = self.intra_scale_gnn[i](x_s, x_d, intra_scale_edges)

                # add skip connection of saved finer scale attributes
                if self.skip_connections:
                    x_d = x_d + x_down * (mask==self.num_scales-i-2)[:,None]
        x = x_up

        # Add non-linearity
        if self.gnn_activation is not None:
            x = self.gnn_activation(x)

        # 3. Decoder
        x = self.node_decoder(x)
                    
        # Add residual connections
        x = x + self._add_residual_connection(x0)
        
        # ReLU because of negative water depth or discharge
        x = torch.relu(x)
        
        # Mask very small water depth
        x = self._mask_small_WD(x, epsilon=0.0001)
        print(f"CUDA mem usage (MiB): ", torch.cuda.memory_allocated("cuda:0")/1024/1024, flush=True)

        return x

class SWEGNN(nn.Module):
    r"""Shallow Water Equations inspired Graph Neural Network

    .. math::
        \mathbf{x}^{\prime}_{di} = \mathbf{x}_{di} + \sum_{j \in \mathcal{N}(i)} 
        \mathbf{s}_{ij} \odot (\mathbf{x}_{dj} - \mathbf{x}_{di})

        \mathbf{s}_{ij} = MLP \left(\mathbf{x}_{si}, \mathbf{x}_{sj},
        \mathbf{x}_{di}, \mathbf{x}_{dj},
        \mathbf{e}_{ij}\right)
    """
    def __init__(self, static_node_features: int, dynamic_node_features: int, edge_features: int, 
                 K: int = 2, normalize=True, with_filter_matrix=True, with_gradient=True,
                 upwind_mode=False, device='cpu', **mlp_kwargs):
        super().__init__()
        self.edge_features = edge_features
        self.edge_input_size = edge_features + static_node_features*2 + dynamic_node_features*2
        self.edge_output_size = dynamic_node_features
        hidden_size = self.edge_output_size*2
        self.normalize = normalize
        self.K = K
        self.with_filter_matrix = with_filter_matrix
        self.device = device
        self.with_gradient = with_gradient
        self.upwind_mode = upwind_mode
        
        self.edge_mlp = make_mlp(self.edge_input_size, self.edge_output_size,
                                hidden_size=hidden_size, device=device, **mlp_kwargs)

        if with_filter_matrix:
            self.filter_matrix = torch.nn.ModuleList([
                nn.Linear(dynamic_node_features, dynamic_node_features, bias=False, device=device) for _ in range(K+1)
            ])



    def forward(self, 
                x_s: Tensor, 
                x_d: Tensor, 
                edge_index: Tensor, 
                edge_attr: Optional[Tensor]=None) -> Tensor:
        '''
        x_s: static node features
        x_d: dynamic node features
        edge_index: edge indices
        edge_attr: edge features
        '''
        nn = lambda t: t.flatten()[~t.flatten().isnan()]
        display = False
        printbuffer = []
        def stats(tensor, txt):
            if tensor is None or tensor.numel() == 0 or nn(tensor).numel() == 0:
                display = True
                printbuffer.append(f"[recompute={PassTrackerMode.status[0]}] {txt}: None")
            elif tensor.numel() == 0:
                display = True
                printbuffer.append(f"[recompute={PassTrackerMode.status[0]}] {txt}: Empty Tensor")
            else:
                if tensor.isnan().any():
                    display = True
                printbuffer.append(f"[recompute={PassTrackerMode.status[0]}] {txt}: min absmin max: {nn(tensor).min().item()} ({nn(tensor).abs().min().item()}) {nn(tensor).max().item()} {f'{tensor.isnan().sum()} NaNs present' if tensor.isnan().sum() > 0 else ''}")
        stats(x_s,"x_s")
        stats(x_d,"x_d")
        stats(edge_index,"edge_index")
        stats(edge_attr,"edge_attr")
        #print(f"[recompute={PassTrackerMode.status[0]}] {hash(x_s)} {hash(x_d)} {hash(edge_index)} {hash(edge_attr)}")
        #print(f"[recompute={PassTrackerMode.status[0]}] Define row")
        row = edge_index[0]
        #print(f"[recompute={PassTrackerMode.status[0]}] Define col")
        col = edge_index[1]
        #print(f"[recompute={PassTrackerMode.status[0]}] Define num_nodes")
        num_nodes = x_d.size(0)
        if self.with_filter_matrix:
            #print(f"[recompute={PassTrackerMode.status[0]}] Clone op")
            #print(x_d)
            #print(self.filter_matrix[0])
            x_d_cloned = x_d.clone()
            #print(f"[recompute={PassTrackerMode.status[0]}] Filter matrix forward op")
            out = self.filter_matrix[0].forward(x_d_cloned) #x_d.clone())
            #print(f"[recompute={PassTrackerMode.status[0]}] Moving on to for loop")
        else:
            out = x_d.clone()
        stats(out,"out (pre for loop")
        
        for k in range(self.K):
            # Filter out zero values
            mask = out.sum(1) != 0
            mask_row = mask[row]
            mask_col = mask[col]
            edge_index_mask = mask_row + mask_col
            #print(f"[recompute={PassTrackerMode.status[0]}] self.edge_features={self.edge_features}, self.normalize={self.normalize}, self.with_gradient={self.with_gradient}, self.upwind_mode={self.upwind_mode}, self.with_filter_matrix={self.with_filter_matrix}, edge_index_mask (shape={edge_index_mask.shape}) at {datetime.datetime.now()}, k={k}, sum={edge_index_mask.sum()}, out.shape={out.shape}, x_s[row].shape={x_s[row].shape}, row.shape={row.shape} ({row.min()}-{row.max()}), col.shape={col.shape} ({col.min()}-{col.max()}), mask.shape={mask.shape}, out.sum(1): 0count={(out.sum(1)==0).sum()}, abs<0.00001count={(out.sum(1).abs()<0.00001).sum()}, abs<eps32({torch.finfo(torch.float32).eps})count={(out.sum(1).abs()<torch.finfo(torch.float32).eps).sum()}") #, abs<0.00001hist={np.histogram(out.sum(1)[out.sum(1).abs()<0.00001].detach().cpu().numpy(), bins=100)}") 
            # Edge update
            e_ij = torch.cat([x_s[row][edge_index_mask], 
                                x_s[col][edge_index_mask], 
                                x_d[row][edge_index_mask], 
                                x_d[col][edge_index_mask]], 1)
            #print(f"[recompute={PassTrackerMode.status[0]}] Test edge_features>0") 
            if self.edge_features > 0:
                e_ij = torch.cat([e_ij, edge_attr[edge_index_mask]], 1)
            #print(f"[recompute={PassTrackerMode.status[0]}] Enter self.edge_mlp: e_ij.shape={e_ij.shape}, edge_attr[edge_index_mask].shape={edge_attr[edge_index_mask].shape if self.edge_features > 0 else 'N/A'}, e_ij.hash_tensor()={e_ij.hash_tensor()}, edge_index_mash.hash_tensor()={edge_index_mask.hash_tensor()}")
            stats(e_ij, f"e_ij (k={k})") 
            for layer in self.edge_mlp:
                stats(layer.weight, str(layer))
            s_ij = self.edge_mlp(e_ij)
            #print(f"[recompute={PassTrackerMode.status[0]}] MLP done!")
            stats(s_ij, "s_ij (initial)")

            if self.normalize:
                if (s_ij==-torch.inf).sum() > 0 or (s_ij==-torch.inf).sum() > 0:
                    print("Screening out +-inf's in s_ij!")
                s_ij.masked_fill_(s_ij==-torch.inf, 0) # Preemptively mask values that will cause NaNs. TODO: remove, this was for testing only.
                s_ij.masked_fill_(s_ij==torch.inf, 0) # Preemptively mask values that will cause NaNs. TODO: remove, this was for testing only.
                s_ij.masked_fill_(s_ij.isnan(), 0) # TODO: remove, this was for testing only
                stats(s_ij, "s_ij (cleaned)")
                vn = vector_norm(s_ij, dim=1, keepdim=True) # TODO: fix. Was part of next line, moved for testing
                stats(vn, "vn")
                s_ij = s_ij/vn
                #s_ij.masked_fill_(torch.isnan(s_ij), 0)

            stats(s_ij, "s_ij (normalised)")
            stats(out, "out")
            # Node update
            if self.with_gradient:
                hydraulic_gradient = out[col][edge_index_mask]-out[row][edge_index_mask]
                stats(hydraulic_gradient, "hydraulic_gradient")
                if self.upwind_mode:
                    hydraulic_gradient[hydraulic_gradient<0] = 0
                shift_sum = hydraulic_gradient*s_ij
            else:
                shift_sum = s_ij*out[row][edge_index_mask]
            stats(shift_sum, "shift_sum")
            scattered = scatter(shift_sum, col[edge_index_mask], reduce='sum', 
                          dim=0, dim_size=num_nodes)
            stats(scattered, "scattered (initial)")

            if self.with_filter_matrix:
                #print(f"scattered min absmin max: {scattered.min().item()} ({scattered.abs().min().item()}) {scattered.max().item()}\n Filter Matrix: {self.filter_matrix[k+1]}")
                stats(self.filter_matrix[k+1].weight, "filter_matrix weights")
                scattered = self.filter_matrix[k+1].forward(scattered)
            
            stats(scattered, "scattered (final)")
            out = out + scattered
        #print(f"[recompute={PassTrackerMode.status[0]}] Done")
        if display:
            print("\n".join(printbuffer))
        return out

    def __repr__(self):
        return '{}(node_features={}, edge_features={}, K={}, with_filter_matrix={}, with_gradient={})'.format(
            self.__class__.__name__, self.edge_output_size, 
            self.edge_features, self.K, self.with_filter_matrix,
            self.with_gradient)

class PassTrackerMode(torch.utils._python_dispatch.TorchDispatchMode):
    status = [None]
    def __init__(self, is_recompute):
        super().__init__()
        self.is_recompute = is_recompute
    def __torch_dispatch__(self,func,types,args=(), kwargs=None):
        PassTrackerMode.status[0] = self.is_recompute
        #print("torch_dispatch:", func, types, args, kwargs, self.is_recompute)
        return func(*args, **kwargs)#, is_recompute=self.is_recompute)

def policy_fn(ctx, op, *args, **kwargs):
    print(op)
    if op == torch.ops.aten.ne.Scalar:
        print("Saving torch.ops.aten.ne.Scalar")
        return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE # There's a <tensor> != 0 operation that creates a mask which determines a later tensor shape in SWEGNN.forward(). It is crucial that this returns identical values each time (to prevent tensor shape mismatch)
    else:
        return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

torch.utils.checkpoint.SAC_IGNORED_OPS.add(torch.ops.aten._to_copy.default) # The _to_copy function get inserted extra times by PyTorch into the backward pass. If it's not ignored, this will cause a function counter mismatch between the forward and backward checkpointing contexts.

def pass_tracker_context_fn(*args, **kwargs):
    #return torch.utils.checkpoint.create_selective_checkpoint_contexts([torch.ops.aten.ne.Scalar], *args, **kwargs)
    import copy, inspect, types
    #fwd, bwd = torch.utils.checkpoint.create_selective_checkpoint_contexts(policy_fn)
    fwd, bwd = torch.utils.checkpoint.create_selective_checkpoint_contexts([torch.ops.aten.ne.Scalar])
    fwd_torch_dispatch = copy.copy(fwd.__torch_dispatch__)
    bwd_torch_dispatch = copy.copy(bwd.__torch_dispatch__)
    def fwd_wrapper(self, *args, **kwargs):
        #print(f"Fwd: {args[0]}")
        PassTrackerMode.status[0] = False
        #if args[0] == torch.ops.aten._to_copy.default:
        #    print("fwd.storage before", fwd.func_counter, fwd.storage[torch.utils.checkpoint._sac_storage_key(args[0],args[2:])])
        out = fwd_torch_dispatch(*args, **kwargs)
        #if args[0] == torch.ops.aten._to_copy.default:
        #    print("fwd.storage after", fwd.func_counter, fwd.storage[torch.utils.checkpoint._sac_storage_key(args[0],args[2:])])
        return out
    def bwd_wrapper(self, *args, **kwargs):
        #print(f"Recomp: {args[0]}")
        PassTrackerMode.status[0] = True 
        #if args[0] == torch.ops.aten._to_copy.default:
        #    print("bwd.storage", bwd.func_counter, bwd.storage[torch.utils.checkpoint._sac_storage_key(args[0],args[2:])])
        return bwd_torch_dispatch(*args, **kwargs)
    fwd.__torch_dispatch__ = types.MethodType(fwd_wrapper, fwd)
    bwd.__torch_dispatch__ = types.MethodType(bwd_wrapper, bwd)
    return (fwd, bwd)
