# Libraries
import torch
from torch_geometric.data import Batch
from typing import overload
from utils.dataset import get_inflow_volume

NUM_WATER_VARS = 2 # water depth and velocity

def get_mean_error(diff_rollout, type_loss, nodes_dim=0):
    '''Calculates mean error between predictions and real values

    Parameters:
    diff_rollout: torch.tensor
        difference between predictions and real values
    type_loss: str
        options: 'RMSE', 'MAE'
    nodes_dim: int (default = 0)
        dimension where nodes are located
    '''
    if type_loss == 'RMSE':
        average_diff_t = torch.sqrt((diff_rollout**2).mean(nodes_dim))
    elif type_loss == 'MAE':
        average_diff_t = diff_rollout.abs().mean(nodes_dim)
    return average_diff_t

def mask_on_water(diff, water_axis=1):
    '''Mask to only calculate loss where there is water

    Parameters:
    diff: torch.tensor
        difference between predictions and real values
    water_axis: int (default = 1)
        axis where water depth is located
    '''
    where_water = (diff != 0).any(water_axis)
    return where_water

def get_loss_variable_scaler(velocity_scaler=1):
    '''Scales loss in velocity terms by a factor velocity_scaler
    
    Parameters:
    velocity_scaler: float (default = 1)
        scales loss in velocity terms by a factor velocity_scaler
    '''
    loss_scaler = torch.ones(NUM_WATER_VARS)
    loss_scaler[1::NUM_WATER_VARS] = velocity_scaler
        
    return loss_scaler

def get_multiscale_loss(diff, data, only_where_water=True, type_loss='RMSE', nodes_dim=0):
    '''Calculates multiscale loss by weighting loss in different scales

    Parameters:
    diff: torch.tensor
        difference between predictions and real values
    only_where_water: bool (default = True)
        if True, only calculates loss where there is water
    type_loss: str (default = 'RMSE')
        options: 'RMSE', 'MAE'
    nodes_dim: int (default = 0)
        dimension where nodes are located
    '''
    node_ptr = data.node_ptr
    if only_where_water:
        where_water = mask_on_water(diff)
    else:
        where_water = torch.ones(diff.shape[0]).bool()
    
    if isinstance(data, Batch):
        multiscale_loss = get_mean_error(torch.cat([diff[data.node_ptr[i,0]:data.node_ptr[i,1]][where_water[node_ptr[i,0]:node_ptr[i,1]]] 
                                                    for i in range(data.num_graphs)]), type_loss, nodes_dim)
    else:
        multiscale_loss = get_mean_error(diff[node_ptr[0]:node_ptr[1]][where_water[node_ptr[0]:node_ptr[1]]], type_loss, nodes_dim)
        
    return multiscale_loss

def loss_function(preds, real, data, BC, type_loss='RMSE', only_where_water=False, 
                  conservation=0, velocity_scaler=1, theoretical_volume_delta=None, log_fn=None):
    '''
    Calculates loss between predictions and real values
    
    Parameters:
    preds: torch.tensor (shape = [num_nodes, num_variables])
        predictions of the model
    real: torch.tensor (shape = [num_nodes, num_variables])
        real values
    data: torch_geometric.data.Data
        data object with the graph information
    BC: torch.tensor
        boundary conditions
    type_loss: str (default = 'RMSE')
        options: 'RMSE', 'MAE'
    only_where_water: bool (default = False)
        if True, only calculates loss where there is water
    conservation: float (default = 0)
        coefficient for mass conservation loss
    velocity_scaler: float (default = 1)
        scales loss in velocity terms by a factor velocity_scaler
    theoretical_volume_delta: float or None
        Global volume delta, positive represents mass gain. If None, BC is used
        to calculate volume delta, disregarding any model outflows.
    log_fn: function
        Called to log conservation loss if applicable.
    '''
    diff = preds - real

    if 'node_ptr' in data.keys():
        loss = get_multiscale_loss(diff, data, only_where_water, type_loss, nodes_dim=0)
    else:
        if only_where_water:
            where_water = mask_on_water(diff)
            diff = diff[where_water]
        loss = get_mean_error(diff, type_loss, nodes_dim=0)

    loss_scaler = get_loss_variable_scaler(velocity_scaler=velocity_scaler).to(diff.device)
    loss = torch.dot(loss, loss_scaler)/loss_scaler.sum()
        
    if conservation != 0:
        WD_index = 2
        input_WD = data.x[:,-WD_index::WD_index] #[m] (only water depth)
        pred_WD = preds[:,0::WD_index] #[m] (only water depth)
        if theoretical_volume_delta is not None:
            # Use supplied theoretical_volume_delta to calculate conservation loss
            conservation_loss = full_conservation_loss(pred_WD, input_WD, data, theoretical_volume_delta, log_fn).abs()
        else:
            # No theoretical volume delta specified, assume model has no outflows and use regular conservation loss.
            conservation_loss = conservation_loss(pred_WD, input_WD, data, BC).abs()
        if log_fn:
            log_fn("conservation_loss", conservation_loss, on_step=True)
        loss = loss + conservation*conservation_loss

    return loss

def conservation_loss(pred_WD, input_WD, data, BC):
    '''
    Calculates loss for mass conservation, calculated as the difference between 
    the predicted volume change (sum(area*(pred_WD - input_WD))) and 
    the theoretical inflow volume (BC[t:t+1]*edge_BC_length). Outflows are not
    accounted for - only use when there are no outflows.

    All calculations are done at the finest scale (in case of multiscale simulations)

    Parameters:
    pred_WD: torch.tensor
        predicted water depth (time t+1), shape (num_nodes, 1)
    input_WD: torch.tensor 
        input water depth (time t), shape (num_nodes, 1)
    data: torch_geometric.data.Data
        data object with the graph information (e.g., batch, node_ptr)
    BC: torch.tensor
        boundary conditions (time t), shape (num_BCs)
    '''
    assert BC.dim() == 1, f"Boundary conditions have wrong dimensions ({BC.dim()})"
    theoretical_volume_delta = get_inflow_volume(data, BC) #[m^3]
    return full_conservation_loss(pred_WD, input_WD, data, theoretical_volume_delta)

def full_conservation_loss(pred_WD, input_WD, data, theoretical_volume_delta, log_fn): #TODO this function doesn't require the log function
    '''
    Calculates loss for mass conservation, calculated as the difference between 
    the predicted volume change (sum(area*(pred_WD - input_WD))) and 
    the theoretical_volume_delta

    All calculations are done at the finest scale (in case of multiscale simulations)

    Parameters:
    pred_WD: torch.tensor
        predicted water depth (time t+1), shape (num_nodes, 1)
    input_WD: torch.tensor 
        input water depth (time t), shape (num_nodes, 1)
    data: torch_geometric.data.Data
        data object with the graph information (e.g., batch, node_ptr)
    theoretical_volume_delta: float
        Global volume delta, positive represents mass gain
    '''
    # Calculate delta_WD
    assert pred_WD.shape == input_WD.shape, f"Input or predictions have wrong dimensions ({pred_WD.shape} != {input_WD.shape})"
    delta_WD = pred_WD - input_WD #[m]
    assert delta_WD.dim() == 2, f"Input or predictions have wrong dimensions ({delta_WD.dim()})"

    # Calculate predicted volume
    area = data.area if data.area.dim() ==2 else data.area.unsqueeze(1) #[m^2]

    # Multiscale (select only the finest scale)
    if 'node_ptr' in data.keys():
        if isinstance(data, Batch):
            predicted_volume_delta = torch.cat([(area*delta_WD)[data.node_ptr[i,0]:data.node_ptr[i,1]] 
                                                 for i in range(data.num_graphs)]).sum() #[m^3]
        else:
            predicted_volume_delta = ((area*delta_WD)[data.node_ptr[0]:data.node_ptr[1]]).sum()

    # Single scale
    else: 
        predicted_volume_delta = (area*delta_WD).sum() #[m^3]
    
    # Theoretical inflow volume
    boundary_correction = ((area*delta_WD)[data.node_BC]).sum() #[m^3] # remove values at ghost cells
    
    log_fn("pred_volume_delta", predicted_volume_delta - boundary_correction, on_step=True)
    # Mass conservation 
    conservation_loss = (predicted_volume_delta - theoretical_volume_delta - boundary_correction)/1e6 #[m^3 * 1e6]

    if isinstance(data, Batch):
        conservation_loss = conservation_loss/data.num_graphs
    
    return conservation_loss
