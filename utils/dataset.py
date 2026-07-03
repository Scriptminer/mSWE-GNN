# Libraries
import torch
from torch_geometric.data import Data
from torch_geometric.utils import scatter
import numpy as np
import os

from database.graph_creation import MultiscaleMesh, rotate_mesh
from utils.load import load_dataset
from utils.scaling import get_scalers
from sklearn.model_selection import train_test_split

NUM_WATER_VARS = 2  # number of water variables (water depth and discharge)

def process_attr(attribute, scaler=None, to_min=False, device='cpu'):
    '''
    processes an attribute for dataset creation (shift to min, shape, scale)
    ------
    scaler: 
        if present, used to scale the attribute
    to_min: bool
        if True, shifts minimum to zero before scaling
    '''
    assert isinstance(attribute, torch.Tensor), "Input attribute is not a tensor"
    
    if attribute.dim() == 1:
        attribute = attribute.reshape(-1,1)

    attr = attribute.clone()

    if to_min:
        attr -= attr.min()

    if scaler is not None:
        attr = torch.cat([torch.FloatTensor(scaler.transform(attr[:,col:col+1])) for col in range(attr.shape[1])], dim=1)

    assert attribute.shape == attr.shape, "Shape has changed during processing: \n"\
        f"Before it was {attribute.shape}, now it is {attr.shape}"
        
    return attr.to(device)

def slopes_from_DEM_grid(DEM):
    '''
    Calculate slope in x and y directions, given a DEM on a grid
    '''
    slope_x, slope_y = torch.gradient(DEM)
    return slope_x.reshape(-1), slope_y.reshape(-1)

def slopes_from_DEM_mesh(mesh, edge_index):
    edge_relative_distance = mesh.face_xy[edge_index[1]] - mesh.face_xy[edge_index[0]]
    edge_slope = (mesh.DEM[edge_index[0]] - mesh.DEM[edge_index[1]])/np.linalg.norm(edge_relative_distance, axis=1)
    directed_slope = torch.FloatTensor(mesh.edge_outward_normal[mesh.edge_type < 3].T) * edge_slope * 1000
    
    slopex = scatter(directed_slope[1,:], edge_index[0], reduce='mean')
    slopey = scatter(directed_slope[0,:], edge_index[0], reduce='mean')

    return slopex, slopey

def get_temporal_res(matrix, temporal_res=60, original_temporal_res=60):
    '''
    extracts a sub-matrix with temporal_res [min] from a temporal matrix [N, T]
    ------
    matrix: torch.tensor
        input temporal matrix with temporal resolution in the second column
    temporal_res: int
        selects the desired time step for the temporal resolution
    original_temporal_res: int
        temporal resolution of the input matrix
    '''
    selected_times = torch.arange(0, matrix.shape[-1], temporal_res/original_temporal_res, dtype=int)
        
    return matrix[:, selected_times]

def get_node_features(data, scalers=None, slopes=False, slope=False, 
                      area=False, DEM=False, device='cpu'):
    '''Return the static node features

    data: torch_geometric.data.data.Data
        dataset sample, containing numerical simulation
    scalers: None, dict
        define how to scale DEM, slopes, and area
        (default=None)
    device: str
        device used to store dataset (default='cpu')

    selected_node_features: dict (of bool)
        slopes: topographic slopes in x and y directions (default=True)
        area: mesh area (default=True)
        DEM: digital elevation model of the topography (default=False)
    '''
    node_features = {}
    
    if scalers is None:
        scalers = {
            'DEM_scaler'  : None, 
            'slope_scaler': None,
            'area_scaler' : None,
        }
    

    # x and y slopes
    if slopes:
        node_features['slopes'] = process_attr(torch.stack((data.slopex, data.slopey),-1).to(torch.float32), scaler=scalers['slope_scaler'], device=device)
    
    # absolute value of slope
    if slope:
        node_features['slope'] = process_attr(torch.sqrt(data.slopex**2 + data.slopey**2).to(torch.float32), 
                                               scaler=scalers['slope_scaler'], device=device)

    # mesh cell area
    if area:
        if isinstance(data.mesh, MultiscaleMesh):
            node_features['area'] = torch.cat([process_attr(data.area[data.node_ptr[i]:data.node_ptr[i+1]], device=device, scaler=scaler) 
                                               for i, scaler in enumerate(scalers['area_scaler'])])
        else:
            node_features['area'] = process_attr(data.area, device=device, scaler=scalers['area_scaler'])
    
    # digital elevation model (DEM)
    if DEM:
        node_features['DEM'] = process_attr(data.DEM, scaler=scalers['DEM_scaler'], to_min=True, device=device)

    selected_node_features = locals()

    selected_nodes = [node_features[key] for key, value in selected_node_features.items() if value==True]
    
    if len(selected_nodes) == 0:
        node_features = torch.ones(data.num_nodes, 1).to(device)
    else:
        node_features = torch.cat(selected_nodes, 1).to(device)
    
    return node_features

def get_edge_features(data, scalers=None, edge_length=False, edge_relative_distance=False,
                      edge_slope=False, device='cpu'):
    '''Return the static edge features

    data: torch_geometric.data.data.Data
        dataset sample, containing numerical simulation
    scalers: None, dict
        define how to scale edge_length, edge_relative_distance and edge_slope
    device: str
        device used to store dataset (default='cpu')

    selected_edge_features: dict (of bool)
        edge_length: distance among two cell centers (default=True)
        edge_relative_distance: relative distance among two cell centers (default=False)
        edge_slope: slope across two neighbouring cells (default=True)
    '''
    if scalers is None:
        scalers = {'edge_length_scaler' : None,
                   'edge_slope_scaler'  : None}
        
    edge_features = {}

    if edge_length:
        if isinstance(data.mesh, MultiscaleMesh):
            if scalers['edge_length_scaler'] is not None:
                edge_features['edge_length'] = torch.cat(
                    [process_attr(data.face_distance[data.edge_ptr[i]:data.edge_ptr[i+1]], device=device, scaler=scaler) 
                     for i, scaler in enumerate(scalers['edge_length_scaler'])])
            else:
                edge_features['edge_length'] = torch.cat(
                    [process_attr(data.face_distance[data.edge_ptr[i]:data.edge_ptr[i+1]], device=device, scaler=None) 
                     for i in range(data.mesh.num_meshes)])
        else:
            edge_features['edge_length'] = process_attr(data.face_distance, scaler=scalers['edge_length_scaler'], device=device)
        
    if edge_relative_distance:
        if isinstance(data.mesh, MultiscaleMesh):
            edge_features['edge_relative_distance'] = torch.cat(
                    [process_attr((data.face_relative_distance/data.face_distance[:,None])[data.edge_ptr[i]:data.edge_ptr[i+1]], device=device, scaler=None) 
                     for i in range(data.mesh.num_meshes)])
        else:
            edge_features['edge_relative_distance'] = process_attr(data.face_relative_distance/data.face_distance[:,None], scaler=scalers['edge_length_scaler'], device=device)
        
    if edge_slope:
        if isinstance(data.mesh, MultiscaleMesh):
            if scalers['edge_slope_scaler'] is not None:
                edge_features['edge_slope'] = torch.cat([process_attr(data.edge_slope[data.edge_ptr[i]:data.edge_ptr[i+1]], device=device, scaler=scaler)
                                                         for i, scaler in enumerate(scalers['edge_slope_scaler'])])
            else:
                edge_features['edge_slope'] = torch.cat(
                    [process_attr(data.edge_slope[data.edge_ptr[i]:data.edge_ptr[i+1]], device=device, scaler=None) 
                                  for i in range(data.mesh.num_meshes)])
        else:
            edge_features['edge_slope'] = process_attr(data.edge_slope, scaler=scalers['edge_slope_scaler'], device=device)

    selected_edge_features = locals()

    selected_edges = [edge_features[key].to(device) for key, value in selected_edge_features.items() if value==True]
    
    if len(selected_edges) == 0:
        edge_features = torch.ones(data.num_edges, 1).to(device)
    else:
        edge_features = torch.cat(selected_edges, 1).float().to(device)
    
    return edge_features

def process_WD_VX_VY(data, temporal_res=60, original_temporal_res=60, 
                     scalers=None, device='cpu'):
    '''Processes dynamic features that will be used to create output node features

    data: torch_geometric.data.data.Data
        dataset sample, containing numerical simulation
    temporal_res: int [min]
        temporal resolution for the dataset (default=60)
    original_temporal_res: int
        temporal resolution of the input matrix
    scalers: None, dict
        define how to scale water depth and velocities
        (default=None)
    device: str
        device used to store dataset (default='cpu')
    '''
    if scalers is None:
        scalers = {
            'WD_scaler' : None, 
            'V_scaler' : None
        }

    temp = Data()

    WD = process_attr(data.WD, scaler=scalers['WD_scaler'], device=device)
    temp.WD = get_temporal_res(WD, temporal_res=temporal_res, original_temporal_res=original_temporal_res)

    VX = process_attr(data.VX, scaler=scalers['V_scaler'], device=device)*WD
    VY = process_attr(data.VY, scaler=scalers['V_scaler'], device=device)*WD

    V = torch.sqrt(VX**2 + VY**2)
    temp.V = get_temporal_res(V, temporal_res=temporal_res, original_temporal_res=original_temporal_res)

    return temp

def create_data_attr(datasets, scalers=None, temporal_res=60, original_temporal_res=60, 
                     device='cpu', **selected_features):
    '''
    Creates x, y, and edge_attr from Data object
    ------
    datasets : list
        each element in the list is a torch_geometric.data.data.Data object
    scalers: dict
        sklearn.preprocessing._data scaler used for normalizing the data
    temporal_res: int [min]
        selects the desired time step for the temporal resolution
    original_temporal_res: int
        temporal resolution of the input matrix
    selected_features:
        selected_node_features: dict (of bool)
            dictionary that specifies node features
        selected_edge_features: dict (of bool)
            dictionary that specifies edge features
    '''
    new_dataset = []
    selected_node_features = {
        key:selected_features[key] for key in 
        ['slopes', 'slope', 'area', 'DEM']}

    selected_edge_features = {
        key:selected_features[key] for key in 
        ['edge_length', 'edge_relative_distance', 'edge_slope']}

    for data in datasets:
        temp = process_WD_VX_VY(data, temporal_res=temporal_res, original_temporal_res=original_temporal_res, scalers=scalers, device=device)
        temp.edge_index = data.edge_index.to(device)
        temp.edge_attr = get_edge_features(data, scalers=scalers, **selected_edge_features, device=device)
        temp.x = get_node_features(data, **selected_node_features, scalers=scalers, device=device)
        temp.DEM = data.DEM
        temp.temporal_res = temporal_res
        temp.area = data.area.to(device)
        if 'BC' in data.keys():
            if data.BC.dim() > 2:   #hydrograph BC
                temp.BC = get_temporal_res(data.BC[:,:,1], temporal_res=temporal_res, original_temporal_res=original_temporal_res)
            else:                   #constant BC
                time_steps = temp.WD.shape[1]
                temp.BC = torch.ones(time_steps)*data.BC
            temp.node_BC = data.node_BC.to(device)
            temp.type_BC = data.type_BC
            temp.edge_BC_length = data.edge_BC_length.to(device)
            temp.BC = temp.BC.to(device)/temp.edge_BC_length
        
        if 'mesh' in data.keys():
            temp.mesh = data.mesh
            if isinstance(temp.mesh, MultiscaleMesh):
                temp.node_ptr = data.node_ptr
                temp.edge_ptr = data.edge_ptr
                temp.intra_edge_ptr = data.intra_edge_ptr
                temp.intra_mesh_edge_index = data.intra_mesh_edge_index.to(device)
        else:
            temp.pos = data.pos

        new_dataset.append(temp)
    
    return new_dataset


def create_model_dataset(train_dataset_name='grid', train_size=100, val_prcnt=0.3, 
                         test_dataset_name='grid', dataset_folder='database/datasets',
                         scalers=None, seed=42, device='cpu', **dataset_parameters):
    '''
    Create dataset with scaled node and edge attributes
    Return training, validation, and testing datasets
    ------
    *_dataset_name: str
        name of the dataset to be loaded for training and testing
    train_size: int
        number of samples used for training
    val_prcnt: float
        percentage of the training dataset used for validation (default=0.3)
    dataset_folder: str
        path to the folder containing the datasets
    scalers: dict
        str or sklearn.preprocessing._data
    seed: int
        fixed randomness for replicability in dataset splits and shuffling
    dataset_parameters:
        selected_node_features: dict (of bool)
            dictionary that specifies node features
        selected_edge_features: dict (of bool)
            dictionary that specifies edge features
        temporal_res: int [min]
            selects the desired time step for the temporal resolution
    '''
    # Load datasets
    train_dataset = load_dataset(train_dataset_name, train_size, seed, os.path.join(dataset_folder, 'train'))
    test_dataset = load_dataset(test_dataset_name, 100, seed=0, dataset_folder=os.path.join(dataset_folder, 'test'))
    # create validation dataset from training
    if val_prcnt != 0:
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=val_prcnt, random_state=seed)
    else:
        print("The validation dataset you are using is the training one. Careful!")
        val_dataset = train_dataset
        
    # Normalization using only training
    scalers = get_scalers(train_dataset, scalers)
    
    # Create x, edge_attr, y
    train_dataset = create_data_attr(train_dataset, scalers=scalers, device=device, **dataset_parameters)
    val_dataset = create_data_attr(val_dataset, scalers=scalers, device=device, **dataset_parameters)
    test_dataset = create_data_attr(test_dataset, scalers=scalers, device=device, **dataset_parameters)
    
    return train_dataset, val_dataset, test_dataset, scalers

def aggregate_WD_V(WD, V, init_time):
    '''Create a tensor that concatenates water depth and velocity (module)
    
    WD and V are taken for a interval [init_time:init_time+1]
    
    Output shape: [num_nodes, 2]
    '''
    return torch.cat((WD[:,init_time:init_time+1], 
                      V[:,init_time:init_time+1]), 1)

def aggregate_BC(BC, previous_t, init_time):
    '''Create a tensor that returns boundary conditions
    
    BC is taken for a interval [init_time:init_time+1]
    
    Output shape: [num_BC, previous_t]
    '''
    return BC[:,init_time:init_time+previous_t]

def get_previous_steps(aggregate_function, init_time, previous_t, *water_variables_args):
    '''Return tensor with input time steps'''
    prev_steps = torch.cat([aggregate_function(*water_variables_args, step) for step in range(init_time,init_time+previous_t)], -1)
    return prev_steps

def get_next_steps(aggregate_function, init_time, rollout_steps, *water_variables_args):
    '''Return tensor with output time steps'''
    next_steps = torch.stack([torch.cat([aggregate_function(*water_variables_args, step_f+step_r) 
                            for step_f in range(init_time,init_time+1)], -1) 
                            for step_r in range(0,rollout_steps)], -1)
    assert next_steps.shape[-1] == rollout_steps, f"The output dimension is wrong: {next_steps.shape}"
    return next_steps

def add_dry_bed_condition(variable, previous_t):
    '''Concatenates a zero vector of size previous_t-1 to the input variable (dry bed condition)'''
    device = variable.device
    if variable.dim() == 1:
        return torch.cat((torch.zeros(previous_t-1, device=device), variable))
    elif variable.dim() == 2:
        num_nodes = variable.shape[0]
        return torch.cat((torch.zeros(num_nodes, previous_t-1, device=device), variable), 1)
    else:
        raise ValueError("Something wrong with the dimensions when adding dry bed conditions")

def get_temporal_samples_size(maximum_time, time_start=0, time_stop=-1, rollout_steps=1):
    '''Returns the number of samples generated when creating the temporal dataset

    maximum_time: int
        number of time steps in a simulation (e.g., 48(*1h))
    time_start: int (default=0)
        initial time step given as input
    time_stop: int (default=-1)
        final time step of the simulation
        if -1, takes all the simulation
    rollout_steps: int (default=1)
        number of times the output is predicted in each temporal sample
    '''
    assert maximum_time > 0, 'The temporal size of the dataset is zero'
    assert time_stop <= maximum_time, 'time_stop cannot be higher than the temporal size of the dataset'
    if time_stop != maximum_time:
        time_stop = time_stop % maximum_time - time_start + 1  # add 1 because rollout_steps starts from 1

    assert time_start <= time_stop, 'time_start cannot be higher than the last selected time'
    assert rollout_steps <= time_stop, 'Number of rollout_steps is too high'

    # if rollout_steps is -1, it takes all the simulation
    temporal_sample_size = time_stop - rollout_steps if rollout_steps > 0 else -rollout_steps

    assert temporal_sample_size >= 0, f'Something went wrong here: the temporal sample size is {temporal_sample_size}'

    return temporal_sample_size

def to_temporal(data, previous_t=2, time_start=0, time_stop=-1, rollout_steps=1):
    '''Converts Data object with temporal signal on graph into multiple graphs
    
    previous_t: int (default=2)
        number of previous time steps given as input
    time_start: int (default=0)
        initial time step given as input
    time_stop: int (default=-1)
        final time step of the simulation
        if -1, takes all the simulation
    rollout_steps: int (default=1)
        number of times the output is predicted
    '''
    temporal_data = []
    device = data.x.device
    maximum_time = data.WD.shape[1]
    temporal_samples_size = get_temporal_samples_size(maximum_time, time_start, time_stop, rollout_steps)
    rollout_steps = (rollout_steps%(time_stop%maximum_time-time_start+1)) if rollout_steps < 0 else rollout_steps

    WD = add_dry_bed_condition(data.WD, previous_t)
    BC = torch.cat((add_dry_bed_condition(data.BC, previous_t), data.BC[:,-1:]), 1) # Also add the last BC because of mass conservation
    V = add_dry_bed_condition(data.V, previous_t)

    for init_time in range(time_start, time_start+temporal_samples_size):
        temp = Data()
        
        temp.edge_index = data.edge_index
        temp.edge_attr = data.edge_attr
        temp.pos = data.pos
        temp.area = data.area
        temp.temporal_res = data.temporal_res
        
    
        prev_steps = get_previous_steps(aggregate_WD_V, init_time, previous_t, WD, V)
        next_steps = get_next_steps(aggregate_WD_V, init_time+previous_t, rollout_steps, WD, V)

        assert prev_steps.shape[1] == NUM_WATER_VARS*previous_t, f"The output dimension is wrong: {prev_steps.shape}"
        assert next_steps.shape[1] == NUM_WATER_VARS , f"The output dimension is wrong: {next_steps.shape}"
        assert next_steps.shape[2] == rollout_steps, f"The output dimension is wrong: {next_steps.shape}"
        if (prev_steps[:,-NUM_WATER_VARS:] != 0).all(): # Except when everything is zero, then no problem
            assert ~torch.isclose(prev_steps[:,-NUM_WATER_VARS:], next_steps[:,:,0]).all(), "You're copying last time step and output"
        
        # current_time = (init_time+previous_t)*data.temporal_res/60
        temp.x = torch.cat((data.x, prev_steps.to(device)), 1)
        temp.y = next_steps.to(device)
        
        temp.BC = get_next_steps(aggregate_BC, init_time, rollout_steps+1, BC, previous_t)[:,::1].to(device)
        temp.time = init_time
        temp.edge_BC_length = data.edge_BC_length
        temp.previous_t = previous_t
        temp.node_BC = data.node_BC
        temp.type_BC = data.type_BC
        
        if 'mesh' in data.keys() and isinstance(data.mesh, MultiscaleMesh):
            temp.node_ptr = data.node_ptr
            temp.edge_ptr = data.edge_ptr
            temp.intra_edge_ptr = data.intra_edge_ptr
            temp.intra_mesh_edge_index = data.intra_mesh_edge_index

        temporal_data.append(temp)
        
    return temporal_data

def to_temporal_dataset(datasets, **temporal_dataset_parameters):
    '''Converts dataset into a list of temporal Data objects'''
    new_dataset = []
    for data in datasets:
        new_dataset += to_temporal(data, **temporal_dataset_parameters)

    return new_dataset
    
def get_edge_BC(node_BC, edge_index):
    '''Returns the edge index id where the boundary conditionare applied'''
    edge_BC = torch.cat([torch.where(node == edge_index)[1] for node in node_BC])
    return edge_BC

def apply_boundary_condition(x_d, BC, node_BC, type_BC=2):
    '''
    Apply inflow boundary condition BC to nodes node_BC
    type_BC:
        1: Inflow water depth h 
        2: Inflow discharge |q|
    '''    
    check_type_BC(type_BC, NUM_WATER_VARS)
    
    x_d[node_BC, (type_BC-1)::NUM_WATER_VARS] = BC

    return x_d

def check_type_BC(type_BC, num_water_vars):
    if type_BC == 1 or type_BC == 2:
        assert type_BC <= num_water_vars, \
            "The boundary conditions are not compatible with the data format you are using."
    elif type_BC == 3:
        raise ValueError("Vector boundary conditions are not yet implemented. Please desist from convincing me to implement them.")
    else:
        raise ValueError(f"BC_type={type_BC} is not a valid input. Please select either:\n1: Inflow water depth\n2: Inflow discharge")

def use_prediction(x, pred, previous_t):
    '''
    Creates a new input by replacing the last input time steps with the model's predictions (pred)
    All the remaining dynamic variables are shifted to the left so that the last input time steps are the predictions

    Args:
    x (torch.tensor): Input tensor (shape: [num_nodes, static_features+dynamic_features])
    pred (torch.tensor): Predictions (shape: [num_nodes, 1])
    previous_t (int): Number of previous time steps
    '''
    out_dim = NUM_WATER_VARS
    assert pred.shape[-1] == out_dim, "The number of predictions is not consistent with the number of future time steps"
    dynaminc_vars = previous_t*NUM_WATER_VARS
    static_vars = x.shape[1]-dynaminc_vars

    if previous_t == 1:
        temp = torch.cat((x[:,:static_vars], pred), 1)
    else:
        temp = torch.cat((x[:,:static_vars], x[:,-dynaminc_vars+out_dim:], pred), 1)
    assert temp.shape == x.shape, f'The shape of the input has changed from {x.shape} to {temp.shape}'

    return temp

def get_real_rollout(dataset, time_start, time_stop):
    '''Return real rollout for the selected time interval'''
    if time_stop == -1:
        real_rollout = dataset.y[:,:,time_start+1:].clone()
    else:
        real_rollout = dataset.y[:,:,time_start+1:time_stop+1].clone()
    
    return real_rollout

def get_input_water(dataset):
    '''Returns water variables used as input'''
    out_dim = 2
    input_water = dataset.x[:,-out_dim:].clone()

    return input_water

def get_temporal_test_dataset_parameters(config, temporal_dataset_parameters):
    '''Returns temporal test dataset parameters by either taking them from the config file if present
    or from the training dataset parameters if otherwise'''
    try:
        temporal_test_dataset_parameters = config['temporal_test_dataset_parameters']
    except:
        temporal_test_dataset_parameters = temporal_dataset_parameters.copy()
        temporal_test_dataset_parameters.pop('rollout_steps')
        # temporal_test_dataset_parameters.pop('previous_t')

    return temporal_test_dataset_parameters

def velocity_from_discharge(discharge, water_depth): #CURRENTLY NOT ACTIVE
    '''Converts discharge to velocity: v = q/h,
    Masking water depth < 0.01 to avoid division by zero
    '''
    # velocity = discharge/water_depth
    # velocity[water_depth<0.01] = 0
    # return velocity
    return discharge

def convert_to_velocity(rollout):
    '''Converts discharge to velocity: v = q/h'''
    if rollout.shape[1] == 2: #scalar
        rollout[:,1,:] = velocity_from_discharge(rollout[:,1,:], rollout[:,0,:])
    if rollout.shape[1] == 3: #vector
        rollout[:,1,:] = velocity_from_discharge(rollout[:,1,:], rollout[:,0,:])
        rollout[:,2,:] = velocity_from_discharge(rollout[:,2,:], rollout[:,0,:])
    return rollout

def get_inflow_volume(data, BC):
    '''Determine input flood volume based on input boundary condition on unit discharge [m^2/s] as:
    
    V = \sum |q| * L_bc

    where |q| is the absolute value of the discharge and L_bc is the length of the boundary condition.

    This function works for a given interval of time which should be implicit in BC. 
    For example, if your BC spans 40 time steps, you hould sum it before passing it to this function.    
    '''
    sec_in_min = 60 #seconds in a minute
    inflow_nodes = BC * data.edge_BC_length  # [m^2/s * m = m^3/s]

    inflow_volume = inflow_nodes.sum() * (sec_in_min * data.temporal_res) #[m^3]
    return inflow_volume 

def get_breach_coordinates(WD, pos):
    '''Returns the coordinates of the breach identified from where the water depth is non-zero at time 0'''
    breach_locations = [loc.item() for loc in torch.where(WD[:,0] != 0)]

    breach_coordinates = [pos[loc] for loc in breach_locations]

    return breach_coordinates

def separate_multiscale_node_features(x, node_ptr):
    """Separates multiscale node features into a list of node features at each scale.
    
    x (torch.tensor): node features of a multiscale mesh
    node_ptr (torch.tensor): partition of the multiscale mesh into scales
    """

    num_scales = len(node_ptr)-1

    x_scales = [x[node_ptr[scale]:node_ptr[scale+1]] for scale in range(num_scales)]

    return x_scales

from torch_geometric.data import Batch
def create_scale_mask(num_nodes, num_scales, node_ptr, data_type, device='cpu'):
    """Creates a mask of shape num_nodes with entry i for each scale i (defined by node_ptr)
    
    mask = e.g., [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, ...]

    num_nodes: int
        Number of nodes in the mesh
    num_scales: int
        Number of scales in the mesh
    node_ptr: torch.tensor
        Pointer to the start of each scale in the mesh
    data_type: torch_geometric.data.Data or torch_geometric.data.Batch
        Data type of the mesh
    device: str
        Device to store the mask
    """                
    mask = torch.zeros(num_nodes, dtype=torch.int, device=device)
    for i in range(num_scales):
        if isinstance(data_type, Batch):
            for j in node_ptr[:,i:i+2]:
                mask[j[0]:j[1]] = i
        else:
            mask[node_ptr[i]:node_ptr[i+1]] = i
    return mask

def rotate_data_sample(data, angle, selected_node_features, selected_edge_features):
    """Data augmentation: rotate the data sample by a given angle
    Use this function after creating the dataset
    
    Args:
        data (torch_geomertic.data.Data): data sample
        angle (float): angle in degrees
        selected_node_features (dict): dictionary of selected node features
        selected_edge_features (dict): dictionary of selected edge features
    """
    rotated_data = data.clone()

    rotated_data.mesh = rotate_mesh(rotated_data.mesh, angle)

    if isinstance(data.mesh, MultiscaleMesh):
        rotated_data.mesh.meshes = [rotate_mesh(mesh, angle) for mesh in data.mesh.meshes]

    angle = np.deg2rad(angle)
    rot_matrix = torch.FloatTensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).to(data.x.device)

    # rotate edge_relative_distance
    if selected_edge_features['edge_relative_distance']:
        edge_length_bool = selected_edge_features['edge_length']
        rotated_data.edge_attr[:,slice(edge_length_bool, 2+edge_length_bool)] = torch.matmul(rot_matrix, data.edge_attr[:,slice(edge_length_bool, 2+edge_length_bool)].T).T

    # rotate slopes
    if selected_node_features['slopes']:
        rotated_data.x[:,:2] = torch.matmul(rot_matrix, data.x[:,:2].T).T

    return rotated_data
