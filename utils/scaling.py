# Libraries
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from database.graph_creation import MultiscaleMesh

def get_none_scalers():
    none_scalers = {'DEM_scaler'        : None, 
                    'slope_scaler'      : None, 
                    'area_scaler'       : None,
                    'edge_length_scaler': None,
                    'edge_slope_scaler' : None, 
                    'WD_scaler'         : None,  
                    'V_scaler'          : None,
                    'mannings_n_scaler' : None}
    return none_scalers

def stack_attributes(dataset, attribute, inverse=False, to_min=False):
    '''
    Returns a vector containing all features 'attribute' contained in dataset
    '''
    if inverse:
        stacked_map = torch.cat([1/data[attribute] for data in dataset])
    else:
        stacked_map = torch.cat([data[attribute] - data[attribute].min()*to_min for data in dataset])
    
    return stacked_map.reshape(-1,1)

def scaler(train_database, attribute, type_scaler='minmax', inverse=False, to_min=False):
    '''
    Returns Scaler for a 2D map
    This function should consider only the training dataset
    -------
    train_database : list
        each element in the list is a torch_geometric.data.data.Data object
    attribute: str or list or tuple
        name of the feature to be scaled
        if list, scaling applies to more features
        if tuple, elements are used for vector norm
    type_scaler: str
        name of the scaler
        options: 'minmax', 'minmax_neg', or 'standard'
    to_min: bool
        subtract the minimum value before scaling (then the minimum is always 0)
    '''    
    assert isinstance(train_database, list), 'train_database must be a list of torch_geometric.data.data.Data objects'
    
    if type_scaler == 'minmax':
        scaler = MinMaxScaler(feature_range=(0,1))
    elif type_scaler == 'minmax_neg':
        scaler = MinMaxScaler(feature_range=(-1,1))
    elif type_scaler == 'standard':
        scaler = StandardScaler()
    elif type_scaler is None:
        return None
    else:
        raise 'type_scaler can be only "minmax", "minmax_neg", or "standard"'
    
    if isinstance(attribute, list):
        all_attrs = torch.cat([stack_attributes(train_database, attr, inverse=inverse, to_min=to_min) for attr in attribute])
    elif isinstance(attribute, tuple):
        all_attrs = torch.cat([stack_attributes(train_database, attr, inverse=inverse, to_min=to_min)**2 for attr in attribute], 1)
        all_attrs = all_attrs.sum(1).sqrt().reshape(-1,1)
    else:
        all_attrs = stack_attributes(train_database, attribute, inverse=inverse, to_min=to_min)

    scaler.fit(all_attrs)
        
    return scaler

def multiscale_scaler(train_database, attribute: str, type_feature:str, type_scaler='minmax'):
    '''
    Returns list of Scalers for a feature of the dataset, at each scale
    This function should consider only the training dataset and only multiscale datasets
    -------
    train_database : list
        each element in the list is a torch_geometric.data.data.Data object
    attribute: str
        name of the feature to be scaled
    type_feature: str
        'node' or 'edge' to indicate if the feature is defined at nodes or edges
    type_scaler: str
        name of the scaler
        options: 'minmax', 'minmax_neg', or 'standard'
    '''
    assert isinstance(train_database, list), 'train_database must be a list of torch_geometric.data.data.Data objects'
    assert isinstance(train_database[0].mesh, MultiscaleMesh), 'train_database must be a list of multiscale datasets'

    num_scales = train_database[0].mesh.num_meshes

    if type_scaler == 'minmax':
        scalers = [MinMaxScaler(feature_range=(0,1)) for _ in range(num_scales)]
    elif type_scaler == 'minmax_neg':
        scalers = [MinMaxScaler(feature_range=(-1,1)) for _ in range(num_scales)]
    elif type_scaler == 'standard':
        scalers = [StandardScaler() for _ in range(num_scales)]
    elif type_scaler is None:
        return None
    else:
        raise ValueError('type_scaler can be only "minmax", "minmax_neg", or "standard", instead got {}'.format(type_scaler) + ' for multiscale_scaler()')
    
    if type_feature == 'node':
        all_attrs = [torch.cat([data[attribute][data.node_ptr[i]:data.node_ptr[i+1]] for data in train_database]).reshape(-1,1) for i in range(num_scales)]
    elif type_feature == 'edge':
        all_attrs = [torch.cat([data[attribute][data.edge_ptr[i]:data.edge_ptr[i+1]] for data in train_database]).reshape(-1,1) for i in range(num_scales)]
    else:
        raise 'type_feature can be only "node" or "edge"'
    
    for attr, scaler in zip(all_attrs, scalers):
        scaler.fit(attr)
        
    return scalers

def get_scalers(dataset, scalers: dict):
    '''
    Returns scaler dictionary with scaler objects as values
    ------
    dataset: list
        training dataset used to obtain the scalers
    scalers: dict
        dict with the type of scaling used for every variable
        options: 'minmax', 'minmax_neg', 'standard', or None
    '''
    if scalers is None:
        scalers = get_none_scalers()

    scalers['DEM_scaler'] = scaler(dataset, 'DEM', type_scaler=scalers['DEM_scaler'], to_min=True)
    scalers['mannings_n_scaler'] = scaler(dataset, 'mannings_n', type_scaler=scalers['mannings_n_scaler'], to_min=False) # to_min False as minimum Manning's N value should not be scaled to 0.
    scalers['WD_scaler'] = scaler(dataset, 'WD', type_scaler=scalers['WD_scaler'])
    scalers['slope_scaler'] = scaler(dataset, ['slopex', 'slopey'], type_scaler=scalers['slope_scaler'])

    # for multiscale datasets use different scalers for area and edges
    if isinstance(dataset[0].mesh, MultiscaleMesh):
        scalers['area_scaler'] = multiscale_scaler(dataset, 'area', type_feature='node', type_scaler=scalers['area_scaler'])
        scalers['edge_length_scaler'] = multiscale_scaler(dataset, 'face_distance', type_feature='edge', type_scaler=scalers['edge_length_scaler'])
        scalers['edge_slope_scaler'] = multiscale_scaler(dataset, 'edge_slope', type_feature='edge', type_scaler=scalers['edge_slope_scaler'])
    else:
        scalers['area_scaler'] = scaler(dataset, 'area', type_scaler=scalers['area_scaler'], inverse=False)
        scalers['edge_length_scaler'] = scaler(dataset, 'face_distance', type_scaler=scalers['edge_length_scaler'])
        scalers['edge_slope_scaler'] = scaler(dataset, 'edge_slope', type_scaler=scalers['edge_slope_scaler'])
    
    scalers['V_scaler'] = scaler(dataset, ('VX', 'VY'), type_scaler=scalers['V_scaler'])

    return scalers