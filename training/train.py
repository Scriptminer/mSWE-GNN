# Libraries
import copy
import numpy as np
import pickle
import torch
import torch.optim as optim
import lightning as L
from lightning.pytorch.callbacks import Callback, BatchSizeFinder
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch
import torch_geometric.data.Dataset
import xarray as xr

from training.loss import loss_function
from utils.miscellaneous import get_rollout_loss, get_CSI
from utils.dataset import use_prediction, apply_boundary_condition, create_data_attr, to_temporal_dataset
from utils.scaling import get_scalers

def adapt_batch_training(batch):
    """Corrects batch features for multiscale batches and so that there are also less if/else conditions later on"""
    assert isinstance(batch, Batch), "This function requires a torch_geometric.data.batch.Batch object as input"
    temp = batch.clone()
    temp.node_BC = torch.cat([temp.ptr[i]+temp[i].node_BC for i in range(temp.num_graphs)])
    temp.temporal_res = temp.temporal_res[0]
    temp.type_BC = temp.type_BC[0]
    temp.previous_t = temp.previous_t[0]
    if 'edge_ptr' in temp.keys():
        update_batch_multiscale(temp)    
        temp.node_BC_ptr = torch.tensor([torch.where(torch.logical_and(temp.node_ptr[:,0] <= node, node <= temp.node_ptr[:,-1]))[0] 
                                                                    for node in temp.node_BC.cpu()])
    else:
        temp.node_BC_ptr = torch.tensor([torch.where(torch.logical_and(temp.ptr[0] <= node, node <= temp.ptr[-1]))[0] 
                                                                    for node in temp.node_BC.cpu()])
    return temp

def update_batch_multiscale(batch):
    """Updates the batch for multiscale models, in particular corrects the partitionings needed in MSGNN"""
    batch_edge_ptr = batch.edge_ptr.reshape(batch.num_graphs, -1)
    batch_intra_edge_ptr = batch.intra_edge_ptr.reshape(batch.num_graphs, -1)
    batch_node_ptr = batch.node_ptr.reshape(batch.num_graphs, -1)
    num_scales = batch_intra_edge_ptr.shape[1]

    updated_batch_edge_ptr = [batch_edge_ptr[0]]
    updated_batch_intra_edge_ptr = [batch_intra_edge_ptr[0]]
    updated_batch_node_ptr = [batch_node_ptr[0]]

    for line in batch_edge_ptr[1:]:
        updated_batch_edge_ptr.append(line + updated_batch_edge_ptr[-1].max())
        
    for line in batch_intra_edge_ptr[1:]:
        updated_batch_intra_edge_ptr.append(line + updated_batch_intra_edge_ptr[-1].max())

    for line in batch_node_ptr[1:]:
        updated_batch_node_ptr.append(line + updated_batch_node_ptr[-1].max())

    updated_batch_edge_ptr = torch.stack(updated_batch_edge_ptr)
    updated_batch_intra_edge_ptr = torch.stack(updated_batch_intra_edge_ptr)
    updated_batch_node_ptr = torch.stack(updated_batch_node_ptr)

    intra_mesh_edge_index = [torch.cat([batch.intra_mesh_edge_index[:,j[0]:j[1]] for j in updated_batch_intra_edge_ptr[:,i:i+2]], 1) 
                             for i in range(num_scales-1)]
    edge_index = [torch.cat([batch.edge_index[:,j[0]:j[1]] for j in updated_batch_edge_ptr[:,i:i+2]], 1) for i in range(num_scales)]
    edge_attr = [torch.cat([batch.edge_attr[j[0]:j[1]] for j in updated_batch_edge_ptr[:,i:i+2]]) for i in range(num_scales)]

    batch.node_ptr = updated_batch_node_ptr
    batch.edge_index = torch.cat(edge_index, 1)
    batch.edge_attr = torch.cat(edge_attr)
    batch.edge_ptr = torch.LongTensor(np.cumsum([0] + [edge.shape[1] for edge in edge_index]))
    batch.intra_edge_ptr = torch.LongTensor(np.cumsum([0] + [edge.shape[1] for edge in intra_mesh_edge_index]))
    batch.intra_mesh_edge_index = torch.cat(intra_mesh_edge_index,1)

@torch.no_grad()
def rollout_test(model, batch):
    '''
    Function that tests a model and returns the rollout prediction
    ------
    model: nn.Model
        e.g., GNN model
    batch: torch_geometric.data.data.Batch
        single or multiple graphs stacked in a batched fashion
    '''
    if isinstance(batch, Batch):
        temp = adapt_batch_training(batch)
    else:
        temp = batch.clone()
        
    dynamic_vars = model.previous_t*model.NUM_WATER_VARS
    assert temp.x.shape[-1] >= dynamic_vars, "The number of dynamic variables is greater than the number of node features"
    final_step = batch.y.shape[-1]
    predicted_rollout = []

    for time_step in range(final_step):
        temp.x[:,-dynamic_vars:] = apply_boundary_condition(temp.x[:,-dynamic_vars:], 
                                                            temp.BC[:,:,time_step], 
                                                            temp.node_BC, type_BC=temp.type_BC)
        pred = model(temp)
        temp.x = use_prediction(temp.x, pred, model.previous_t)
        predicted_rollout.append(pred)
    
    return torch.stack(predicted_rollout, -1)

class LightningTrainer(L.LightningModule):
    def __init__(self, model, lr_info, trainer_options, temporal_test_dataset_parameters):
        '''
        model: nn.Model
            e.g., GNN model
        lr_info: dict
            learning rate information
        trainer_options: dict
            trainer options
        temporal_test_dataset_parameters: dict
            temporal test dataset parameters
        '''
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.dynamic_vars = self.model.previous_t*self.model.NUM_WATER_VARS
        self.lr_info = lr_info
        self.learning_rate = lr_info['learning_rate']
        self.batch_size = trainer_options['batch_size']
        self.only_where_water = trainer_options['only_where_water']
        self.conservation = trainer_options['conservation']
        self.velocity_scaler = trainer_options['velocity_scaler']
        self.type_loss = trainer_options['type_loss']
        self.temporal_test_dataset_parameters = temporal_test_dataset_parameters
        self.rollout_steps = 1
        assert self.type_loss in ['RMSE','MAE'], "loss_type must be either 'RMSE' or 'MAE'"        
        self.curriculum_epoch = trainer_options['curriculum_epoch']
        
    def training_step(self, batch):
        self.log("rollout_steps", torch.tensor(self.rollout_steps, dtype=torch.float32), on_step=False, on_epoch=True)
        temp = adapt_batch_training(batch)
        roll_loss = []

        for i in range(self.rollout_steps):
            temp.x[:,-self.dynamic_vars:] = apply_boundary_condition(temp.x[:,-self.dynamic_vars:], 
                                                                temp.BC[:,:,i], temp.node_BC, 
                                                                type_BC=temp.type_BC)
            # Model prediction
            preds = self.model(temp)
            temp.x = use_prediction(temp.x, preds, self.model.previous_t)

            loss = loss_function(preds, temp.y[:,:,i], temp, temp.BC[:,-2:,i+1].mean(1), type_loss=self.type_loss, 
                           only_where_water=self.only_where_water, conservation=self.conservation, 
                           velocity_scaler=self.velocity_scaler)
            roll_loss.append(loss)

        loss = torch.stack(roll_loss).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 
                                lr=self.lr_info['learning_rate'], 
                                weight_decay=self.lr_info['weight_decay'])
        
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                               step_size=self.lr_info['step_size'], 
                                               gamma=self.lr_info['gamma'])
        return [optimizer], [lr_scheduler]
        
    def validation_step(self, batch, batch_idx):     
        predicted_rollout = rollout_test(self.model, batch)
        real_rollout = batch.y

        # Masking the output to only consider finest scale
        if 'node_ptr' in batch.keys():
            temp = adapt_batch_training(batch)
            mask = self.model._create_scale_mask(temp) == 0
            predicted_rollout = predicted_rollout[mask]
            real_rollout = real_rollout[mask]
        
        assert real_rollout.shape == predicted_rollout.shape, "Real and predicted rollout must have the same dimensions\n"\
                                                f"Intead there is {real_rollout.shape} == {predicted_rollout.shape}"

        val_loss = get_rollout_loss(predicted_rollout, real_rollout, type_loss=self.type_loss, 
                                only_where_water=self.only_where_water).mean()

        # CSI validation
        CSI_005 = get_CSI(predicted_rollout, real_rollout, water_threshold=0.05).nanmean()
        CSI_03 = get_CSI(predicted_rollout, real_rollout, water_threshold=0.3).nanmean()
        
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_CSI_005", CSI_005, prog_bar=True)
        self.log("val_CSI_03", CSI_03, prog_bar=False)

    def predict_step(self, batch, batch_idx):     
        predicted_rollout = rollout_test(self.model, batch)
        return [predicted_rollout[batch.ptr[i]:batch.ptr[i+1]] 
                for i in range(batch.num_graphs)]

class Dataset(torch_geometric.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, dataset_parameters=None, temporal_dataset_parameters=None, device="cpu", simulation_names=[], mesh_common_file=""):
        print("Warning: not yet implemented all parameters from dataset.py, create_model_dataset()")
        if device == "cpu":
            print("Warning: using device CPU for dataset")
        self.dataset_parameters = dataset_parameters
        self.temporal_dataset_parameters = temporal_dataset_parameters
        self.device = device
        self.simulation_names = simulation_names
        with open(mesh_common_file, "rb") as f:
            self.mesh_common = pickle.load(f)
        super().__init__(root, transform, pre_transform, pre_filter)
    
    def get(self, idx):
        data = copy.deepcopy(self.mesh_common)
        zarr_data = xr.open_dataset(f"{self.simulation_names[idx]}.zarr")
        for data_var in zarr_data.data_vars:
            data[data_var] = torch.FloatTensor(zarr_data[data_var])
        
        scalers = get_scalers([data], scalers)
        # Create x, edge_attr, y
        dataset = create_data_attr([data], scalers=scalers, device=self.device, **self.dataset_parameters)
        temporal_dataset = to_temporal_dataset(dataset, **self.temporal_dataset_parameters)
        return temporal_dataset[0]

    def len(self):
        return len(self.simulation_names)


class DataModule(L.LightningDataModule):
    def __init__(self, temporal_train_dataset, temporal_val_dataset, 
                 batch_size: int = 8, train_dataloader_params: dict = {},
                 val_dataloader_params: dict = {}):
        super().__init__()
        self.batch_size = batch_size
        self.temporal_train_dataset = temporal_train_dataset
        self.temporal_val_dataset = temporal_val_dataset
        self.train_dataloader_params = train_dataloader_params
        self.val_dataloader_params = val_dataloader_params

    def train_dataloader(self):
        return DataLoader(self.temporal_train_dataset, batch_size=self.batch_size, 
                          **self.train_dataloader_params,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.temporal_val_dataset, batch_size=self.batch_size, 
                          **self.val_dataloader_params,
                          shuffle=False)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.temporal_train_dataset = self.temporal_train_dataset
            self.temporal_val_dataset = self.temporal_val_dataset

class CurriculumLearning(Callback):
    def __init__(self, max_rollout_steps, mode='epoch', patience=5, patience_buffer=1) -> None:
        '''Curriculum learning callback
        Increases the number of training rollout steps depending on the mode.
        mode: str, options:
            - 'epoch': increases the number of rollout steps every epoch
            - 'loss'  : increases the number of rollout steps when the training loss is below a threshold
            - 'plateau': increases the number of rollout steps when the training loss is not decreasing
        patience: int, number of epochs without improvement before increasing the number of rollout steps
        patience_buffer: int, number of epochs with small improvement that are not considered for the patience counter
        '''
        super().__init__()
        self.max_rollout_steps = max_rollout_steps
        self.mode = mode
        assert self.mode in ['epoch', 'loss', 'plateau'], "Invalid curriculum learning mode. Please choose between 'epoch', 'loss', or 'plateau'"
        self.patience = patience
        self.patience_counter = 0
        self.patience_buffer = patience_buffer
        self.patience_buffer_counter = 0

    def on_train_epoch_start(self, trainer, pl_module):
        if pl_module.curriculum_epoch == 0:
            rollout_steps = self.max_rollout_steps
        else:
            rollout_steps = trainer.current_epoch//pl_module.curriculum_epoch+1
        self._check_steps(pl_module, rollout_steps)

    def _check_steps(self, pl_module, rollout_steps):
        if rollout_steps > self.max_rollout_steps:
            rollout_steps = self.max_rollout_steps
        pl_module.rollout_steps = rollout_steps

class CurriculumBatchSizeFinder(BatchSizeFinder):
    def __init__(self, max_rollout_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_rollout_steps = max_rollout_steps

    def on_fit_start(self, trainer, pl_module):
        pl_module.rollout_steps = self.max_rollout_steps
        self.scale_batch_size(trainer, pl_module)
        pl_module.rollout_steps = 1