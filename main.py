import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
import torchviz
from torchviz import make_dot
from torch_geometric.utils import to_networkx
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from plotting_module import DesignExplorationPlotter
from advanced_analysis import AdvancedDesignAnalyzer


# Constants
# BOOM_PREDICTORS = ['Boom2', 'TAGEL', 'Alpha21264']
BOOM_PREDICTORS = [0, 1, 2]

# ============================
# CPU Architecture Graph Construction
# ============================
class CPUComponent:
    """Represents a component in the CPU architecture with its properties"""
    def __init__(self, component_id, type_name, features):
        self.id = component_id
        self.type = type_name  # e.g., 'FetchUnit', 'DecodeUnit', etc.
        self.features = features  # Component parameters
        self.idx = None  # Index in the graph
        
class CPUArchDAG:
    """Directed acyclic graph representation of CPU architecture"""
    def __init__(self):
        self.components = []
        self.connections = []  # list of (src_idx, tgt_idx)
        
    def add_component(self, component):
        component.idx = len(self.components)
        self.components.append(component)
        return component
        
    def add_connection(self, src_component, tgt_component):
        """Add a directed connection from source to target component"""
        if src_component.idx is None or tgt_component.idx is None:
            raise ValueError("Components must be added to the graph first")
            
        # Ensure we maintain DAG property (no cycles)
        if src_component.idx == tgt_component.idx:
            raise ValueError("Self-loops not allowed")
            
        # Add the connection
        self.connections.append((src_component.idx, tgt_component.idx))
        
    def to_pyg_data(self):
        """Convert to PyTorch Geometric Data object"""
    # 找出最大特征维度
        max_feature_dim = 0
        for component in self.components:
            max_feature_dim = max(max_feature_dim, len(component.features))
    
    # Node features with padding
        x = torch.zeros(len(self.components), max_feature_dim)
        for i, component in enumerate(self.components):
        # 复制现有特征
            feat_len = len(component.features)
            x[i, :feat_len] = torch.tensor(component.features, dtype=torch.float)
        # 其余位置保持为0（填充）
    
    # Edge index
        if not self.connections:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(self.connections, dtype=torch.long).t()
    
    # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)
    
        return data

def build_boom_cpu_graph(arch_params):
    """Build a DAG representation of BOOM CPU based on architecture parameters"""
    dag = CPUArchDAG()
    
    # Extract key parameters
    predictor_type = arch_params.get('predictor_type', 0)
    fetch_width = arch_params.get('fetchWidth', 4)
    decode_width = arch_params.get('decodeWidth', 4)
    num_rob_entries = arch_params.get('numRobEntries', 64)
    int_issue_width = arch_params.get('intIssueWidth', 2)
    mem_issue_width = arch_params.get('memIssueWidth', 1)
    
    # Create standardized feature vector encodings
    # Component type + numerical parameters
    
    # 1. Fetch Unit # 6
    fetch_defaults = [4, 8, 16, 8, 4, 4]
    fetch_features = [1, 0, 0, 0, 0, 0, 0, 0]  # One-hot encoding for component type
    fetch_features.extend(normalize_features([
        fetch_width, 
        arch_params.get('numFetchBufferEntries', 8),
        arch_params.get('numRasEntries', 16),
        arch_params.get('maxBrCount', 8),
        arch_params.get('nICacheWays', 4),
        arch_params.get('nICacheTLBWays', 4) 
    ], fetch_defaults))
    fetch_unit = dag.add_component(CPUComponent(0, 'FetchUnit', fetch_features))
    
    # 2. Branch Predictor - features depend on type # 3
    bp_features = [0, 1, 0, 0, 0, 0, 0, 0]
    # One-hot encoding for predictor type
    # if predictor_type == 'Boom2':
    if predictor_type == 0:
        bp_type_enc = [1, 0, 0]
    # elif predictor_type == 'TAGEL':
    elif predictor_type == 1:
        bp_type_enc = [0, 1, 0]
    else:  # Alpha21264
        bp_type_enc = [0, 0, 1]
    bp_features.extend(bp_type_enc)
    bp_defaults = [16, 8]
    bp_features.extend(normalize_features([
        arch_params.get('numRasEntries', 16),
        arch_params.get('maxBrCount', 8)
    ], bp_defaults))
    bp_unit = dag.add_component(CPUComponent(1, 'BranchPredictor', bp_features))
    
    # 3. Decode Unit # 1
    decode_defaults = [4]
    decode_features = [0, 0, 1, 0, 0, 0, 0, 0]
    decode_features.extend(normalize_features([
        decode_width
    ], decode_defaults))
    decode_unit = dag.add_component(CPUComponent(2, 'DecodeUnit', decode_features))
    
    # 4. Rename Unit # 1
    rename_defaults = [64]
    rename_features = [0, 0, 0, 1, 0, 0, 0, 0]
    rename_features.extend(normalize_features([
        arch_params.get('numIntPhysRegisters', 64)
    ], rename_defaults))
    rename_unit = dag.add_component(CPUComponent(3, 'RenameUnit', rename_features))
    
    # 5. Reorder Buffer # 3
    rob_defaults = [64, 4, 4]
    rob_features = [0, 0, 0, 0, 1, 0, 0, 0]
    rob_features.extend(normalize_features([
        num_rob_entries,
        arch_params.get('numRXQEntries', 4), 
        arch_params.get('numRCQEntries', 4)
    ], rob_defaults))
    rob_unit = dag.add_component(CPUComponent(4, 'ROB', rob_features))
    
    # 6. Integer Issue Unit # 1
    int_defaults = [2]
    int_features = [0, 0, 0, 0, 0, 1, 0, 0]
    int_features.extend(normalize_features([
        int_issue_width
    ], int_defaults))
    int_unit = dag.add_component(CPUComponent(5, 'IntIssueUnit', int_features))
    
    # 7. Memory Issue Unit # 8
    mem_defaults = [1, 12, 1, 1, 4, 4, 4, 512, 4]
    mem_features = [0, 0, 0, 0, 0, 0, 1, 0]
    mem_features.extend(normalize_features([
        mem_issue_width,
        arch_params.get('numLdqEntries', 12),
        arch_params.get('enablePrefetching', 0),
        arch_params.get('enableSFBOpt', 0),
        arch_params.get('nDCacheWays', 4),
        arch_params.get('nDCacheMSHRs', 4),
        arch_params.get('nDCacheTLBWays', 4), 
        arch_params.get('nL2TLBEntries', 512), 
        arch_params.get('nL2TLBWays', 4)
    ], mem_defaults))
    mem_unit = dag.add_component(CPUComponent(6, 'MemIssueUnit', mem_features))
    
    # 8. Commit Unit # 1
    commit_defaults = [4]
    commit_features = [0, 0, 0, 0, 0, 0, 0, 1]
    commit_features.extend(normalize_features([
        decode_width  # Commit width typically matches decode width
    ], commit_defaults))
    commit_unit = dag.add_component(CPUComponent(7, 'CommitUnit', commit_features))
    
    # Add connections to create the DAG
    # Fetch -> BP
    dag.add_connection(fetch_unit, bp_unit)
    # BP -> Fetch (feedback)
    dag.add_connection(bp_unit, fetch_unit)
    # Fetch -> Decode
    dag.add_connection(fetch_unit, decode_unit)
    # Decode -> Rename
    dag.add_connection(decode_unit, rename_unit)
    # Rename -> ROB
    dag.add_connection(rename_unit, rob_unit)
    # ROB -> Int Issue
    dag.add_connection(rob_unit, int_unit)
    # ROB -> Mem Issue
    dag.add_connection(rob_unit, mem_unit)
    # Int Issue -> Commit
    dag.add_connection(int_unit, commit_unit)
    # Mem Issue -> Commit
    dag.add_connection(mem_unit, commit_unit)
    # Commit -> ROB (feedback)
    dag.add_connection(commit_unit, rob_unit)
    
    return dag

def normalize_features(features, default_values):
    """Normalization by dividing features by their default values"""
    normalized = []
    
    for i, feat in enumerate(features):
        if isinstance(feat, bool):
            normalized.append(1.0 if feat else 0.0)
        else:
            # 使用默认值进行归一化，避免除以0
            default = default_values[i] if i < len(default_values) and default_values[i] != 0 else 1.0
            normalized.append(float(feat) / default)
            
    return normalized

# ============================
# Feature Extraction & Encoding
# ============================
def encode_boom_architecture(arch_params):
    """Encode BOOM architecture parameters as a feature vector"""
    # Predictor type one-hot encoding
    predictor_type = arch_params.get('predictor_type', 0)
    predictor_encoding = [0, 0, 0]  # [Boom2, TAGEL, Alpha21264]
    if predictor_type == 0:
        predictor_encoding[0] = 1
    elif predictor_type == 1:
        predictor_encoding[1] = 1
    else:  # Alpha21264
        predictor_encoding[2] = 1
    
    # Normalize numerical parameters
    discrete_params = [
    # numerical_params = [
        ('fetchWidth', 1), 
        ('numFetchBufferEntries', 5),
        ('numRasEntries', 2),
        ('maxBrCount', 3),
        ('decodeWidth', 4),
        ('numRobEntries', 4),
        ('numIntPhysRegisters', 4),
        ('memIssueWidth', 1),
        ('intIssueWidth', 4),
        ('numLdqEntries', 3),
        ('numRXQEntries', 2),
        ('numRCQEntries', 2),
        ('nL2TLBEntries', 3),
        ('nL2TLBWays', 2),
        ('nICacheWays', 2),
        ('nICacheTLBWays', 2),
        ('nDCacheWays', 2),
        ('nDCacheMSHRs', 2),
        ('nDCacheTLBWays', 2)
    ]
    numerical_encoding = []
    # for param, max_val in numerical_params:
    for param, max_val in discrete_params:
        if param in arch_params:
            # Normalize to [0,1]
            norm_value = float(arch_params[param]) / max_val
            numerical_encoding.append(norm_value)
        else:
            numerical_encoding.append(0.0)
    
    # Boolean parameters
    boolean_params = ['enablePrefetching', 'enableSFBOpt']
    boolean_encoding = [1.0 if arch_params.get(param, False) else 0.0 
                       for param in boolean_params]
    
    # Combine all features
    return predictor_encoding + numerical_encoding + boolean_encoding

def encode_genus_parameters(genus_params):
    """Encode GENUS synthesis parameters as a feature vector"""
    # Process switch-value pairs
    switch_value_pairs = [
        ('enable_latch_max_borrow', 'latch_max_borrow', 499),
        ('enable_max_fanout', 'max_fanout', 62),
        ('enable_lp_clock_gating_max_flops', 'lp_clock_gating_max_flops', 2),
        ('enable_lp_power_optimization_weight', 'lp_power_optimization_weight', 8)
    ]
    
    switch_value_encoding = []
    for switch, value, max_val in switch_value_pairs:
        if switch in genus_params and genus_params[switch]:
            switch_value_encoding.append(1.0)  # Switch is on
            if value in genus_params:
                # Normalize value
                norm_value = float(genus_params[value]) / max_val
                switch_value_encoding.append(norm_value)
            else:
                switch_value_encoding.append(0.5)  # Default value
        else:
            switch_value_encoding.append(0.0)  # Switch is off
            switch_value_encoding.append(0.0)  # Value is irrelevant
    
    # Encode effort levels
    effort_params = {
        # 'syn_generic_effort': ['low', 'medium', 'high'],
        # 'syn_map_effort': ['low', 'medium', 'high'],
        # 'syn_opt_effort': ['low', 'medium', 'high'],
        # 'leakage_power_effort': ['low', 'medium', 'high'],
        # 'lp_power_analysis_effort': ['low', 'medium', 'high'],
        # 'retime_effort_level': ['low', 'medium', 'high']
        'syn_generic_effort': [0, 1, 2],
        'syn_map_effort': [0, 1, 2],
        'syn_opt_effort': [0, 1, 2],
        'leakage_power_effort': [0, 1, 2],
        'lp_power_analysis_effort': [0, 1, 2],
        'iopt_lp_power_analysis_effort': [0, 1, 2],
        'retime_effort_level': [0, 1, 2],
        'dp_analytical_opt': [0, 1, 2],
        'dp_rewriting': [0, 1, 2],
        'dp_sharing': [0, 1, 2],
    }
    ###########################
    num_value_pairs = [
        ('max_dynamic_power', 1470),
        ('max_leakage_power', 950),
    ]
    num_value_encoding = []
    for value, max_val in num_value_pairs:
        if value in genus_params:
            # Normalize value
            norm_value = float(genus_params[value]) / max_val
            num_value_encoding.append(norm_value)
    ###########################
    effort_encoding = []
    for param, values in effort_params.items():
        if param in genus_params:
            # One-hot encoding
            one_hot = [1.0 if genus_params[param] == val else 0.0 for val in values]
        else:
            # Default: medium
            one_hot = [0.0, 1.0, 0.0]
        effort_encoding.extend(one_hot)
    
    # Boolean parameters
    boolean_params = {
        'time_recovery_arcs':[0, 1], 
        'auto_partition':[0, 1], 
        'dp_area_mode':[0, 1], 
        'dp_csa':[0, 1], 
        'exact_match_seq_async_ctrls':[0, 1],  
        'exact_match_seq_sync_ctrls':[0, 1], 
        'iopt_enable_floating_output_check':[0, 1], 
        'iopt_force_constant_removal':[0, 1], 
        'lbr_seq_in_out_phase_opto':[0, 1], 
        'optimize_net_area':[0, 1], 
        'retime_optimize_reset':[0, 1]
    }
    
    # boolean_encoding = [1.0 if genus_params.get(param, False) else 0.0 
    #                    for param in boolean_params]
    boolean_encoding = []
    for param, values in boolean_params.items():
        if param in genus_params:
            # One-hot encoding
            one_hot = [1.0 if genus_params[param] == val else 0.0 for val in values]
        else:
            # Default: medium
            one_hot = [0.0, 1.0]
        effort_encoding.extend(one_hot)
    
    # Return combined encoding
    return switch_value_encoding + effort_encoding + boolean_encoding + num_value_encoding

# ============================
# Neural Network Components
# ============================
class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism to fuse architecture and GENUS parameter representations"""
    def __init__(self, arch_dim, param_dim, output_dim):
        super(CrossAttentionFusion, self).__init__()
        
        # Projection layers
        self.arch_proj = nn.Linear(arch_dim, output_dim)
        self.param_proj = nn.Linear(param_dim, output_dim)
        
        # Cross-attention mechanism
        self.arch_to_param_attn = nn.Linear(output_dim * 2, 1)
        self.param_to_arch_attn = nn.Linear(output_dim * 2, 1)
        
        # Output projection
        self.output_proj = nn.Linear(output_dim * 2, output_dim)
        
        # Layernorm for stability
        self.layernorm = nn.LayerNorm(output_dim)
    def forward(self, arch_embedding, param_embedding):
        # Project to common dimension
        arch_proj = self.arch_proj(arch_embedding)
        param_proj = self.param_proj(param_embedding)
    
    # 检查张量维度
        if len(arch_proj.shape) == 1:
            # 如果是一维张量，添加批次维度
            arch_proj = arch_proj.unsqueeze(0)
            param_proj = param_proj.unsqueeze(0)
        
            # 现在张量的形状是 [1, feature_dim]，可以在维度 1 上连接
            arch_to_param_input = torch.cat([arch_proj, param_proj], dim=1)
            param_to_arch_input = torch.cat([param_proj, arch_proj], dim=1)
        else:
        # 如果已经是多维张量，则按原计划进行
            arch_to_param_input = torch.cat([arch_proj, param_proj], dim=1)
            param_to_arch_input = torch.cat([param_proj, arch_proj], dim=1)
    
    # 继续后面的代码...
        arch_to_param_score = torch.sigmoid(self.arch_to_param_attn(arch_to_param_input))
        param_to_arch_score = torch.sigmoid(self.param_to_arch_attn(param_to_arch_input))
    
    # Apply attention weights
        arch_weighted = arch_proj * param_to_arch_score
        param_weighted = param_proj * arch_to_param_score
    
    # 这里的连接也需要考虑维度
        if len(arch_weighted.shape) == 1:
        # 如果是一维张量，先变成二维再连接
            fusion = torch.cat([arch_weighted.unsqueeze(0), param_weighted.unsqueeze(0)], dim=1)
        else:
            fusion = torch.cat([arch_weighted, param_weighted], dim=1)
    
        output = self.output_proj(fusion)
    
    # Residual connection and layernorm
        output = self.layernorm(output + arch_proj + param_proj)
    
    # 如果需要，移除批次维度
        if len(arch_embedding.shape) == 1:
            output = output.squeeze(0)
    
        return output        


class CPUArchGAT(nn.Module):
    """Graph Attention Network for CPU architecture representation"""
    def __init__(self, in_channels, hidden_channels=64, out_channels=64, num_layers=2):
        super(CPUArchGAT, self).__init__()
        
        self.initial_proj = nn.Linear(in_channels, hidden_channels)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False))
        for _ in range(num_layers-1):
            self.gat_layers.append(GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False))
            
        # Final projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index):
        # Initial projection
        h = F.leaky_relu(self.initial_proj(x))
        
        # Graph attention layers
        for gat in self.gat_layers:
            h = gat(h, edge_index)
            h = F.leaky_relu(h)
            h = self.dropout(h)
            
        # Final projection
        out = self.output_proj(h)
        
        # Global pooling (mean of all nodes)
        graph_embedding = out.mean(dim=0)
        
        return graph_embedding

class GENUSParameterEncoder(nn.Module):
    """Encoder for GENUS synthesis parameters"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GENUSParameterEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class PPAPredictor(nn.Module):
    """Multi-task predictor for Performance, Power, and Area"""
    def __init__(self, input_dim, hidden_dim):
        super(PPAPredictor, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # Task-specific heads
        self.perf_head = nn.Linear(hidden_dim, 1)
        self.power_head = nn.Linear(hidden_dim, 1)
        self.area_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        shared_features = self.shared(x)
        
        perf = self.perf_head(shared_features)
        power = self.power_head(shared_features)
        area = self.area_head(shared_features)
        
        return {
            'performance': perf,
            'power': power,
            'area': area
        }

#    arch_model = BOOMMetaGAT(arch_feature_dim=arch_feature_dim, genus_feature_dim=genus_feature_dim, hidden_dim=64, output_dim=64)
class BOOMMetaGAT(nn.Module):
    """Complete Meta-GAT model for CPU architecture and GENUS parameter optimization"""
    def __init__(self, arch_feature_dim, genus_feature_dim, hidden_dim=64, output_dim=64):
        super(BOOMMetaGAT, self).__init__()
        
        # CPU architecture encoder (graph-based)
        self.cpu_encoder = CPUArchGAT(
            in_channels=arch_feature_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim
        )        
        # GENUS parameter encoder
        self.genus_encoder = GENUSParameterEncoder(
            input_dim=genus_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            arch_dim=output_dim,
            param_dim=output_dim,
            output_dim=output_dim
        )
        
        # PPA predictor
        self.ppa_predictor = PPAPredictor(
            input_dim=output_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, arch_data, genus_params, use_graph=True):
        # Use graph-based architecture encoding
        arch_x, arch_edge_index = arch_data.x, arch_data.edge_index
        arch_embedding = self.cpu_encoder(arch_x, arch_edge_index)
        # Encode GENUS parameters
        genus_embedding = self.genus_encoder(genus_params)
        
        # Fuse representations
        fused_embedding = self.fusion(arch_embedding, genus_embedding)
        # print(f"Fused embedding shape: {fused_embedding.shape}")
        
        # Predict PPA metrics
        ppa_predictions = self.ppa_predictor(fused_embedding)
        
        return ppa_predictions

# ============================
# Meta-Learning Framework
# ============================
class ImprovedBOOMMetaLearner:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, first_order=False):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
        self.first_order = first_order  # Option for first-order approximation
        
    def adapt_to_new_architecture(self, support_set, num_steps=5):
        """Adapt to a new architecture with few-shot learning"""
        device = next(self.model.parameters()).device
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Inner-loop adaptation
        for step in range(num_steps):
            support_loss = 0
            for arch_data, genus_params, target_ppa in support_set:
                # Move data to device
                arch_data = arch_data.to(device)
                genus_params = genus_params.to(device)
                target_ppa = {k: v.to(device) for k, v in target_ppa.items()}
                
                # Forward pass
                pred_ppa = self.model(arch_data, genus_params)
                
                # Compute loss
                loss = self.compute_ppa_loss(pred_ppa, target_ppa)
                support_loss += loss / len(support_set)
            
            # Compute gradients and update parameters manually
            self.model.zero_grad()
            support_loss.backward(retain_graph=not self.first_order)
            
            # Manual parameter update
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.data = param.data - self.inner_lr * param.grad
            
            print(f"Inner-loop step {step+1}/{num_steps}, loss: {support_loss.item():.4f}")
        
        # Return the adapted model (which is just the updated self.model)
        # And restore the original parameters for the meta-model
        adapted_model = copy.deepcopy(self.model)
        
        # Restore original parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(original_params[name])
        
        return adapted_model
        
    def meta_train_step(self, task_batch, num_inner_steps=3):
        """Perform meta-training step with proper bi-level optimization"""
        device = next(self.model.parameters()).device
        meta_loss = 0
        
        self.outer_optimizer.zero_grad()
        
        for task_idx, task in enumerate(task_batch):
            # Split into support and query sets
            support_set, query_set = task
            
            # Store original parameters
            original_params = {name: param.clone() for name, param in self.model.named_parameters()}
            
            # Inner-loop optimization
            for step in range(num_inner_steps):
                support_loss = 0
                for arch_data, genus_params, target_ppa in support_set:
                    # Move data to device
                    arch_data = arch_data.to(device)
                    genus_params = genus_params.to(device)
                    target_ppa = {k: v.to(device) for k, v in target_ppa.items()}
                    
                    # Forward pass
                    pred_ppa = self.model(arch_data, genus_params)
                    
                    # Compute loss
                    loss = self.compute_ppa_loss(pred_ppa, target_ppa)
                    support_loss += loss / len(support_set)
                
                # Update parameters manually
                self.model.zero_grad()
                grads = torch.autograd.grad(support_loss, self.model.parameters(), 
                                           create_graph=not self.first_order, 
                                           allow_unused=True)
                
                # Manual update
                with torch.no_grad():
                    for param, grad in zip(self.model.parameters(), grads):
                        if grad is not None:
                            param.data = param.data - self.inner_lr * grad
            
            # Evaluate on query set with adapted parameters
            query_loss = 0
            for arch_data, genus_params, target_ppa in query_set:
                # Move data to device
                arch_data = arch_data.to(device)
                genus_params = genus_params.to(device)
                target_ppa = {k: v.to(device) for k, v in target_ppa.items()}
                
                # Forward pass with adapted model
                pred_ppa = self.model(arch_data, genus_params)
                
                # Compute loss
                loss = self.compute_ppa_loss(pred_ppa, target_ppa)
                query_loss += loss / len(query_set)
            
            # Add to meta-loss
            meta_loss += query_loss / len(task_batch)
            
            # Restore original parameters for next task
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data.copy_(original_params[name])
        
        # Meta-update using meta-loss
        meta_loss.backward()
        self.outer_optimizer.step()
        
        print(f"Meta-train step, meta loss: {meta_loss.item():.4f}")
        return meta_loss.item()
    
    def compute_ppa_loss(self, pred_ppa, target_ppa):
        """Compute weighted MSE loss for PPA predictions"""
        perf_loss = F.mse_loss(pred_ppa['performance'], target_ppa['performance'])
        power_loss = F.mse_loss(pred_ppa['power'], target_ppa['power'])
        area_loss = F.mse_loss(pred_ppa['area'], target_ppa['area'])
        
        # Weighted sum (can adjust weights based on importance)
        total_loss = perf_loss + power_loss + area_loss
        
        return total_loss

# ============================
# Reference Point Adaptation
# ============================
class ReferencePointManager:
    def __init__(self, exploration_factor=0.1, decay_rate=0.95):
        self.exploration_factor = exploration_factor
        self.decay_rate = decay_rate
        self.pareto_history = []
        
    def get_reference_point(self, pareto_front):
        """Compute adaptive reference point for hypervolume calculation"""
        if not pareto_front:
            # Default reference point if no Pareto front exists yet
            return np.array([10.0, 10.0, 10.0])  # Assumes minimization
        
        # Extract PPA values based on the structure of Pareto front elements
        front_array = []
        for p in pareto_front:
            if isinstance(p[1], dict):
                # Dictionary format
                front_array.append([
                    p[1]['performance'].item(), 
                    p[1]['power'].item(), 
                    p[1]['area'].item()
                ])
            elif isinstance(p[1], tuple):
                # Tuple format
                front_array.append([p[1][0], p[1][1], p[1][2]])
            else:
                print(f"Warning: Unknown PPA structure in reference point: {type(p[1])}")
        
        front_array = np.array(front_array)
        
        # Store current front in history
        self.pareto_history.append(front_array)
        if len(self.pareto_history) > 10:
            self.pareto_history.pop(0)  # Keep only last 10 fronts
        
        # Compute ideal and nadir points
        ideal_point = np.min(front_array, axis=0)
        nadir_point = np.max(front_array, axis=0)
        
        # Apply user preferences if provided
        preference_vector = np.ones(3)  # Default: equal weights
        
        # Apply short-term adaptation based on front shape
        if len(self.pareto_history) >= 2:
            # Calculate front movement direction
            prev_front = self.pareto_history[-2]
            prev_front_array = prev_front
            
            # Simple movement calculation (this could be more sophisticated)
            prev_nadir = np.max(prev_front_array, axis=0)
            movement = nadir_point - prev_nadir
            
            # Scale preference vector based on movement
            movement_factor = 1.0 + 0.5 * np.abs(movement) / (nadir_point - ideal_point + 1e-10)
            preference_vector = preference_vector * movement_factor
        
        # Decay exploration factor over time
        self.exploration_factor *= self.decay_rate
        
        # Compute final reference point
        offset = (nadir_point - ideal_point) * preference_vector * (1.0 + self.exploration_factor)
        reference_point = nadir_point + offset
        
        return reference_point
    def get_reference_point_new(self, pareto_front):
        """Compute adaptive reference point for hypervolume calculation"""
        if not pareto_front:
            # Default reference point if no Pareto front exists yet
            return np.array([1.0, 1.0, 1.0])  # Assumes minimization
        
        # Extract PPA values
        front_array = []
        for p in pareto_front:
            if isinstance(p[1], dict):
                # Dictionary format
                front_array.append([
                    p[1]['performance'].item(), 
                    p[1]['power'].item(), 
                    p[1]['area'].item()
                ])
            elif isinstance(p[1], tuple):
                # Tuple format
                front_array.append([p[1][0], p[1][1], p[1][2]])
            else:
                print(f"Warning: Unknown PPA structure in reference point: {type(p[1])}")
        
        front_array = np.array(front_array)
        
        # Compute maximum values in each dimension
        nadir_point = np.max(front_array, axis=0)
        
        # Add a margin to create the reference point (e.g., 10% more than max)
        reference_point = nadir_point * 1.1
        
        return reference_point
# ============================
# Design Space Exploration
# ============================
class DesignSpaceExplorer:
    def __init__(self, model, exploration_data, target_point=None):
        """
        Initialize the Design Space Explorer with a model and exploration dataset
        
        Args:
            model: The trained neural network model
            arch_param_space: Architecture parameter space
            genus_param_space: GENUS parameter space
            exploration_data: Dataset to explore (arch_A_data[51:1927])
        """
        self.model = model
        self.exploration_data = exploration_data  # The offline dataset to explore
        self.pareto_front = []
        self.explored_designs = []
        self.sampled_indices = set()  # Keep track of indices already sampled
        self.sampled_designs = []  # Track designs in order of sampling
        self.sampled_design_indices = []  # Track indices of sampled designs
        self.num_mc_samples = 24
        self.exploration_weight = 0.5        
        # Reference point manager
        # self.reference_manager = ReferencePointManager()
        self.reference_point = [1.0, 1.0, 1.0]
        #  target_point = (0.3, 0.4, 0.5)
        self.target_point = target_point
        self.dominating_points = []
        

        
        # Store original tasks for meta-learning
        self.original_target_tasks = None
        self.plotter = DesignExplorationPlotter()
        self.analyzer = AdvancedDesignAnalyzer()


    def enable_dropout(self, model):
        """Enable dropout layers during inference"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Set dropout layers to training mode
        return model
    def predict_with_uncertainty(self, cpu_graph, genus_features):
        """Predict PPA values with uncertainty estimates using MC dropout"""
        # Enable dropout for inference
        self.enable_dropout(self.model)
        
        # Store predictions from multiple forward passes
        perf_preds = []
        power_preds = []
        area_preds = []
        
        device = next(self.model.parameters()).device
        cpu_graph = cpu_graph.to(device)
        genus_features = genus_features.to(device)
        
        # Perform multiple forward passes
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                pred_ppa = self.model(cpu_graph, genus_features)
                
                perf_preds.append(pred_ppa['performance'].item())
                power_preds.append(pred_ppa['power'].item())
                area_preds.append(pred_ppa['area'].item())
        
        # Calculate mean and variance for each objective
        perf_mean = np.mean(perf_preds)
        perf_var = np.var(perf_preds)
        # print("perf_mean", perf_mean)
        # print("sqrt_perf_var", np.sqrt(perf_var))
        
        power_mean = np.mean(power_preds)
        power_var = np.var(power_preds)
        # print("power_mean", power_mean)
        # print("sqrt_power_var", np.sqrt(power_var))  

        area_mean = np.mean(area_preds)
        area_var = np.var(area_preds)
        # print("area_mean", area_mean)
        # print("sqrt_area_var", np.sqrt(area_var))
        
        # Return both predictions and uncertainties
        means = {
            'performance': perf_mean,
            'power': power_mean,
            'area': area_mean
        }
        
        uncertainties = {
            'performance': perf_var,
            'power': power_var,
            'area': area_var
        }
        
        return means, uncertainties        
    def sample_from_dataset(self):
        """
        Sample a design point from the offline dataset that maximizes potential 
        improvement to the current Pareto front based on model predictions
        """
        available_indices = [i for i in range(len(self.exploration_data)) 
                           if i not in self.sampled_indices]
        
        if not available_indices:
            print("All designs in the exploration dataset have been sampled!")
            return None
        
        # Get current Pareto front for evaluation
        current_ppa_values = []
        
        for p in self.pareto_front:
            if isinstance(p[1], dict):
                # Dictionary format
                current_ppa_values.append([
                    p[1]['performance'].item(), 
                    p[1]['power'].item(), 
                    p[1]['area'].item()
                ])
            elif isinstance(p[1], tuple):
                # Tuple format
                current_ppa_values.append([p[1][0], p[1][1], p[1][2]])
        
        current_ppa_values = np.array(current_ppa_values) if current_ppa_values else np.empty((0, 3))
        
        # Calculate hypervolume of current front
        current_hv = self.calculate_hypervolume(current_ppa_values, self.reference_point)
        
        # Find point with maximum expected hypervolume contribution
        best_idx = None
        max_contribution = -1.0
        best_ppa = None
        
        device = next(self.model.parameters()).device
        
        # Evaluate each candidate
        print(f"Evaluating {len(available_indices)} candidate designs...")
        
        # Process in batches to speed up evaluation
        batch_size = 8
        for i in range(0, len(available_indices), batch_size):
            batch_indices = available_indices[i:i+batch_size]
            batch_data = [self.exploration_data[idx] for idx in batch_indices]
            
            # Extract batch components - already in correct format
            batch_cpu_graphs = [data[0] for data in batch_data]
            batch_genus_features = [data[1] for data in batch_data]
            
            # Evaluate each design in the batch
            with torch.no_grad():
                for idx, cpu_graph, genus_features in zip(batch_indices, batch_cpu_graphs, batch_genus_features):
                    # Move to device
                    cpu_graph = cpu_graph.to(device)
                    genus_features = genus_features.to(device)
                    
                    # Get model prediction
                    # pred_ppa = self.model(cpu_graph, genus_features)
                    mean_ppa, uncertainty_ppa = self.predict_with_uncertainty(cpu_graph, genus_features)
                    total_uncertainty = (
                        uncertainty_ppa['performance'] + 
                        uncertainty_ppa['power'] + 
                        uncertainty_ppa['area']
                    )                    
                    # Extract PPA values
                    pred_perf = mean_ppa['performance'].item() 
                    pred_power = mean_ppa['power'].item()
                    pred_area = mean_ppa['area'].item()
                    uncertainty__perf = np.sqrt(uncertainty_ppa['performance'])
                    uncertainty_power = np.sqrt(uncertainty_ppa['power'])
                    uncertainty_area = np.sqrt(uncertainty_ppa['area'])
                    # print("pre = ", pred_perf, pred_power, pred_area)
                    # print("real = ")   
                    # print("unc", uncertainty__perf, uncertainty_power, uncertainty_area)                   
                    pred_perf = mean_ppa['performance'] - self.exploration_weight * np.sqrt(uncertainty_ppa['performance'])
                    pred_power = mean_ppa['power'] - self.exploration_weight * np.sqrt(uncertainty_ppa['power'])
                    pred_area = mean_ppa['area']  - self.exploration_weight * np.sqrt(uncertainty_ppa['area'])                       
                    # Extract PPA values
                    # pred_perf = pred_ppa['performance'].item()
                    # pred_power = pred_ppa['power'].item()
                    # pred_area = pred_ppa['area'].item()
                    
                    # Calculate hypervolume contribution
                    temp_front = np.vstack([
                        current_ppa_values,
                        [pred_perf, pred_power, pred_area]
                    ]) if current_ppa_values.size > 0 else np.array([[
                        pred_perf, pred_power, pred_area
                    ]])
                    
                    new_hv = self.calculate_hypervolume(temp_front, self.reference_point)
                    contribution = new_hv - current_hv
                    
                    if contribution > max_contribution:
                        max_contribution = contribution
                        best_idx = idx
                        pred_PPA = [pred_perf, pred_power, pred_area]
                        # print("pred_PPA = ", pred_PPA)
                        best_ppa = self.exploration_data[idx][2]  # Real PPA values
                        # print("best_ppa = ", [best_ppa['performance'].item(), best_ppa['power'].item(), best_ppa['area'].item()])
                        # print("contribution = ",contribution)
        
        if best_idx is not None:
            print(f"Selected design {best_idx} with expected hypervolume contribution: {max_contribution:.4f}")
            self.sampled_indices.add(best_idx)
            # print("best_idx = ", best_idx)
            # print("pred_PPA = ", pred_PPA)
            # print("best_ppa = ", best_ppa)
            # Return the data in its original format - no conversion needed
            return self.exploration_data[best_idx], best_idx
        else:
            # print("Failed to find a design with positive hypervolume contribution!")
            # Fallback to random selection if no improvement found
            idx = random.choice(available_indices)
            self.sampled_indices.add(idx)
            return self.exploration_data[idx], idx
    
    def initialize_pareto_front(self, initial_data):
        """Initialize Pareto front from initial data points"""
        initial_pareto_front = []
        
        for cpu_graph, genus_features, ppa in initial_data:
            # Extract architecture parameters
            
            # Design tuple
            design = (cpu_graph, genus_features)
            
            # Check if this design is dominated by any existing design in the front
            dominated = False
            for i, (existing_design, existing_ppa) in enumerate(initial_pareto_front):
                # If existing design dominates this design
                if (existing_ppa['performance'] <= ppa['performance'] and
                    existing_ppa['power'] <= ppa['power'] and
                    existing_ppa['area'] <= ppa['area'] and
                    (existing_ppa['performance'] < ppa['performance'] or
                    existing_ppa['power'] < ppa['power'] or
                    existing_ppa['area'] < ppa['area'])):
                    dominated = True
                    break
                
                # If this design dominates an existing design
                if (ppa['performance'] <= existing_ppa['performance'] and
                    ppa['power'] <= existing_ppa['power'] and
                    ppa['area'] <= existing_ppa['area'] and
                    (ppa['performance'] < existing_ppa['performance'] or
                    ppa['power'] < existing_ppa['power'] or
                    ppa['area'] < existing_ppa['area'])):
                    initial_pareto_front[i] = None  # Mark for removal
            
            # Add if not dominated
            if not dominated:
                initial_pareto_front.append((design, ppa))
            
            # Remove dominated points
            initial_pareto_front = [p for p in initial_pareto_front if p is not None]
        
        return initial_pareto_front
    def run_exploration(self, num_iterations=100, update_interval=5, initial_data=None):
        """
        Run design space exploration for the specified number of iterations
        
        Args:
            num_iterations: Total number of iterations to run
            update_interval: Number of iterations after which to update the model
            initial_data: Initial data points to use for Pareto front
        """
        # Initialize with designs from initial_data if provided
        if initial_data:
            print(f"Initializing with {len(initial_data)} initial designs...")
            
            # Add initial designs to explored_designs
            for cpu_graph, genus_features, ppa in initial_data:
                design = (cpu_graph, genus_features)
                self.explored_designs.append((design, ppa))
            
            # Initialize Pareto front from initial designs
            self.pareto_front = self.initialize_pareto_front(initial_data)
            
            # Calculate and print initial hypervolume
            if self.pareto_front:
                ppa_values = []
                for p in self.pareto_front:
                    if isinstance(p[1], dict):
                        ppa_values.append([
                            p[1]['performance'].item(), 
                            p[1]['power'].item(), 
                            p[1]['area'].item()
                        ])
                    elif isinstance(p[1], tuple):
                        ppa_values.append([p[1][0], p[1][1], p[1][2]])
                
                ppa_values = np.array(ppa_values)
                initial_hv = self.calculate_hypervolume(ppa_values, self.reference_point)
                print(f"Initial hypervolume: {initial_hv:.6f}")
            
            print(f"Initial Pareto front contains {len(self.pareto_front)} designs")
        
        # Continue with the regular exploration
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration+1}/{num_iterations}")
            
            # Sample a design from the dataset
            sample = self.sample_from_dataset()
            if sample is None:
                print("Exploration stopped: no more designs available")
                break
                
            design_data, design_idx = sample
            cpu_graph, genus_features, real_ppa = design_data
            
            # Update with real PPA values
            self.update_with_real_ppa((cpu_graph, genus_features), real_ppa)
            
            # Calculate and print current hypervolume
            if self.pareto_front:
                ppa_values = []
                for p in self.pareto_front:
                    if isinstance(p[1], dict):
                        ppa_values.append([
                            p[1]['performance'].item(), 
                            p[1]['power'].item(), 
                            p[1]['area'].item()
                        ])
                    elif isinstance(p[1], tuple):
                        ppa_values.append([p[1][0], p[1][1], p[1][2]])
                
                ppa_values = np.array(ppa_values)
                current_hv = self.calculate_hypervolume(ppa_values, self.reference_point)
                print(f"Current hypervolume: {current_hv:.6f}")
            
            print(f"Current Pareto front contains {len(self.pareto_front)} designs")
            
            # Update model every update_interval iterations
            if (iteration + 1) % update_interval == 0:
                print(f"Updating model at iteration {iteration+1}")
                self.update_model()
        
        # Final visualization
        print("\nExploration complete! Final Pareto front:")
        if self.target_point is not None:
            # Extract PPA values from sampled designs
            sampled_perf, sampled_power, sampled_area = self.plotter.extract_ppa_values(self.sampled_designs)
            
            # Find designs that dominate the target point
            self.dominating_points = self.plotter.identify_dominating_points(
                sampled_perf, sampled_power, sampled_area, self.target_point
            )
            
            if self.dominating_points:
                print(f"Found {len(self.dominating_points)} designs dominating the target point")
                for idx in self.dominating_points:
                    print(f"  Design {idx+1} dominates the target")       
        # Print final Pareto front designs
        for i, (design, ppa) in enumerate(self.pareto_front):
            # Print PPA based on its structure
            if isinstance(ppa, dict):
                print(f"Pareto Design {i+1}: Performance: {ppa['performance'].item():.4f}, Power: {ppa['power'].item():.4f}, Area: {ppa['area'].item():.4f}")
            elif isinstance(ppa, tuple):
                print(f"Pareto Design {i+1}: Performance: {ppa[0]:.4f}, Power: {ppa[1]:.4f}, Area: {ppa[2]:.4f}")

            # Generate visualizations
        print("\nGenerating visualizations...")
        self.plotter.plot_design_space(self.exploration_data, self.sampled_designs, file_prefix="dse_exploration", target_point=self.target_point,dominated_points=self.dominating_points)
        
        # Generate advanced analysis
        print("\nGenerating advanced analysis plots...")
        self.analyzer.analyze_exploration_progress(self.sampled_designs, file_prefix="dse_advanced")
        self.analyzer.create_summary_table(self.sampled_designs, file_prefix="dse_summary")
        
        # Print all sampled designs in order
        self.plotter.print_sampled_designs(self.sampled_designs, self.sampled_design_indices)
                       
        return self.pareto_front

    def update_with_real_ppa(self, design, real_ppa):
        """Update model and Pareto front with real PPA values"""
        # Store design and PPA
        self.explored_designs.append((design, real_ppa))
        self.sampled_designs.append((design, real_ppa))
        
        # Update Pareto front
        self.update_pareto_front()

    def update_model(self, meta_lr=0.0005, inner_lr=0.01, 
                    num_inner_steps=3, num_outer_steps=5, 
                    first_order=True):
        """Update the model using the collected designs with proper meta-learning approach"""
        if len(self.explored_designs) < 5:
            print("Not enough data to update model yet (minimum 5 designs required)")
            return
            
        print(f"Updating model with {len(self.explored_designs)} designs...")
        
        # Create training data from explored designs
        train_data = [(design[0], design[1], ppa) for design, ppa in self.explored_designs]
        
        # Get device and set model to training mode
        device = next(self.model.parameters()).device
        self.model.train()
        
        # Create optimizer for meta-updates
        meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        
        # Create validation set
        val_size = max(2, int(len(train_data) * 0.1))  # 10% for validation
        random.shuffle(train_data)
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]
        
        # Track best model
        best_val_loss = float('inf')
        best_model_params = None
        patience_counter = 0
        patience = 3  # Early stopping patience
        
        print(f"Meta-training with {len(train_data)} designs, validating with {len(val_data)} designs")
        
        # Outer loop (meta-training steps)
        for outer_step in range(num_outer_steps):
            meta_loss = 0.0
            
            # Create mini-batches for tasks
            batch_size = min(16, len(train_data))
            num_batches = max(1, len(train_data) // batch_size)
            
            # Process mini-batches
            for batch_idx in range(num_batches):
                # Get batch data
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_data))
                batch = train_data[start_idx:end_idx]
                
                # Split into support/query sets (80/20 split)
                split_idx = int(len(batch) * 0.8)
                support_set = batch[:split_idx]
                query_set = batch[split_idx:]
                
                if len(support_set) == 0 or len(query_set) == 0:
                    continue
                
                # Store original parameters
                original_params = {name: param.clone() for name, param in self.model.named_parameters()}
                
                # Inner loop adaptation
                for inner_step in range(num_inner_steps):
                    support_loss = 0.0
                    
                    # Process support set
                    for cpu_graph, genus_features, target_ppa in support_set:
                        # Move data to device
                        cpu_graph = cpu_graph.to(device)
                        genus_features = genus_features.to(device)
                        
                        # Handle different types of target_ppa
                        if isinstance(target_ppa, dict):
                            target_ppa = {k: v.to(device) if isinstance(v, torch.Tensor) 
                                        else torch.tensor([v], dtype=torch.float32, device=device) 
                                        for k, v in target_ppa.items()}
                        
                        # Forward pass
                        pred_ppa = self.model(cpu_graph, genus_features)
                        
                        # Compute loss
                        perf_loss = torch.nn.functional.mse_loss(pred_ppa['performance'], target_ppa['performance'])
                        power_loss = torch.nn.functional.mse_loss(pred_ppa['power'], target_ppa['power'])
                        area_loss = torch.nn.functional.mse_loss(pred_ppa['area'], target_ppa['area'])
                        loss = perf_loss + power_loss + area_loss
                        
                        # Add to batch loss
                        support_loss += loss / len(support_set)
                    
                    # Update model parameters using gradients
                    self.model.zero_grad()
                    
                    # Different handling based on first_order approximation
                    if first_order:
                        # First-order approximation (faster but less accurate)
                        support_loss.backward()
                        
                        # Manual parameter update
                        with torch.no_grad():
                            for param in self.model.parameters():
                                if param.grad is not None:
                                    param.data.sub_(inner_lr * param.grad)
                    else:
                        # Second-order MAML (slower but more accurate)
                        grads = torch.autograd.grad(
                            support_loss, 
                            self.model.parameters(),
                            create_graph=True, 
                            allow_unused=True
                        )
                        
                        # Manual parameter update
                        with torch.no_grad():
                            for param, grad in zip(self.model.parameters(), grads):
                                if grad is not None:
                                    param.data.sub_(inner_lr * grad)
                
                # Evaluate on query set with adapted model
                query_loss = 0.0
                for cpu_graph, genus_features, target_ppa in query_set:
                    # Move data to device
                    cpu_graph = cpu_graph.to(device)
                    genus_features = genus_features.to(device)
                    
                    # Handle different types of target_ppa
                    if isinstance(target_ppa, dict):
                        target_ppa = {k: v.to(device) if isinstance(v, torch.Tensor) 
                                    else torch.tensor([v], dtype=torch.float32, device=device) 
                                    for k, v in target_ppa.items()}
                    
                    # Forward pass with adapted model
                    pred_ppa = self.model(cpu_graph, genus_features)
                    
                    # Compute loss
                    perf_loss = torch.nn.functional.mse_loss(pred_ppa['performance'], target_ppa['performance'])
                    power_loss = torch.nn.functional.mse_loss(pred_ppa['power'], target_ppa['power'])
                    area_loss = torch.nn.functional.mse_loss(pred_ppa['area'], target_ppa['area'])
                    loss = perf_loss + power_loss + area_loss
                    
                    # Add to query loss
                    query_loss += loss / len(query_set)
                
                # Restore original parameters for meta-update
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        param.data.copy_(original_params[name])
                
                # Accumulate meta-loss
                meta_loss += query_loss / num_batches
            
            # Meta-update
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
            
            # Validate model
            val_loss = self._evaluate_model(val_data, device)
            # print(f"  Outer step {outer_step+1}/{num_outer_steps}, Meta loss: {meta_loss.item():.6f}, Val loss: {val_loss:.6f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_params = {name: param.clone() for name, param in self.model.named_parameters()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("  Early stopping triggered!")
                    break
        
        # Restore best model parameters
        if best_model_params is not None:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data.copy_(best_model_params[name])
        
        # Final evaluation on validation set
        final_val_loss = self._evaluate_model(val_data, device)
        # print(f"Model updated successfully. Final validation loss: {final_val_loss:.6f}")
        return self.model

    def _evaluate_model(self, val_data, device):
        """Evaluate model on validation data"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for cpu_graph, genus_features, target_ppa in val_data:
                # Move data to device
                cpu_graph = cpu_graph.to(device)
                genus_features = genus_features.to(device)
                
                # Handle different types of target_ppa
                if isinstance(target_ppa, dict):
                    target_ppa = {k: v.to(device) if isinstance(v, torch.Tensor) 
                                else torch.tensor([v], dtype=torch.float32, device=device) 
                                for k, v in target_ppa.items()}
                
                # Forward pass
                pred_ppa = self.model(cpu_graph, genus_features)
                
                # Compute loss
                perf_loss = torch.nn.functional.mse_loss(pred_ppa['performance'], target_ppa['performance'])
                power_loss = torch.nn.functional.mse_loss(pred_ppa['power'], target_ppa['power'])
                area_loss = torch.nn.functional.mse_loss(pred_ppa['area'], target_ppa['area'])
                loss = perf_loss + power_loss + area_loss
                
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(val_data) if val_data else float('inf')
             
    def update_pareto_front(self):
        """Update Pareto front based on explored designs using pymoo for correct dominance calculation"""        
        # Extract PPA values for all designs
        all_designs = []
        ppa_values = []
        
        for design, ppa in self.explored_designs:
            # Handle different PPA structures
            if isinstance(ppa, dict):
                perf = ppa['performance'].item()
                power = ppa['power'].item() 
                area = ppa['area'].item()
            elif isinstance(ppa, tuple):
                perf, power, area = ppa
            else:
                print(f"Warning: Unknown PPA structure: {type(ppa)}")
                continue
                
            all_designs.append((design, (perf, power, area)))
            ppa_values.append([perf, power, area])
        
        if not ppa_values:
            self.pareto_front = []
            return
            
        # Convert to numpy array
        ppa_values = np.array(ppa_values)
        
        # Use Non-dominated sorting to find Pareto front (rank 0)
        nds = NonDominatedSorting()
        fronts = nds.do(ppa_values)
        
        # Get indices of Pareto optimal solutions (first front)
        pareto_indices = fronts[0]
        
        # Update Pareto front
        self.pareto_front = [all_designs[i] for i in pareto_indices]
        
        print(f"Pareto front updated: {len(self.pareto_front)} designs")
            
    def calculate_hypervolume(self, front, reference_point):
        """
        Calculate hypervolume of a Pareto front using pymoo
        
        Args:
            front: numpy array of shape (n_points, n_objectives)
            reference_point: numpy array of shape (n_objectives,)
                
        Returns:
            float: hypervolume value
        """
        if front.shape[0] == 0:
            return 0.0
        from pymoo.indicators.hv import HV
        indicator = HV(ref_point=reference_point)
        return indicator.do(front)

# ============================
# Data Handling & Training Utilities
# ============================
def load_cpu_data(filename, start_idx=22):
    """Load CPU architecture data from a CSV file without a header"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Parse line into architecture parameters and PPA values
            parts = line.strip().split(',')
            
            # Extract architecture parameters
            arch_params = {
                'predictor_type': int(parts[0]),
                'fetchWidth': int(parts[1]),
                'numFetchBufferEntries': int(parts[2]),
                'numRasEntries': int(parts[3]),
                'maxBrCount': int(parts[4]),
                'decodeWidth': int(parts[5]),
                'numRobEntries': int(parts[6]),
                'numIntPhysRegisters': int(parts[7]),
                'memIssueWidth': int(parts[8]),
                'intIssueWidth': int(parts[9]),
                'numLdqEntries': int(parts[10]), 
                'enablePrefetching': int(parts[11]),
                'enableSFBOpt': int(parts[12]),
                'numRXQEntries': int(parts[13]),
                'numRCQEntries': int(parts[14]),
                'nL2TLBEntries': int(parts[15]),
                'nL2TLBWays': int(parts[16]),
                'nICacheWays': int(parts[17]), 
                'nICacheTLBWays': int(parts[18]), 
                'nDCacheWays': int(parts[19]), 
                'nDCacheMSHRs': int(parts[20]), 
                'nDCacheTLBWays': int(parts[21])
            }
            
            # Extract GENUS parameters
            genus_params = {
                'time_recovery_arcs': int(parts[start_idx + 0]),
                'auto_partition': int(parts[start_idx + 1]),
                'dp_analytical_opt': int(parts[start_idx + 2]),
                'dp_area_mode': int(parts[start_idx + 3]),
                'dp_csa': int(parts[start_idx + 4]),
                'dp_rewriting': int(parts[start_idx + 5]),
                'dp_sharing': int(parts[start_idx + 6]),
                'exact_match_seq_async_ctrls': int(parts[start_idx + 7]),
                'exact_match_seq_sync_ctrls': int(parts[start_idx + 8]),
                'iopt_enable_floating_output_check': int(parts[start_idx + 9]),
                'iopt_force_constant_removal': int(parts[start_idx + 10]),
                'iopt_lp_power_analysis_effort': int(parts[start_idx + 11]),
                'lbr_seq_in_out_phase_opto': int(parts[start_idx + 12]),
                'optimize_net_area': int(parts[start_idx + 13]),
                'retime_effort_level': int(parts[start_idx + 14]),
                'retime_optimize_reset': int(parts[start_idx + 15]),
                'syn_generic_effort': int(parts[start_idx + 16]),
                'syn_map_effort': int(parts[start_idx + 17]),
                'syn_opt_effort': int(parts[start_idx + 18]),
                'leakage_power_effort': int(parts[start_idx + 19]),
                'lp_power_analysis_effort': int(parts[start_idx + 20]),
                'enable_latch_max_borrow': int(parts[start_idx + 21]),
                'latch_max_borrow': float(parts[start_idx + 22]),
                'enable_max_fanout': int(parts[start_idx + 23]),
                'max_fanout': int(parts[start_idx + 24]),
                'enable_lp_clock_gating_max_flops': int(parts[start_idx + 25]),
                'lp_clock_gating_max_flops': int(parts[start_idx + 26]),
                'enable_lp_power_optimization_weight': int(parts[start_idx + 27]),
                'lp_power_optimization_weight': float(parts[start_idx + 28]),
                'max_dynamic_power': float(parts[start_idx + 29]),
                'max_leakage_power': float(parts[start_idx + 30])
            }
            
            # Extract PPA values
            # BOOM 
            # CPI 0.727344 1.5367069
            # Power 9.01266 42.482
            # Area 4714505.956 7584237.065
            # Rocket 
            # CPI 1.298293317 2.266998559
            # Power 3.5736943 7.43694
            # Area 3947933.763 4232238.102

            # For GENUS
            # BOOM 
            # CPI 1.42660255918044
            # Power 9.01266 42.482
            # Area 4735870.833 4840131.693
            # Rocket 
            # CPI 2.26699790464907
            # Power 3.6514412 5.155712
            # Area 3948156.652 3997550.697
            BOOM_max_CPI = 1.5367069
            BOOM_min_CPI = 0.727344
            BOOM_max_Power = 42.482
            BOOM_min_Power = 9.01266
            BOOM_max_Area = 7584237.065
            BOOM_min_Area = 4714505.956
            ppa = {
                'performance': torch.tensor([(float(parts[-6]) - BOOM_min_CPI) / (BOOM_max_CPI - BOOM_min_CPI)], dtype=torch.float32),
                'power': torch.tensor([(float(parts[-5]) - BOOM_min_Power) / (BOOM_max_Power - BOOM_min_Power)], dtype=torch.float32),
                'area': torch.tensor([(float(parts[-4]) - BOOM_min_Area) / (BOOM_max_Area - BOOM_min_Area)], dtype=torch.float32)
            }
            # Build CPU graph
            genus_features_tensor = torch.tensor(encode_genus_parameters(genus_params), dtype=torch.float32)
            cpu_graph = build_boom_cpu_graph(arch_params)
            pyg_data = cpu_graph.to_pyg_data()
            
            # Add to dataset
            # data.append((pyg_data, genus_params, ppa))
            data.append((pyg_data, genus_features_tensor, ppa))
    
    return data


def create_tasks_for_meta_learning(arch_A_data, n_tasks=10, n_support=5, n_query=10):
    """Create tasks for meta-learning from architecture A and B data"""
    tasks = []
    
    # Combine data
    all_data = arch_A_data
    
    for _ in range(n_tasks):
        # Randomly sample data for this task
        sampled_data = random.sample(all_data, n_support + n_query)
        
        # Split into support and query sets
        support_set = sampled_data[:n_support]
        query_set = sampled_data[n_support:]
        
        tasks.append((support_set, query_set))
    
    return tasks
# epochs=100
# train_model(arch_model, arch_A_data + arch_B_data, epochs=3)
def train_model(model, train_data, val_data=None, epochs=20, batch_size=32):
    device = next(model.parameters()).device
    """Train the model directly on CPU architecture and PPA data"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Shuffle data
        random.shuffle(train_data)
        
        # Process in batches
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            batch_loss = 0
            
            optimizer.zero_grad()
            
            for cpu_graph, genus_features, target_ppa in batch:
                cpu_graph = cpu_graph.to(device)
                genus_features = genus_features.to(device)
                target_ppa = {k: v.to(device) for k, v in target_ppa.items()}
                # Forward pass
                pred_ppa = model(cpu_graph, genus_features)
                
                target_perf = target_ppa['performance'].view_as(pred_ppa['performance'])
                target_power = target_ppa['power'].view_as(pred_ppa['power'])
                target_area = target_ppa['area'].view_as(pred_ppa['area'])

                perf_loss = F.mse_loss(pred_ppa['performance'], target_perf)
                power_loss = F.mse_loss(pred_ppa['power'], target_power)
                area_loss = F.mse_loss(pred_ppa['area'], target_area)                
                loss = perf_loss + power_loss + area_loss
                batch_loss += loss
            
            # Backward pass
            batch_loss /= len(batch)
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
        
        # Print progress
        avg_loss = total_loss / (len(train_data) // batch_size)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        # Validation
        if val_data is not None and (epoch + 1) % 5 == 0:
            val_loss = evaluate_model(model, val_data)
            print(f"Validation Loss: {val_loss:.4f}")
    
    return model

def evaluate_model(model, data):
    """Evaluate the model on a dataset"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for cpu_graph, genus_features, target_ppa in data:
            # Forward pass
            pred_ppa = model(cpu_graph, genus_features)
            
            # Compute loss
            perf_loss = F.mse_loss(pred_ppa['performance'], target_ppa['performance'])
            power_loss = F.mse_loss(pred_ppa['power'], target_ppa['power'])
            area_loss = F.mse_loss(pred_ppa['area'], target_ppa['area'])
            
            loss = perf_loss + power_loss + area_loss
            total_loss += loss.item()
    
    return total_loss / len(data)

# ============================
# Main Workflow
# ============================

# save_model(arch_model, "boom_arch_model.pth")
# save_model(rocket_genus_model, "genus_model.pth")
def save_model(model, filepath):
    """保存模型到文件"""
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存到: {filepath}")

# arch_model = BOOMMetaGAT(arch_feature_dim, genus_feature_dim, hidden_dim=64, output_dim=64)
# arch_model = load_model(arch_model, "boom_arch_model.pth")
def load_model(model, filepath, device=None):
    """从文件加载模型权重"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    print(f"模型已从 {filepath} 加载")
    return model


######## dse
def main():
    """Main workflow for BOOM CPU Meta-GAT with knowledge transfer"""
    arch_A_data = load_cpu_data('dataA.csv')
    #arch_B_data = load_cpu_data('dataB.csv')
    # arch_C_data = load_cpu_data('dataC.csv')
    # print(arch_A_data)
    
    # 2. Define parameter spaces
    # 3. Initialize model
    print("Initializing Meta-GAT model...")
    # Determine feature dimensions
    sample_cpu_graph = arch_A_data[0][0]
    sample_genus_features = arch_A_data[0][1]
    genus_feature_dim = len(encode_genus_parameters({}))  # 计算特征维度
    # print('genus_feature_dim = ', genus_feature_dim)
    
    # 训练GENUS参数模型

    arch_feature_dim = sample_cpu_graph.x.shape[1]
    genus_feature_dim = len(sample_genus_features)
    # print('arch_feature_dim = ', arch_feature_dim)
    print('genus_feature_dim = ', genus_feature_dim)
    arch_model = BOOMMetaGAT(
        arch_feature_dim=arch_feature_dim,
        genus_feature_dim=genus_feature_dim,
        # genus_feature_dim=40,
        hidden_dim=64,
        output_dim=64
    )

    # 5. 训练 Architecture A+B 架构模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arch_model = load_model(arch_model, "trained_boom_arch_model.pth")
    
    
    # 7. 创建元学习器
    meta_learner = ImprovedBOOMMetaLearner(arch_model)
    
    # 8. 适应到 C 架构
    print("Adapting to architecture C with few-shot learning...")
    # adapted_model = meta_learner.adapt_to_new_architecture(arch_A_data[0:50])
    task = create_tasks_for_meta_learning(arch_A_data[0:50], n_tasks=10, n_support=10, n_query=5)
    # print(task)
    meta_learner.meta_train_step(task)
    adapted_model = meta_learner.model 
    initial_data = arch_A_data[0:50]
    exploration_data = arch_A_data[51:1926]
    target_point = (0.3, 0.3, 0.4) 
    explorer = DesignSpaceExplorer(
        model=adapted_model,
        # model=arch_model,
        exploration_data=exploration_data,
        target_point=target_point
    )
    # 8. Run design space exploration
    print("Starting design space exploration...")
    
    # Run for 100 iterations, update model every 5 iterations
    # print(initial_data)
    pareto_front = explorer.run_exploration(num_iterations = 50, update_interval=2,initial_data=initial_data)
    
    # 9. Save final model and results
    final_model_path = "boom_dse_final_model_prefer.pth"
    save_model(adapted_model, final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    print("Exploration complete!")
if __name__ == "__main__":
    main()

