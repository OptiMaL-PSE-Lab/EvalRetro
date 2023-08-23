import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr.basic import SumAggregation
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn.inits import glorot
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# First try simple GCN model - inspired from PyTorch geometric tutorial

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.init_parameters()

    def init_parameters(self):
        self.linear.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # x size is N_nodes x node_features
        x = self.linear(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        out += self.bias

        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
        
# This class is used to classify the nodes and edges in the graph (binary classification)
class GCN(nn.Module):
    def __init__(self, num_n_features, num_hidden, n_layers):
        super(GCN, self).__init__()
        # Apply GCNConv layer n_layers times
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_n_features, num_hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(num_hidden, num_hidden))
        
        self.node_classifier = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(2*num_hidden, 2*num_hidden),
            nn.ReLU(),
            nn.Linear(2*num_hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr, info_batch, return_type="Both", train=True):

        for conv in self.convs:
            x = conv(x, edge_index) # shape of x is N_nodes x num_hidden
            # for batches, features are concatenated in node dimension. hence, the shape of N * batch_size x num_hidden
        node_logits = self.node_classifier(x) 
        # calculate x_edge by concatenating the node features of the two nodes connected by the edge
        new_edge_index = edge_index[:,::2]
        x_edge = torch.cat((x[new_edge_index[0]], x[new_edge_index[1]]), dim=-1) # shape of x_edge is N_edges x 2 * num_hidden
        edge_logits = self.edge_classifier(x_edge)
        # if logits negative, set to 1e-5
        if train:
            row, col = edge_index
            edge_batch = info_batch[row]
            concatenated_logits = torch.cat((node_logits, edge_logits), dim=0)
            concatenated_batch = torch.cat((info_batch, edge_batch[::2]), dim=0)
            len_node = node_logits.shape[0]
            logits = softmax(concatenated_logits, index=concatenated_batch)
            node_logits, edge_logits = logits[:len_node], logits[len_node:]
        else:
            soft_unbatched = nn.Softmax(dim=0)
            node_logits, edge_logits = node_logits.squeeze(), edge_logits.squeeze()
            len_node = node_logits.shape[0]
            concatenated = torch.cat((node_logits, edge_logits), dim=0)
            logits = soft_unbatched(concatenated)
            node_logits, edge_logits = logits[:len_node], logits[len_node:]
        if return_type == "Both":
            return (node_logits, edge_logits) # Shape of node_logits and edge_logits is N_nodes x 1 and N_edges x 1
        elif return_type == "Node":
            return node_logits
        elif return_type == "Edge":
            return edge_logits
        

class EdgeGAT(MessagePassing):
    def __init__(self, in_channels, in_channels_edge, out_channels, drop_prob=0.1, negative_slope=0.2):
        super(EdgeGAT, self).__init__(aggr='add')
        self.Wm_1 = nn.Linear(in_channels, out_channels, bias=False)
        self.Wm_2 = nn.Linear(in_channels_edge, out_channels, bias=False)
        self.Wm_3 = nn.Linear(out_channels+in_channels_edge, out_channels, bias=False) 
        self.attn_fc = nn.Linear(2*out_channels+in_channels_edge, 1, bias=False)
        self.dropout = nn.Dropout(p=drop_prob)
        self.mlp = nn.Sequential(
            nn.Linear(2*out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.negative_slope = negative_slope
        
        self.init_parameters()
    
    def init_parameters(self):
        self.Wm_1.reset_parameters()
        self.Wm_2.reset_parameters()
        self.Wm_3.reset_parameters()
        self.attn_fc.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.Wm_1(x)
        #edge_index, _ = add_self_loops(edge_index)  # Add self-loops for undirected graphs
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

    def message(self, x_i, x_j, edge_attr, index):
        alpha = self.attn_fc(torch.cat((x_i, x_j, edge_attr), dim=-1)) # shape of alpha is N_edges x 1
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = softmax(alpha, index)
        alpha = self.dropout(alpha) # shape of alpha is N_edges x 1
        out = alpha * self.Wm_3(torch.cat([x_j, edge_attr], dim=-1)) # shape of out is N_edges x out_channels
        return out
    
    def update(self, aggr_out, x):
        # Use mlp to update node features
        output = self.mlp(torch.cat([aggr_out, x], dim=-1))
        return output



class EGAT(nn.Module):
    def __init__(self, num_n_features, num_edge_features, num_hidden, n_layers):
        super(EGAT, self).__init__()
        self.linear = nn.Linear(num_edge_features, num_hidden)
        # Apply GCNConv layer n_layers times
        self.convs = nn.ModuleList()
        self.convs.append(EdgeGAT(num_n_features, num_edge_features, num_hidden))
        for _ in range(n_layers - 1):
            self.convs.append(EdgeGAT(num_hidden, num_hidden, num_hidden))
        
        self.node_classifier = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(2*num_hidden, 2*num_hidden),
            nn.ReLU(),
            nn.Linear(2*num_hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr,  return_type="Both", info_batch=None, train=True):
        new_edge_attr = self.linear(edge_attr)
        for i, gat in enumerate(self.convs):
            if i == 0:
                x = gat(x, edge_index, edge_attr)
            else:
                x = gat(x, edge_index, new_edge_attr)

        node_logits = self.node_classifier(x) 
        # calculate x_edge by concatenating the node features of the two nodes connected by the edge
        new_edge_index = edge_index[:,::2]
        node_diff, node_add = torch.abs(x[new_edge_index[0]] - x[new_edge_index[1]]), x[new_edge_index[0]] + x[new_edge_index[1]]
        x_edge = torch.cat((node_diff, node_add), dim=1)
        edge_logits = self.edge_classifier(x_edge)
        if train:
            row, col = edge_index
            edge_batch = info_batch[row]
            concatenated_logits = torch.cat((node_logits, edge_logits), dim=0)
            concatenated_batch = torch.cat((info_batch, edge_batch[::2]), dim=0)
            len_node = node_logits.shape[0]
            logits = softmax(concatenated_logits, index=concatenated_batch)
            node_logits, edge_logits = logits[:len_node], logits[len_node:]
        else:
            soft_unbatched = nn.Softmax(dim=0)
            node_logits, edge_logits = node_logits.squeeze(), edge_logits.squeeze()
            len_node = node_logits.shape[0]
            concatenated = torch.cat((node_logits, edge_logits), dim=0)
            logits = soft_unbatched(concatenated)
            node_logits, edge_logits = logits[:len_node], logits[len_node:]
            # node_logits, edge_logits = torch.squeeze(node_logits), torch.squeeze(edge_logits)

        if return_type == "Both":
            return (node_logits, edge_logits) # Shape of node_logits and edge_logits is N_nodes x 1 and N_edges x 1
        elif return_type == "Explain":
            # find index of max edge_logits
            return edge_logits

        else:
            raise ValueError("return_type must be either 'Both' or 'Explain'")
        

class DMPNN_layer(MessagePassing):
    """ 
    This layer only returns messages along the edges rather than updating node features
    A GRU layer is 
    """
    def __init__(self, in_channels, in_channels_edge, channels_message, graph_features):
        super(DMPNN_layer, self).__init__()
        self.Wm_z = nn.Linear(in_channels+in_channels_edge+channels_message, channels_message, bias=True)
        self.Wm_r = nn.Linear(in_channels+in_channels_edge+channels_message, channels_message, bias=True)
        self.W = nn.Linear(in_channels+in_channels_edge, channels_message, bias=False)
        self.U = nn.Linear(channels_message, channels_message, bias=True) 
        
        self.init_parameters()

        self.dropout = nn.Dropout(p=0.15)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels + channels_message, graph_features),
            nn.ReLU(),
        )

    def init_parameters(self):
        self.Wm_z.reset_parameters()
        self.Wm_r.reset_parameters()
        self.W.reset_parameters()
        self.U.reset_parameters()

    def forward(self, x, edge_index, edge_attr, neighbour_index, message_edge, layer_type):
        # create s_uv zeros tensor of same shape as message_edge
        # add 0 vector to beginning of message_edge
        message_upd = torch.cat((torch.zeros(1, message_edge.shape[1], device=x.device), message_edge), dim=0)
        s_uv = message_upd.index_select(0, neighbour_index.view(-1))
        s_uv = s_uv.view(neighbour_index.shape[0], neighbour_index.shape[1], -1)
        s_uv = s_uv.sum(dim=1)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, s_uv=s_uv, message_edge=message_edge, neighbour_index=neighbour_index, layer_type=layer_type, size=None)

    def message(self, x_i, edge_attr, s_uv, message_edge, neighbour_index):
        
        z_uv = F.sigmoid(self.Wm_z(torch.cat([x_i, edge_attr, s_uv], dim=-1)))
        r_uv = F.sigmoid(self.Wm_r(torch.cat([x_i, edge_attr, message_edge], dim=-1)))
        
        # remove the for loop like we have done for the forward pass
        message_upd = torch.cat((torch.zeros(1, message_edge.shape[1], device=x_i.device), message_edge), dim=0)
        r_uv_upd = torch.cat((torch.zeros(1, r_uv.shape[1], device=x_i.device), r_uv), dim=0)

        idx = neighbour_index.view(-1)
        message_upd = message_upd.index_select(0, idx)
        r_uv_upd = r_uv_upd.index_select(0, idx)
        r_uv_upd, message_upd = r_uv_upd.view(neighbour_index.shape[0], neighbour_index.shape[1], -1), message_upd.view(neighbour_index.shape[0], neighbour_index.shape[1], -1)
        r_uv_dash = r_uv_upd * message_upd
        r_uv_dash = torch.sum(r_uv_dash, dim=1)

        m_uv_dash = F.tanh(self.W(torch.cat([x_i, edge_attr], dim=-1)) + self.U(r_uv_dash))
        m_uv_t = (1-z_uv)*s_uv + z_uv*m_uv_dash
        m_uv_t = self.dropout(m_uv_t)

        return m_uv_t
        

    def aggregate(self, inputs, layer_type, index):
        if layer_type == 'final':
            # add inputs based on index
            aggr = SumAggregation()
            return aggr.forward(inputs, index)
        else: 
            return inputs
        
    def update(self, aggr_out, x, layer_type):
        if layer_type == 'final':
            try :
                aggr_out = self.mlp(torch.cat([x, aggr_out], dim=-1))
            except: # some error originating from pyg
                print("Exception")
                if x.shape[0] > aggr_out.shape[0]:
                    zeros = torch.zeros(x.shape[0] - aggr_out.shape[0], aggr_out.shape[1], device=x.device)
                    aggr_out = torch.cat([aggr_out, zeros], dim=0)
                    aggr_out = self.mlp(torch.cat([x, aggr_out], dim=-1))
                else:
                    zeros = torch.zeros(aggr_out.shape[0] - x.shape[0], x.shape[1], device=x.device)
                    x = torch.cat([x, zeros], dim=0)
                    aggr_out = self.mlp(torch.cat([x, aggr_out], dim=-1))
            
            return aggr_out
            
        return aggr_out
    

class DMPNN(nn.Module):
    def __init__(self, num_n_features, num_edge_features, num_hidden, graph_features, n_layers):
        super(DMPNN, self).__init__()

        self.convs = nn.ModuleList()
        self.num_hidden = num_hidden
        
        for _ in range(n_layers-1):
            self.convs.append(DMPNN_layer(num_n_features, num_edge_features, num_hidden, graph_features))

        self.final_dmpnn = DMPNN_layer(num_n_features, num_edge_features, num_hidden, graph_features)
        
        self.node_classifier = nn.Sequential(
            nn.Linear(graph_features, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(2*graph_features, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x, edge_index, edge_attr,  return_type="Both", info_batch=None, train=True):
        
        neighbour_index = self.get_edge_indices(edge_index)
        # initialise messages from edge features
        message_edge = torch.zeros(edge_attr.shape[0], self.num_hidden, device=x.device)

        for dmpnn in self.convs:
                message_edge = dmpnn(x, edge_index, edge_attr, neighbour_index, message_edge, "initial")

        x = self.final_dmpnn(x, edge_index, edge_attr, neighbour_index, message_edge, "final")

        node_logits = self.node_classifier(x) 
        # calculate x_edge by concatenating the node features of the two nodes connected by the edge
        new_edge_index = edge_index[:,::2]
        node_diff, node_add = torch.abs(x[new_edge_index[0]] - x[new_edge_index[1]]), x[new_edge_index[0]] + x[new_edge_index[1]]
        x_edge = torch.cat((node_diff, node_add), dim=1)
        edge_logits = self.edge_classifier(x_edge)
        if train:
            row, col = edge_index
            edge_batch = info_batch[row]
            concatenated_logits = torch.cat((node_logits, edge_logits), dim=0)
            concatenated_batch = torch.cat((info_batch, edge_batch[::2]), dim=0)
            len_node = node_logits.shape[0]
            logits = softmax(concatenated_logits, index=concatenated_batch)
            node_logits, edge_logits = logits[:len_node], logits[len_node:]
        else:
            soft_unbatched = nn.Softmax(dim=0)
            node_logits, edge_logits = node_logits.squeeze(), edge_logits.squeeze()
            len_node = node_logits.shape[0]
            concatenated = torch.cat((node_logits, edge_logits), dim=0)
            logits = soft_unbatched(concatenated)
            node_logits, edge_logits = logits[:len_node], logits[len_node:]

        if return_type == "Both":
            return (node_logits, edge_logits) # Shape of node_logits and edge_logits is N_nodes x 1 and N_edges x 1
        elif return_type == "Explain":
            # find index of max edge_logits
            return edge_logits

        else:
            raise ValueError("return_type must be either 'Both' or 'Explain'")
        
    def get_edge_indices(self, edge_indices):
        row, col = edge_indices
        padding_len = 3
        indices_tensor = torch.zeros((len(row), padding_len), dtype=torch.long, device=row.device)
        for i, (source_idx, target_idx) in enumerate(zip(*edge_indices)): 
            x = (source_idx == col) 
            y = (target_idx != row)
            z = torch.logical_and(x,y)
            z = z.nonzero(as_tuple=True)[0]
            # add ones to all elements in z to account for the padding
            z += 1
            # pad z with zeros to make it of length padding_len
            z = F.pad(z, (0, padding_len - len(z)), 'constant', 0)
            indices_tensor[i] = z

        return indices_tensor