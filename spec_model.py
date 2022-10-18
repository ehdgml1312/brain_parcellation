import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import EdgeConv, GATConv

class spec_ln(torch.nn.Module):
    def __init__(self, config):
        super(spec_ln, self).__init__()
        self.k = config.K
        self.in_channels = config.in_channels
        hidden_channels = config.hidden_channels
        self.spectral_encode1 = nn.Linear(1, 64)
        self.spectral_encode2 = nn.Linear(64, 64)
        self.spatial_encode1 = nn.Linear(self.in_channels, 64)
        self.spatial_encode2 = nn.Linear(64,64)
        
        self.ln = nn.LayerNorm(64)

        self.conv1 = EdgeConv(nn.Linear(2*64, hidden_channels[0]))
        self.conv2 = EdgeConv(nn.Linear(2*(64+hidden_channels[0]), hidden_channels[1]))
        self.conv3 = EdgeConv(nn.Linear(2*(64+hidden_channels[0]+hidden_channels[1]), hidden_channels[2]))
        self.conv1 = EdgeConv(nn.Linear(2*64, hidden_channels[0]))
        self.conv2 = EdgeConv(nn.Linear(2*192, hidden_channels[1]))
        self.conv3 = EdgeConv(nn.Linear(2*384, hidden_channels[2]))

        self.conv5 = EdgeConv(nn.Linear(2*64, hidden_channels[0]))
        self.conv6 = EdgeConv(nn.Linear(2*(64+hidden_channels[0]), hidden_channels[1]))
        self.conv7 = EdgeConv(nn.Linear(2*(64+hidden_channels[0]+hidden_channels[1]), hidden_channels[2]))
        self.conv8 = EdgeConv(nn.Linear(2*(64+hidden_channels[0]+hidden_channels[1]+hidden_channels[2]), hidden_channels[3]))
        
        self.conv5 = EdgeConv(nn.Linear(2*64, hidden_channels[0]))
        self.conv6 = EdgeConv(nn.Linear(2*192, hidden_channels[1]))
        self.conv7 = EdgeConv(nn.Linear(2*384, hidden_channels[2]))
        self.conv8 = EdgeConv(nn.Linear(2*640, hidden_channels[3]))

        self.decode1 = nn.Linear(5*64+4*hidden_channels[0]+3*hidden_channels[1]+2*hidden_channels[2]+hidden_channels[3], 64)
        self.decode2 = nn.Linear(64,32)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data, spec_edge):
        x0, edge_index = data.x[:, :self.in_channels], data.edge_index
        u, e = data.u[:,:self.k], data.e[:self.k]

        s0 = self.ln(F.leaky_relu(self.spectral_encode2(self.ln(F.leaky_relu(self.spectral_encode1(e))))))
        x0 = self.ln(F.leaky_relu(self.spatial_encode2(self.ln(F.leaky_relu(self.spatial_encode1(x0))))))

        s1 = self.conv1(s0, spec_edge)
        x1 = self.conv5(x0, edge_index)

        s1 = self.ln(F.leaky_relu(s1))
        x1 = self.ln(F.leaky_relu(x1))
        

        s1 = torch.cat([s1, torch.matmul(u.transpose(0, 1), x0)], 1)
        x1 = torch.cat([x1, torch.matmul(u, s0)], 1)

        s2 = torch.cat([s0,s1],1)
        x2 = torch.cat([x0,x1],1)
        s2 = self.conv2(s2, spec_edge)
        x2 = self.conv6(x2, edge_index)
        s2 = self.ln(F.leaky_relu(s2))
        x2 = self.ln(F.leaky_relu(x2))
        
        s2 = torch.cat([s2, torch.matmul(u.transpose(0, 1), x1)], 1)
        x2 = torch.cat([x2, torch.matmul(u, s1)], 1)

        s3 = torch.cat([s0,s1,s2],1)
        x3 = torch.cat([x0,x1,x2],1)
        s3 = self.conv3(s3, spec_edge)
        x3 = self.conv7(x3, edge_index)
        s3 = self.ln(F.leaky_relu(s3))
        x3 = self.ln(F.leaky_relu(x3))

        s3 = torch.cat([s3, torch.matmul(u.transpose(0, 1), x2)], 1)
        x3 = torch.cat([x3, torch.matmul(u, s2)], 1)

        x4 = torch.cat([x0,x1,x2,x3],1)
        x4 = self.conv8(x4, edge_index)
        x4 = self.ln(F.leaky_relu(x4))
        x4 = torch.cat([x4, torch.matmul(u, s3)], 1)
        

        x4 = torch.cat([x0,x1,x2,x3,x4],1)
        out = self.ln(F.leaky_relu(self.decode1(self.dropout(x4))))
        out = self.decode2(self.dropout(out))

        return out


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

class mpnn(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') 
        self.msg = nn.Linear(in_channels,in_channels)
        self.upd = nn.Linear(2*in_channels,out_channels)

    def forward(self, x, edge_index):

        out = self.propagate(edge_index, x=x)
        out = self.upd(torch.cat((x,out),1))

        return out

    def message(self, x_j):

        return self.msg(x_j)

class GATEdge(MessagePassing):
    def __init__(self, in_channels:int,
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATEdge, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)


        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None


    def forward(self, x, edge_index,
                size=None, return_attention_weights=None):
        print(x)
        
        # if self.add_self_loops:
           
        #     num_nodes = x.size(0)
        #     edge_index, _ = remove_self_loops(edge_index)
        #     edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)


        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j, x_i):
        
        H, C = self.heads, self.out_channels
        x_l = self.lin(x_j-x_i).view(-1, H, C)
        x_r = self.lin(x_i).view(-1, H, C)
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1) 
        alpha = alpha_l + alpha_r
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha,0)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (x_j-x_i) * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
        
class spec_ln_msg(torch.nn.Module):
    def __init__(self, config):
        super(spec_ln_msg, self).__init__()
        self.k = config.K
        self.in_channels = config.in_channels
        hidden_channels = config.hidden_channels
        self.spectral_encode1 = nn.Linear(1, 64)
        self.spectral_encode2 = nn.Linear(64, 64)
        self.spatial_encode1 = nn.Linear(self.in_channels, 64)
        self.spatial_encode2 = nn.Linear(64,64)
        
        self.ln = nn.LayerNorm(64)


        self.conv1 = GATEdge(64, 8,8)
        self.conv2 = GATEdge(192, 8,8)
        self.conv3 = GATEdge(384, 8,8)
        
        self.conv5 = GATEdge(64, 8,8)
        self.conv6 = GATEdge(192, 8,8)
        self.conv7 = GATEdge(384, 8,8)
        self.conv8 = GATEdge(640, 8,8)


        self.decode1 = nn.Linear(5*64+4*hidden_channels[0]+3*hidden_channels[1]+2*hidden_channels[2]+hidden_channels[3], 64)
        self.decode2 = nn.Linear(64,32)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data, spec_edge):
        x0, edge_index = data.x[:, :self.in_channels], data.edge_index
        u, e = data.u[:,:self.k], data.e[:self.k]

        s0 = self.ln(F.leaky_relu(self.spectral_encode2(self.ln(F.leaky_relu(self.spectral_encode1(e))))))
        x0 = self.ln(F.leaky_relu(self.spatial_encode2(self.ln(F.leaky_relu(self.spatial_encode1(x0))))))

        s1 = self.conv1(s0, spec_edge)
        x1 = self.conv5(x0, edge_index)

        s1 = self.ln(F.leaky_relu(s1))
        x1 = self.ln(F.leaky_relu(x1))
        

        s1 = torch.cat([s1, torch.matmul(u.transpose(0, 1), x0)], 1)
        x1 = torch.cat([x1, torch.matmul(u, s0)], 1)

        s2 = torch.cat([s0,s1],1)
        x2 = torch.cat([x0,x1],1)
        s2 = self.conv2(s2, spec_edge)
        x2 = self.conv6(x2, edge_index)
        s2 = self.ln(F.leaky_relu(s2))
        x2 = self.ln(F.leaky_relu(x2))
        
        s2 = torch.cat([s2, torch.matmul(u.transpose(0, 1), x1)], 1)
        x2 = torch.cat([x2, torch.matmul(u, s1)], 1)

        s3 = torch.cat([s0,s1,s2],1)
        x3 = torch.cat([x0,x1,x2],1)
        s3 = self.conv3(s3, spec_edge)
        x3 = self.conv7(x3, edge_index)
        s3 = self.ln(F.leaky_relu(s3))
        x3 = self.ln(F.leaky_relu(x3))

        s3 = torch.cat([s3, torch.matmul(u.transpose(0, 1), x2)], 1)
        x3 = torch.cat([x3, torch.matmul(u, s2)], 1)

        x4 = torch.cat([x0,x1,x2,x3],1)
        x4 = self.conv8(x4, edge_index)
        x4 = self.ln(F.leaky_relu(x4))
        x4 = torch.cat([x4, torch.matmul(u, s3)], 1)
        

        x4 = torch.cat([x0,x1,x2,x3,x4],1)
        out = self.ln(F.leaky_relu(self.decode1(self.dropout(x4))))
        out = self.decode2(self.dropout(out))

        return out

