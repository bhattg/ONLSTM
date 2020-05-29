import torch
import torch.nn as nn


class CumMax(nn.Module):
        def __init__(self):
                super(CumMax, self).__init__()

        def forward(self, input):
                return torch.cumsum(nn.Softmax(dim=-1)(input), -1)


class ONLSTM_cell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(ONLSTM_cell, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.wi = nn.Linear(hidden_size + input_size, hidden_size)
        self.wf = nn.Linear(hidden_size + input_size, hidden_size)
        self.wg = nn.Linear(hidden_size + input_size, hidden_size)
        self.wo = nn.Linear(hidden_size + input_size, hidden_size)
        self.wftilde = nn.Linear(hidden_size + input_size, hidden_size)
        self.witilde = nn.Linear(hidden_size + input_size, hidden_size)

    def forward(self, input, hidden):
        #in ON LSTM cell, its input should be of dim (bsz, input_dim), so addition shoudl be performed along dim = -1

        hx, cx = hidden
        input_plus_hidden = torch.cat((input, hx), dim=-1)

        f_t = nn.Sigmoid()(self.wf(input_plus_hidden))
        i_t = nn.Sigmoid()(self.wi(input_plus_hidden))
        o_t = nn.Sigmoid()(self.wo(input_plus_hidden))
        c_hat_t = nn.Tanh()(self.wg(input_plus_hidden))

        f_tilde_t = CumMax()(self.wftilde(input_plus_hidden))
        i_tilde_t = 1 - CumMax()(self.witilde(input_plus_hidden))

        omega_t = f_tilde_t * i_tilde_t
        f_hat_t = f_t * omega_t + (f_tilde_t - omega_t)
        i_hat_t = i_t * omega_t + (i_tilde_t - omega_t)

        cx = f_hat_t * cx + i_hat_t * c_hat_t
        hx = o_t * nn.Tanh()(cx)

        return hx, cx

class ONLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1, dropout=0.2):
        super(ONLSTM, self).__init__()

        self.input_size  = input_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        for layer in range(num_layers):
            layer_input_size  = self.input_size if layer == 0 else self.hidden_size
            cell = ONLSTM_cell(input_size  = layer_input_size, hidden_size = self.hidden_size)
            setattr(self, 'cell_{}'.format(layer), cell)

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))


    def forward_single_layer(self, input_, layer, hidden, max_time=50):
        # hidden, a tuple of (h, c)
        # assumed input shape : (time_stamps, batch_size, input_emb_dim)
        # assumed output shape: (time_stamps, batch_size, hidden_dim) , h_n shape = (1, batch_size, hidden_dim) --> corresponding to the last time stamp 
        all_hidden = []
        cell = self.get_cell(layer)
        max_time, M = input_.shape[0], input_.shape[1]
        state = hidden

        assert (hidden[0].shape[0],hidden[0].shape[1]) == (M, self.hidden_size)

        for time in range(max_time):
            state = cell(input_[time, :, :], state)
            if layer==self.num_layers-1:
                all_hidden.append(state[0])
            else:
                all_hidden.append(self.dropout_layer(state[0]))

        h_n, c_n = state #last time stamp state (M, H)
        all_hidden = torch.stack(all_hidden)
        assert (h_n.shape[0], h_n.shape[1]) ==(M, self.hidden_size)
        return all_hidden, h_n, c_n

    def forward(self, input_, h_0 = None, c_0= None, max_time=50):
        # we assume that the input shape is (time_stamps, batch_size, input_sizes)
        # for every example the h_init will serve as none. H_init will be none in each layer and for all examples. 
        # the inputs that will be passed to layer_0 will be the input_, for the subsequent layers, we will pass the processed 
        # hidden layer outputs. 
        # h_0 is the inital state to be used for the dynamics. 
        # h_0, c_0 = hidden
        max_time, M = input_.shape[0], input_.shape[1]

        if not torch.is_tensor(h_0):
            h_0 = torch.zeros((self.num_layers, M, self.hidden_size)).to(device)

        if not torch.is_tensor(c_0):
            c_0 = torch.zeros((self.num_layers, M, self.hidden_size)).to(device)

        h_n = []
        c_n = []
   
        for layer in range(self.num_layers):
            if layer == 0: 
                all_hidden, h_n_layer, c_n_layer = self.forward_single_layer(input_, layer, (h_0[layer, :, :],c_0[layer, :, :]))
            else:
                all_hidden, h_n_layer, c_n_layer = self.forward_single_layer(all_hidden, layer, (h_0[layer, :, :],c_0[layer, :, :]))

            h_n.append(h_n_layer)
            c_n.append(c_n_layer)


        h_n = torch.stack(h_n)
        c_n = torch.stack(c_n)

        assert (h_n.shape[0],h_n.shape[1],h_n.shape[2]) ==( self.num_layers, M , self.hidden_size )
        assert (c_n.shape[0],c_n.shape[1],c_n.shape[2]) ==( self.num_layers, M , self.hidden_size )
        assert (all_hidden.shape[0],all_hidden.shape[1],all_hidden.shape[2])==(max_time, M, self.hidden_size)

        return all_hidden, (h_n, c_n)
