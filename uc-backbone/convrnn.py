import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    ''' Initialize ConvGRU cell '''

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()

        # 256
        self.input_size = input_size

        # 256
        self.hidden_size = hidden_size

        # 1
        self.kernel_size = kernel_size

        # 0
        padding = kernel_size // 2

        # forget gate
        self.reset_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size,
            padding=padding)

        # input gate
        self.update_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size,
            padding=padding)

        self.out_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size,
            padding=padding)

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_tensor, hidden_state):

        # for the first prediction
        if hidden_state is None:
            B, C, *spatial_dim = input_tensor.size()
            hidden_state = torch.zeros(
                [B, self.hidden_size, *spatial_dim]).cuda()
        # [B, C, H, W]
        combined = torch.cat([input_tensor, hidden_state],
                             dim=1)  # concat in C

        # operations
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        out = torch.tanh(self.out_gate(
            torch.cat([input_tensor, hidden_state * reset], dim=1)))
        new_state = hidden_state * (1 - update) + out * update
        return new_state


class ConvGRU(nn.Module):
    ''' Initialize a multi-layer Conv GRU '''

    def __init__(self, input_size, hidden_size, kernel_size, num_layers,
                 dropout=0.1, radius_type='linear'):
        super(ConvGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_size
            cell = ConvGRUCell(input_dim, self.hidden_size, self.kernel_size)
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cell_list.append(getattr(self, name))

        self.cell_list = nn.ModuleList(cell_list)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.radius_type = radius_type

        # make the radius positive
        if self.radius_type == 'log':
            print('[convrnn.py] Using log as radius_type')
            self.activation = exp_activation()
        else:
            print('[convrnn.py] Using linear as radius_type')

    def forward(self, x, hidden_state=None):
        [B, seq_len, *_] = x.size()

        if hidden_state is None:
            hidden_state = [None] * self.num_layers
        # input: image sequences [B, T, C, H, W]
        current_layer_input = x
        del x

        last_state_list = []

        for idx in range(self.num_layers):
            cell_hidden = hidden_state[idx]
            output_inner = []
            for t in range(seq_len):

                # give the hidden state and input to the current layer
                cell_hidden = self.cell_list[idx](
                    current_layer_input[:, t, :], cell_hidden)
                cell_hidden = self.dropout_layer(
                    cell_hidden)  # dropout in each time step

                # make the radius positive
                if self.radius_type == 'logi':
                    cell_hidden = self.activation(cell_hidden)

                output_inner.append(cell_hidden)

            layer_output = torch.stack(output_inner, dim=1)
            current_layer_input = layer_output

            last_state_list.append(cell_hidden)

        last_state_list = torch.stack(last_state_list, dim=1)

        # change the radius here
        return layer_output, last_state_list


class exp_activation(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()  # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return exp_radius(input)


def exp_radius(input):

    input[:, -1, :, :] = torch.exp(input[:, -1, :, :])
    return input


if __name__ == '__main__':
    crnn = ConvGRU(input_size=257, hidden_size=257,
                   kernel_size=1, num_layers=1)
    crnn = crnn.cuda()

    # [B, seq_len, C, H, W], temporal axis=1
    data = torch.randn(4, 5, 257, 4, 4)
    data = data.cuda()
    output, hn = crnn(data)

    print(output.shape, hn.shape, hn[:, :, -1, :, :])
