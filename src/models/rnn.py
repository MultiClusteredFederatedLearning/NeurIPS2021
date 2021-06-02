import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, num_embeddings=80, rnn_type='lstm', tie_weights=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.rnn_type = rnn_type
        self.num_input = 8
        self.hidden_size = 256
        self.num_layer = 2

        self.emb = nn.Embedding(self.num_embeddings, self.num_input)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.num_input, hidden_size=self.hidden_size,
                            num_layers=self.num_layer, bias=False,
                            batch_first=True, bidirectional=False)
        else:
            raise NotImplementedError
        
        self.fc = nn.Linear(self.hidden_size, self.num_embeddings)

        if tie_weights:
            self.fc.weight = self.emb.weight

    def forward(self, x, hidden):
        x = self.emb(x) # (batch, seq_len) -> (batch, seq_len, num_input)
        x, hidden = self.rnn(x, hidden) # -> (seq_len, batch, hidden_size)
        # 出力の最後のみをfc層に入力
        x = self.fc(x[:,-1,:]) # (batch, hidden_size) -> (batch, num_emb)
        return x, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layer, batch_size, self.hidden_size),
                    weight.new_zeros(self.num_layer, batch_size, self.hidden_size))
        else:
            return weight.new_zeros(self.num_layer, batch_size, self.hidden_size)

    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
            
if __name__ == "__main__":
    num_emb = 80
    seq_len = 80
    batch_size = 50
    x = torch.randint(num_emb, (seq_len, batch_size))

    model = RNNModel(num_emb)
    hidden = model.init_hidden(batch_size)
    model(x, hidden)
    