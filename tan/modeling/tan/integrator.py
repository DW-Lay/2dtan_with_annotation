import torch
from torch import nn
from torch.functional import F

class Integrator(nn.Module):
    def __init__(self, feat_hidden_size, query_input_size, query_hidden_size, 
            bidirectional, num_layers):
        super(Integrator, self).__init__()
        if bidirectional:
            # python3以后  '/'表示浮点数除法   '//'表示整数除法
            query_hidden_size //= 2
        # input_size ：输入的维度
        # hidden_size：h的维度
        # num_layers：堆叠LSTM的层数，默认值为1
        # bias：偏置 ，默认值：True
        # batch_first： 如果是True，则input为(batch(batch数), seq(单词个数), input_size(向量尺寸))。默认值为：False（seq_len, batch, input_size）
        # bidirectional ：是否双向传播，默认值为False
        self.lstm = nn.LSTM(
            query_input_size, query_hidden_size, num_layers=num_layers, 
            bidirectional=bidirectional, batch_first=True
        )
        # 一个全链接层
        self.fc = nn.Linear(query_hidden_size, feat_hidden_size)
        # 将map2d按照（512）的深度进行一次卷积操作
        self.conv = nn.Conv2d(feat_hidden_size, feat_hidden_size, 1, 1)

    def encode_query(self, queries, wordlens):
        # 当我们有多块GPU，并且全部使用时，训练lstm，需要进行参数展平的操作，否则会有一堆提示信息
        self.lstm.flatten_parameters()
        # 输出 
        # output ：（batch, seq_len,  num_directions * hidden_size）
        # h_n：(num_layers * num_directions, batch, hidden_size)
        # c_n ：（num_layers * num_directions, batch, hidden_size） 
        # 所以下面的[0]是为了取出output   queries 为 eg. （4(batch),22(seq_len),512(hidden_size))
        queries = self.lstm(queries)[0] 
        # 以下结果为  (4,512)
        # wordlens.long() 为每个batch里句子的长度， 如果batch为4 则wordlens.long()=[23,54,12,34]其中的数字为每个句子中单词的个数
        # queries.size(0) 为batch数
        # queries[range(queries.size(0)), wordlens.long() - 1] 比如queries[0,15] 取得就是第一个batch中经过lstm的最后一个向量，代表了整个句子 
        queries = queries[range(queries.size(0)), wordlens.long() - 1]
        return self.fc(queries)

    def forward(self, queries, wordlens, map2d):
        queries = self.encode_query(queries, wordlens)[:,:,None,None]
        map2d = self.conv(map2d)
        # print('map2d.shape',map2d.shape)
        return F.normalize(queries * map2d)

def build_integrator(cfg):
    feat_hidden_size = cfg.MODEL.TAN.FEATPOOL.HIDDEN_SIZE 
    query_input_size = cfg.INPUT.PRE_QUERY_SIZE
    query_hidden_size = cfg.MODEL.TAN.INTEGRATOR.QUERY_HIDDEN_SIZE 
    bidirectional = cfg.MODEL.TAN.INTEGRATOR.LSTM.BIDIRECTIONAL
    num_layers = cfg.MODEL.TAN.INTEGRATOR.LSTM.NUM_LAYERS
    return Integrator(feat_hidden_size, query_input_size, query_hidden_size, 
        bidirectional, num_layers) 
