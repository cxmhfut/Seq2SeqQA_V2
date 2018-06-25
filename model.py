import tensorflow as tf
from data_helpers import loadDataset


class Seq2SeqModel:
    def __init__(self, flags):
        self.flags = flags
        word2id, id2word, trainingSamples = loadDataset(flags.dataset_filename)

        self.word2id = word2id
        self.id2word = id2word
        self.vocab_size = len(word2id)

        # hyper parameters
        self.rnn_size = flags.rnn_size  # Number of hidden units in each layer
        self.keep_prob_dropout = flags.keep_prob_dropout  # dropout
        self.num_layers = flags.num_layers  # Number of layers in each encoder and decoder

        self.build_model()

    def create_rnn_cell(self):
        def single_rnn_cell():
            """
            创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell函数，
            不然直接把cell放在MultiRNNCell的列表中最终模型会发生错误
            :return:
            """
            single_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(single_cell,
                                                 input_keep_prob=1.0,
                                                 output_keep_prob=self.keep_prob_dropout
                                                 )
            return cell

        # 列表中每个元素都是调用sing_rnn_cell函数
        cell = tf.nn.rnn_cell.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_model(self):
        """
        build model
        :return:
        """

