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
        self.embedding_size = flags.embedding_size  # Embedding matrix size [vocab_size,embedding_size]

        # model
        self.batch_size = None
        self.encoder_inputs = None
        self.encoder_inputs_length = None
        self.decoder_targets = None
        self.decoder_targets_length = None
        self.max_target_sequence_length = None
        self.mask = None

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
        print('Building model...')
        # 1 定义模型的placeholder
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')

        # 根据目标序列长度，选出其中最大值，然后使用该值构建序列长度的mask标志。
        """
        tf.sequence_mask():
            tf.sequence_mask([1, 3, 2], 5)
            [[ True False False False False]
             [ True  True True False False]
             [ True  True  False False False]]
        """
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length,
                                     self.max_target_sequence_length,
                                     dtype=tf.float32,
                                     name='masks')
        # 2 定义模型的encoder部分
        with tf.variable_scope('encoder'):
            # 创建LSTMCell，两层+dropout
            encoder_cell = self.create_rnn_cell()
            # 构建Embedding矩阵，encoder和decoder共用该词向量矩阵
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
            encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
            # 使用dynamic_rnn构建LSTM模型，将输入编码成隐层向量
            # encoder_outputs用于attention，batch_size*encoder_inputs_length*rnn_size
            # encoder_state用于decoder的初始状态，batch_size*rnn_size
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                               encoder_inputs_embedded,
                                                               sequence_length=self.encoder_inputs_length,
                                                               dtype=tf.float32)
