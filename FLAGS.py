import tensorflow as tf

tf.app.flags.DEFINE_string(
    'word_id_filename',
    'data/word_id.pkl',
    "the path of dict data: word2id = data['word2id'], id2word = data['id2word']"
)

tf.app.flags.DEFINE_string(
    'train_samples_filename',
    'data/train_samples.pkl',
    "the path of train samples: train_samples = data['train_samples']"
)

tf.app.flags.DEFINE_string(
    'dataset_filename',
    'data/dataset-70425-vocabSize33050.pkl',
    "the path of dataset: word2id = data['word2id], id2word = data['id2word'], trainingSamples = data['trainingSamples']"
)

tf.app.flags.DEFINE_integer(
    'rnn_size',
    1024,
    'Number of hidden units in each layer'
)

tf.app.flags.DEFINE_float(
    'keep_prob_dropout',
    0.9,
    'dropout'
)

tf.app.flags.DEFINE_integer(
    'num_layers',
    2,
    'Number of layers in each encoder and decoder'
)

tf.app.flags.DEFINE_integer(
    'embedding_size',
    1024,
    'Embedding matrix shape = (vocab_size, embedding_size)'
)

flags = tf.app.flags.FLAGS
