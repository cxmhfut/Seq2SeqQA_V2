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

tf.app.flags.DEFINE_string(
    'model_dir',
    'model',
    'Path to save model checkpoints')

tf.app.flags.DEFINE_string(
    'model_name',
    'model.ckpt',
    'File name used for model checkpoints')

tf.app.flags.DEFINE_string(
    'padToken',
    '<pad>',
    'pad token'
)

tf.app.flags.DEFINE_string(
    'goToken',
    '<go>',
    'go token'
)

tf.app.flags.DEFINE_string(
    'endToken',
    '<eos>',
    'end token'
)

tf.app.flags.DEFINE_string(
    'unkToken',
    '<unk>',
    'unknow token'
)

tf.app.flags.DEFINE_integer(
    'num_epochs',
    50,
    'the number of epochs'
)

tf.app.flags.DEFINE_float(
    'learning_rate',
    0.0001,
    'learning rate'
)

tf.app.flags.DEFINE_integer(
    'rnn_size',
    1024,
    'Number of hidden units in each layer'
)

tf.app.flags.DEFINE_integer(
    'batch_size',
    128,
    'Batch size'
)

tf.app.flags.DEFINE_integer(
    'beam_size',
    3,
    'Beam size'
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

tf.app.flags.DEFINE_integer(
    'steps_per_checkpoint',
    100,
    'Save model checkpoint every this iteration'
)

flags = tf.app.flags.FLAGS
