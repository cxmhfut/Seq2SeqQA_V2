import tensorflow as tf

tf.app.flags.DEFINE_string(
    'word_id_filename',
    'data/word_id.pkl',
    "the path of dict data: word2id = data['word2id'] and id2word = data['id2word']"
)

tf.app.flags.DEFINE_string(
    'train_samples_filename',
    'data/train_samples.pkl',
    "the path of train samples: train_samples = data['train_samples']"
)

flags = tf.app.flags.FLAGS