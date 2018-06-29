import numpy as np
import tensorflow as tf
from model import Seq2SeqModel
from FLAGS import flags
from data_helpers import sentence2enco
import sys

def predict_ids_seq(predict_ids, id2word, beam_size):
    """
    将beam_search返回的结果转化为字符串
    :param predict_ids:列表，长度为batch_size，每个元素都是decoder_len*beam_size的数组
    :param id2word:
    :param beam_size:
    :return:
    """
    for single_predict in predict_ids:
        for i in range(beam_size):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = [id2word[idx] for idx in predict_list[0]]
            print(" ".join(predict_seq))

def predict():
    with tf.Session() as sess:
        model = Seq2SeqModel(flags,mode='predict',beam_search=True)
        ckpt = tf.train.get_checkpoint_state(flags.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters...')
            model.saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(flags.model_dir))
        sys.stdout.write(">")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            batch = sentence2enco(sentence,model.word2id)
            predict_ids = model.infer(sess,batch)
            predict_ids_seq(predict_ids,model.id2word,model.beam_size)
            print(">")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

if __name__ == '__main__':
    predict()