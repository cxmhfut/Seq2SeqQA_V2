import tensorflow as tf
from model import Seq2SeqModel
from FLAGS import flags
from data_helpers import getBatches
from tqdm import tqdm
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train():
    with tf.Session() as sess:
        model = Seq2SeqModel(flags, mode='train', beam_search=False)
        ckpt = tf.train.get_checkpoint_state(flags.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters...')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Create new model parameters...')
            sess.run(tf.global_variables_initializer())
        current_step = 0
        summary_writer = tf.summary.FileWriter(flags.model_dir, graph=sess.graph)
        num_epochs = flags.num_epochs
        for epoch in range(num_epochs):
            print("----- Epoch {}/{} -----".format(epoch + 1, num_epochs))
            batches = getBatches(model.trainingSamples, flags.batch_size)
            for batch in tqdm(batches, desc='Training'):
                loss, summary = model.train(sess, batch)
                current_step += 1
                if current_step % flags.steps_per_checkpoint == 0:
                    perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                    tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step,loss,perplexity))
                    summary_writer.add_summary(summary,current_step)
                    checkpoint_path = os.path.join(flags.model_dir,flags.model_name)
                    model.saver.save(sess,checkpoint_path,global_step=current_step)

if __name__ == '__main__':
    train()