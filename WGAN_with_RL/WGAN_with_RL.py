from __future__ import division
from __future__ import print_function
import tensorflow as tf
import cPickle as pkl
import numpy as np

CORPUS = 'arxiv.txt' # set the CORPUS for train
EPOCH_NUM = 100  # default 100
TRAIN_STEPS_NUM = 100 # default 100
USING_CPU = False # set False if having GPU
BATCH_SIZE = 32 # default 1
SAVE_PATH = 'save/'
LEN = 100 # set the length for training
PG = 1000 # the maximum length of training seqs for policy gradient
EMB = 'emb_mapping.pkl'
ST, ED = '|', '^'
STI, EDI = 0, 1

# one-hot embedding
def embedding(corpus, emb=EMB):
    with open(corpus, 'r') as f:
        mapping_dic = set(f.read())
    # start token
    if ST in mapping_dic:
        mapping_dic.remove(ST)
    # end token
    if ED in mapping_dic:
        mapping_dic.remove(ED)
    mapping_dic = [ST, ED] + list(mapping_dic)
    mapping_dic = dict((c, i) for i, c in enumerate(mapping_dic))
    with open(emb, 'wb') as f:
        pkl.dump(mapping_dic, f)
    inverse_mapping_dic = dict((i, c) for c, i in mapping_dic.items())
    return mapping_dic, inverse_mapping_dic

def mapping(corpus, mapping_dic, batch_size=10, trian_len=50):
    with open(corpus, 'r') as f:
        corpus_content = f.read()
    start_i = np.arange(0, len(corpus_content) - trian_len)
    while True:
        batch_i = np.random.choice(start_i, size=(trian_len,), replace=False)
        lines = [corpus_content[i:i + trian_len] for i in batch_i]
        # tokens
        tkn = [[mapping_dic[i] for i in tx] for tx in lines]
        yield np.asarray(tkn)

def inverse_mapping(idex, inverse_mapping_dic, argmax=False):
    if argmax:
        idex = np.argmax(idex, axis=-1)
    char = [inverse_mapping_dic.get(i, None) for i in idex]
    # text
    tx = ''.join(c for c in char if c)
    return tx

def scope_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

# define WGAN model
class WGAN_RL(object):
    def __init__(self, sess, words_num, D_times=None, using_cpu=False, sava_path='save/', prior_dim=100):
        # text lens
        self.tl_ = tf.placeholder(dtype='int32', shape=())
        # text
        self.t_ = tf.placeholder(dtype='int32', shape=(None, None))
        # latent dimensions
        self.l_ = tf.placeholder(dtype='float32', shape=(None, prior_dim))
        self.time = tf.Variable(0, name='time')
        # sample (bool)
        self.for_sam = tf.placeholder(dtype='bool', shape=())
        self.sess = sess
        self.words_num = words_num
        # the dimensions of prior
        self.prior_dim = prior_dim
        self.save_path = sava_path
        self.using_cpu = using_cpu
        self.weights = []
        self.D_times = D_times # the times(num) for training Discriminator
    @property
    def current_time(self):
        return self.sess.run(self.time)

    # generate the prior vector for generator
    def get_prior_dim(self, batch_size):
        return np.random.normal(size=(batch_size, self.prior_dim))

    # the project martix
    def random_W(self, name, shape, cpu_or_gpu='gpu', trainable=True):
        on_gpu = True
        if cpu_or_gpu == 'gpu':
            on_gpu = True
        elif cpu_or_gpu == 'cpu':
            on_gpu = False
        if self.using_cpu:
            on_gpu = False
        with tf.device('/gpu:0' if on_gpu else '/cpu:0'):
            W = tf.get_variable(name=name,
                                shape=shape,
                                initializer=(lambda shape, dtype, partition_info:
                                                  tf.random_normal(shape, stddev=np.sqrt(6. / sum(shape)))),
                                trainable=trainable)
        self.weights.append(W)
        return W

    def G(self, rnn_stack=2, hidden_size=100):
        with tf.variable_scope('generator'):
            output_W = self.random_W('output_W', (hidden_size, self.words_num))
            output_fn = lambda x: tf.matmul(x, output_W)
            cells = [tf.contrib.rnn.GRUCell(hidden_size)] * rnn_stack
            cell = tf.contrib.rnn.MultiRNNCell(cells)

            def proWB(i, activation=tf.tanh):
                W = self.random_W('rnn_proj_%d_W' % i, (self.prior_dim, hidden_size))
                b = self.random_W('rnn_proj_%d_b' % i, (hidden_size,))
                # projection
                p = activation(tf.matmul(self.l_, W) + b)
                return p

            encoder_state = tuple(proWB(i) for i in range(rnn_stack))
            batch_size = tf.shape(self.l_)[0]
            supervise_train = tf.concat([tf.zeros_like(self.t_[:, :1]), self.t_[:, :-1]], axis=1)
            supervise_train = tf.one_hot(supervise_train, self.words_num)
            # supervised training fn function (tensorflow)
            spf = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
            LENGTH = tf.ones((batch_size,), 'int32') * self.tl_
            # s, t can be ignore
            super_anser, s, t = tf.contrib.seq2seq.dynamic_rnn_decoder(cell=cell, inputs=supervise_train, decoder_fn=spf, sequence_length=LENGTH)
            super_anser = tf.einsum('ijk,kl->ijl', super_anser, output_W)
            superLoss = tf.contrib.seq2seq.sequence_loss(logits=super_anser, targets=self.t_, weights=tf.ones((batch_size, self.tl_)))
            superLoss = tf.reduce_mean(superLoss)
            tf.get_variable_scope().reuse_variables()
            # embeddings
            embs = tf.eye(self.words_num)
            sampling = tf.contrib.seq2seq.simple_decoder_fn_inference(
                                                output_fn=output_fn,
                                                encoder_state=encoder_state,
                                                embeddings=embs,
                                                start_of_sequence_id=0,
                                                end_of_sequence_id=-1,
                                                maximum_length=self.tl_ - 1,
                                                num_decoder_symbols=self.words_num,
                                                name='decoder_inference_fn')
            sampling_seq, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(cell=cell, decoder_fn=sampling)
            softmax_ans = tf.nn.softmax(sampling_seq)
            sampling_seq = tf.argmax(sampling_seq, axis=-1)

        tf.summary.scalar('loss/NLL', superLoss)
        return softmax_ans, sampling_seq, superLoss

    def D(self, seq_for_D, reuse=False, rnn_stack=3, rnn_dims=200):
        with tf.variable_scope('discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            convert_to_one = tf.one_hot(seq_for_D, self.words_num)
            cells = [tf.contrib.rnn.GRUCell(rnn_dims)] * rnn_stack
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            cell = tf.contrib.rnn.FusedRNNCellAdaptor(cell, True)
            convert_to_one = tf.transpose(convert_to_one, (1, 0, 2)) # set that if seq too long
            the_output, _ = cell(convert_to_one, dtype='float32')
            the_output = tf.transpose(the_output, (1, 0, 2)) # set that if seq too long
            pred_W = self.random_W('pred_W', (rnn_dims, 1))
            D_output = tf.einsum('ijk,kl->ijl', the_output, pred_W)
        return D_output

    def D_train(self, real, generate, DW):
        with tf.variable_scope('loss/discriminator'):
            # build opt
            D_OP = tf.train.RMSPropOptimizer(1e-4)
            # real data
            RV = -tf.reduce_mean(real)
            # generated data
            GV = tf.reduce_mean(generate)
            DLoss = RV + GV
            # regularization terms
            with tf.variable_scope('REGU'):
                D_REGU = sum([tf.nn.l2_loss(w) for w in DW]) * 1e-4
            total_loss = DLoss + D_REGU
            tf.summary.scalar('D_loss', total_loss + 350)
            D_train = D_OP.minimize(total_loss, var_list=DW)
            # weights clipping
            clip_D = [p.assign(tf.clip_by_value(p, -1, 1)) for p in DW]
        return D_train, clip_D

    def G_train(self, result, rewards, softmax_value, GW):
        with tf.variable_scope('loss/generator'):
            # reward opt
            R_OP = tf.train.GradientDescentOptimizer(1e-3)
            # generator opt
            G_OP = tf.train.RMSPropOptimizer(1e-4)
            result = tf.one_hot(result, self.words_num)
            softmax_value = tf.clip_by_value(softmax_value * result, 1e-15, 1)
            # expected reward by MCS
            MCS = tf.Variable(tf.zeros((PG,)))
            reward = rewards - MCS[:tf.shape(rewards)[1]]
            threshold = tf.reduce_mean(tf.abs(reward))
            # expected
            EOP = R_OP.minimize(
                threshold, var_list=[MCS])
            reward = tf.expand_dims(tf.cumsum(reward, axis=1, reverse=True), -1)
            # generator rewards
            GR = tf.log(softmax_value) * reward
            # mean reward
            GR = tf.reduce_mean(GR)
            GLoss = -GR
            with tf.variable_scope('REGU'):
                G_REGU = sum([tf.nn.l2_loss(w) for w in GW]) * 1e-5
            # loss + regularization
            G_ALL = GLoss + G_REGU
            G_train = G_OP.minimize(G_ALL, var_list=GW)
            G_train = tf.group(G_train, EOP)
        return G_train

    def build(self, REGU=1e-4):
        softmax_ans, sampling_seq, superLoss = self.G()
        real = self.D(self.t_)
        # generated sentences
        gene = self.D(sampling_seq, reuse=True)
        # generator params
        GWs = scope_variables('generator')
        # discriminator params
        DWs = scope_variables('discriminator')
        self.do_sampling = sampling_seq
        D_train, clip_D = self.D_train(real, gene, DWs)
        G_train = self.G_train(sampling_seq, gene, softmax_ans, GWs)
        LR = 10000. / (10000. + tf.cast(self.time, 'float32'))
        # learning rate
        LR *= 1e-3
        OPT = tf.train.AdamOptimizer(LR)
        super_train = OPT.minimize(superLoss)
        G_train = tf.group(G_train, super_train)
        count = self.time.assign(self.time + 1)
        # decide the phase of discriminator and generator
        switch = tf.cond(tf.equal(tf.mod(self.time, self.D_times), 0), lambda: G_train, lambda: D_train)
        self.WGAN_train = tf.group(switch, count)
        self.clip_D = clip_D
        self.summary_writer = tf.summary.FileWriter(self.save_path, self.sess.graph)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.done = True

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save(self):
        self.saver.save(self.sess, self.save_path + 'CKPT.ckpt')

    def do_train(self, batch):
        batchs, leng = batch.shape
        prior = self.get_prior_dim(batchs)
        feed_dict = {self.t_: batch, self.tl_: leng, self.l_: prior, self.for_sam: False}
        t = self.sess.run(self.time)
        _, _, summary = self.sess.run([self.WGAN_train, self.clip_D, self.summary_op], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, t)

    def gene(self, sample_len):
        latent = self.get_prior_dim(1)
        sample, = self.sess.run([self.do_sampling], feed_dict={self.l_: latent, self.tl_: sample_len, self.for_sam: True})
        return sample[0]

# main function
if __name__ == '__main__':
    map, inv_map = embedding(CORPUS)
    words_num = len(map)
    sampler = mapping(CORPUS, map, batch_size=BATCH_SIZE, trian_len=LEN)
    sess = tf.Session()
    WGAN = WGAN_RL(sess, words_num, sava_path=SAVE_PATH, D_times=5, using_cpu=USING_CPU)
    WGAN.build()
    WGAN.load()
    f = open("record.txt", 'w')
    for epoch in xrange(1, EPOCH_NUM + 1):
        for step in xrange(1, TRAIN_STEPS_NUM + 1):
            print('%d steps of epoch %d' % (step, epoch))
            f.write('%d steps of epoch %d\n' % (step, epoch))
            WGAN.do_train(sampler.next())
        the_sample = WGAN.gene(500)
        print(inverse_mapping(the_sample, inv_map))
        f.write('=====' * 10 + '\n' + inverse_mapping(the_sample, inv_map) + '\n' + '=====' * 10 + '\n')
        WGAN.save()
    f.write("finished\n")
    f.close()
