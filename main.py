from config import FLAGS

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from qa_utils.logger import logger
from tqdm import tqdm

from bert import tokenization, modeling
from bert_joint.run_nq import AnswerType,read_nq_examples,convert_examples_to_features,FeatureWriter,\
    input_fn_builder,validate_flags_or_throw,read_candidates,compute_pred_dict,make_submission,make_tfrecords


class QADense(tf.keras.layers.Layer):
    def __init__(self,
                 output_size,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                **kwargs):
        super(QADense,self).__init__(**kwargs)
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self,input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
          raise TypeError("Unable to build `TDense` layer with "
                          "non-floating point (and non-complex) "
                          "dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
          raise ValueError("The last dimension of the inputs to "
                           "`TDense` should be defined. "
                           "Found `None`.")
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.output_size,last_dim],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            "bias",
            shape=[self.output_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        super(QADense, self).build(input_shape)
    def call(self,x):
        return tf.matmul(x,self.kernel,transpose_b=True)+self.bias


def get_model(transform_variable_names=False,is_training=False):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    validate_flags_or_throw(bert_config)
    tf.io.gfile.makedirs(FLAGS.output_dir)

    seq_len = FLAGS.max_seq_length
    unique_id = tf.keras.Input(shape=(1,), dtype=tf.int64, name='unique_id')
    input_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='segment_ids')

    BERT = modeling.BertModel(config=bert_config, name='bert')
    pooled_output, sequence_output = BERT(inputs=[input_ids,input_mask,segment_ids])

    position_layer = QADense(2, name='logits')
    logits = position_layer(sequence_output)
    start_logits, end_logits = tf.split(logits, axis=-1, num_or_size_splits=2, name='split')
    start_logits = tf.squeeze(start_logits, axis=-1, name='start_squeeze')
    end_logits = tf.squeeze(end_logits, axis=-1, name='end_squeeze')

    # num_answer_types = 5  # YES, NO, UNKNOWN, SHORT, LONG
    answer_type_layer = QADense(5, name='answer_types')
    ans_type = answer_type_layer(pooled_output)

    start_logits = tf.identity(start_logits, 'start_positions')
    end_logits = tf.identity(end_logits, 'end_positions')

    # ans_type = tf.identity(start_logits, 'answer_types')
    if  is_training:
        qa_model = tf.keras.Model([input_ for input_ in [input_ids, input_mask, segment_ids]
                           if input_ is not None],
                          [start_logits, end_logits, ans_type],
                          name='bert-baseline')
    else:
        qa_model = tf.keras.Model([input_ for input_ in [input_ids, input_mask, segment_ids]
                                   if input_ is not None],
                                  [unique_id,start_logits, end_logits, ans_type],
                                  name='bert-baseline')
    print(qa_model.summary())

    if transform_variable_names:
        model_params = {v.name: v for v in qa_model.trainable_variables}
        model_roots = np.unique([v.name.split('/')[0] for v in qa_model.trainable_variables])

        saved_names = [k for k, v in tf.train.list_variables(FLAGS.raw_init_checkpoint)]
        a_map = {v: v + ':0' for v in saved_names}
        model_roots = np.unique([v.name.split('/')[0] for v in qa_model.trainable_variables])

        def transform(x):
            x = x.replace('attention/self', 'attention')
            x = x.replace('attention', 'self_attention')
            x = x.replace('attention/output', 'attention_output')

            x = x.replace('/dense', '')
            x = x.replace('/LayerNorm', '_layer_norm')
            x = x.replace('embeddings_layer_norm', 'embeddings/layer_norm')

            x = x.replace('attention_output_layer_norm', 'attention_layer_norm')
            x = x.replace('embeddings/word_embeddings', 'word_embeddings/embeddings')

            x = x.replace('/embeddings/', '/embeddings_postprocessor/')
            #x = x.replace('/token_type_embeddings', '/type_embeddings')
            x = x.replace('/pooler/', '/pooler_transform/')
            x = x.replace('answer_type_output_bias', 'answer_types/bias')
            x = x.replace('answer_type_output_', 'answer_types/')
            x = x.replace('cls/nq/output_', 'logits/')
            x = x.replace('/weights', '/kernel')
            return x

        a_map = {k: model_params.get(transform(v), None) for k, v in a_map.items() if k != 'global_step'}

        tf.compat.v1.train.init_from_checkpoint(ckpt_dir_or_file=FLAGS.raw_init_checkpoint,
                                                assignment_map=a_map)

        cpkt = tf.train.Checkpoint(model=qa_model)
        cpkt.save(FLAGS.init_checkpoint)
    else:
        pass
        # cpkt = tf.train.Checkpoint(model=qa_model)
        # cpkt.restore(FLAGS.init_checkpoint).assert_consumed()

    if is_training:
        # Computes the loss for positions.
        def compute_loss(positions,logits):
            positions = tf.cast(positions, tf.int32)
            logits = tf.cast(logits, tf.float32)
            one_hot_positions = tf.one_hot(
                positions, depth=FLAGS.max_seq_length, dtype=tf.float32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            loss = -tf.reduce_mean(
                tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
            return loss

        # Computes the loss for labels.
        def compute_label_loss(labels,logits):
            labels = tf.cast(labels, tf.int32)
            logits = tf.cast(logits, tf.float32)
            one_hot_labels = tf.one_hot(
                labels, depth=len(AnswerType), dtype=tf.float32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            loss = -tf.reduce_mean(
                tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
            return loss

        # Specify the training configuration (optimizer, loss, metrics)
        qa_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=FLAGS.learning_rate),  # Optimizer
                      # Loss function to minimize
                      loss={'tf_op_layer_start_positions': compute_loss,
                            'tf_op_layer_end_positions': compute_loss,
                            'answer_types': compute_label_loss},
                         # Loss function to minimize
                      loss_weights={'tf_op_layer_start_positions': 0.3,
                            'tf_op_layer_end_positions': 0.3,
                            'answer_types': 0.4},
                      )


    return qa_model

def process_train_nq_file():
    logger.info('Process train file %s', FLAGS.train_file)
    if os.path.exists(FLAGS.train_tfrecord_file.replace('??','00')):
        logger.info('train tfrecord file already processed')
        return FLAGS.train_tfrecord_file

    make_tfrecords(FLAGS.train_file, is_training=True, num_shards=FLAGS.num_shards)

    return

def process_test_nq_file():
    logger.info('Process test file %s', FLAGS.predict_file)
    if os.path.exists(FLAGS.predict_tfrecord_file):
        logger.info('test file already processed')
        return FLAGS.predict_tfrecord_file

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    eval_examples = read_nq_examples(
        input_file=FLAGS.predict_file, is_training=False)
    eval_writer = FeatureWriter(
        filename=FLAGS.predict_tfrecord_file,
        is_training=False)
    eval_features = []

    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

    num_spans_to_ids = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    for spans, ids in num_spans_to_ids.items():
        logger.info("  Num split into %d = %d", spans, len(ids))

    return eval_writer.filename


def prediction():
    prediction_tfrecord_fname = process_test_nq_file()
    logger.info('Predict file: %s', prediction_tfrecord_fname)
    test_dataset, test_dataset_batch = input_fn_builder(
        input_file=prediction_tfrecord_fname,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        batch_size=FLAGS.predict_batch_size)

    # print(next(iter(test_dataset_batch.take(1))))

    # estimator = get_estimator()
    qa_model = get_model(transform_variable_names=True)

    logger.info("***** Running predictions *****")

    results = qa_model.predict_generator(test_dataset_batch, verbose=True)
    results = dict(zip(['uniqe_id', 'start_logits', 'end_logits', 'answer_type_logits'], results))

    all_results = []
    for result in zip(results['uniqe_id'],results['start_logits'],results['end_logits'],
                      results['answer_type_logits']):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        unique_id, start_logits, end_logits, answer_type_logits = result
        all_results.append({
                'unique_id' : unique_id[0],
                'start_logits' : start_logits.tolist(),
                'end_logits' : end_logits.tolist(),
                'answer_type_logits' : answer_type_logits.tolist()})

    candidates_dict = read_candidates(FLAGS.predict_file)
    eval_features = test_dataset.map(
        lambda record: tf.io.parse_single_example(serialized=record,
                                  features={
                                      "unique_id": tf.io.FixedLenFeature([], tf.int64),
                                      "token_map": tf.io.FixedLenFeature([FLAGS.max_seq_length],
                                                                         tf.int64),
                                                  })
    )
    eval_features = list(eval_features)

    with open('./all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    with open('./candidates_dict.pkl','wb') as f:
        pickle.dump(candidates_dict, f)

    with open('./eval_features.pkl','wb') as f:
        pickle.dump(eval_features, f)

    # with open('./all_results.pkl', 'rb') as f:
    #     all_results = pickle.load(f)
    #
    # with open('./candidates_dict.pkl','rb') as f:
    #     candidates_dict = pickle.load(f)
    #
    # with open('./eval_features.pkl','rb') as f:
    #     eval_features = pickle.load(f)
    #
    # print(all_results)

    nq_pred_dict = compute_pred_dict(candidates_dict, eval_features,all_results)
    predictions_json = {"predictions": list(nq_pred_dict.values())}
    with open(FLAGS.output_prediction_file, "w") as f:
        json.dump(predictions_json, f, indent=4)

    make_submission(FLAGS.output_prediction_file)


def train():
    train_tfrecord_fname = process_train_nq_file()
    n_training_file = int(FLAGS.train_valid_split_ratio*FLAGS.num_shards)
    training_fnames = [train_tfrecord_fname.replace('??','%02d'%i) for i in range(n_training_file)]
    validation_fnames = [train_tfrecord_fname.replace('??','%02d'%i) for i in range(n_training_file,FLAGS.num_shards)]

    for i,fname in enumerate(training_fnames):
        logger.info('Training file %d/%d: %s',i,n_training_file,fname)
    for i,fname in enumerate(validation_fnames):
        logger.info('Validation file %d/%d: %s',i,FLAGS.num_shards-n_training_file,fname)

    train_dataset, train_dataset_batch = input_fn_builder(
        input_file=training_fnames,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        n_repeats=FLAGS.num_train_epochs,
        drop_remainder=False,
        batch_size=FLAGS.train_batch_size)

    valid_dataset, valid_dataset_batch = input_fn_builder(
        input_file=validation_fnames,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        n_repeats=FLAGS.num_train_epochs,
        drop_remainder=False,
        batch_size=FLAGS.train_batch_size)

    # n_training_data,n_training_batch = 0,0
    # for v_data in tqdm(train_dataset_batch):
    #     n_training_batch += 1
    #     n_training_data += v_data[0]['input_ids'].shape[0]
    # logger.info('Training data : %d samples. %d batch', n_training_data, n_training_batch)
    #
    # n_valid_data,n_valid_batch = 0,0
    # for v_data in valid_dataset_batch:
    #     n_valid_batch += 1
    #     n_valid_data += v_data[0]['input_ids'].shape[0]
    # logger.info('Validation data : %d samples. %d batch', n_valid_data, n_valid_batch)

    qa_model = get_model(transform_variable_names=True,is_training=True)
    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath=FLAGS.checkpoint_path,
        verbose=1,
        save_weights_only=True,
        # save_best_only=True,
        save_freq=10000)

    logger.info("***** Training *****")
    qa_model.fit(x=train_dataset_batch,steps_per_epoch=54577,epochs=100,#11939
                           validation_data=valid_dataset_batch,validation_steps=13644,
                           verbose=1, callbacks=[cp_callback])

if __name__ == '__main__':
    train()
    prediction()