from config import FLAGS

import os
import json
import pickle
import numpy as np
from absl import logging, app
import tensorflow as tf

from bert import tokenization, modeling
from bert_joint.run_nq import read_nq_examples,convert_examples_to_features,FeatureWriter,\
    input_fn_builder,validate_flags_or_throw,read_candidates,compute_pred_dict,make_submission

logging.set_verbosity(logging.INFO)


class QADense(tf.keras.layers.Layer):
    def __init__(self,
                 output_size,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                **kwargs):
        super().__init__(**kwargs)
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
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: last_dim})
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


def get_model(transform_variable_names=False):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    validate_flags_or_throw(bert_config)
    tf.io.gfile.makedirs(FLAGS.output_dir)

    seq_len = FLAGS.max_seq_length
    unique_id = tf.keras.Input(shape=(1,), dtype=tf.int64, name='unique_id')
    input_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='segment_ids')

    BERT = modeling.BertModel(config=bert_config, name='bert')
    pooled_output, sequence_output = BERT(input_word_ids=input_ids,
                                          input_mask=input_mask,
                                          input_type_ids=segment_ids)

    position_layer = QADense(2, name='logits')
    logits = position_layer(sequence_output)
    start_logits, end_logits = tf.split(logits, axis=-1, num_or_size_splits=2, name='split')
    start_logits = tf.squeeze(start_logits, axis=-1, name='start_squeeze')
    end_logits = tf.squeeze(end_logits, axis=-1, name='end_squeeze')

    # num_answer_types = 5  # YES, NO, UNKNOWN, SHORT, LONG
    answer_type_layer = QADense(5, name='ans_type')
    ans_type = answer_type_layer(pooled_output)

    # logits = QADense(2, name='logits')(sequence_output)
    # start_logits, end_logits = tf.split(logits, axis=-1, num_or_size_splits=2, name='split')
    # start_logits = tf.squeeze(start_logits, axis=-1, name='start_squeeze')
    # end_logits = tf.squeeze(end_logits, axis=-1, name='end_squeeze')
    #
    # ans_type = QADense(5, name='ans_type')(pooled_output)

    qa_model = tf.keras.Model([input_ for input_ in [unique_id, input_ids, input_mask, segment_ids]
                           if input_ is not None],
                          [unique_id, start_logits, end_logits, ans_type],
                          name='bert-baseline')

    # print(qa_model.summary())

    if transform_variable_names:
        model_params = {v.name: v for v in qa_model.trainable_variables}
        model_roots = np.unique([v.name.split('/')[0] for v in qa_model.trainable_variables])
        for k in model_params:
            print(k)
        print(model_roots)
        #
        saved_names = [k for k, v in tf.train.list_variables(FLAGS.raw_init_checkpoint)]
        a_map = {v: v + ':0' for v in saved_names}
        model_roots = np.unique([v.name.split('/')[0] for v in qa_model.trainable_variables])
        #for k in saved_names:
        #    print(k)
        print(model_roots)

        def transform(x):
            print(x, end=' : ')
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
            x = x.replace('answer_type_output_bias', 'ans_type/bias')
            x = x.replace('answer_type_output_', 'ans_type/')
            x = x.replace('cls/nq/output_', 'logits/')
            x = x.replace('/weights', '/kernel')
            print(x)
            return x

        a_map = {k: model_params.get(transform(v), None) for k, v in a_map.items() if k != 'global_step'}
        for k,v in a_map.items():
            if v is None:
                print(k,v)
            #print(k, ' : ',v)
        tf.compat.v1.train.init_from_checkpoint(ckpt_dir_or_file=FLAGS.raw_init_checkpoint,
                                                assignment_map=a_map)

        cpkt = tf.train.Checkpoint(model=qa_model)
        cpkt.save(FLAGS.init_checkpoint)
    else:
        cpkt = tf.train.Checkpoint(model=qa_model)
        cpkt.restore(FLAGS.init_checkpoint).assert_consumed()

    return qa_model

def process_test_nq_file():
    logging.info('Process test file %s', FLAGS.predict_file)
    if os.path.exists(FLAGS.predict_tfrecord_file):
        logging.info('test file already processed')
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

    logging.info("  Num orig examples = %d", len(eval_examples))
    logging.info("  Num split examples = %d", len(eval_features))
    for spans, ids in num_spans_to_ids.items():
        logging.info("  Num split into %d = %d", spans, len(ids))

    return eval_writer.filename


def prediction(_):
    prediction_tfrecord_fname = process_test_nq_file()
    logging.info('Predict file: %s', prediction_tfrecord_fname)
    test_dataset, test_dataset_batch = input_fn_builder(
        input_file=prediction_tfrecord_fname,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        batch_size=FLAGS.predict_batch_size)

    # estimator = get_estimator()
    qa_model = get_model(transform_variable_names=False)

    logging.info("***** Running predictions *****")

    results = qa_model.predict_generator(test_dataset_batch, verbose=True)
    results = dict(zip(['uniqe_id', 'start_logits', 'end_logits', 'answer_type_logits'], results))

    print(results)

    all_results = []
    for result in zip(results['uniqe_id'],results['start_logits'],results['end_logits'],
                      results['answer_type_logits']):
        if len(all_results) % 1000 == 0:
            logging.info("Processing example: %d" % (len(all_results)))
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

    nq_pred_dict = compute_pred_dict(candidates_dict, eval_features,all_results)
    predictions_json = {"predictions": list(nq_pred_dict.values())}
    with open(FLAGS.output_prediction_file, "w") as f:
        json.dump(predictions_json, f, indent=4)

    make_submission(FLAGS.output_prediction_file)
if __name__ == '__main__':
    app.run(prediction)
