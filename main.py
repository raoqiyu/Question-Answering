

from absl import logging, flags, app
from bert import tokenization, modeling
from bert_joint.run_nq import read_nq_examples,convert_examples_to_features,FeatureWriter,\
    input_fn_builder,validate_flags_or_throw,model_fn_builder,RawResult,read_candidates,compute_pred_dict

import os
import json
import numpy as np
import tensorflow as tf

logging.set_verbosity(logging.INFO)

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

# del_all_flags(flags.FLAGS)
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", './data/bert-joint-baseline/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "predict_file", './data/tensorflow2-question-answering/simplified-nq-test.jsonl',
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")
flags.DEFINE_string(
    "predict_tfrecord_file", './data/tensorflow2-question-answering/simplified-nq-test.tfrecord',
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")
flags.DEFINE_string(
    "output_dir", './model/',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")

flags.DEFINE_string(
    "raw_init_checkpoint", './data/bert-joint-baseline/bert_joint.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string(
    "init_checkpoint", './model/bert-joint-baseline/bert_joint.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")


flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("train_precomputed_file", None,
                    "Precomputed tf records for training.")

flags.DEFINE_integer("train_num_precomputed", None,
                     "Number of precomputed tf records for training.")


flags.DEFINE_string("vocab_file", './data/bert-joint-baseline/vocab-nq.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_boolean(
    "skip_nested_contexts", True,
    "Completely ignore context that are not top level nodes in the page.")
flags.DEFINE_integer(
    "max_position", 50,
    "Maximum context position for which to generate special tokens.")
flags.DEFINE_integer("max_contexts", 48,
                     "Maximum number of contexts to output for an example.")
flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")
flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")
flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")
flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

class TDense(tf.keras.layers.Layer):
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
        super(TDense, self).build(input_shape)
    def call(self,x):
        return tf.matmul(x,self.kernel,transpose_b=True)+self.bias


def get_model(transform_variable_names=False):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    validate_flags_or_throw(bert_config)
    tf.io.gfile.makedirs(FLAGS.output_dir)

    seq_len = bert_config.max_position_embeddings
    unique_id = tf.keras.Input(shape=(1,), dtype=tf.int64, name='unique_id')
    input_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='segment_ids')

    BERT = modeling.BertModel(config=bert_config, name='bert')
    pooled_output, sequence_output = BERT(input_word_ids=input_ids,
                                          input_mask=input_mask,
                                          input_type_ids=segment_ids)

    logits = TDense(2, name='logits')(sequence_output)
    start_logits, end_logits = tf.split(logits, axis=-1, num_or_size_splits=2, name='split')
    start_logits = tf.squeeze(start_logits, axis=-1, name='start_squeeze')
    end_logits = tf.squeeze(end_logits, axis=-1, name='end_squeeze')

    ans_type = TDense(5, name='ans_type')(pooled_output)

    qa_model = tf.keras.Model([input_ for input_ in [unique_id, input_ids, input_mask, segment_ids]
                           if input_ is not None],
                          [unique_id, start_logits, end_logits, ans_type],
                          name='bert-baseline')

    print(qa_model.summary())

    if transform_variable_names:
        model_params = {v.name: v for v in qa_model.trainable_variables}
        # model_roots = np.unique([v.name.split('/')[0] for v in qa_model.trainable_variables])
        # print(model_roots)
        #
        saved_names = [k for k, v in tf.train.list_variables(FLAGS.raw_init_checkpoint)]
        a_map = {v: v + ':0' for v in saved_names}
        # model_roots = np.unique([v.name.split('/')[0] for v in qa_model.trainable_variables])

        def transform(x):
            x = x.replace('attention/self', 'attention')
            x = x.replace('attention', 'self_attention')
            x = x.replace('attention/output', 'attention_output')

            x = x.replace('/dense', '')
            x = x.replace('/LayerNorm', '_layer_norm')
            x = x.replace('embeddings_layer_norm', 'embeddings/layer_norm')

            x = x.replace('attention_output_layer_norm', 'attention_layer_norm')
            x = x.replace('embeddings/word_embeddings', 'word_embeddings/embeddings')

            x = x.replace('/embeddings/', '/embedding_postprocessor/')
            x = x.replace('/token_type_embeddings', '/type_embeddings')
            x = x.replace('/pooler/', '/pooler_transform/')
            x = x.replace('answer_type_output_bias', 'ans_type/bias')
            x = x.replace('answer_type_output_', 'ans_type/')
            x = x.replace('cls/nq/output_', 'logits/')
            x = x.replace('/weights', '/kernel')

            return x

        a_map = {k: model_params.get(transform(v), v) for k, v in a_map.items() if k != 'global_step'}
        tf.compat.v1.train.init_from_checkpoint(ckpt_dir_or_file=FLAGS.raw_init_checkpoint,
                                                assignment_map=a_map)

        cpkt = tf.train.Checkpoint(model=qa_model)
        cpkt.save(FLAGS.init_checkpoint).assert_consumed()

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

def get_estimator():
    with open(FLAGS.bert_config_file, 'r') as f:
        bert_config = modeling.BertConfig.from_dict(json.load(f))
    # bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    validate_flags_or_throw(bert_config)
    tf.io.gfile.makedirs(FLAGS.output_dir)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        num_train_features = FLAGS.train_num_precomputed
        num_train_steps = int(num_train_features / FLAGS.train_batch_size *
                              FLAGS.num_train_epochs)

        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    return estimator

def prediction(_):
    prediction_tfrecord_fname = process_test_nq_file()
    predict_input_fn = input_fn_builder(
        input_file=prediction_tfrecord_fname,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        batch_size=FLAGS.predict_batch_size)

    # estimator = get_estimator()
    qa_model = get_model(transform_variable_names=True)

    logging.info("***** Running predictions *****")

    # If running eval on the TPU, you will need to specify the number of steps.
    all_results = []
    for result in qa_model.predict_generator(
            predict_input_fn(), yield_single_examples=True):
        if len(all_results) % 1000 == 0:
            logging.info("Processing example: %d" % (len(all_results)))
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        answer_type_logits = [float(x) for x in result["answer_type_logits"].flat]
        all_results.append(
            RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits,
                answer_type_logits=answer_type_logits))

    candidates_dict = read_candidates(FLAGS.predict_file)
    eval_features = [
        tf.train.Example.FromString(r)
        for r in tf.python_io.tf_record_iterator(prediction_tfrecord_fname)
    ]
    nq_pred_dict = compute_pred_dict(candidates_dict, eval_features,
                                     [r._asdict() for r in all_results])
    predictions_json = {"predictions": nq_pred_dict.values()}
    with tf.gfile.Open(FLAGS.output_prediction_file, "w") as f:
        json.dump(predictions_json, f, indent=4)

if __name__ == '__main__':
    print(FLAGS)
    app.run(prediction)
