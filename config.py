import os,sys

is_kaggle_server = os.path.exists('/kaggle')

if is_kaggle_server and not sys.path[-1].startswith('/kaggle'):
    sys.path.append('/kaggle/input/questionansering/')

input_data_path, output_data_path = '', ''
if is_kaggle_server:
    input_data_path = '/kaggle/input/'
    output_data_path = '/kaggle/working/'
else:
    input_data_path = './data/'
    output_data_path = './data/'


class QAFlags:
    def __init__(self):
        self.bert_config_file = input_data_path + 'bert-joint-baseline/bert_config.json'

        self.train_file = input_data_path + 'tensorflow2-question-answering/simplified-nq-train.jsonl'
        self.train_tfrecord_file = input_data_path + 'tensorflow2-question-answering/simplified-nq-train.tfrecord'

        self.sample_submission = input_data_path+'tensorflow2-question-answering/sample_submission.csv'
        self.predict_file = input_data_path + 'tensorflow2-question-answering/simplified-nq-test.jsonl'
        self.predict_tfrecord_file = input_data_path+'tensorflow2-question-answering/simplified-nq-test.tfrecord'
        #
        # self.predict_file = 'data/bert-joint-baseline/tiny-dev/nq-dev-sample.no-annot.jsonl.gz'
        # self.predict_tfrecord_file = 'data/bert-joint-baseline/tiny-dev/nq-dev-sample.no-annot.tfrecord'

        self.output_dir = output_data_path+'model/'
        self.output_prediction_file = output_data_path+'prediction/competition_prediction.csv'

        self.do_train = False
        self.do_predict =  True

        self.raw_init_checkpoint = input_data_path+'bert-joint-baseline/bert_joint.ckpt'
        self.init_checkpoint = self.output_dir+'bert-joint-baseline/bert_joint.ckpt-1'

        self.train_batch_size = 32
        self.predict_batch_size = 32

        self.learning_rate = 5e-5
        self.num_train_epochs = 3.0
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 1000
        self.iterations_per_loop = 1000
        self.train_precomputed_file = None
        self.train_num_precomputed = None

        self.vocab_file = input_data_path+'bert-joint-baseline/vocab-nq.txt'
        self.do_lower_case = True
        self.skip_nested_contexts =  True

        self.max_position = 50
        self.max_contexts = 48
        self.max_query_length = 64
        self.max_seq_length = 384
        self.doc_stride = 128
        self.n_best_size = 20
        self.max_answer_length = 30
        self.include_unknowns = -1.0

        self.long_answer_score_threshold = 0.5
        self.short_answer_score_threshold = 0.9

FLAGS = QAFlags()
