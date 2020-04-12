from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

# for running in jupyter env
flags.DEFINE_string('f', '', 'kernel')

## Required parameters
flags.DEFINE_string(
    "bert_config_file", "../bert/model_52000/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "../bert/model_52000/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "../bert/bert_out/10001/",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("coqa_train_file", "../coqa/coqa-train-v1.0.json",
                    "CoQA json for training. E.g., coqa-train-v1.0.json")

flags.DEFINE_string(
    "coqa_predict_file", "../coqa/coqa-dev-v1.0.json",
    "CoQA json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string("quac_train_file", "../quac/train_v0.2.json",
                    "QuAC json for training.")

flags.DEFINE_string(
    "quac_predict_file", "../quac/val_v0.2.json",
    "QuAC json for predictions.")

flags.DEFINE_string(
    "init_checkpoint", "../bert/model_52000/model_52000.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

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
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 16,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 2.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("evaluation_steps", 1000,
                     "How often to do evaluation.")

flags.DEFINE_integer("evaluate_after", 18000,
                     "we do evaluation after centain steps.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 50,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_integer(
    "history", 6,
    "Number of conversation history to use.")

flags.DEFINE_bool(
    "only_history_answer", True,
    "only prepend history answers without questions?")

flags.DEFINE_bool(
    "use_history_answer_marker", True,
    "use markers for hisotory answers instead of prepending them."
    "This flag surpasses the only_history_answer flag.")

flags.DEFINE_bool(
    "load_small_portion", False,
    "during develping, we only want to load a very small portion of "
    "the data to see if the code works.")

flags.DEFINE_bool(
    "use_RL", True,
    "whether to use the reinforced backtracker."
    "this flag supasses the history flag, because we will choose history freely with RL")

flags.DEFINE_string(
    "dataset", 'quac',
    "QuAC or CoQA")

# no longer used
flags.DEFINE_integer(
    "max_history_turns", 11,
    "what is the max history turns a question can have "
    "e.g. in QuAC data, a dialog has a maximum of 12 turns,"
    "so a question has a maximum of 11 history turns")

# no longer used
flags.DEFINE_integer("example_batch_size", 4,
                     "when using RL, we want the batch size to be smaller because one example can gen multiple features")

flags.DEFINE_string(
    "cache_dir", "./cache_large/",
    "we store generated features here, so that we do not need to generate them every time")

flags.DEFINE_integer(
    "pretrain_steps", 2000,
    "we pretrain the CQA model for some steps before the reinforced backtracker kicks in")

flags.DEFINE_integer(
    "episode_steps", 1000,
    "we use a validation subset to generate reward, we change this set every k steps")

flags.DEFINE_integer(
    "reward_set_batches_num", 3,
    "for every training step, we run k validation steps on a validation subset to get reward,"
    "training will be slow if this is set to a large number")

flags.DEFINE_integer(
    "max_considered_history_turns", 11,
    "we only consider k history turns that immediately proceed the current turn,"
    "training will be slow if this is set to a large number")

flags.DEFINE_integer(
    "save_actions_and_states_step", 22000,
    "we store the states and actions when training is almost done for visulazation")

flags.DEFINE_float("history_penalty", 0.000,
                   "how much we want to penalize the reinforced backtracker when it selects to much history")

flags.DEFINE_bool(
    "actor_critic", False,
    "True: actor critic, False: REINFORCE")

flags.DEFINE_float("reward_decay", 0.8, "reward discount factor")

flags.DEFINE_string(
    "state_features", "1234",
    "1: bert_representations, 2: start, end probs, 3: cur_turn, his_turn, diff, 4: tfidf cosine")

flags.DEFINE_string(
    "reward_type", "loss",
    "loss: the loss gap on reward set, f1: the f1 on reward set")

flags.DEFINE_integer(
    "train_steps", 22000,
    "loss: the loss gap on reward set, f1: the f1 on reward set")

flags.DEFINE_bool(
    "better_hae", False,
    "assign different history answer embedding to differet previous turns")

flags.DEFINE_string(
    "history_selection", "tfidf_sim",
    "previous_j: select the immediate previous j turns"
    "tfidf_sim: select FLAGS.more_history turns according to the ranking of "
    "cosine similarity better the current question and a history answer."
    "RL: reinforcement learning"
)

flags.DEFINE_integer(
    "more_history", 2,
    "Number of conversation history to use. applicable to other rules except for previous_j")

flags.DEFINE_integer(
    "max_question_len_for_matching", 20,
    "applicable for the interaction matrix")

flags.DEFINE_integer(
    "max_answer_len_for_matching", 40,
    "applicable for the interaction matrix")

flags.DEFINE_string(
    "glove", '../glove/glove.840B.300d.pkl',
    "glove pre-trained word embedding, we use 840B.300d")

flags.DEFINE_integer(
    "embedding_dim", 300,
    "dimension for glove pre-trained word embedding")

flags.DEFINE_integer(
    "kernel_size", 3,
    "cnn kernel size for the cnn in policy net")

flags.DEFINE_integer(
    "kernel_count", 16,
    "cnn kernel count for the cnn in policy net")

flags.DEFINE_integer(
    "pool_size", 3,
    "cnn kernel size for the cnn in policy net")

flags.DEFINE_float("rl_learning_rate", 1e-4, "The initial learning rate for the policy net and value net.")

flags.DEFINE_bool("MTL", False, "multi-task learning. jointly learn the dialog acts (followup, yesno)")

flags.DEFINE_bool("DomainL", False, "domain learning. jointly learn the domain type (science, literature, etc.)")

flags.DEFINE_float("MTL_lambda", 0.0,
                   "total loss = (1 - 2 * lambda) * convqa_loss + lambda * followup_loss + lambda * yesno_loss")

flags.DEFINE_float("Domain_gamma", 0.0,
                   "total loss = (1 - 2 * lambda - gamma ) * convqa_loss + lambda * followup_loss + lambda * "
                   "yesno_loss + gamma * domain_loss")

flags.DEFINE_float("MTL_mu", 0.0, "total loss = mu * convqa_loss + lambda * followup_loss + lambda * yesno_loss + gamma * domain_loss")

flags.DEFINE_integer(
    "ideal_selected_num", 1,
    "ideal # selected history turns per example/question")

flags.DEFINE_bool("aux", False, "use aux loss or not")

flags.DEFINE_float("aux_lambda", 0.0, "auxiliary loss")

flags.DEFINE_bool("aux_shared", False, "wheter to share the aux prediction layer with the main convqa model")

flags.DEFINE_bool("disable_attention", False, "dialable the history attention module")

flags.DEFINE_bool("history_attention_hidden", False, "dialable the history attention module")

flags.DEFINE_string("history_attention_input", "CLS", "CLS, reduce_mean, reduce_max")

flags.DEFINE_string("mtl_input", "CLS", "CLS, reduce_mean, reduce_max")

flags.DEFINE_integer("history_ngram", 1,
                     "in history attention, we attend to groups of history turns, this param indicate how many histories in one group"
                     "if set to 1, it's equivalent to attend to every history turns independently")

flags.DEFINE_bool("reformulate_question", False,
                  "prepend the immediate previous history question to the current question")

flags.DEFINE_bool("front_padding", False, "pad the BERT input sequence at the front")

flags.DEFINE_bool("freeze_bert", False, "freeze BERT")

flags.DEFINE_bool("fine_grained_attention", False, "use fine grained attention")

flags.DEFINE_bool("append_self", False,
                  "when converting an example to variations, whether to append a variation without any history (self)")

flags.DEFINE_float("null_score_diff_threshold", 1.8, "null_score_diff_threshold")

flags.DEFINE_integer("bert_hidden", 1024, "bert hidden units, 768 or 1024")

flags.DEFINE_list("domain_array", ['Others','Literature', 'CreativeArts', 'Music', 'MusicGroup', 'Humanities', 'Politics',
                                   'Social Studies', 'Business & Management', 'Sports-Adventure', 'Natural Sciences',
                                   'Fiction'], "The possible domain types for context.")
