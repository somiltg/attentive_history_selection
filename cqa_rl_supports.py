from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from cqa_flags import FLAGS
from cqa_gen_batches import *
from cqa_model import *
from cqa_supports import *
from sklearn.metrics.pairwise import cosine_similarity


# from cqa_selection_supports import *

def get_selected_examples(examples_matrix, actions_reshaped):
    selected_examples = []
    for variations, variations_action in zip(examples_matrix, actions_reshaped):

        selected_example = deepcopy(variations[0])
        selected_example.history_answer_marker = np.zeros(len(selected_example.history_answer_marker))
        for variation, action in zip(variations, variations_action):
            if variation is not None and action == 1:
                selected_example.history_answer_marker = np.asarray(selected_example.history_answer_marker) + \
                                                                np.asarray(variation.history_answer_marker)
        selected_examples.append(selected_example)
        
    return selected_examples
    
def convert_features_to_feed_dict(features):
    batch_unique_ids, batch_input_ids, batch_input_mask = [], [], []
    batch_segment_ids, batch_start_positions, batch_end_positions, batch_history_answer_marker = [], [], [], []
    batch_yesno, batch_followup = [], []
    batch_metadata = []
    
    yesno_dict = {'y': 0, 'n': 1, 'x': 2}
    followup_dict = {'y': 0, 'n': 1, 'm': 2}
    
    for feature in features:
        batch_unique_ids.append(feature.unique_id)
        batch_input_ids.append(feature.input_ids)
        batch_input_mask.append(feature.input_mask)
        batch_segment_ids.append(feature.segment_ids)
        batch_start_positions.append(feature.start_position)
        batch_end_positions.append(feature.end_position)
        batch_history_answer_marker.append(feature.history_answer_marker)
        batch_yesno.append(yesno_dict[feature.metadata['yesno']])
        batch_followup.append(followup_dict[feature.metadata['followup']])
        batch_metadata.append(feature.metadata)
    
    feed_dict = {'unique_ids': batch_unique_ids, 'input_ids': batch_input_ids, 
              'input_mask': batch_input_mask, 'segment_ids': batch_segment_ids, 
              'start_positions': batch_start_positions, 'end_positions': batch_end_positions, 
              'history_answer_marker': batch_history_answer_marker, 'yesno': batch_yesno, 'followup': batch_followup, 
              'metadata': batch_metadata}
    return feed_dict

def convert_examples_to_example_variations(examples, max_considered_history_turns):
    # an example is "question + passage + markers (M3 + M4) + markers_list (M3, M4)"
    # an example variation is "question + passage + markers (M3)"
    # meaning that we only have one marker for each example variation
    # because we want to make a binary choice for every example variation,
    # and combine all variations to form an example
    
    new_examples = []
    for example in examples:
        # if the example is the first question in the dialog, it does not contain history answers, 
        # so we simply append it.
        if len(example.metadata['tok_history_answer_markers']) == 0:
            example.metadata['history_turns'] = []
            new_examples.append(example)
        else:
            for history_turn, marker, history_turn_text in zip(
                    example.metadata['history_turns'][- max_considered_history_turns:], 
                    example.metadata['tok_history_answer_markers'][- max_considered_history_turns:],
                    example.metadata['history_turns_text'][- max_considered_history_turns:]):
                each_new_example = deepcopy(example)
                each_new_example.history_answer_marker = marker
                each_new_example.metadata['history_turns'] = [history_turn]
                each_new_example.metadata['tok_history_answer_markers'] = [marker]
                each_new_example.metadata['history_turns_text'] = [history_turn_text]
                new_examples.append(each_new_example)
                
            if FLAGS.append_self:
                # after the variations that contain histories, we append an example that is without any 
                # history. If the the current question is topic shift, all the attention weights should be
                # on this no-history variation.
                each_new_example = deepcopy(example)
                each_new_example.history_answer_marker = [0] * len(example.metadata['tok_history_answer_markers'][0])
                each_new_example.metadata['history_turns'] = []
                each_new_example.metadata['tok_history_answer_markers'] = []
                each_new_example.metadata['history_turns_text'] = []
                new_examples.append(each_new_example)
             
    return new_examples

def convert_examples_to_example_variations_with_question_reformulated(examples, max_considered_history_turns):
    # an example is "question + passage + markers (M3 + M4) + markers_list (M3, M4)"
    # an example variation is "question + passage + markers (M3)"
    # meaning that we only have one marker for each example variation
    # because we want to make a binary choice for every example variation,
    # and combine all variations to form an example
    
    new_examples = []
    for example in examples:
        # if the example is the first question in the dialog, it does not contain history answers, 
        # so we simply append it.
        if len(example.metadata['tok_history_answer_markers']) == 0:
            example.metadata['history_turns'] = []
            new_examples.append(example)
        else:
            immediate_previous_history_question = example.metadata['history_turns_text'][-1][0]
            for history_turn, marker, history_turn_text in zip(
                    example.metadata['history_turns'][- max_considered_history_turns:], 
                    example.metadata['tok_history_answer_markers'][- max_considered_history_turns:],
                    example.metadata['history_turns_text'][- max_considered_history_turns:]):
                each_new_example = deepcopy(example)
                each_new_example.history_answer_marker = marker
                each_new_example.metadata['history_turns'] = [history_turn]
                each_new_example.metadata['tok_history_answer_markers'] = [marker]
                each_new_example.metadata['history_turns_text'] = [history_turn_text]
                each_new_example.question_text = immediate_previous_history_question + each_new_example.question_text
                new_examples.append(each_new_example)
    return new_examples

# def convert_examples_to_variations_and_then_features(examples, tokenizer, max_seq_length, 
#                                 doc_stride, max_query_length, max_considered_history_turns, glove, tfidf_vectorizer, is_training):
def convert_examples_to_variations_and_then_features(examples, tokenizer, max_seq_length, 
                                doc_stride, max_query_length, max_considered_history_turns, is_training):
    # different from the "convert_examples_to_features" in cqa_supports.py, we return two masks with the feature (example/variaton trackers).
    # the first mask is the example index, and the second mask is the variation index. Wo do this to keep track of the features generated
    # by different examples and variations.
    
    all_features = []
    example_features_nums = [] # keep track of how many features are generated from the same example (regardless of example variations)
    example_tracker = []
    variation_tracker = []
    # matching_signals_dict = {}
    unique_id = 1000000000
    
    
    # when training, we shuffle the data for more stable training.
    # we shuffle here so that we do not need to shuffle when generating batches
    num_examples = len(examples)    
    if is_training:
        np.random.seed(0)
        idx = np.random.permutation(num_examples)
        examples_shuffled = np.asarray(examples)[idx]
    else:
        examples_shuffled = np.asarray(examples)
    
    for example_index, example in enumerate(examples_shuffled):
        example_features_num = []
        if FLAGS.reformulate_question:
            variations = convert_examples_to_example_variations_with_question_reformulated([example], max_considered_history_turns)
        else:
            variations = convert_examples_to_example_variations([example], max_considered_history_turns)
        for variation_index, variation in enumerate(variations):
            features = convert_examples_to_features([variation], tokenizer, max_seq_length, doc_stride, max_query_length, is_training)
            # matching_signals = extract_matching_signals(variation, glove, tfidf_vectorizer)
            # matching_signals_dict[(example_index, variation_index)] = matching_signals
            
            # the example_index and unique_id in features are wrong due to the generation of example variations.
            # we fix them here.
            for i in range(len(features)):
                features[i].example_index = example_index
                features[i].unique_id = unique_id
                unique_id += 1
            all_features.extend(features)
            variation_tracker.extend([variation_index] * len(features))
            example_tracker.extend([example_index] * len(features))
            example_features_num.append(len(features))
        # every variation of the same example should generate the same amount of features
        assert len(set(example_features_num)) == 1
        example_features_nums.append(example_features_num[0]) 
    assert len(all_features) == len(example_tracker)
    assert len(all_features) == len(variation_tracker)
    # return all_features, example_tracker, variation_tracker, example_features_nums, matching_signals_dict
    return all_features, example_tracker, variation_tracker, example_features_nums

def get_merged_features(batch_bert_rep, batch_example_tracker, batch_variation_tracker):
    # we merge the rep of features that are generated by the same example variation
    prev_e_tracker, prev_v_tracker = None, None
    merged_example_tracker, merged_variation_tracker, merged_bert_rep = [], [], []
    same_variation_count = 1 # whether the current rep is generated by the same variation of the previous rep
    for rep, e_tracker, v_tracker in zip(batch_bert_rep, batch_example_tracker, batch_variation_tracker):
        rep = np.asarray(rep)
        if e_tracker == prev_e_tracker and v_tracker == prev_v_tracker:                    
            merged_bert_rep[-1] = (merged_bert_rep[-1] * same_variation_count + rep) / (same_variation_count + 1)
            same_variation_count += 1
        else:
            same_variation_count = 1
            merged_example_tracker.append(e_tracker)
            merged_variation_tracker.append(v_tracker)
            merged_bert_rep.append(rep)
        prev_e_tracker, prev_v_tracker = e_tracker, v_tracker

    assert len(merged_example_tracker) == len(merged_bert_rep)
    assert len(merged_variation_tracker) == len(merged_bert_rep)
    
    return merged_bert_rep, merged_example_tracker, merged_variation_tracker

def get_merged_tfidf(batch_tfidf, batch_example_tracker, batch_variation_tracker):
    # we merge the tfidf features that are generated by the same example variation
    prev_e_tracker, prev_v_tracker = None, None
    merged_example_tracker, merged_variation_tracker, merged_tfidf = [], [], []
    same_variation_count = 1 # whether the current rep is generated by the same variation of the previous rep
    for tfidf, e_tracker, v_tracker in zip(batch_tfidf, batch_example_tracker, batch_variation_tracker):
        if e_tracker == prev_e_tracker and v_tracker == prev_v_tracker:                    
            merged_tfidf[-1] = (merged_tfidf[-1] * same_variation_count + tfidf) / (same_variation_count + 1)
            same_variation_count += 1
        else:
            same_variation_count = 1
            merged_example_tracker.append(e_tracker)
            merged_variation_tracker.append(v_tracker)
            merged_tfidf.append(tfidf)
        prev_e_tracker, prev_v_tracker = e_tracker, v_tracker

    assert len(merged_example_tracker) == len(merged_tfidf)
    assert len(merged_variation_tracker) == len(merged_tfidf)
    
    return merged_tfidf, merged_example_tracker, merged_variation_tracker

def get_turn_features(metadata):
    # extract current turn id, history turn id from metadata as a part of states
    res = []
    for m in metadata:
        if len(m['history_turns']) > 0:
            history_turn_id = m['history_turns'][0]
        else:
            history_turn_id = 0
        res.append([m['turn'], history_turn_id, m['turn'] - history_turn_id])
    return res

def get_selected_example_features(batch_features, batch_example_tracker, batch_variation_tracker, 
                                  merged_example_tracker, merged_variation_tracker, actions):
    
    action_dict = {} # a dictionary, key: (e_tracker, v_tracker), value: action
    for action, e_tracker, v_tracker in zip(np.squeeze(actions, axis=1), merged_example_tracker, merged_variation_tracker):
        action_dict[(e_tracker, v_tracker)] = action
    
    prev_e_tracker, prev_v_tracker = None, None
    relative_selected_pos = [] # the relative position of the selected history compared to the current turn
    handled_variations = {} # each example variation could have more than one features, but we only store the "relative_selected_pos" once 
    f_tracker = 0 # feature tracker, denotes the feature index for each variation
    selected_example_features_dict = {}
    for feature, e_tracker, v_tracker in zip(batch_features, batch_example_tracker, batch_variation_tracker):
        # get the f_tracker
        if e_tracker == prev_e_tracker and v_tracker == prev_v_tracker:
            f_tracker += 1
        else:
            f_tracker = 0
        prev_e_tracker, prev_v_tracker = e_tracker, v_tracker

        # we append a feature that is without any history, so we can build upon this if other histories are chosen
        if e_tracker not in selected_example_features_dict:                    
            feature_without_marker = deepcopy(feature)
            feature_without_marker.history_answer_marker = np.zeros(len(feature.history_answer_marker))
            feature_without_marker.metadata['history_turns'] = []
            selected_example_features_dict[e_tracker] = [feature_without_marker]
        elif f_tracker >= len(selected_example_features_dict[e_tracker]):
            feature_without_marker = deepcopy(feature)
            feature_without_marker.history_answer_marker = np.zeros(len(feature.history_answer_marker))
            feature_without_marker.metadata['history_turns'] = []
            selected_example_features_dict[e_tracker].append(feature_without_marker)

        if len(feature.metadata['history_turns']) > 0 and action_dict[(e_tracker, v_tracker)] == 1:
            curr_feature = deepcopy(selected_example_features_dict[e_tracker][f_tracker])
            
            if not FLAGS.better_hae:
                curr_feature.history_answer_marker = curr_feature.history_answer_marker + \
                                                     np.asarray(feature.history_answer_marker)
                # when the markers have overlaps, the corresponding bit would be 2 after added together. we need to set it to one.
                curr_feature.history_answer_marker = curr_feature.history_answer_marker != 0
                curr_feature.history_answer_marker = curr_feature.history_answer_marker.astype(int)
                
            else:
                curr_feature.history_answer_marker[np.asarray(feature.history_answer_marker) == 1] = \
                                        int(feature.metadata['turn'] - feature.metadata['history_turns'][0])
                curr_feature.history_answer_marker = curr_feature.history_answer_marker.astype(int)
            
            curr_feature.metadata['history_turns'].append(feature.metadata['history_turns'][0])
            selected_example_features_dict[e_tracker][f_tracker] = curr_feature
            
            if (e_tracker, v_tracker) not in handled_variations:
                # we always choose the current turn
                relative_selected_pos.append(0)
                
                handled_variations[(e_tracker, v_tracker)] = 1
                if len(feature.metadata['history_turns']) > 0:
                    relative_selected_pos.append(feature.metadata['turn'] - feature.metadata['history_turns'][0])

    selected_example_features = []
    reward_penalty = [] # abs(selected_number - ideal_selected_number)
    reward_penalty_example_dict = {}
    for e_tracker, features in selected_example_features_dict.items():
        selected_example_features.extend(features)
        if e_tracker not in reward_penalty_example_dict:
            selected_history_num_this_example = len(features[0].metadata['history_turns'])
            total_history_num_this_example = features[0].metadata['turn'] - 1
            reward_penalty_example_dict[e_tracker] = abs(selected_history_num_this_example - FLAGS.ideal_selected_num)
            
    for e_tracker in merged_example_tracker:
        reward_penalty.append(reward_penalty_example_dict[e_tracker])
    assert len(reward_penalty) == len(actions)
    return selected_example_features, relative_selected_pos, reward_penalty

def get_selected_example_features_without_actions(batch_features, batch_example_tracker, batch_variation_tracker, tfidf_vectorizer=None):
    
    prev_e_tracker, prev_v_tracker = None, None
    relative_selected_pos = [] # the relative position of the selected history compared to the current turn
    handled_variations = {} # each example variation could have more than one features, but we only store the "relative_selected_pos" once 
    f_tracker = 0 # feature tracker, denotes the feature index for each variation
    
    large_tfidf_dict = {}
    if FLAGS.history_selection == 'tfidf_sim' and FLAGS.more_history != 0:
        fd = convert_features_to_feed_dict(batch_features)
        tfidf_features = get_tfidf_features(fd['input_ids'], fd['history_answer_marker'], tfidf_vectorizer)
        # print('tfidf_features', tfidf_features)
        # print('batch_example_tracker', batch_example_tracker)
        # print('batch_variation_tracker', batch_variation_tracker)
        merged_tfidf, merged_example_tracker, merged_variation_tracker = get_merged_features(
                                        tfidf_features, batch_example_tracker, batch_variation_tracker)
        # print('merged_tfidf', merged_tfidf)
        # print('merged_example_tracker', merged_example_tracker)
        # print('merged_variation_tracker', merged_variation_tracker)
        tfidf_dict = {}
        for tfidf_score, e_tracker, v_tracker in zip(merged_tfidf, merged_example_tracker, merged_variation_tracker):
            if e_tracker not in tfidf_dict:
                tfidf_dict[e_tracker] = []
            tfidf_dict[e_tracker].append(tfidf_score)
        # print('tfidf_dict', tfidf_dict)
        
        for e_tracker, score_list in tfidf_dict.items():
            score_list_arg_sorted = np.argsort(np.asarray(score_list))
            selected_variation_ids = score_list_arg_sorted[- FLAGS.more_history :]
            for v_id in selected_variation_ids:
                if tfidf_dict[e_tracker][v_id] != 0:
                    large_tfidf_dict[(e_tracker, v_id)] = True
        # print('large_tfidf_dict', large_tfidf_dict)
    
    selected_example_features_dict = {}
    for feature, e_tracker, v_tracker in zip(batch_features, batch_example_tracker, batch_variation_tracker):
        # get the f_tracker
        if e_tracker == prev_e_tracker and v_tracker == prev_v_tracker:
            f_tracker += 1
        else:
            f_tracker = 0
        prev_e_tracker, prev_v_tracker = e_tracker, v_tracker

        # we append a feature that is without any history, so we can build upon this if other histories are chosen
        if e_tracker not in selected_example_features_dict:                    
            feature_without_marker = deepcopy(feature)
            feature_without_marker.history_answer_marker = np.zeros(len(feature.history_answer_marker))
            feature_without_marker.metadata['history_turns'] = []
            selected_example_features_dict[e_tracker] = [feature_without_marker]
        elif f_tracker >= len(selected_example_features_dict[e_tracker]):
            feature_without_marker = deepcopy(feature)
            feature_without_marker.history_answer_marker = np.zeros(len(feature.history_answer_marker))
            feature_without_marker.metadata['history_turns'] = []
            selected_example_features_dict[e_tracker].append(feature_without_marker)
        
        # add history answer markers, a turn is selected if it either in immediate previous j turns or have large tfidf score
        if len(feature.metadata['history_turns']) > 0 and \
             (feature.metadata['history_turns'][0] >= feature.metadata['turn'] - FLAGS.history or \
             (e_tracker, v_tracker) in large_tfidf_dict):
                
            curr_feature = deepcopy(selected_example_features_dict[e_tracker][f_tracker])
            
            if not FLAGS.better_hae:
                curr_feature.history_answer_marker = curr_feature.history_answer_marker + \
                                                     np.asarray(feature.history_answer_marker)
                # when the markers have overlaps, the corresponding bit would be 2 after added together. we need to set it to one.
                curr_feature.history_answer_marker = curr_feature.history_answer_marker != 0
                curr_feature.history_answer_marker = curr_feature.history_answer_marker.astype(int)
                
            else:
                curr_feature.history_answer_marker[np.asarray(feature.history_answer_marker) == 1] = \
                                        int(feature.metadata['turn'] - feature.metadata['history_turns'][0])
                curr_feature.history_answer_marker = curr_feature.history_answer_marker.astype(int)
            
            curr_feature.metadata['history_turns'].append(feature.metadata['history_turns'][0])
            selected_example_features_dict[e_tracker][f_tracker] = curr_feature
            
            if (e_tracker, v_tracker) not in handled_variations:
                # we always choose the current turn
                relative_selected_pos.append(0)
                
                handled_variations[(e_tracker, v_tracker)] = 1
                if len(feature.metadata['history_turns']) > 0:
                    relative_selected_pos.append(feature.metadata['turn'] - feature.metadata['history_turns'][0])

    selected_example_features = []
    for features in selected_example_features_dict.values():
        selected_example_features.extend(features)
        
    return selected_example_features, relative_selected_pos

def get_features_to_be_selected(batch_features, batch_example_tracker, batch_variation_tracker):
    # features that generated by the first question of a dialog do not have histories. 
    # thus these features can bypass the history selection module.
    batch_features_to_select, batch_example_tracker_to_select, batch_variation_tracker_to_select = [], [], []
    batch_features_to_bypass, batch_example_tracker_to_bypass, batch_variation_tracker_to_bypass = [], [], []
    for feature, e_tracker, v_tracker in zip(batch_features, batch_example_tracker, batch_variation_tracker):
        if len(feature.metadata['history_turns']) > 0:
            batch_features_to_select.append(feature)
            batch_example_tracker_to_select.append(e_tracker)
            batch_variation_tracker_to_select.append(v_tracker)
        else:
            batch_features_to_bypass.append(feature)
            batch_example_tracker_to_bypass.append(e_tracker)
            batch_variation_tracker_to_bypass.append(v_tracker)
            
    return batch_features_to_select, batch_example_tracker_to_select, batch_variation_tracker_to_select, \
            batch_features_to_bypass, batch_example_tracker_to_bypass, batch_variation_tracker_to_bypass

# no longer used
def get_histo(values, bins=range(13)):
    """Get the histogram of a list/vector of values."""
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy        
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)
        
    return hist

def get_discounted_future_rewards(ep_rewards):
    ep_rs = np.array(ep_rewards)
    discounted_ep_rs = np.zeros_like(ep_rs)
    running_add = 0

    for t in reversed(range(0, len(ep_rs))):
        running_add = np.average(running_add)
        running_add = running_add * FLAGS.reward_decay + ep_rs[t]
        discounted_ep_rs[t] = running_add

    return discounted_ep_rs

def get_tfidf_features(sub_batch_input_ids, sub_batch_history_answer_marker, tfidf_vectorizer):
    questions, histories = [], []
    for input_ids, history_answer_marker in zip(sub_batch_input_ids, sub_batch_history_answer_marker):
        sep_idx = input_ids.index(102)
        question = input_ids[1:sep_idx]
        questions.append(' '.join(map(str, question)))
        
        history_answer_idx = np.asarray(history_answer_marker) == 1
        history = np.asarray(input_ids)[history_answer_idx]
        histories.append(' '.join(map(str, history.tolist())))
    
    questions_tfidf = tfidf_vectorizer.transform(questions)
    histories_tfidf = tfidf_vectorizer.transform(histories)
    
    cosine = []
    for q, h in zip(questions_tfidf, histories_tfidf):
        cosine.append(cosine_similarity(q, h)[0][0])
        
    return cosine

def get_state_dim(state_features):
    features_dim = [None, 768, 2, 3, 1] # how many dims for each feature
    state_dim = 0
    for c in state_features:
        state_dim += features_dim[int(c)]
    return state_dim

def get_state_rep(bert_representations, start_prob_res, end_prob_res, turn_features, tfidf_features, state_features):
    features = [None, 
            np.asarray(bert_representations), 
            np.hstack((np.expand_dims(np.asarray(start_prob_res), axis=-1), np.expand_dims(np.asarray(end_prob_res), axis=-1))), 
            np.asarray(turn_features), 
            np.expand_dims(np.asarray(tfidf_features), axis=-1)]
        
    state_rep = features[int(state_features[0])]
    for state_feature in state_features[1:]:
        state_rep = np.hstack((state_rep, features[int(state_feature)]))
    
    return state_rep

def fix_history_answer_marker_for_bhae(sub_batch_history_answer_marker, turn_features):
    res = []
    for marker, turn_feature in zip(sub_batch_history_answer_marker, turn_features):
        turn_diff = turn_feature[2]
        marker = np.asarray(marker)
        marker[marker == 1] = turn_diff
        res.append(marker.tolist())
        
    return res