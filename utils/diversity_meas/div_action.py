# May 2020
# Uncertainty metrics derived from action vectors

import os, sys, pickle
import scipy
import numpy as np
from tqdm import tqdm


def measure_action_entropy_results(results, results_from_path=False):
    '''
    NOTE: You need to call measure_vae_uncertainty(...) with collect_actions=True first.
    Given a results dictionary with keys mean_var, actions, preds, contexts,
    returns the same information with extra key action_entropy.
    '''

    if results_from_path:
        with open(results, 'rb') as f:
            results = pickle.load(results)

    for key in tqdm(results.keys()):
        cur_info = results[key]
        cur_info['action_entropy'] = calculate_action_entropy(cur_info['actions'])
        results[key] = cur_info

    return results


def calculate_action_entropy(actions):
    '''
    actions: (class type, path, future step, class) => logit.
    Returns: (class type, future step) => mean entropy over time.
    '''

    if not(isinstance(actions, list)):
        actions = [actions] # single class type
    
    result = []
    for cur_actions in actions:
        (paths, pred_step, num_class) = cur_actions.shape
        action_probs = scipy.special.softmax(cur_actions, axis=2)
        all_entropy = (-action_probs * np.log2(action_probs + 1e-12)).sum(axis=2)
        cur_entropy = np.zeros(pred_step) # separately over time
        for t in range(pred_step):
            cur_entropy[t] = all_entropy[:, t].mean()
        result.append(cur_entropy)

    return result
