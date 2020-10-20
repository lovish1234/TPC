import torch
import numpy
import math
import os


parser.add_argument('--model_verb', default='',
                    type=str, help='Path to verb model')
parser.add_argument('--model_noun', default='',
                    type=str, help='Path to noun model')
parser.add_argument('--epoch', default=0, type=int,
                    help='Number of epoch to calculate accuracy for')


def main():
    args = parser.parse_args()

    path_to_verb_result = os.path.join(
        args.model_verb, 'val_result', 'epoch%d' % args.epoch)
    path_to_noun_result = os.path.join(
        args.model_noun, 'val_result', 'epoch%d' % args.epoch)

    target_list_verb = torch.load(os.path.join(
        path_to_verb_result, 'ground_truth.pt'))
    pred_verb = torch.load(os.path.join(path_to_verb_result, 'prediction.pt'))
    target_list_noun = torch.load(os.path.join(
        path_to_noun_result, 'ground_truth.pt'))
    pred_noun = torch.load(os.path.join(path_to_noun_result, 'prediction.pt'))

    verb_correct = []
    noun_correct = []
    both_correct = []
    for i in range(len(pred_noun)):
        if pred_noun[i] == target_list_noun[i]:
            noun_correct.append(i)
        if pred_verb[i] == target_list_verb[i]:
            verb_correct.append(i)
        if pred_verb[i] == target_list_verb[i] and pred_noun[i] == target_list_noun[i]:
            both_correct.append(i)

    print('verb accuracy: %0.5f' % (len(verb_correct) / len(pred_verb)))
    print('noun accuracy: %0.5f' % (len(noun_correct) / len(pred_verb)))
    print('action accuracy: %0.5f' % (len(both_correct) / len(pred_verb)))
