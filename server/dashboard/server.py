import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

import numpy as np
from gulpio import GulpDirectory
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset, EpicVideoFlowDataset, GulpVideoSegment
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from datetime import datetime, timedelta
from math import log, e
from epic_kitchens import meta
import plotly.graph_objects as go
import plotly.express as px


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def pd_to_dict(df, entity='verbs'):
    x_coarse_dict = {}
    for index, row in df.iterrows():
        x_list = row[entity]
        x_class = row['class_key']
        for i in x_list:
            x_coarse_dict[i] = x_class

    return x_coarse_dict


def get_unique_keys_2(dictionary, class_mapping):

    unique_dictionary = {}
    for key in dictionary:
        if key in class_mapping:
            if class_mapping[key] in unique_dictionary:
                unique_dictionary[class_mapping[key]] += dictionary[key]
            else:
                unique_dictionary[class_mapping[key]] = dictionary[key]
        else:
            pass
    return unique_dictionary


def get_unique_entity_nested(dictionary, class_mapping):

    unique_dictionary = {}
    for key in dictionary:
        entity = key
        if entity in class_mapping:
            unique_action = class_mapping[entity]
            if unique_action in unique_dictionary:
                unique_dictionary[unique_action] = combine_dict(
                    unique_dictionary[unique_action], dictionary[key])
            else:
                unique_dictionary[unique_action] = dictionary[key]
        else:
            pass

    for key in unique_dictionary:
        temp_dict = {}
        for x in unique_dictionary[key]:
            if x in class_mapping:
                unique_action = class_mapping[x]
                if unique_action in temp_dict:
                    temp_dict[unique_action] += unique_dictionary[key][x]
                else:
                    temp_dict[unique_action] = unique_dictionary[key][x]
            else:
                pass
        unique_dictionary[key] = temp_dict

    return unique_dictionary


def combine_dict(dict_1, dict_2):

    comb_dict = {}

    for key in dict_1:
        if key in dict_2:
            comb_dict[key] = dict_1[key] + dict_2[key]
        else:
            comb_dict[key] = dict_1[key]

    for key in dict_2:
        if key not in comb_dict:
            comb_dict[key] = dict_2[key]

    return comb_dict


def change_dict_format(dict_1, type='verb'):

    key_list = []
    value_list = []
    dict_final = {}

    for key, value in dict_1.items():
        key_list.append(key)
        value_list.append(value)

    if type == 'verb':
        dict_final = {'Verbs': key_list, 'Number of instances': value_list}
    elif type == 'noun':
        dict_final = {'Nouns': key_list, 'Number of instances': value_list}
    elif type == 'action':
        dict_final = {'Actions': key_list, 'Number of instances': value_list}

    return dict_final


gulp_root = Path(
    '/proj/vondrick/datasets/epic-kitchens/data/processed/gulp')
class_type = 'verb+noun'
rgb_train = EpicVideoDataset(gulp_root / 'rgb_train', class_type)
segment_uids = list(rgb_train.gulp_dir.merged_meta_dict.keys())


def get_dataframe_verb(entity='open', participant='All', video='All', degree=1):

    verb_dict = {}
    verb_dict_nested = {}

    for i in range(len(segment_uids) - int(degree)):

        # for each video, this should be broken down

        example_meta_entry = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i]]
        example_meta_entry_next = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i + int(
            degree)]]

        vid = example_meta_entry['meta_data'][0]['video_id']
        vid_next = example_meta_entry_next['meta_data'][0]['video_id']

        pid, vid = vid.split('_')
        pid_next, vid_next = vid_next.split('_')

        # if the action is not from the same sequence, do not add
        if pid != pid_next or vid != vid_next:
            continue

        participant_id = example_meta_entry['meta_data'][0]['participant_id']
        video_id = example_meta_entry['meta_data'][0]['video_id'].split(
            '_')[-1]

        if participant == 'All' or participant_id == participant:
            pass
        elif participant_id != participant:
            continue

        if video == 'All' or video_id == video:
            pass
        elif video_id != video:
            continue

        verb = example_meta_entry['meta_data'][0]['verb']
        verb_next = example_meta_entry_next['meta_data'][0]['verb']
        # verbs
        if verb in verb_dict:
            verb_dict[verb] += 1
        else:
            verb_dict[verb] = 1

        if verb in verb_dict_nested:
            if verb_next in verb_dict_nested[verb]:
                verb_dict_nested[verb][verb_next] += 1
            else:
                verb_dict_nested[verb][verb_next] = 1
        else:
            verb_dict_nested[verb] = {}
            verb_dict_nested[verb][verb_next] = 1

    verb_dict_nested = get_unique_entity_nested(
        verb_dict_nested, verb_coarse_dict)
    verb_dict = get_unique_keys_2(verb_dict, verb_coarse_dict)
    verb_dict = change_dict_format(verb_dict, type='verb')

    entity_dict = change_dict_format(verb_dict_nested[entity], type='verb')

    return entity_dict, verb_dict


def get_average_time_histogram(action_1, action_2):

    time_list = []
    FMT = '%H:%M:%S.%f'
    time_dict_action = {}

    for i in range(len(segment_uids)):

        example_meta_entry = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i]]
        start_time = example_meta_entry['meta_data'][0]['start_timestamp']
        stop_time = example_meta_entry['meta_data'][0]['stop_timestamp']

        noun = example_meta_entry['meta_data'][0]['noun']
        verb = example_meta_entry['meta_data'][0]['verb']
        action = verb + ' ' + noun

        tdelta = datetime.strptime(stop_time, FMT) - \
            datetime.strptime(start_time, FMT)

        if action in time_dict_action:
            time_dict_action[action].append(tdelta.total_seconds())
        else:
            time_dict_action[action] = [tdelta.total_seconds()]

        time_list.append(tdelta.total_seconds())

    time_dict_action = get_unique_actions(
        time_dict_action, noun_coarse_dict, verb_coarse_dict)

    df_action1 = pd.DataFrame.from_dict(time_dict_action[action_1])
    df_action1 = df_action1.rename(columns={0: "time"})

    df_action2 = pd.DataFrame.from_dict(time_dict_action[action_2])
    df_action2 = df_action2.rename(columns={0: "time"})

    df_action1['action'] = [action_1] * len(df_action1.index)
    df_action2['action'] = [action_2] * len(df_action2.index)

    df_action1 = df_action1.append(df_action2)

    return (df_action1)


def get_dataframe_noun_seconds(entity='open', participant='All', video='All', degree=1, a=1, b=5):

    noun_dict = {}
    noun_dict_nested = {}

    for i in range(len(segment_uids) - 1):

        # for each video, this should be broken down
        example_meta_entry = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i]]
        actions_tube = get_future_actions(i, a, b)

        participant_id = example_meta_entry['meta_data'][0]['participant_id']
        video_id = example_meta_entry['meta_data'][0]['video_id'].split(
            '_')[-1]

        if participant == 'All' or participant_id == participant:
            pass
        elif participant_id != participant:
            continue

        if video == 'All' or video_id == video:
            pass
        elif video_id != video:
            continue

        # case where there is no annotation
        if actions_tube == []:

            noun = example_meta_entry['meta_data'][0]['noun']
            verb = example_meta_entry['meta_data'][0]['verb']
            action = verb + ' ' + noun

            noun_next = 'None'
            verb_next = 'None'
            action_next = 'None None'

            # nouns
            if noun in noun_dict:
                noun_dict[noun] += 1
            else:
                noun_dict[noun] = 1

            if noun in noun_dict_nested:
                if noun_next in noun_dict_nested[noun]:
                    noun_dict_nested[noun][noun_next] += 1
                else:
                    noun_dict_nested[noun][noun_next] = 1
            else:
                noun_dict_nested[noun] = {}
                noun_dict_nested[noun][noun_next] = 1

        for j in range(len(actions_tube)):

            example_meta_entry_next = rgb_train.gulp_dir.merged_meta_dict[
                segment_uids[actions_tube[j]]]

            vid = example_meta_entry['meta_data'][0]['video_id']
            vid_next = example_meta_entry_next['meta_data'][0]['video_id']

            pid, vid = vid.split('_')
            pid_next, vid_next = vid_next.split('_')

            # if the action is not from the same sequence, do not add
            if pid != pid_next or vid != vid_next:
                continue

            noun = example_meta_entry['meta_data'][0]['noun']
            verb = example_meta_entry['meta_data'][0]['verb']
            action = verb + ' ' + noun

            noun_next = example_meta_entry_next['meta_data'][0]['noun']
            verb_next = example_meta_entry_next['meta_data'][0]['verb']
            action_next = verb_next + ' ' + noun_next

            # nouns
            if noun in noun_dict:
                noun_dict[noun] += 1
            else:
                noun_dict[noun] = 1

            if noun in noun_dict_nested:
                if noun_next in noun_dict_nested[noun]:
                    noun_dict_nested[noun][noun_next] += 1
                else:
                    noun_dict_nested[noun][noun_next] = 1
            else:
                noun_dict_nested[noun] = {}
                noun_dict_nested[noun][noun_next] = 1

    noun_dict_nested = get_unique_entity_nested(
        noun_dict_nested, noun_coarse_dict)
    noun_dict = get_unique_keys_2(noun_dict, noun_coarse_dict)
    noun_dict = change_dict_format(noun_dict, type='noun')

    #verb, noun = act.split(' ')
    entity_dict = change_dict_format(noun_dict_nested[entity], type='noun')
    return entity_dict, noun_dict


def get_dateframe_verb_seconds(entity='open', participant='All', video='All', degree=1, a=1, b=5):

    verb_dict = {}
    verb_dict_nested = {}

    for i in range(len(segment_uids) - 1):

        # for each video, this should be broken down
        example_meta_entry = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i]]
        actions_tube = get_future_actions(i, 1, 5)

        participant_id = example_meta_entry['meta_data'][0]['participant_id']
        video_id = example_meta_entry['meta_data'][0]['video_id'].split(
            '_')[-1]

        if participant == 'All' or participant_id == participant:
            pass
        elif participant_id != participant:
            continue

        if video == 'All' or video_id == video:
            pass
        elif video_id != video:
            continue

        # case where there is no annotation
        if actions_tube == []:

            noun = example_meta_entry['meta_data'][0]['noun']
            verb = example_meta_entry['meta_data'][0]['verb']
            action = verb + ' ' + noun

            noun_next = 'None'
            verb_next = 'None'
            action_next = 'None None'

            # verbs
            if verb in verb_dict:
                verb_dict[verb] += 1
            else:
                verb_dict[verb] = 1

            if verb in verb_dict_nested:
                if verb_next in verb_dict_nested[verb]:
                    verb_dict_nested[verb][verb_next] += 1
                else:
                    verb_dict_nested[verb][verb_next] = 1
            else:
                verb_dict_nested[verb] = {}
                verb_dict_nested[verb][verb_next] = 1

        for j in range(len(actions_tube)):

            example_meta_entry_next = rgb_train.gulp_dir.merged_meta_dict[
                segment_uids[actions_tube[j]]]

            vid = example_meta_entry['meta_data'][0]['video_id']
            vid_next = example_meta_entry_next['meta_data'][0]['video_id']

            pid, vid = vid.split('_')
            pid_next, vid_next = vid_next.split('_')

            # if the action is not from the same sequence, do not add
            if pid != pid_next or vid != vid_next:
                continue

            noun = example_meta_entry['meta_data'][0]['noun']
            verb = example_meta_entry['meta_data'][0]['verb']
            action = verb + ' ' + noun

            noun_next = example_meta_entry_next['meta_data'][0]['noun']
            verb_next = example_meta_entry_next['meta_data'][0]['verb']
            action_next = verb_next + ' ' + noun_next

            # verbs
            if verb in verb_dict:
                verb_dict[verb] += 1
            else:
                verb_dict[verb] = 1

            if verb in verb_dict_nested:
                if verb_next in verb_dict_nested[verb]:
                    verb_dict_nested[verb][verb_next] += 1
                else:
                    verb_dict_nested[verb][verb_next] = 1
            else:
                verb_dict_nested[verb] = {}
                verb_dict_nested[verb][verb_next] = 1

    verb_dict_nested = get_unique_entity_nested(
        verb_dict_nested, verb_coarse_dict)
    verb_dict = get_unique_keys_2(verb_dict, verb_coarse_dict)
    verb_dict = change_dict_format(verb_dict, type='verb')

    entity_dict = change_dict_format(verb_dict_nested[entity], type='verb')
    return entity_dict, verb_dict


def get_dataframe_noun(entity='open', participant='All', video='All', degree=1):

    noun_dict = {}
    noun_dict_nested = {}

    for i in range(len(segment_uids) - int(degree)):

        # for each video, this should be broken down

        example_meta_entry = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i]]
        example_meta_entry_next = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i + int(
            degree)]]

        vid = example_meta_entry['meta_data'][0]['video_id']
        vid_next = example_meta_entry_next['meta_data'][0]['video_id']

        pid, vid = vid.split('_')
        pid_next, vid_next = vid_next.split('_')

        # if the action is not from the same sequence, do not add
        if pid != pid_next or vid != vid_next:
            continue

        participant_id = example_meta_entry['meta_data'][0]['participant_id']
        video_id = example_meta_entry['meta_data'][0]['video_id'].split(
            '_')[-1]

        if participant == 'All' or participant_id == participant:
            pass
        elif participant_id != participant:
            continue

        if video == 'All' or video_id == video:
            pass
        elif video_id != video:
            continue

        noun = example_meta_entry['meta_data'][0]['noun']
        noun_next = example_meta_entry_next['meta_data'][0]['noun']
        # nouns
        if noun in noun_dict:
            noun_dict[noun] += 1
        else:
            noun_dict[noun] = 1

        if noun in noun_dict_nested:
            if noun_next in noun_dict_nested[noun]:
                noun_dict_nested[noun][noun_next] += 1
            else:
                noun_dict_nested[noun][noun_next] = 1
        else:
            noun_dict_nested[noun] = {}
            noun_dict_nested[noun][noun_next] = 1

    noun_dict_nested = get_unique_entity_nested(
        noun_dict_nested, noun_coarse_dict)
    noun_dict = get_unique_keys_2(noun_dict, noun_coarse_dict)
    noun_dict = change_dict_format(noun_dict, type='noun')

    entity_dict = change_dict_format(noun_dict_nested[entity], type='noun')

    return entity_dict, noun_dict


def get_unique_actions(dictionary, class_mapping_nouns, class_mapping_verbs):

    unique_dictionary = {}
    for key in dictionary:

        # print(key)
        verb, noun = key.split(' ')
        if noun in class_mapping_nouns and verb in class_mapping_verbs:
            unique_action = class_mapping_verbs[verb] + \
                ' ' + class_mapping_nouns[noun]
            if unique_action in unique_dictionary:
                unique_dictionary[unique_action] += dictionary[key]
            else:
                unique_dictionary[unique_action] = dictionary[key]
        else:
            pass
    return unique_dictionary


def get_unique_actions_nested(dictionary, class_mapping_nouns, class_mapping_verbs):

    unique_dictionary = {}
    for key in dictionary:

        verb, noun = key.split(' ')
        if noun in class_mapping_nouns and verb in class_mapping_verbs:
            unique_action = class_mapping_verbs[verb] + \
                ' ' + class_mapping_nouns[noun]
            if unique_action in unique_dictionary:
                unique_dictionary[unique_action] = combine_dict(
                    unique_dictionary[unique_action], dictionary[key])
            else:
                unique_dictionary[unique_action] = dictionary[key]
        else:
            pass

    for key in unique_dictionary:
        temp_dict = {}
        for x in unique_dictionary[key]:
            verb, noun = x.split(' ')
            if noun in class_mapping_nouns and verb in class_mapping_verbs:

                unique_action = class_mapping_verbs[verb] + \
                    ' ' + class_mapping_nouns[noun]
                if unique_action in temp_dict:
                    temp_dict[unique_action] += unique_dictionary[key][x]
                else:
                    temp_dict[unique_action] = unique_dictionary[key][x]
            else:
                pass
        unique_dictionary[key] = temp_dict

    return unique_dictionary


def get_future_actions(i, a, b):

    example_meta_entry = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i]]

    present = example_meta_entry['meta_data'][0]['video_id']
    pid, vid = present.split('_')

    start_time = example_meta_entry['meta_data'][0]['start_timestamp']
    stop_time = example_meta_entry['meta_data'][0]['stop_timestamp']

    start_time_present = datetime.strptime(start_time, "%H:%M:%S.%f")
    stop_time_present = datetime.strptime(stop_time, "%H:%M:%S.%f")

    start_time_future_intervel = stop_time_present + timedelta(seconds=int(a))
    stop_time_future_intervel = stop_time_present + timedelta(seconds=int(b))
    #print (start_time_future_intervel, stop_time_future_intervel)

    list_future = []

    for j in range(i + 1, len(segment_uids) - 1):
        # if the action is not from the same sequence, do not add

        example_meta_entry_next = rgb_train.gulp_dir.merged_meta_dict[segment_uids[j]]
        future = example_meta_entry_next['meta_data'][0]['video_id']
        pid_next, vid_next = future.split('_')

        start_time = example_meta_entry_next['meta_data'][0]['start_timestamp']
        stop_time = example_meta_entry_next['meta_data'][0]['stop_timestamp']

        start_time_future = datetime.strptime(start_time, "%H:%M:%S.%f")
        stop_time_future = datetime.strptime(stop_time, "%H:%M:%S.%f")

        #print (start_time_future, stop_time_future)

        if pid != pid_next or vid != vid_next:
            break

        if (start_time_future_intervel > stop_time_future) or (stop_time_future_intervel < start_time_future):
            break
        else:
            list_future.append(j)

    return list_future


def get_dataframe_action(entity='open fridge', participant='All', video='All', degree=1):

    action_dict = {}
    action_dict_nested = {}

    for i in range(len(segment_uids) - int(degree)):

        # for each video, this should be broken down

        example_meta_entry = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i]]
        example_meta_entry_next = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i + int(
            degree)]]

        vid = example_meta_entry['meta_data'][0]['video_id']
        vid_next = example_meta_entry_next['meta_data'][0]['video_id']

        pid, vid = vid.split('_')
        pid_next, vid_next = vid_next.split('_')

        # if the action is not from the same sequence, do not add
        if pid != pid_next or vid != vid_next:
            continue

        participant_id = example_meta_entry['meta_data'][0]['participant_id']
        video_id = example_meta_entry['meta_data'][0]['video_id'].split(
            '_')[-1]

        if participant == 'All' or participant_id == participant:
            pass
        elif participant_id != participant:
            continue

        if video == 'All' or video_id == video:
            pass
        elif video_id != video:
            continue

        noun = example_meta_entry['meta_data'][0]['noun']
        verb = example_meta_entry['meta_data'][0]['verb']
        action = verb + ' ' + noun

        noun_next = example_meta_entry_next['meta_data'][0]['noun']
        verb_next = example_meta_entry_next['meta_data'][0]['verb']
        action_next = verb_next + ' ' + noun_next

        if action in action_dict:
            action_dict[action] += 1
        else:
            action_dict[action] = 1

        if action in action_dict_nested:
            if action_next in action_dict_nested[action]:
                action_dict_nested[action][action_next] += 1
            else:
                action_dict_nested[action][action_next] = 1
        else:
            action_dict_nested[action] = {}
            action_dict_nested[action][action_next] = 1

    action_dict_nested = get_unique_actions_nested(
        action_dict_nested, noun_coarse_dict, verb_coarse_dict)
    action_dict = get_unique_actions(
        action_dict, noun_coarse_dict, verb_coarse_dict)
    #action_dict = change_dict_format(action_dict, type='action')

    entity_dict = change_dict_format(action_dict_nested[entity], type='action')

    return entity_dict


def get_dataframe_action_seconds(entity='open fridge', participant='All', video='All', a=1, b=5):

    action_dict = {}
    action_dict_nested = {}

    for i in range(len(segment_uids) - 1):

        # for each video, this should be broken down
        example_meta_entry = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i]]
        actions_tube = get_future_actions(i, a, b)

        participant_id = example_meta_entry['meta_data'][0]['participant_id']
        video_id = example_meta_entry['meta_data'][0]['video_id'].split(
            '_')[-1]

        if participant == 'All' or participant_id == participant:
            pass
        elif participant_id != participant:
            continue

        if video == 'All' or video_id == video:
            pass
        elif video_id != video:
            continue

        noun = example_meta_entry['meta_data'][0]['noun']
        verb = example_meta_entry['meta_data'][0]['verb']
        action = verb + ' ' + noun
        # case where there is no annotation
        if actions_tube == []:

            noun_next = 'None'
            verb_next = 'None'
            action_next = 'None None'

            if action in action_dict:
                action_dict[action] += 1
            else:
                action_dict[action] = 1

            if action in action_dict_nested:
                if action_next in action_dict_nested[action]:
                    action_dict_nested[action][action_next] += 1
                else:
                    action_dict_nested[action][action_next] = 1
            else:
                action_dict_nested[action] = {}
                action_dict_nested[action][action_next] = 1

        for j in range(len(actions_tube)):

            example_meta_entry_next = rgb_train.gulp_dir.merged_meta_dict[
                segment_uids[actions_tube[j]]]

            vid = example_meta_entry['meta_data'][0]['video_id']
            vid_next = example_meta_entry_next['meta_data'][0]['video_id']

            pid, vid = vid.split('_')
            pid_next, vid_next = vid_next.split('_')

            # if the action is not from the same sequence, do not add
            if pid != pid_next or vid != vid_next:
                continue

            noun_next = example_meta_entry_next['meta_data'][0]['noun']
            verb_next = example_meta_entry_next['meta_data'][0]['verb']
            action_next = verb_next + ' ' + noun_next

            # actions
            if action in action_dict:
                action_dict[action] += 1
            else:
                action_dict[action] = 1

            if action in action_dict_nested:
                if action_next in action_dict_nested[action]:
                    action_dict_nested[action][action_next] += 1
                else:
                    action_dict_nested[action][action_next] = 1
            else:
                action_dict_nested[action] = {}
                action_dict_nested[action][action_next] = 1

    action_dict_nested = get_unique_actions_nested(
        action_dict_nested, noun_coarse_dict, verb_coarse_dict)
    action_dict = get_unique_actions(
        action_dict, noun_coarse_dict, verb_coarse_dict)
    #action_dict = change_dict_format(action_dict)
    entity_dict = change_dict_format(action_dict_nested[entity], type='action')
    return entity_dict


def get_noun_verb_dict(coarse_flag=True,
                       range_min_noun=1,
                       range_max_noun=320,
                       range_min_verb=1,
                       range_max_verb=119,
                       range_min_action=1,
                       range_max_action=2200,
                       participant='All',
                       video='All'):

    noun_dict = {}
    verb_dict = {}
    action_dict = {}

    for i in range(len(segment_uids)):

        example_meta_entry = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i]]
        noun = example_meta_entry['meta_data'][0]['noun']
        verb = example_meta_entry['meta_data'][0]['verb']
        participant_id = example_meta_entry['meta_data'][0]['participant_id']
        video_id = example_meta_entry['meta_data'][0]['video_id'].split(
            '_')[-1]
        # print('Video_id: ', video_id)
        # print('Video: ', video)

        if participant == 'All' or participant_id == participant:
            pass
        elif participant_id != participant:
            continue

        if video == 'All' or video_id == video:
            pass
        elif video_id != video:
            continue

        action = verb + ' ' + noun

        if noun in noun_dict:
            noun_dict[noun] += 1
        else:
            noun_dict[noun] = 1

        if verb in verb_dict:
            verb_dict[verb] += 1
        else:
            verb_dict[verb] = 1

        if action in action_dict:
            action_dict[action] += 1
        else:
            action_dict[action] = 1

    if coarse_flag:
        noun_dict = get_unique_keys_2(noun_dict, noun_coarse_dict)
        verb_dict = get_unique_keys_2(verb_dict, verb_coarse_dict)
        action_dict = get_unique_actions(
            action_dict, noun_coarse_dict, verb_coarse_dict)

    noun_dict_sorted = {k: v for k, v in sorted(
        noun_dict.items(), key=lambda item: item[1], reverse=True)[range_min_noun - 1:range_max_noun - 1]}
    verb_dict_sorted = {k: v for k, v in sorted(
        verb_dict.items(), key=lambda item: item[1], reverse=True)[range_min_verb - 1:range_max_verb - 1]}
    action_dict_sorted = {k: v for k, v in sorted(
        action_dict.items(), key=lambda item: item[1], reverse=True)[range_min_action - 1:range_max_action - 1]}

    noun_dict_sorted = change_dict_format(noun_dict_sorted)
    verb_dict_sorted = change_dict_format(verb_dict_sorted)
    action_dict_sorted = change_dict_format(action_dict_sorted)
    return noun_dict_sorted, verb_dict_sorted, action_dict_sorted


# actions which entain most number of actions divided by the number of times they occur
def get_entropy(action_dict, action_count_dict):

    list_actions = []
    list_entropy = []
    list_count = []

    for key in action_dict:

        values = list(action_dict[key].values())
        sum_values = sum(action_dict[key].values())
        count = action_count_dict[key]

        probs = [i / sum_values for i in values]

        # Compute entropy
        ent = 0.
        base = None
        base = e if base is None else base
        for i in probs:
            ent -= i * log(i, base)

        list_actions.append(key)
        list_entropy.append(ent)
        list_count.append(count)

    return list_actions, list_entropy, list_count


def get_most_uncertain_actions():

    noun_dict = {}
    verb_dict = {}

    action_dict = {}
    action_dict_nested = {}

    for i in range(len(segment_uids) - 1):

        # for each video, this should be broken down

        example_meta_entry = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i]]
        example_meta_entry_next = rgb_train.gulp_dir.merged_meta_dict[segment_uids[i + 1]]

        vid = example_meta_entry['meta_data'][0]['video_id']
        vid_next = example_meta_entry_next['meta_data'][0]['video_id']

        pid, vid = vid.split('_')
        pid_next, vid_next = vid_next.split('_')

        # if the action is not from the same sequence, do not add
        if pid != pid_next or vid != vid_next:
            continue

        noun = example_meta_entry['meta_data'][0]['noun']
        verb = example_meta_entry['meta_data'][0]['verb']
        action = verb + ' ' + noun

        noun_next = example_meta_entry_next['meta_data'][0]['noun']
        verb_next = example_meta_entry_next['meta_data'][0]['verb']
        action_next = verb_next + ' ' + noun_next

        if action in action_dict:
            action_dict[action] += 1
        else:
            action_dict[action] = 1

        if action in action_dict_nested:
            if action_next in action_dict_nested[action]:
                action_dict_nested[action][action_next] += 1
            else:
                action_dict_nested[action][action_next] = 1
        else:
            action_dict_nested[action] = {}
            action_dict_nested[action][action_next] = 1

    action_dict_nested = get_unique_actions_nested(
        action_dict_nested, noun_coarse_dict, verb_coarse_dict)
    action_dict = get_unique_actions(
        action_dict, noun_coarse_dict, verb_coarse_dict)
    #print(action_dict_nested, action_dict)

    list_actions, list_entropy, list_count = get_entropy(
        action_dict_nested, action_dict)
    #print (list_actions[0], list_entropy[0], list_count[0])

    list_common = zip(list_actions, list_entropy, list_count)
    list_common = sorted(list_common, key=lambda tup: tup[1], reverse=True)

    list_actions = [i[0] for i in list_common]
    list_entropy = [i[1] for i in list_common]
    list_count = [i[2] for i in list_common]

    dict_entropy = {}
    dict_scatter = {'Action': list_actions,
                    'Entropy': list_entropy, 'Number of Instances': list_count}
    pd_scatter = pd.DataFrame.from_dict(dict_scatter)

    for i in range(len(list_actions)):
        dict_entropy[list_actions[i]] = list_entropy[i]

    entropy_sorted = {k: v for k, v in sorted(
        dict_entropy.items(), key=lambda item: item[1], reverse=True)}
    entropy_sorted = change_dict_format(entropy_sorted)
    return (pd_scatter, entropy_sorted)


#    get_scatter_plot(dict_scatter)
#     print (statistics.mean(list_entropy), statistics.median(list_entropy),  statistics.stdev(list_entropy))


# available_indicators = df['Indicator Name'].unique()
df_noun = meta.noun_classes()
df_verb = meta.verb_classes()

# include None as an action
df2_noun = pd.DataFrame({"class_key": ['None'], "nouns": [['None']]})
df_noun = df_noun.append(df2_noun)

df2_verb = pd.DataFrame({"class_key": ['None'], "verbs": [['None']]})
df_verb = df_verb.append(df2_verb)

noun_coarse_dict = pd_to_dict(df_noun, entity='nouns')
verb_coarse_dict = pd_to_dict(df_verb, entity='verbs')

# available verbs
_, verb_dict = get_dataframe_verb(entity='open')
available_entities_verb = pd.DataFrame.from_dict(verb_dict)['Verbs'].unique()

# available nouns
_, noun_dict = get_dataframe_noun(entity='fridge')
available_entities_noun = pd.DataFrame.from_dict(noun_dict)['Nouns'].unique()

action_dict = get_dataframe_action(entity='open fridge')


# video and participant info
df_meta = meta.video_info()

df_meta['participant'] = df_meta.index.str.split('_')
df_meta['video'] = df_meta.index.str.split('_')

df_meta['participant'] = df_meta['participant'].apply(lambda x: x[0])
df_meta['video'] = df_meta['video'].apply(lambda x: x[1])

# available participants
df_meta = df_meta.append(
    {'participant': 'All', 'video': 'All'}, ignore_index=True)
available_participants = df_meta['participant'].unique()


# available_videos for a particular participant
def available_videos(participant_id):
    # number of videos for one participant
    if participant_id == 'All':
        return ['All']
    else:
        available_vid = list(df_meta[df_meta['participant']
                                     == participant_id]['video'].unique())
        available_vid.append('All')
        return (available_vid)


app.layout = html.Div([

    html.Div([
        html.Div([
            "Dashboard : Epic-Kitchens Dataset"
        ],
            style={'width': '90%',
                   'text-align': 'center',
                   'display': 'inline-block',
                   'fontWeight': 'bold',
                   'text-decoration': 'underline',
                   'font-family': 'Arial, Helvetica, sans-serif'})
    ], style={
        #'borderWidth': 'medium',
        #'borderColor': 'blue',
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '40px 40px',
        'border-radius': '15px',
        'margin-bottom': ' 5px',
    }),

    html.Div([

        html.Div([
            "Participants"
        ],
            style={'width': '99%',
                   'text-align': 'center',
                   'font-weight': 'bold',
                   'padding': '5px 5px',
                   'backgroundColor': 'rgb(250, 250, 250)',
                   }),

        html.Div([

            html.Div([
                "Participant ID"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='general-entity-1',
                options=[{'label': i, 'value': i}
                         for i in available_participants],
                value='All'
            ),
        ],
            style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            html.Div([
                "Video ID"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),
            dcc.Dropdown(
                id='general-entity-2',
                # options=[{'label': 'All',
                #           'value': 'All'}],
                # value='All'
            ),
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'border-radius': '15px',
        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),

    html.Div([
        html.Div([
            "Participants: 32"
        ],
            style={'width': '23%',
                   'text-align': 'center',
                   'display': 'inline-block',
                   'padding': '5px 5px',
                   'borderTop': 'thin lightgrey solid',
                   'borderLeft': 'thin lightgrey solid',
                   'borderRight': 'thin lightgrey solid',
                   'borderBottom': 'thin lightgrey solid',
                   'backgroundColor': 'rgb(250, 250, 250)',
                   'border-radius': '15px',
                   'margin-right': ' 5px',
                   }),
        html.Div([
            "Videos: " + str(len(df_meta.index) - 1)
            # len(example_meta_entry)
        ],
            style={'width': '23%',
                   'text-align': 'center',
                   'display': 'inline-block',
                   'padding': '5px 5px',
                   'borderTop': 'thin lightgrey solid',
                   'borderLeft': 'thin lightgrey solid',
                   'borderRight': 'thin lightgrey solid',
                   'borderBottom': 'thin lightgrey solid',
                   'backgroundColor': 'rgb(250, 250, 250)',
                   'border-radius': '15px',
                   'margin-right': ' 5px',
                   }),
        html.Div([
            "Nouns"
        ],
            style={'width': '23%',
                   'text-align': 'center',
                   'display': 'inline-block',
                   'padding': '5px 5px',
                   'borderTop': 'thin lightgrey solid',
                   'borderLeft': 'thin lightgrey solid',
                   'borderRight': 'thin lightgrey solid',
                   'borderBottom': 'thin lightgrey solid',
                   'backgroundColor': 'rgb(250, 250, 250)',
                   'border-radius': '15px',
                   'margin-right': ' 5px',
                   }),
        html.Div([
            "Verbs"
        ], style={'width': '23%',
                  'text-align': 'center',
                  'display': 'inline-block',
                  'padding': '5px 5px',
                  'borderTop': 'thin lightgrey solid',
                  'borderLeft': 'thin lightgrey solid',
                  'borderRight': 'thin lightgrey solid',
                  'borderBottom': 'thin lightgrey solid',
                  'backgroundColor': 'rgb(250, 250, 250)',
                  'border-radius': '15px',
                  })
    ], style={

        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),

    html.Div([
        html.Div([

            html.Div([
                "Noun Plots"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='bar-entity-1',
                options=[{'label': 'Noun Distribution',
                          'value': 'Noun Distribution'}],
                value='Noun Distribution'
            ),


            dcc.RadioItems(
                id='bar-entity-1-radio',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Log',
                labelStyle={'display': 'inline-block'}
            )
        ],
            style={'width': '49%', 'display': 'inline-block'}),
        html.Div([

            html.Div([
                "Verb Plots"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='bar-entity-2',
                options=[{'label': 'Verb Distribution',
                          'value': 'Verb Distribution'}],
                value='Verb Distribution'
            ),
            dcc.RadioItems(
                id='bar-entity-2-radio',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Log',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'border-radius': '15px',
        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),

    html.Div([
        html.Div([
            dcc.Graph(
                id='bar-chart-1',
                #hoverData={'points': [{'customdata': 'Japan'}]}
            ),

            dcc.RangeSlider(
                id='bar-chart-1-slider',
                min=1,
                max=320,
                # marks={'2':'2','5':'5'},
                value=[1, 320],
                step=10,
            ),

            # html.Div([
            #     dcc.Slider(
            #         id='bar-chart-1-slider',
            #         min=0,
            #         max=10,
            #         value=10,
            #         marks={'2':'2','5':'5'},
            #         step=None
            #     ),
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

        html.Div([
            dcc.Graph(
                id='bar-chart-2'
            ),

            dcc.RangeSlider(
                id='bar-chart-2-slider',
                min=1,
                max=119,
                # marks={'2':'2','5':'5'},
                value=[1, 119],
                step=10,
            ),

        ], style={'display': 'inline-block', 'width': '49%'}),
    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '40px 40px',
        'border-radius': '15px',
        'margin-bottom': ' 5px',
    }),


    html.Div([
        html.Div([

            html.Div([
                "Actions Plots"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='bar-entity-3',
                options=[{'label': 'Action Distribution',
                          'value': 'Action Distribution'}],
                value='Action Distribution'
            ),
            dcc.RadioItems(
                id='bar-entity-3-radio',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Log',
                labelStyle={'display': 'inline-block', 'align': 'center'}
            )
        ],
            style={'width': '49%', 'display': 'inline-block'}),
        html.Div([

            html.Div([
                "Action Entropy Plots"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='bar-entity-4',
                options=[{'label': 'Action Entropy Distribution',
                          'value': 'Action Entropy Distribution'},
                         {'label': 'Action Entropy Scatter Plot',
                          'value': 'Action Entropy Scatter Plot'}],
                value='Action Entropy Distribution'
            )  # ,
            # dcc.RadioItems(
            #     id='bar-entity-4-radio',
            #     options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
            #     value='Log',
            #     labelStyle={'display': 'inline-block'}
            # )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'border-radius': '15px',
        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),

    html.Div([
        html.Div([

            dcc.Graph(
                id='bar-chart-3',
                #hoverData={'points': [{'customdata': 'Japan'}]}
            ),

            dcc.RangeSlider(
                id='bar-chart-3-slider',
                min=1,
                max=2000,
                # marks={'2':'2','5':'5'},
                value=[1, 2000],
                step=20,
            ),

        ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
        html.Div([
            dcc.Graph(
                id='bar-chart-4'
            ),

            dcc.RangeSlider(
                id='bar-chart-4-slider',
                min=1,
                max=2000,
                # marks={'2':'2','5':'5'},
                value=[1, 2000],
                step=20,
            ),
        ], style={'display': 'inline-block', 'width': '49%'}),
    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '40px 40px',
        'border-radius': '15px',
        'margin-bottom': ' 5px',
    }),

    html.Div([

        html.Div([
            "Future Action Module"
        ],
            style={'width': '99%',
                   'text-align': 'center',
                   'font-weight': 'bold',
                   'padding': '5px 5px',
                   'backgroundColor': 'rgb(250, 250, 250)',
                   }),

        html.Div([

            html.Div([
                "Verb"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='entity-1',
                options=[{'label': i, 'value': i}
                         for i in available_entities_verb],
                value='open'
            )
        ], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([

            html.Div([
                "Noun"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='entity-2',
                options=[{'label': i, 'value': i}
                         for i in available_entities_noun],
                value='fridge'
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

        html.Div([

            html.Div([
                "Degree"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='entity-3',
                options=[{'label': i, 'value': i}
                         for i in [1, 2, 3, 4, 5]],
                value='1'
            )
        ], style={'width': '99%', 'display': 'inline-block'}),

    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'border-radius': '15px',
        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),

    # html.Div([
    #     html.Div([
    #         dcc.Dropdown(
    #             id='entity-3',
    #             options=[{'label': i, 'value': i}
    #                      for i in [1,2,3,4,5]],
    #             value='1'
    #         )
    #     ],
    #         style={'width': '99%', 'display': 'inline-block'}),
    # ], style={
    #     'borderTop': 'thin lightgrey solid',
    #     'borderLeft': 'thin lightgrey solid',
    #     'borderRight': 'thin lightgrey solid',
    #     'borderBottom': 'thin lightgrey solid',
    #     'backgroundColor': 'rgb(250, 250, 250)',
    #     'border-radius': '15px',
    #     'padding': '10px 5px',
    #     'margin-bottom': ' 5px',
    # }),



    html.Div([
        html.Div([
            dcc.Graph(
                id='pie-chart',
                #hoverData={'points': [{'customdata': 'Japan'}]}
            )
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
        html.Div([
            dcc.Graph(
                id='pie-chart-2'
            )
        ], style={'display': 'inline-block', 'width': '49%'}),
    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'border-radius': '15px',
        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),

    html.Div([
        html.Div([
            dcc.Graph(
                id='pie-chart-3'
            )
        ], style={'display': 'inline-block', 'width': '100%', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'border-radius': '15px',
        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),


    html.Div([

        html.Div([
            "Future Action Module [ in seconds ]"
        ],
            style={'width': '99%',
                   'text-align': 'center',
                   'font-weight': 'bold',
                   'padding': '5px 5px',
                   'backgroundColor': 'rgb(250, 250, 250)',
                   }),

        html.Div([

            html.Div([
                "Verb"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='entity-1-seconds',
                options=[{'label': i, 'value': i}
                         for i in available_entities_verb],
                value='open'
            )
        ], style={'width': '49%', 'display': 'inline-block'}),

        html.Div([

            html.Div([
                "Noun"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='entity-2-seconds',
                options=[{'label': i, 'value': i}
                         for i in available_entities_noun],
                value='fridge'
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

        html.Div([

            html.Div([
                "Start time offset"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='time-offset-1',
                options=[{'label': i, 'value': i}
                         for i in [1, 2, 3, 4, 5]],
                value='1'
            )
        ], style={'width': '49%', 'display': 'inline-block'}),

        html.Div([

            html.Div([
                "End time offset"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='time-offset-2',
                options=[{'label': i, 'value': i}
                         for i in [1, 2, 3, 4, 5]],
                value='2'
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'border-radius': '15px',
        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),

    html.Div([
        html.Div([
            dcc.Graph(
                id='pie-chart-4',
                #hoverData={'points': [{'customdata': 'Japan'}]}
            )
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
        html.Div([
            dcc.Graph(
                id='pie-chart-5'
            )
        ], style={'display': 'inline-block', 'width': '49%'}),
    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'border-radius': '15px',
        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),

    html.Div([
        html.Div([
            dcc.Graph(
                id='pie-chart-6'
            )
        ], style={'display': 'inline-block', 'width': '100%', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'border-radius': '15px',
        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),


    html.Div([

        html.Div([
            "Action-time Module"
        ],
            style={'width': '99%',
                   'text-align': 'center',
                   'font-weight': 'bold',
                   'padding': '5px 5px',
                   'backgroundColor': 'rgb(250, 250, 250)',
                   }),

        html.Div([

            html.Div([
                "Verb-1"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='verb-action-1',
                options=[{'label': i, 'value': i}
                         for i in available_entities_verb],
                value='open'
            )
        ], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([

            html.Div([
                "Noun-1"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='noun-action-1',
                options=[{'label': i, 'value': i}
                         for i in available_entities_noun],
                value='fridge'
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),


        html.Div([

            html.Div([
                "Verb-2"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='verb-action-2',
                options=[{'label': i, 'value': i}
                         for i in available_entities_verb],
                value='open'
            )
        ], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([

            html.Div([
                "Noun-2"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='noun-action-2',
                options=[{'label': i, 'value': i}
                         for i in available_entities_noun],
                value='drawer'
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),


    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'border-radius': '15px',
        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),


    html.Div([
        html.Div([
            dcc.Graph(
                id='hist-1'
            )
        ], style={'display': 'inline-block', 'width': '100%', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ], style={
        'borderTop': 'thin lightgrey solid',
        'borderLeft': 'thin lightgrey solid',
        'borderRight': 'thin lightgrey solid',
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'border-radius': '15px',
        'padding': '10px 5px',
        'margin-bottom': ' 5px',
    }),

])


@app.callback(
    dash.dependencies.Output('general-entity-2', 'options'),
    [dash.dependencies.Input('general-entity-1', 'value')])
def set_video_options(entity_name_1):
    return [{'label': i, 'value': i} for i in available_videos(entity_name_1)]


@app.callback(
    dash.dependencies.Output('general-entity-2', 'value'),
    [dash.dependencies.Input('general-entity-2', 'options')])
def set_cities_value(entity_name_1):
    return entity_name_1[-1]['value']


@app.callback(
    dash.dependencies.Output('bar-chart-1', 'figure'),
    [dash.dependencies.Input('bar-entity-1', 'value'),
     dash.dependencies.Input('bar-entity-1-radio', 'value'),
     dash.dependencies.Input('bar-chart-1-slider', 'value'),
     dash.dependencies.Input('general-entity-1', 'value'),
     dash.dependencies.Input('general-entity-2', 'value'),
        #dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(entity_name_1, entity_name_2, entity_name_3, entity_name_4, entity_name_5):

    range_min_noun, range_max_noun = entity_name_3[0], entity_name_3[1]
    dictionary, _, _ = get_noun_verb_dict(
        coarse_flag=True, range_min_noun=range_min_noun, range_max_noun=range_max_noun, participant=entity_name_4, video=entity_name_5)
    df = pd.DataFrame.from_dict(dictionary)
    barchart = px.bar(df, x="Verbs", y="Number of instances", labels={
                      'Verbs': 'Nouns', 'Number of instances': 'Number of instances'}, title="Most frequent nouns")
    if entity_name_2 == 'Log':
        barchart = barchart.update_layout(yaxis_type="log")
    elif entity_name_2 == 'Linear':
        pass
    return (barchart)


@app.callback(
    dash.dependencies.Output('bar-chart-2', 'figure'),
    [dash.dependencies.Input('bar-entity-2', 'value'),
     dash.dependencies.Input('bar-entity-2-radio', 'value'),
     dash.dependencies.Input('bar-chart-2-slider', 'value'),
     dash.dependencies.Input('general-entity-1', 'value'),
     dash.dependencies.Input('general-entity-2', 'value'),
        #dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
        #dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
        #dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(entity_name_1, entity_name_2, entity_name_3, entity_name_4, entity_name_5):

    range_min_verb, range_max_verb = entity_name_3[0], entity_name_3[1]
    _, dictionary, _ = get_noun_verb_dict(
        coarse_flag=True, range_min_verb=range_min_verb, range_max_verb=range_max_verb, participant=entity_name_4, video=entity_name_5)
    df = pd.DataFrame.from_dict(dictionary)
    barchart = px.bar(df, x="Verbs", y="Number of instances", labels={
                      'Verbs': 'Verbs', 'Number of instances': 'Number of instances'}, title="Most frequent verbs")
    if entity_name_2 == 'Log':
        barchart = barchart.update_layout(yaxis_type="log")
    elif entity_name_2 == 'Linear':
        pass
    return (barchart)


@app.callback(
    dash.dependencies.Output('bar-chart-3', 'figure'),
    [dash.dependencies.Input('bar-entity-3', 'value'),
     dash.dependencies.Input('bar-entity-3-radio', 'value'),
     dash.dependencies.Input('bar-chart-3-slider', 'value'),
     dash.dependencies.Input('general-entity-1', 'value'),
     dash.dependencies.Input('general-entity-2', 'value'),
        #dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
        #dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
        #dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(entity_name_1, entity_name_2, entity_name_3, entity_name_4, entity_name_5):

    range_min_action, range_max_action = entity_name_3[0], entity_name_3[1]
    _, _, dictionary = get_noun_verb_dict(
        coarse_flag=True, range_min_action=range_min_action, range_max_action=range_max_action, participant=entity_name_4, video=entity_name_5)
    df = pd.DataFrame.from_dict(dictionary)
    barchart = px.bar(df, x="Verbs", y="Number of instances", labels={
                      'Verbs': 'Actions', 'Number of instances': 'Number of instances'}, title="Most frequent actions")
    if entity_name_2 == 'Log':
        barchart = barchart.update_layout(yaxis_type="log")
    elif entity_name_2 == 'Linear':
        pass
    return (barchart)


# action entropy
@app.callback(
    dash.dependencies.Output('bar-chart-4', 'figure'),
    [dash.dependencies.Input('bar-entity-4', 'value'),
     #dash.dependencies.Input('bar-entity-4-radio', 'value')
        #dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
        #dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
        #dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(entity_name_1):

    if entity_name_1 == 'Action Entropy Distribution':
        _, dictionary = get_most_uncertain_actions()
        df = pd.DataFrame.from_dict(dictionary)
        barchart = px.bar(df, x="Verbs", y="Number of instances", labels={
                          'Verbs': 'Actions', 'Number of instances': 'Entropy'}, title="Most diverse actions")
        return (barchart)
    elif entity_name_1 == 'Action Entropy Scatter Plot':
        df, _ = get_most_uncertain_actions()
        #df = pd.DataFrame.from_dict(dictionary)
        scatterplot = px.scatter(df, x="Entropy", y="Number of Instances", hover_name="Action", labels={
            'Action': 'Action', 'Number of Instances': 'Number of samples', 'Entropy': 'Entropy'}, title="Entropy vs. Samples")
        return (scatterplot)


@app.callback(
    dash.dependencies.Output('pie-chart', 'figure'),
    [dash.dependencies.Input('entity-1', 'value'),
     dash.dependencies.Input('general-entity-1', 'value'),
     dash.dependencies.Input('entity-3', 'value'),
     dash.dependencies.Input('general-entity-2', 'value'),
     #dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     #dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     #dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(entity_name_1, entity_name_2, entity_name_3, entity_name_4):

    entity_dict, verb_dict = get_dataframe_verb(
        entity=entity_name_1, participant=entity_name_2, degree=entity_name_3, video=entity_name_4)
    piechart = px.pie(entity_dict, values='Number of instances', labels={
        'Verbs': 'Next verbs', 'Number of instances': 'Number of instances'}, names='Verbs', title='Next verb given present action:')
    return (piechart)


@app.callback(
    dash.dependencies.Output('pie-chart-2', 'figure'),
    [dash.dependencies.Input('entity-2', 'value'),
     dash.dependencies.Input('general-entity-1', 'value'),
     dash.dependencies.Input('entity-3', 'value'),
     dash.dependencies.Input('general-entity-2', 'value'),
     #dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     #dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     #dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(entity_name_1, entity_name_2, entity_name_3, entity_name_4):

    entity_dict, noun_dict = get_dataframe_noun(
        entity=entity_name_1, participant=entity_name_2, degree=entity_name_3, video=entity_name_4)
    piechart = px.pie(entity_dict, values='Number of instances', labels={
        'Verbs': 'Next nouns', 'Number of instances': 'Number of instances'}, names='Nouns', title='Next noun given present action:')
    return (piechart)


@app.callback(
    dash.dependencies.Output('pie-chart-3', 'figure'),
    [dash.dependencies.Input('entity-1', 'value'),
     dash.dependencies.Input('entity-2', 'value'),
     dash.dependencies.Input('general-entity-1', 'value'),
     dash.dependencies.Input('entity-3', 'value'),
     dash.dependencies.Input('general-entity-2', 'value'),
     #dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     #dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(entity_name_1, entity_name_2, entity_name_3, entity_name_4, entity_name_5):

    action_dict = get_dataframe_action(
        entity=entity_name_1 + ' ' + entity_name_2, participant=entity_name_3, degree=entity_name_4, video=entity_name_5)
    piechart = px.pie(action_dict, values='Number of instances', labels={
        'Actions': 'Next actions', 'Number of instances': 'Number of instances'}, names='Actions', title='Next action given present action:')
    return (piechart)


@app.callback(
    dash.dependencies.Output('pie-chart-4', 'figure'),
    [dash.dependencies.Input('entity-1-seconds', 'value'),
     dash.dependencies.Input('general-entity-1', 'value'),
     dash.dependencies.Input('time-offset-1', 'value'),
     dash.dependencies.Input('time-offset-2', 'value'),
     dash.dependencies.Input('general-entity-2', 'value'),
     #dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     #dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(entity_name_1, entity_name_2, entity_name_3, entity_name_4, entity_name_5):

    entity_dict, verb_dict = get_dateframe_verb_seconds(
        entity=entity_name_1, participant=entity_name_2, a=entity_name_3, b=entity_name_4, video=entity_name_5)
    piechart = px.pie(entity_dict, values='Number of instances', labels={
        'Verbs': 'Next verbs', 'Number of instances': 'Number of instances'}, names='Verbs', title='Next verb given present action:')
    return (piechart)


@app.callback(
    dash.dependencies.Output('pie-chart-5', 'figure'),
    [dash.dependencies.Input('entity-2-seconds', 'value'),
     dash.dependencies.Input('general-entity-1', 'value'),
     dash.dependencies.Input('time-offset-1', 'value'),
     dash.dependencies.Input('time-offset-2', 'value'),
     dash.dependencies.Input('general-entity-2', 'value'),
     #dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     #dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     #dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(entity_name_1, entity_name_2, entity_name_3, entity_name_4, entity_name_5):

    entity_dict, noun_dict = get_dataframe_noun_seconds(
        entity=entity_name_1, participant=entity_name_2, a=entity_name_3, b=entity_name_4, video=entity_name_5)
    piechart = px.pie(entity_dict, values='Number of instances', labels={
        'Verbs': 'Next nouns', 'Number of instances': 'Number of instances'}, names='Nouns', title='Next noun given present action:')
    return (piechart)


@app.callback(
    dash.dependencies.Output('pie-chart-6', 'figure'),
    [dash.dependencies.Input('entity-1-seconds', 'value'),
     dash.dependencies.Input('entity-2-seconds', 'value'),
     dash.dependencies.Input('general-entity-1', 'value'),
     dash.dependencies.Input('time-offset-1', 'value'),
     dash.dependencies.Input('time-offset-2', 'value'),
     dash.dependencies.Input('general-entity-2', 'value'),
     #dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     #dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(entity_name_1, entity_name_2, entity_name_3, entity_name_4, entity_name_5, entity_name_6):

    action_dict = get_dataframe_action_seconds(
        entity=entity_name_1 + ' ' + entity_name_2, participant=entity_name_3, a=entity_name_4, b=entity_name_5, video=entity_name_6)
    piechart = px.pie(action_dict, values='Number of instances', labels={
        'Actions': 'Next actions', 'Number of instances': 'Number of instances'}, names='Actions', title='Next action given present action:')
    return (piechart)


@app.callback(
    dash.dependencies.Output('hist-1', 'figure'),
    [dash.dependencies.Input('noun-action-1', 'value'),
     dash.dependencies.Input('verb-action-1', 'value'),
     dash.dependencies.Input('noun-action-2', 'value'),
     dash.dependencies.Input('verb-action-2', 'value'),
     #dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     #dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(entity_name_1, entity_name_2, entity_name_3, entity_name_4):

    action_1 = entity_name_2 + ' ' + entity_name_1
    action_2 = entity_name_4 + ' ' + entity_name_3

    df = get_average_time_histogram(action_1, action_2)
    hist = px.histogram(df, x="time", color="action",
                        opacity=0.6, barmode="overlay")
    return (hist)


if __name__ == '__main__':

    app.run_server(debug=True)
