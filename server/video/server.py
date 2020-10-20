import dash
import dash_core_components as dcc
import dash_html_components as html

import flask
from flask import render_template_string
import glob
import os
import pickle

image_directory = 'static/videos/'



list_of_images = [os.path.basename(x) for x in glob.glob(
    '{}*.mp4'.format(image_directory))]
# static_image_route = 'static/gifs/'


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

from PIL import Image
from io import StringIO


WIDTH = 200
HEIGHT = 120

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#app = dash.Dash()

import pandas as pd
from epic_kitchens import meta
from flask import Flask, request, render_template, send_from_directory


def pd_to_dict(df, entity='verbs'):
  x_coarse_dict = {}
  for index, row in df.iterrows():
    x_list = row[entity]
    x_class = row['class_key']
    for i in x_list:
      x_coarse_dict[i] = x_class

  return x_coarse_dict


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
verb_dict = {'Verbs': ['open', 'take', 'get']}
available_entities_verb = pd.DataFrame.from_dict(verb_dict)['Verbs'].unique()

# available nouns
noun_dict = {'Nouns': ['onion', 'cupboard', 'tomato']}
available_entities_noun = pd.DataFrame.from_dict(noun_dict)['Nouns'].unique()


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


with open('verb_noun.pkl', 'rb') as f:
  available_verbs, available_nouns = pickle.load(f)


app.layout = html.Div([


    html.Div([
        html.Div([
            "Dashboard : Epic-Kitchens Dataset - Qualitative Explorer"
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
            "Sort by Instance:"
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
                value='All'
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
            "Sort by Action:"
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
                id='verb',
                options=[{'label': i, 'value': i}
                         for i in available_verbs],
                value='open'
            ),
        ],
            style={'width': '49%', 'display': 'inline-block'}),

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
                id='noun',
                options=[{'label': i, 'value': i}
                         for i in available_nouns],
                value='cupboard'
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
            "Display Settings:"
        ],
            style={'width': '99%',
                   'text-align': 'center',
                   'font-weight': 'bold',
                   'padding': '5px 5px',
                   'backgroundColor': 'rgb(250, 250, 250)',
                   }),
        html.Div([

            html.Div([
                "Clips per page"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

            dcc.Dropdown(
                id='pagelimit',
                options=[{'label': i, 'value': i}
                         for i in [10, 20, 30, 40, 50]],
                value=10
            ),
        ],
            style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            html.Div([
                "Page Number"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),
            dcc.Dropdown(
                id='pagenumber',
                # options=[{'label': i, 'value': i}
                #          for i in available_entities_noun],
                value=1
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
            "Qualitative Data"
        ],
            style={'width': '99%',
                   'text-align': 'center',
                   'font-weight': 'bold',
                   'padding': '5px 5px',
                   'backgroundColor': 'rgb(250, 250, 250)',
                   }),
        html.Div([

            html.Div([
                "Clip"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),
        ],
            style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            html.Div([
                "Ground Truth"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),
        ],
            style={'width': '33%', 'float': 'center', 'display': 'inline-block'}),
        html.Div([
            html.Div([
                "Predicted"
            ],
                style={'width': '99%',
                       'text-align': 'center',
                       'font-weight': 'bold',
                       'padding': '5px 5px',
                       'backgroundColor': 'rgb(250, 250, 250)',
                       }),

        ],
            style={'width': '30%', 'float': 'right', 'display': 'inline-block'})


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

    html.Div(id='container'),


])


@app.callback(
    dash.dependencies.Output('general-entity-2', 'options'),
    [dash.dependencies.Input('general-entity-1', 'value')])
def set_video_options(entity_name_1):
  return [{'label': i, 'value': i} for i in available_videos(entity_name_1)]




def get_images(participant='All', video='All', verb='open', noun='cupboard'):
  images = []

  for root, dirs, files in os.walk('static'):
    # for root, dirs, files in os.walk('.'):

    try:
      participant_video = root.split('/')[-1]
      participant_id, video_id = participant_video.split('_')
    except:
      continue

    if participant == 'All' or participant == participant_id:
      pass
    elif participant != participant_id:
      continue

    if video == 'All' or video_id == video:
      pass
    elif video != video_id:
      continue

    for filename in [os.path.join(root, name) for name in files]:
      # verb_id = filename.split('.')[1].split('_')[-2]
      # noun_id = filename.split('.')[1].split('_')[-1]

      verb_id = filename.split('.')[0].split('_')[-2]
      noun_id = filename.split('.')[0].split('_')[-1]

      if verb == 'All' or verb == verb_id:
        pass
      elif verb != verb_id:
        continue

      if noun == 'All' or noun == noun_id:
        pass
      elif noun != noun_id:
        continue

      if not filename.endswith('.mp4'):
        continue
      images.append(filename)
  return images


@app.callback(
    [dash.dependencies.Output('container', 'children'),
     dash.dependencies.Output('pagenumber', 'options')],
    [dash.dependencies.Input('pagenumber', 'value'),
     dash.dependencies.Input('pagelimit', 'value'),
     dash.dependencies.Input('general-entity-1', 'value'),
     dash.dependencies.Input('general-entity-2', 'value'),
     dash.dependencies.Input('verb', 'value'),
     dash.dependencies.Input('noun', 'value')])
def display_images(entity_1, entity_2, entity_3, entity_4, entity_5, entity_6):
  images = []
  list_images = get_images(entity_3, entity_4, entity_5, entity_6)
  len_list = len(list_images)

  start = (entity_1 - 1) * entity_2
  end = entity_1 * entity_2
  list_images = list_images[start:end]

  for i in range(int(entity_2)):
    images.append(
        html.Div([
            html.Div([
                html.Plaintext(i + 1)
            ], style={'width': '3%',
                      'text-align': 'center',
                      'vertical-align': 'middle',
                      'display': 'inline-block',
                      'fontWeight': 'bold',
                      'text-decoration': 'underline',
                      'font-family': 'Arial, Helvetica, sans-serif'}),

            html.Div([
                html.Video(src=list_images[i],
                           autoPlay=True,
                           loop=True)
            ], style={'width': '20%',
                      'text-align': 'center',
                      'vertical-align': 'middle',
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

    )
  return html.Div(images), [{'label': i, 'value': i} for i in range(1, (int((len_list) / int(entity_2))) + 2)]


if __name__ == '__main__':

  app.run_server(debug=True, port=4555)
