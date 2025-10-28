import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

# Load CSV Files
stephen = pd.read_csv('3_stephen_curry_shot_chart_2023.csv')
james   = pd.read_csv('2_james_harden_shot_chart_2023.csv')
lebron  = pd.read_csv('1_lebron_james_shot_chart_1_2023.csv')

# Add Player Names
stephen['Player'] = 'Stephen_Curry'
james['Player']   = 'James_Harden'
lebron['Player']  = 'Lebron_James'

# Combine All Data
df = pd.concat([stephen, james, lebron], axis=0)
df['date'] = pd.to_datetime(df['date']).dt.date

# Shot distance
bins = [-1, 3, 9, 16, 23, 26, 30, 99]
labels = ['0–3 ft', '4–9 ft', '10–16 ft', '17–23 ft', '24–26 ft', '27–30 ft', '31+ ft']
df['distance_bin'] = pd.cut(df['distance_ft'], bins=bins, labels=labels, right=True)

# Attempts per quarter
att_qtr = (
    df.groupby(['Player', 'qtr', 'shot_type'])
      .agg(result=('result', 'sum'))
      .reset_index()
)

# Shot type per distance
shot_dist = (
    df.groupby(['Player', 'distance_bin', 'shot_type'])
      .agg(result=('result', 'sum'))
      .reset_index()
)

# Quarter groups
q_all = ['1st Qtr','2nd Qtr','3rd Qtr','4th Qtr','1st OT','2nd OT']
q_reg = ['1st Qtr','2nd Qtr','3rd Qtr','4th Qtr']
q_ot  = ['1st OT','2nd OT']

def build(data, path, val_key, title_suffix):
    fig_tmp = px.sunburst(
        data,
        path=path,
        values=val_key,
        color='Player',
        color_discrete_map={
            'Lebron_James':  '#334668',
            'Stephen_Curry': '#6D83AA',
            'James_Harden':  '#C8D0DF'
        }
    )
    tr = fig_tmp.data[0]
    tr.name = title_suffix
    return tr

# Attempts
att_all = build(att_qtr[att_qtr['qtr'].isin(q_all)], ['Player','qtr','shot_type'], 'result', 'Attempts — All (Quarter)')
att_reg = build(att_qtr[att_qtr['qtr'].isin(q_reg)], ['Player','qtr','shot_type'], 'result', 'Attempts — Regulation (Quarter)')
att_ot  = build(att_qtr[att_qtr['qtr'].isin(q_ot )], ['Player','qtr','shot_type'], 'result', 'Attempts — OT (Quarter)')

def distance(qtrs):
    sub = df[df['qtr'].isin(qtrs)]
    d = (sub.groupby(['Player','distance_bin','shot_type'])
             .agg(result=('result','sum'))
             .reset_index())
    return d

# Distance
Dist_all = build(distance(q_all), ['Player','distance_bin','shot_type'], 'result', 'Distance — All')
Dist_reg = build(distance(q_reg), ['Player','distance_bin','shot_type'], 'result', 'Distance — Regulation')
Dist_ot  = build(distance(q_ot ), ['Player','distance_bin','shot_type'], 'result', 'Distance — OT')
fig_addon = go.Figure(data=[att_all, att_reg, att_ot, Dist_all, Dist_reg, Dist_ot])

for i, t in enumerate(fig_addon.data):
    t.visible = (i == 0)

def visible(idx):
    v = [False]*6
    v[idx] = True
    return v

# Layout and dropdown
fig_addon.update_layout(
    updatemenus=[dict(
        type="dropdown",
        x=0.92, xanchor="right",
        y=1.05, yanchor="top",
        showactive=True,
        buttons=[
            dict(label="Attempts — All",        method="update",
                 args=[{"visible": visible(0)},
                       {"title": {"text": "Shot Attempts (All)", "x": 0.5, "xanchor": "center"}}]),
            dict(label="Attempts — Regulation", method="update",
                 args=[{"visible": visible(1)},
                       {"title": {"text": "Shot Attempts (Regulation)", "x": 0.5, "xanchor": "center"}}]),
            dict(label="Attempts — OT",         method="update",
                 args=[{"visible": visible(2)},
                       {"title": {"text": "Shot Attempts (OT)", "x": 0.5, "xanchor": "center"}}]),
            dict(label="Distance — All",        method="update",
                 args=[{"visible": visible(3)},
                       {"title": {"text": "Shot Distance (All)", "x": 0.5, "xanchor": "center"}}]),
            dict(label="Distance — Regulation", method="update",
                 args=[{"visible": visible(4)},
                       {"title": {"text": "Shot Distance (Regulation)", "x": 0.5, "xanchor": "center"}}]),
            dict(label="Distance — OT",         method="update",
                 args=[{"visible": visible(5)},
                       {"title": {"text": "Shot Distance (OT)", "x": 0.5, "xanchor": "center"}}]),
        ],
    )],
    height=824,
    margin=dict(b=140, r=20, l=20, t=90),
    plot_bgcolor='#fafafa',
    paper_bgcolor='#fafafa',
    title={"text": "Shot Attempts (All)", "x": 0.5, "xanchor": "center", "y": 0.98, "yanchor": "top"},
    title_font=dict(size=24, color='#8a8d93', family="Lato, sans-serif"),
    font=dict(color='#8a8d93'),
    hoverlabel=dict(
        bgcolor="#f2f2f2",
        font_size=13,
        font_family="Lato, sans-serif",
        align="left",
        namelength=-1,
    ),
    showlegend=False
)

for tr in fig_addon.data:
    tr.update(domain=dict(y=[0.12, 1.0]))

fig_addon.show()
