import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

# Load CSV Files
stephen = pd.read_csv('3_stephen_curry_shot_chart_2023.csv', low_memory=False)
james   = pd.read_csv('2_james_harden_shot_chart_2023.csv',   low_memory=False)
lebron  = pd.read_csv('1_lebron_james_shot_chart_1_2023.csv', low_memory=False)

# Add Player Names (keep your labels)
stephen['Player'] = 'Stephen_Curry'
james['Player']   = 'James_Harden'
lebron['Player']  = 'Lebron_James'

# Combine All Data
df = pd.concat([stephen, james, lebron], axis=0, ignore_index=True)
if 'date' in df.columns:
    try:
        df['date'] = pd.to_datetime(df['date']).dt.date
    except Exception:
        pass

# Shot distance
if 'distance_ft' in df.columns:
    bins = [-1, 3, 9, 16, 23, 26, 30, 99]
    labels = ['0–3 ft', '4–9 ft', '10–16 ft', '17–23 ft', '24–26 ft', '27–30 ft', '31+ ft']
    df['distance_bin'] = pd.cut(df['distance_ft'], bins=bins, labels=labels, right=True)
else:
    df['distance_bin'] = np.nan
# After creating df['distance_bin'] …
df['shot_type'] = df['shot_type'].astype(str)

# Keep only rows that have a distance bin for distance-based views
# and avoid categorical gotchas by using plain strings
df = df.copy()
df['distance_bin'] = df['distance_bin'].astype(str)  # cast away categorical
df.loc[df['distance_bin'].isin(['nan', 'NaN']), 'distance_bin'] = np.nan  # clean accidental "nan" strings

# -------------------------------
# Made flag & simple FG% groups
# -------------------------------
# Map common encodings to 1/0 in a single line (simple approach)
df['Made'] = df['result'].astype(str).str.lower().isin(
    ['1','true','made','make','shot made','made shot','hit']
).astype(int)

def fg_agg(group_cols, data):
    # drop rows missing any grouping key
    data2 = data.dropna(subset=[c for c in group_cols if c in data.columns]).copy()
    g = (
        data2.groupby(group_cols, as_index=False, observed=True)   # observed=True fixes categorical combos
            .agg(Attempts=('Made','count'), Made=('Made','sum'))
    )
    g['FG%'] = (g['Made'] / g['Attempts'] * 100).round(1)
    return g


# Quarter filters (unchanged)
q_all = ['1st Qtr','2nd Qtr','3rd Qtr','4th Qtr','1st OT','2nd OT']
q_reg = ['1st Qtr','2nd Qtr','3rd Qtr','4th Qtr']
q_ot  = ['1st OT','2nd OT']

# Quarter views (ok to group directly)
fg_qtr_all = fg_agg(['Player','qtr','shot_type'], df[df['qtr'].isin(q_all)])
fg_qtr_reg = fg_agg(['Player','qtr','shot_type'], df[df['qtr'].isin(q_reg)])
fg_qtr_ot  = fg_agg(['Player','qtr','shot_type'], df[df['qtr'].isin(q_ot)])

# Distance views — filter first, then group
df_dist_all = df[df['distance_bin'].notna()]
df_dist_reg = df[(df['distance_bin'].notna()) & (df['qtr'].isin(q_reg))]
df_dist_ot  = df[(df['distance_bin'].notna()) & (df['qtr'].isin(q_ot))]

fg_dist_all = fg_agg(['Player','distance_bin','shot_type'], df_dist_all)
fg_dist_reg = fg_agg(['Player','distance_bin','shot_type'], df_dist_reg)
fg_dist_ot  = fg_agg(['Player','distance_bin','shot_type'], df_dist_ot)


# -------------------------------
# Sunburst builder (simple)
# - values = Attempts (size)
# - color  = FG%       (hot/cold)
# -------------------------------
def build(data, path, title_suffix):
    if data.empty:
        # keep layout stable if a view is empty
        data = pd.DataFrame({'Player':[], 'Attempts':[], 'FG%':[]})
        path = ['Player']
    fig_tmp = px.sunburst(
        data,
        path=path,
        values='Attempts',
        color='FG%',
        color_continuous_scale='RdYlGn',
        range_color=[0,100],
        branchvalues='total',
        maxdepth=-1
    )
    
    tr = fig_tmp.data[0]
    tr.name = title_suffix

    tr.update(
        marker=dict(
            colorscale='RdYlGn',
            cmin=0, cmax=100,
            showscale=False  # set True on one trace if you want a colorbar
        )
    )

    tr.hovertemplate = (
        "<b>%{label}</b><br>" +
        "Attempts: %{value}<br>" +
        "FG%: %{color:.1f}%<br>" 
    )
    tr.textinfo = "label"
    return tr
    
# Build traces (use your same dropdown structure)
att_all = build(fg_qtr_all, ['Player','qtr','shot_type'], 'FG% — All (Quarter)')
att_reg = build(fg_qtr_reg, ['Player','qtr','shot_type'], 'FG% — Regulation (Quarter)')
att_ot  = build(fg_qtr_ot,  ['Player','qtr','shot_type'], 'FG% — OT (Quarter)')

Dist_all = build(fg_dist_all, ['Player','distance_bin','shot_type'], 'FG% — Distance (All)')
Dist_reg = build(fg_dist_reg, ['Player','distance_bin','shot_type'], 'FG% — Distance (Reg)')
Dist_ot  = build(fg_dist_ot,  ['Player','distance_bin','shot_type'], 'FG% — Distance (OT)')

fig_addon = go.Figure(data=[att_all, att_reg, att_ot, Dist_all, Dist_reg, Dist_ot])
fig_addon.update_layout(coloraxis_colorscale='RdYlGn')  # 0% red → 100% green


for i, t in enumerate(fig_addon.data):
    t.visible = (i == 0)

def visible(idx):
    v = [False]*6
    v[idx] = True
    return v

# Layout and dropdown (text updated to reflect FG%)
fig_addon.update_layout(
    updatemenus=[dict(
        type="dropdown",
        x=0.92, xanchor="right",
        y=1.05, yanchor="top",
        showactive=True,
        buttons=[
            dict(label="FG% — All (Quarter)",        method="update",
                 args=[{"visible": visible(0)},
                       {"title": {"text": "FG% (color) & Attempts (size) — All (Quarter)", "x": 0.5, "xanchor": "center"}}]),
            dict(label="FG% — Regulation (Quarter)", method="update",
                 args=[{"visible": visible(1)},
                       {"title": {"text": "FG% (color) & Attempts (size) — Regulation (Quarter)", "x": 0.5, "xanchor": "center"}}]),
            dict(label="FG% — OT (Quarter)",         method="update",
                 args=[{"visible": visible(2)},
                       {"title": {"text": "FG% (color) & Attempts (size) — OT (Quarter)", "x": 0.5, "xanchor": "center"}}]),
            dict(label="FG% — Distance (All)",       method="update",
                 args=[{"visible": visible(3)},
                       {"title": {"text": "FG% (color) & Attempts (size) — Distance (All)", "x": 0.5, "xanchor": "center"}}]),
            dict(label="FG% — Distance (Regulation)", method="update",
                 args=[{"visible": visible(4)},
                       {"title": {"text": "FG% (color) & Attempts (size) — Distance (Regulation)", "x": 0.5, "xanchor": "center"}}]),
            dict(label="FG% — Distance (OT)",         method="update",
                 args=[{"visible": visible(5)},
                       {"title": {"text": "FG% (color) & Attempts (size) — Distance (OT)", "x": 0.5, "xanchor": "center"}}]),
        ],
    )],
    height=824,
    margin=dict(b=140, r=20, l=20, t=90),
    plot_bgcolor='#fafafa',
    paper_bgcolor='#fafafa',
    coloraxis_colorbar=dict(title="FG%"),
    title={"text": "FG% (color) & Attempts (size) — All (Quarter)", "x": 0.5, "xanchor": "center", "y": 0.98, "yanchor": "top"},
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
