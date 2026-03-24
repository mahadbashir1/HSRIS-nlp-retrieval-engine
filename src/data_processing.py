import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from src.config import TICKET_CSV

REQUIRED_COLS = [
    'Ticket ID', 'Ticket Description', 'Ticket Subject',
    'Ticket Priority', 'Ticket Type', 'Ticket Channel', 'Resolution',
    'Customer Age', 'Ticket Status'
]

def load_and_clean_data(file_path=TICKET_CSV):
    raw_df = pd.read_csv(file_path)
    # Ensure all required columns exist, drop optional ones if missing
    cols_to_keep = [c for c in REQUIRED_COLS if c in raw_df.columns]
    tickets = raw_df[cols_to_keep].copy()
    
    tickets.dropna(subset=['Ticket Description'], inplace=True)
    tickets.reset_index(drop=True, inplace=True)

    tickets['Ticket Description'] = (
        tickets['Ticket Description']
        .apply(lambda txt: re.sub(r'\{[^}]+\}', '', str(txt)).strip())
    )
    if 'Resolution' in tickets.columns:
        tickets['Resolution'] = tickets['Resolution'].fillna('Resolution not yet recorded.')
    
    return tickets, raw_df

def generate_eda_plot(raw_df, output_path='dataset_overview.json'):
    output_path = output_path.replace('.png', '.json')
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "domain"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "histogram"}, {"type": "domain"}]],
        subplot_titles=("Priority Distribution", "Submission Channel", "Ticket Type", 
                        "Description Length", "Customer Age", "Ticket Status"),
        horizontal_spacing=0.08, vertical_spacing=0.15
    )
    
    pri_counts = raw_df['Ticket Priority'].value_counts()
    fig.add_trace(go.Pie(labels=pri_counts.index, values=pri_counts.values, 
                         marker=dict(colors=['#00ff99', '#bf00ff', '#fcee0a']),
                         textinfo='label+percent', textposition='outside', textfont=dict(color='#ffffff', size=12)), 
                  row=1, col=1)
    
    ch_counts = raw_df['Ticket Channel'].value_counts()
    fig.add_trace(go.Bar(x=ch_counts.values, y=ch_counts.index, orientation='h', marker_color='#bf00ff'), row=1, col=2)
    
    ty_counts = raw_df['Ticket Type'].value_counts()
    fig.add_trace(go.Bar(x=ty_counts.index, y=ty_counts.values, marker_color='#00ff99'), row=1, col=3)
    
    raw_df['word_count'] = raw_df['Ticket Description'].fillna('').apply(lambda x: len(str(x).split()))
    fig.add_trace(go.Histogram(x=raw_df['word_count'], nbinsx=45, marker_color='#00ff99'), row=2, col=1)
    
    fig.add_trace(go.Histogram(x=raw_df['Customer Age'].dropna(), nbinsx=22, marker_color='#bf00ff'), row=2, col=2)
    
    st_counts = raw_df['Ticket Status'].value_counts()
    fig.add_trace(go.Pie(labels=st_counts.index, values=st_counts.values, 
                         marker=dict(colors=['#fcee0a', '#00ff99', '#bf00ff']),
                         textinfo='label+percent', textposition='outside', textfont=dict(color='#ffffff', size=12)), 
                  row=2, col=3)
    
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor="#0a0a0a", 
        plot_bgcolor="#0a0a0a", 
        font=dict(family="Courier New", color="#ffffff"), 
        showlegend=False, 
        title_text="Support Data Analysis & Overview (Full Telemetry)",
        height=850,
        bargap=0.3
    )
    pio.write_json(fig, output_path)
