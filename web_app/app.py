import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
from dash import Dash


file_path = "../DataCollection/rock_all_tracks.csv"
data = pd.read_csv(file_path)

app = Dash(__name__, external_stylesheets=["/assets/style.css"])

#mean of each key feature
def calculate_mean_values(data):
    return data[["Tempo", "Popularity", "Danceability", "Energy", "Loudness", 
                 "Acousticness", "Speechiness", "Liveness", "Valence"]].mean()

mean_values = calculate_mean_values(data)


app = dash.Dash(__name__)

#layout
app.layout = html.Div([
   
    html.H1("Track Analysis Dashboard", style={'text-align': 'center', 'font-size': '2em'}),

    html.H2(f"Currently Analyzing: rock_all_tracks.csv", style={'text-align': 'center', 'margin-top': '10px'}),

    html.Div([
        html.H2("Summary of Key Features (Mean Values)", style={'text-align': 'center'}),
        dcc.Graph(id='summary-bar-chart', figure=px.bar(
            x=mean_values.index, y=mean_values.values,
            title="Mean Values of Key Features",
            labels={"x": "Features", "y": "Mean Value"}
        ).update_traces(text=mean_values.values.round(2), textposition='outside').update_layout(title_x=0.5))
    ], className="graph-container"),


    html.H3("Top 5 Songs by Popularity"),
    html.Ul(id="top-5-songs-list"),  #list for top 5 songs
])
@app.callback(
    Output("top-5-songs-list", "children"),
    Input("top-5-songs-list", "id")
)
def display_top_5_songs(_):
    
    top_5_songs = data.sort_values(by="Popularity", ascending=False).head(5)
    
    return [
        html.Li(f"{row['Track_Name']} - Popularity: {row['Popularity']}")
        for _, row in top_5_songs.iterrows()
    ]


if __name__ == "__main__":
    app.run_server(debug=True)