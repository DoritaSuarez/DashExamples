import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input,Output

# Use the 'Cars' dataset

cars = pd.read_csv('https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv')

# Build simple Dash Layout

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Button(id='ignore',style={'display':'none'}), #Create a hidden button just so the first callback will have an input.  It doesn't so anything.
    dcc.Graph(id='carGraph'), #Graph that displays all data
    html.Div(id='display'),  #To show format of selectData
    dcc.Graph(id='filterGraph') #Graph that shows only filtered data    
])


# Create graph with data
@app.callback(Output('carGraph','figure'),[Input('ignore','n_clicks')])            
def testfunc(clicks):
    trace1 = go.Scattergl(x=cars['disp'],y=cars['hp'],mode='markers',text=cars['model'])
    layout=go.Layout(title='All Data')
    return {'data':[trace1],'layout':layout}



# Show result of selecting data with either box select or lasso
    
@app.callback(Output('display','children'),[Input('carGraph','selectedData')])
def selectData(selectData):
    return str('Selecting points produces a nested dictionary: {}'.format(selectData))


#Extract the 'text' component and use it to filter the dataframe and then create another graph
    
@app.callback(Output('filterGraph','figure'),[Input('carGraph','selectedData')])
def selectData3(selectData):
    filtList = []
    for i in range(len(selectData['points'])):
        filtList.append(selectData['points'][i]['text'])

    
    filtCars = cars[cars['model'].isin(filtList)]
    
    trace2 = go.Scattergl(x=filtCars['disp'],y=filtCars['hp'],mode='markers',text=filtCars['model'])
    layout2 = go.Layout(title='Filtered Data')
    
    return {'data':[trace2],'layout':layout2}  
    
      

if __name__ == '__main__':
    app.run_server(debug=True)