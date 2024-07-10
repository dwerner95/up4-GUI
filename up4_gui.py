from dash import Dash, html, dcc, Input, Output, callback, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import up4
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Create a root window but keep it hidden
root = tk.Tk()
root.withdraw()

# Open the file dialog and get the full path of the selected file
filename = filedialog.askopenfilename()
print(filename)
root.destroy()


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')

app.layout = html.Div([

    html.Div([

        html.Div([
            dcc.Dropdown(
                ['Number Field', "Velocity Field", "Occupancy Field", "Dispersion Field"],
                "Velocity Field",
                id='filter-field',
            ),
            dcc.RadioItems( 
                ['Cartesian', 'Cylindrical'],
                'Cartesian',
                id='filter-gridtype',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            ),
            dcc.RadioItems( 
                ['X', 'Y', 'Z'],
                'X',
                id='filter-grid_axis',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            ),

            dcc.RadioItems( 
                ['Depth Average', 'Slice'],
                'Depth Average',
                id='filter-plot_type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            ),
            html.Div([
                "Select Slice Position",
                dcc.Slider(
                    0,#data.dimensions()["xmin"],
                    1,#data.dimensions()["xmax"],
                    #step=None,
                    id='filter-slice_position',
                    value=0,#(data.dimensions()["xmin"] + data.dimensions()["xmax"])/2,
                    marks={i: f'{i:.02f}' for i in np.linspace(0,1, 10)}
                ),
            ],id='slice-container', style={'width': '99%', 'display': 'none'}),

            html.Div([
                "Cell Size x,y,z [mm] ", # cell size needs to be dfferent, choosen!
                dcc.Input(id='filter-cellsize', value='5,5,5', type='text'),
            ],id='cell_size_container', style={'display': 'inline-block'}),
            # update button
            html.Button('Update', id='update-button', n_clicks=0),
            html.Div([
                dcc.Checklist(
                    ['Show Particle Trajectories'],
                    ['Show Particle Trajectories'],
                    id='filter-show_trajectories',
                ),
                dcc.Checklist(
                    ['Show Dispersion Distribution'],
                    [],
                    id='filter-dispersion_show_dist',
                ),
                dcc.Input(id='filter-num_traj', value=10, type='text'),
            ], id = 'dispersion_particle_container', style={'display': 'inline-block'}),
            html.Div([
                "Select time range:",
                dcc.RangeSlider(
                    0, #np.min(time),
                    1,#np.max(time),
                    #step=None,
                    id='filter-time',
                    value=[0,1],#[np.min(time), np.max(time)],
                    allowCross=False,
                    marks={i: f'{i:.02f}' for i in np.linspace(0,1, 10)}
                ),
            ], style={'width': '99%', 'display': 'inline-block'}),
            html.Div(
                     [
                "Select time range for Dispersion:",
                dcc.Slider(
                    0.0,
                    1,#np.max(time)-np.min(time),
                    #step=None,
                    id='filter-time_for_dispersion',
                    value=1.0,
                    marks={i: f'{i:.02f}' for i in np.linspace(0.0, 1, 10)}
                ),
            ],id='slider-container', style={'width': '99%', 'display': 'none'}),
            
        ],
        style={'width': '99%', 'display': 'inline-block'}),

    ], style={
        'padding': '10px 5px', 'display': 'flex'
    }),

    html.Div([
        dcc.Graph(
            id='main-graph',
        )
    ], style={'width': '99%'}),

    html.Div([
        html.Button('Upload Data', id='upload-data', n_clicks=0),
    ])

])


# show or hide slider for the Dispersion time
@callback(
    Output(component_id='slider-container', component_property='style'),
    Input('filter-field', 'value'),
    prevent_initial_call=True
)
def show_hide_element(visibility_state):
    if visibility_state == 'Dispersion Field':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

# show or hide cell size input
@callback(
    Output(component_id='cell_size_container', component_property='style'),
    Input('filter-gridtype', 'value'),
    prevent_initial_call=True
)
def show_hide_cellsize(visibility_state):
    if visibility_state == 'Cartesian':
        return {'display': 'block'}
    else:
        return {'display': 'none'}
    
# show or hide cell size input
@callback(
    Output(component_id='slice-container', component_property='style'),
    Input('filter-plot_type', 'value'),
    prevent_initial_call=True
)
def show_hide_slice_container(visibility_state):
    if visibility_state == 'Slice':
        return {'width': '99%', 'display': 'inline-block'}
    else:
        return {'width': '99%', 'display': 'none'}

# show or hide cell size input
@callback(
    Output(component_id='dispersion_particle_container', component_property='style'),
    Input('filter-field', 'value'),
    prevent_initial_call=True
)
def show_hide_dispersion_particle_container(visibility_state):
    if visibility_state == 'Dispersion Field':
        return {'display': 'inline-block'}
    else:
        return {'display': 'none'}


@callback( 
    Output('filter-slice_position', 'min'),
    Input('filter-grid_axis', 'value'),
    prevent_initial_call=True
)
def update_slider_min(axis):
    return np.min(data.dimensions()[f"{axis.lower()}min"])

@callback(
    Output('filter-slice_position', 'max'),
    Input('filter-grid_axis', 'value'),
    prevent_initial_call=True
)
def update_slider_max(axis):
    return np.max(data.dimensions()[f"{axis.lower()}max"])

###### Hover 
@callback(
    Output('main-graph', 'figure',allow_duplicate=True),
    State('filter-field', 'value'),
    State('filter-num_traj', 'value'),
    Input('main-graph', 'clickData'),
    State('main-graph', 'figure'),
    State('filter-slice_position', 'value'),
    State('filter-time_for_dispersion', 'value'),
    State('filter-grid_axis', 'value'),
    State('filter-show_trajectories', 'value'),
    State('filter-dispersion_show_dist', 'value'),
    prevent_initial_call=True
)
def display_hover_data(filed_type,num_particles,hoverData, fig,slice_position, tfd,axis, show_traj, show_dist):
    if axis == 'X':
        x=slice_position
        y=hoverData["points"][0]['x'] 
        z=hoverData["points"][0]['y']
    elif axis == 'Y':
        x=hoverData["points"][0]['x']
        y=slice_position
        z=hoverData["points"][0]['y']
    else:
        x=hoverData["points"][0]['x']
        y=hoverData["points"][0]['y']
        z=slice_position
    print("in hover data")

    print(filed_type, num_particles, hoverData)
    if filed_type == 'Dispersion Field':
        fig = go.Figure(fig)
        if not (show_dist or show_traj):
            return fig
        # find particles in the cell
        cell_id = grid.cell_id(x,y,z)

        count = 0
        traj_x = [np.nan]
        traj_y = [np.nan]
        traj_z = [np.nan]
        print("Calling rust")
        trajectories = data.extract_dispersion_trajectories(
            grid,tfd,int(num_particles),cell_id)
        print("done")
        end_points = np.asarray([traj[-1] for traj in trajectories if len(traj) > 0])
        trajectories = np.asarray([np.asarray(traj) for traj in trajectories], dtype=object)
        if show_traj:
            
            for id,traj in enumerate(trajectories):
                if len(traj) == 0:
                    continue
                print(traj)
                traj_x.extend(traj[:,0])
                traj_y.extend(traj[:,1])
                traj_z.extend(traj[:,2])
                traj_x.append(np.nan)
                traj_y.append(np.nan)
                traj_z.append(np.nan)

            
            if axis == 'X':
                fig.add_trace(go.Scatter(x=traj_y, y=traj_z, mode='lines', name='Trajectory'))
            elif axis == 'Y':
                fig.add_trace(go.Scatter(x=traj_x, y=traj_z, mode='lines', name='Trajectory'))
            else:
                fig.add_trace(go.Scatter(x=traj_x, y=traj_y, mode='lines', name='Trajectory'))
        if show_dist:
            # get all the end points of the trajectories
            
            # get the mean and standard deviation of the end points
            mean = np.mean(end_points, axis=0)
            std = np.std(end_points, axis=0)
            # plot mean point with a circle indicating the standard deviation
            # use this:
            kwargs = {'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': 'black', 'opacity': 0.3, 'line': {'color': 'black'}}
            # points = [go.layout.Shape(x0=x-r, y0=y-r, x1=x+r, y1=y+r, **kwargs) for x, y in xy]
            # fig.update_layout(shapes=points)
            if axis == 'X':
                points = [go.layout.Shape(x0=mean[1]-std[1], y0=mean[2]-std[2], x1=mean[1]+std[1], y1=mean[2]+std[2], **kwargs)]
            elif axis == 'Y':
                points = [go.layout.Shape(x0=mean[0]-std[0], y0=mean[2]-std[2], x1=mean[0]+std[0], y1=mean[2]+std[2], **kwargs)]
            else:
                points = [go.layout.Shape(x0=mean[0]-std[0], y0=mean[1]-std[1], x1=mean[0]+std[0], y1=mean[1]+std[1], **kwargs)]
            if not show_traj:
                # show a single line from start to end point
                x_start = hoverData["points"][0]['x']
                y_start = hoverData["points"][0]['y']
                if axis == 'X':
                    x_end = mean[1]
                    y_end = mean[2]
                elif axis == 'Y':
                    x_end = mean[0]
                    y_end = mean[2]
                else:
                    x_end = mean[0]
                    y_end = mean[1]
                fig.add_trace(go.Scatter(x=[x_start, x_end], y=[y_start, y_end], mode='lines', name='Dispersion'))
            
            fig.add_shape(points[0])
        print('done', traj_x, traj_y, traj_z)


            
    return fig

@callback(
    Output('filter-slice_position', 'value', allow_duplicate=True),
    Output('filter-slice_position', 'min', allow_duplicate=True),
    Output('filter-slice_position', 'max', allow_duplicate=True),
    Output('filter-slice_position', 'marks', allow_duplicate=True),

    Output('filter-time', 'value', allow_duplicate=True),
    Output('filter-time', 'min', allow_duplicate=True),
    Output('filter-time', 'max', allow_duplicate=True),
    Output('filter-time', 'marks', allow_duplicate=True),

    Output('filter-time_for_dispersion', 'value', allow_duplicate=True),
    Output('filter-time_for_dispersion', 'min', allow_duplicate=True),
    Output('filter-time_for_dispersion', 'max', allow_duplicate=True),
    Output('filter-time_for_dispersion', 'marks', allow_duplicate=True),

    Input('upload-data', 'n_clicks'),
    prevent_initial_call=True
)
def update_filename(n_clicks):
    
    print(filename)
    global data ,time, dividor
    data = up4.Data(filename)
    time = data.time()
    dividor = 1000 if data.dimensions()["xmax"] -  data.dimensions()["xmin"] < 10 else 1 # crude unit check
    filter_slice_position_value = (data.dimensions()["xmin"] + data.dimensions()["xmax"])/2
    filter_slice_position_min = data.dimensions()["xmin"]
    filter_slice_position_max = data.dimensions()["xmax"]
    filter_slice_position_marks = {i: f'{i:.02f}' for i in np.linspace(data.dimensions()["xmin"], data.dimensions()["xmax"], 10)}
    
    filter_time_value = [np.min(time), np.max(time)]
    filter_time_min = np.min(time)
    filter_time_max = np.max(time)
    filter_time_marks = {i: f'{i:.02f}' for i in np.linspace(np.min(time), np.max(time), 10)}
    
    filter_time_for_dispersion = (np.max(time)-np.min(time))/2
    filter_time_for_dispersion_min = 0.0
    filter_time_for_dispersion_max = 10_000 if (np.max(time)-np.min(time)) > 10_000 else 10
    filter_time_for_dispersion_marks = {i: f'{i:.02f}' for i in np.linspace(0.0, filter_time_for_dispersion_max , 10)}
    
    return (filter_slice_position_value,
            filter_slice_position_min, 
            filter_slice_position_max, 
            filter_slice_position_marks,
            filter_time_value,
            filter_time_min,
            filter_time_max,
            filter_time_marks,
            filter_time_for_dispersion,
            filter_time_for_dispersion_min,
            filter_time_for_dispersion_max,
            filter_time_for_dispersion_marks
            )
     

@callback(
    Output('main-graph', 'figure'),
    State('filter-field', 'value'),
    State('filter-gridtype', 'value'),
    State('filter-time', 'value'),
    State('filter-cellsize', 'value'),
    State('upload-data', 'filename'),
    State('filter-grid_axis', 'value'),
    State('filter-plot_type', 'value'),
    Input('filter-slice_position', 'value'),
    Input('filter-time_for_dispersion', 'value'),
    Input('update-button', 'n_clicks'),
    prevent_initial_call=True
    )
def update_graph(field, gridtype, time, size,filename,axis, plot_type, slice_pos, tfd, n_clicks):
    global grid
    data.set_time(time[0]-0.001, time[1]+0.001)
    print(size, plot_type, axis, slice_pos, tfd)
    if gridtype == 'Cartesian':
        cell_size = [int(x) for x in size.split(",")]
        assert len(cell_size) == 3
        grid = up4.Grid(data, cell_size=[x/dividor for x in cell_size], grid_style='cartesian')
    else:
        grid = up4.Grid(data, grid_style='cylindrical')
    print(1)
    if field == 'Number Field':
        field = data.numberfield(grid)
        colorbar_title="Number of Particles"
    elif field == 'Velocity Field':
        field = data.velocityfield(grid)
        colorbar_title="Velocity (m s<sup>-1</sup>)"
    elif field == 'Occupancy Field':
        field = data.occupancyfield(grid)
        colorbar_title="Occupancy"
    elif field == 'Dispersion Field':
        field, me = data.dispersion(grid, tfd)
        colorbar_title="Dispersion (m<sup>2</sup> )"
    print(2)
    plotter = up4.Plotter2D(field) # create a Plotter2D instance
    axis = 0 if axis == 'X' else 1 if axis == 'Y' else 2
    print(3)
    plot_layout = dict(
        width=800, height=800,
        xaxis_title="y position (m)",
        yaxis_title="z position (m)",
    ) # set layout parameters for plot
    plot_style = dict(
        colorbar_title=colorbar_title
    ) # style the trace(s) in the plot
    print(4)
    print(plot_type, axis, slice_pos,)
    if plot_type == 'Slice':
        pos = [0,0,0]
        pos[axis] = slice_pos
        print(slice_pos, pos)
        fig = plotter.scalar_map(
            axis = axis,
            selection = "plane",
            index = grid.cell_id(pos[0],pos[1], pos[2])[axis],
            layout = plot_layout,
            style = plot_style
        )
    else:
        fig = plotter.scalar_map(
            axis = axis,
            selection = "depth_average",
            layout = plot_layout,
            style = plot_style
        )

    if gridtype != 'Cylindrical':
        fig.update_layout(
            # make equal aspect ratio
            yaxis_scaleanchor = "x",
            yaxis_scaleratio = 1,
        )
    fig.update_layout(

        template = "plotly_white",
    )
    print(5)

    return fig



if __name__ == '__main__':
    app.run(debug=False)
