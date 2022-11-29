#!/usr/bin/env python
# coding: utf-8

# In[170]:


# Carga de librerias basicas
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

#Carga de libreria para entrenar y probar
from sklearn.model_selection import train_test_split

#Carga de libreria para Evaluacion
from sklearn.metrics import r2_score, mean_squared_error

#Carga de libreria para Lineal
from sklearn.linear_model import LinearRegression

#Carga de libreria para KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors

#Carga de libreria para Random Forest
from sklearn.ensemble import RandomForestRegressor

#Carga de librerias de Dash para correr Dashboard
import dash_bootstrap_components as dbc
from dash import Dash, dash_table, dcc, html, Input, Output, State
from jupyter_dash import JupyterDash


# In[171]:


# Carga de bases de datos de Whirlpool
df = pd.read_excel(r'WORKFILE_Supsa_Energy_Audit_Information_Actualizada.xlsx')
#Cambio de nombre para evitar espacios
df = df.set_axis(['ID','Production_Line','Platform','Familia','Test_Date','Refrigerant','Model_Number','Serial_Number','Sensores','Posicion','Target','Energy_Consumed(kWh/yr)','Porc_Below_Rating_Point','RC_Temp_Average_P1','RC1_Temp_P1','RC2_Temp_P1','RC3_Temp_P1','FC_Temp_Average_P1','FC1_Temp_P1','FC2_Temp_P1','FC3_Temp_P1','Energy_Usage(kWh/day)_P1','Porc_Run_Time_P1','Avg_Ambient_Temp_P1','Temp_Setting_P2','RC_Temp_Average_P2','RC1_Temp_P2','RC2_Temp_P2','RC3_Temp_P2','FC_Temp_Average_P2','FC1_Temp_P2','FC2_Temp_P2','FC3_Temp_P2','Energy_Usage(kWh/day)_P2','Porc_Run_Time_P2','Avg_Ambient_Temp_P2','Ability','Compressor','Supplier','E-star/Std.'], axis=1)
# Cambio de % a valor sin el % para evitar temas en análisis
df['Porc_Below_Rating_Point'] = df['Porc_Below_Rating_Point'].replace("%","")
df['Porc_Below_Rating_Point'] = pd.to_numeric(df['Porc_Below_Rating_Point'])*100
del df['Temp_Setting_P2']
#df['Test_Date'] = df['Test_Date'].dt.strftime("%d/%m/%Y") no usar... dejar para fut ref
df.head()


# In[172]:


app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Tablero Tendencias"
server = app.server

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "23rem",
    "padding": "1rem 1rem",
    "background-color": "#1a1919",
}

CONTENT_STYLE = {
    "margin-left": "23rem",
#    "margin-right": "",
     "width": "90rem"
}

sidebar = html.Div(
    [
        html.H2("Tablero Tendencias", style={'color': '#f4b610', "font-weight": "bold", "font-size": "36px"}),
        html.Hr(style={'color': 'white'}),
        html.H4("Fecha Inicial -> Fecha Final", style={"color": "white", "font-weight": "bold"}),
        dcc.DatePickerRange(id='dates-picker', display_format='DD/MM/YYYY', 
                min_date_allowed=df['Test_Date'].min(), max_date_allowed=df['Test_Date'].max(),
                initial_visible_month=df['Test_Date'].min(), 
                #final_visible_month=df['Test_Date'].max(),
                start_date=df['Test_Date'].min(), end_date=df['Test_Date'].max(), 
                with_portal=True, number_of_months_shown=4,
                style={"background": "white", "color": "black"}),
        html.Br(),
        html.Br(),
#         html.Button('Últimos 15 días', id='ult_15d_button'),
#         html.Button('Últimos 30 días', id='ult_30d_button'),
#         html.Button('Últimos 6 meses', id='ult_6m_button'),
#         html.Button('Todo 2022', id='ult_2022_button'),
#         html.Br(),
#         html.Br(),
#         html.H4("Posición",  style={"color": "white", "font-weight": "bold"}),
#         dcc.Checklist(options=[1,2], value=[1,2], style={"color": "white","font-size": "20px"}, inline=True, id="pos_checklist", labelStyle={'background':'#1a1919','padding':'0rem 1rem','border-radius':'0.3rem'}),
#         html.Br(),
        html.H4("Test Size", style={"color": "white", "font-weight": "bold"}),
        dcc.Dropdown([0.2,0.25,0.3,0.35,0.4],0.2, id="test_size_dropdown"),
        html.Br(),
        html.H4("Familia", style={"color": "white", "font-weight": "bold"}),
        dcc.Dropdown(np.append('Todas',df['Familia'].unique()),'Todas', id="fam_dropdown"),
        html.Br(),
        html.H4("Refrigerante", style={"color": "white", "font-weight": "bold"}),
        dcc.Checklist(options=df['Refrigerant'].unique(), value=df['Refrigerant'].unique(), style={"color": "white","font-size": "20px"}, inline=True, id="ref_checklist", labelStyle={'background':'#1a1919','padding':'0rem 1rem','border-radius':'0.3rem'}),
        html.Br(),
        html.H4("Lineas de Producción", style={"color": "white", "font-weight": "bold"}),
        dcc.Checklist(options=df['Production_Line'].unique(), value=df['Production_Line'].unique(), style={"color": "white","font-size": "20px"}, inline=True, id="lineaprod_checklist", labelStyle={'background':'#1a1919','padding':'0rem 1rem','border-radius':'0.3rem'}),
        html.Br(),
        html.H4("Plataforma", style={"color": "white", "font-weight": "bold"}),
        dcc.Dropdown(np.append('Todas',df['Platform'].unique()), 'Todas', id="plat_dropdown"),
        html.Br(),
        html.H4("Proveedor", style={"color": "white", "font-weight": "bold"}),
        dcc.Dropdown(np.append('Todas',df['Supplier'].unique()), 'Todas', id="prov_dropdown"),
        #html.Br(),
        #html.Button('Resetear Filtros', id='reset_filters_button', style={"background": "#8a0404", "color": "white","font-size": "16px","width":"21rem"}),
    ],
    style=SIDEBAR_STYLE,id='sidebar',
)
content = html.Div([
    
    html.Div(dbc.Row([
    dbc.Col([dbc.Row(html.H4("Selecciona variable 'x'", style={"color": "#1a1919", "font-weight": "bold", "textAlign": "center", "font-size": "22px"})), 
             dbc.Row(dcc.Dropdown(df.iloc[: , 11:].columns, value=df.iloc[: , 11:].columns[0],id="x_dropdown", style={"color": "#1a1919", "textAlign": "center", "font-size": "16px"})),
            dbc.Row(html.H4(''))],
            width=5),
    dbc.Col([dbc.Row(html.H4("Selecciona variable 'y'", style={"color": "#1a1919", "font-weight": "bold", "textAlign": "center", "font-size": "22px"})), 
             dbc.Row(dcc.Dropdown(df.iloc[: , 11:].columns, value=df.iloc[: , 11:].columns[6], id="y_dropdown", style={"color": "#1a1919", "textAlign": "center", "font-size": "16px"})),
            dbc.Row(html.H4(''))],
            width=5),
    ], justify="center"), 
    style={"background-color": "#f4b610","margin-left": "23rem", 'verticalAlign': 'top'}
    ),
    
    html.Div([
        dbc.Row(html.Div(id='tendencies_graph')), 
        dbc.Row([
            dbc.Col(html.Div(id='tendencies_hist1'), width=3),
            dbc.Col(html.Div(id='tendencies_hist2'), width=3),
            dbc.Col(html.Div(id='tendencies_boxpt1'), width=3),
            dbc.Col(html.Div(id='tendencies_boxpt2'), width=3)
                ])
        ],
        style={"margin-left": "23rem"}
    )
])

app.layout = html.Div(
    [
        sidebar,
        content
    ]
)


# In[173]:


df_sorted = df.sort_values(by='Test_Date')

@app.callback(
    Output("tendencies_graph", "children"),
    [Input("fam_dropdown", component_property="value"),
    Input("ref_checklist", component_property="value"), Input("lineaprod_checklist", component_property="value"),
    Input("plat_dropdown", component_property="value"), Input("prov_dropdown", component_property="value"),
    Input('dates-picker', 'start_date'), Input('dates-picker', 'end_date'), Input('x_dropdown', 'value'), 
    Input('y_dropdown', 'value'), Input("test_size_dropdown", component_property="value")
    ]
)

def update_graph_tendencies(familia,refrigerante,linea_prod,plataforma,proveedor,f_inicio,f_final,x_var,y_var,test_size):
    df_sorted = df.sort_values(by='Test_Date')
#     #Filter para posicion
#     if isinstance(posicion, list):
#         if len(posicion)==1 and posicion[0]== 1:
#             rc_y = ['RC1_Temp_P1','RC2_Temp_P1','RC3_Temp_P1']
#         elif len(posicion)==1 and posicion[0]== 2:
#             rc_y = ['RC1_Temp_P2','RC2_Temp_P2','RC3_Temp_P2']
#         else:
#             rc_y = ['RC1_Temp_P1','RC2_Temp_P1','RC3_Temp_P1','RC1_Temp_P2','RC2_Temp_P2','RC3_Temp_P2']
    #Filter para familias
    if familia == "Todas":
        temp=df_sorted
    else:
        temp = df_sorted[df_sorted['Familia'] == familia]
    
    #Filter para refrigerantes
    if len(refrigerante)==1:
        temp = temp[temp['Refrigerant'] == refrigerante[0]]
    else:
        pass
    
    #Filter para lineas de producc
    if isinstance(linea_prod, list):
        if len(linea_prod) == 1:
            temp = temp[temp['Production_Line'] == linea_prod[0]]
        elif len(linea_prod) == 2:
            temp = temp[(temp['Production_Line'] == linea_prod[0])
                              | (temp['Production_Line'] == linea_prod[1])]
    else:
        pass
    
    #Filtro para plataforma
    if plataforma == "Todas":
        temp=temp
    else:
        temp = temp[temp['Platform'] == plataforma]
    
    #Filtro para proveedor
    if proveedor == "Todas":
        temp=temp
    else:
        temp = temp[temp['Supplier'] == proveedor]
    
    #Filtro fecha inicial y final
    temp = temp[(temp['Test_Date'] >= f_inicio) & (temp['Test_Date'] <= f_final)]
    
    x_train, x_test, y_train, y_test = train_test_split(temp[x_var].values.reshape(-1,1),temp[y_var].values.reshape(-1,1),test_size=test_size,random_state=42)
    regress_lin = LinearRegression()
    regress_lin.fit(x_train, y_train.squeeze())
    y_pred_lin = regress_lin.predict(x_test)
    
    regress_knn = neighbors.KNeighborsRegressor(n_neighbors=20)
    regress_knn.fit(x_train,y_train.squeeze())
    y_pred_knn = regress_knn.predict(x_test)

    regress_rf = RandomForestRegressor(n_estimators = 1000, max_depth = 5, random_state = 0)
    regress_rf.fit(x_train, y_train.squeeze())
    y_pred_rf = regress_rf.predict(x_test)
    
    r2_lin = r2_score(y_test,y_pred_lin)
    rmse_lin = mean_squared_error(y_test,y_pred_lin,squared=False) #square=False para que sea RMSE
    r2_knn = r2_score(y_test,y_pred_knn)
    rmse_knn = mean_squared_error(y_test,y_pred_knn,squared=False)
    r2_rf = r2_score(y_test,y_pred_rf)
    rmse_rf = mean_squared_error(y_test,y_pred_rf,squared=False)
    
    #Inicio de creacion de gráficas y datos duros
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temp[x_var], y=temp[y_var], mode="markers", name="Datos", opacity=0.3,
            marker=dict(line=dict(color='black', width=1)))
                 )
    x_test, y_pred_lin, y_pred_knn, y_pred_rf = zip(*sorted(zip(x_test.squeeze(), y_pred_lin, y_pred_knn, y_pred_rf))) #ordenar variables para que se grafiquen en orden
    fig.add_trace(go.Scatter(x=x_test, y=y_pred_lin, name="Lineal", line=dict(color='#eb2400')))
    fig.add_trace(go.Scatter(x=x_test, y=y_pred_knn, name="KNN", line=dict(color='#000000')))
    fig.add_trace(go.Scatter(x=x_test, y=y_pred_rf, name="RFF", line=dict(color='#007504')))
    fig.update_layout(
    title=f'{x_var} vs. {y_var}',#' \n Lineal - R2: {r2_lin}, RMSE: {rmse_lin} \n KNN - R2: {r2_knn}, RMSE: {rmse_knn} \n Random Forest - R2: {r2_rf}, RMSE: {rmse_rf}',
    xaxis_title=x_var,
    yaxis_title=y_var,
    legend=dict(
        title="Leyenda",
        orientation="h",
        yanchor="top",
        y=1.15,
        xanchor="right",
        x=1
))
    return [dcc.Graph(id='linegraph',figure=fig)]


# In[174]:


@app.callback(
    Output("tendencies_hist1", "children"),
    [Input("fam_dropdown", component_property="value"),
    Input("ref_checklist", component_property="value"), Input("lineaprod_checklist", component_property="value"),
    Input("plat_dropdown", component_property="value"), Input("prov_dropdown", component_property="value"),
    Input('dates-picker', 'start_date'), Input('dates-picker', 'end_date'), Input('x_dropdown', 'value')
    ]
)

def update_hist1_tendencies(familia,refrigerante,linea_prod,plataforma,proveedor,f_inicio,f_final,x_var):
    df_sorted = df.sort_values(by='Test_Date')
#     #Filter para posicion
#     if isinstance(posicion, list):
#         if len(posicion)==1 and posicion[0]== 1:
#             rc_y = ['RC1_Temp_P1','RC2_Temp_P1','RC3_Temp_P1']
#         elif len(posicion)==1 and posicion[0]== 2:
#             rc_y = ['RC1_Temp_P2','RC2_Temp_P2','RC3_Temp_P2']
#         else:
#             rc_y = ['RC1_Temp_P1','RC2_Temp_P1','RC3_Temp_P1','RC1_Temp_P2','RC2_Temp_P2','RC3_Temp_P2']
    #Filter para familias
    if familia == "Todas":
        temp=df_sorted
    else:
        temp = df_sorted[df_sorted['Familia'] == familia]
    
    #Filter para refrigerantes
    if len(refrigerante)==1:
        temp = temp[temp['Refrigerant'] == refrigerante[0]]
    else:
        pass
    
    #Filter para lineas de producc
    if isinstance(linea_prod, list):
        if len(linea_prod) == 1:
            temp = temp[temp['Production_Line'] == linea_prod[0]]
        elif len(linea_prod) == 2:
            temp = temp[(temp['Production_Line'] == linea_prod[0])
                              | (temp['Production_Line'] == linea_prod[1])]
    else:
        pass
    
    #Filtro para plataforma
    if plataforma == "Todas":
        temp=temp
    else:
        temp = temp[temp['Platform'] == plataforma]
    
    #Filtro para proveedor
    if proveedor == "Todas":
        temp=temp
    else:
        temp = temp[temp['Supplier'] == proveedor]
    
    #Filtro fecha inicial y final
    temp = temp[(temp['Test_Date'] >= f_inicio) & (temp['Test_Date'] <= f_final)]
    
    #Inicio de creacion de gráficas y datos duros
    fig = px.histogram(temp, x=x_var)
    fig.update_layout(
    title=f'Histograma de {x_var}',
    xaxis_title=x_var,
)
    return [dcc.Graph(id='histogram1',figure=fig)]


# In[175]:


@app.callback(
    Output("tendencies_hist2", "children"),
    [Input("fam_dropdown", component_property="value"),
    Input("ref_checklist", component_property="value"), Input("lineaprod_checklist", component_property="value"),
    Input("plat_dropdown", component_property="value"), Input("prov_dropdown", component_property="value"),
    Input('dates-picker', 'start_date'), Input('dates-picker', 'end_date'), Input('y_dropdown', 'value')
    ]
)

def update_hist2_tendencies(familia,refrigerante,linea_prod,plataforma,proveedor,f_inicio,f_final,y_var):
    df_sorted = df.sort_values(by='Test_Date')
#     #Filter para posicion
#     if isinstance(posicion, list):
#         if len(posicion)==1 and posicion[0]== 1:
#             rc_y = ['RC1_Temp_P1','RC2_Temp_P1','RC3_Temp_P1']
#         elif len(posicion)==1 and posicion[0]== 2:
#             rc_y = ['RC1_Temp_P2','RC2_Temp_P2','RC3_Temp_P2']
#         else:
#             rc_y = ['RC1_Temp_P1','RC2_Temp_P1','RC3_Temp_P1','RC1_Temp_P2','RC2_Temp_P2','RC3_Temp_P2']
    #Filter para familias
    if familia == "Todas":
        temp=df_sorted
    else:
        temp = df_sorted[df_sorted['Familia'] == familia]
    
    #Filter para refrigerantes
    if len(refrigerante)==1:
        temp = temp[temp['Refrigerant'] == refrigerante[0]]
    else:
        pass
    
    #Filter para lineas de producc
    if isinstance(linea_prod, list):
        if len(linea_prod) == 1:
            temp = temp[temp['Production_Line'] == linea_prod[0]]
        elif len(linea_prod) == 2:
            temp = temp[(temp['Production_Line'] == linea_prod[0])
                              | (temp['Production_Line'] == linea_prod[1])]
    else:
        pass
    
    #Filtro para plataforma
    if plataforma == "Todas":
        temp=temp
    else:
        temp = temp[temp['Platform'] == plataforma]
    
    #Filtro para proveedor
    if proveedor == "Todas":
        temp=temp
    else:
        temp = temp[temp['Supplier'] == proveedor]
    
    #Filtro fecha inicial y final
    temp = temp[(temp['Test_Date'] >= f_inicio) & (temp['Test_Date'] <= f_final)]
    
    #Inicio de creacion de gráficas y datos duros
    fig = px.histogram(temp, x=y_var)
    fig.update_layout(
    title=f'Histograma de {y_var}',
    xaxis_title=y_var,
)
    return [dcc.Graph(id='histogram2',figure=fig)]


# In[176]:


@app.callback(
    Output("tendencies_boxpt1", "children"),
    [Input("fam_dropdown", component_property="value"),
    Input("ref_checklist", component_property="value"), Input("lineaprod_checklist", component_property="value"),
    Input("plat_dropdown", component_property="value"), Input("prov_dropdown", component_property="value"),
    Input('dates-picker', 'start_date'), Input('dates-picker', 'end_date'), Input('x_dropdown', 'value')
    ]
)

def update_box1_tendencies(familia,refrigerante,linea_prod,plataforma,proveedor,f_inicio,f_final,x_var):
    df_sorted = df.sort_values(by='Test_Date')
#     #Filter para posicion
#     if isinstance(posicion, list):
#         if len(posicion)==1 and posicion[0]== 1:
#             rc_y = ['RC1_Temp_P1','RC2_Temp_P1','RC3_Temp_P1']
#         elif len(posicion)==1 and posicion[0]== 2:
#             rc_y = ['RC1_Temp_P2','RC2_Temp_P2','RC3_Temp_P2']
#         else:
#             rc_y = ['RC1_Temp_P1','RC2_Temp_P1','RC3_Temp_P1','RC1_Temp_P2','RC2_Temp_P2','RC3_Temp_P2']
    #Filter para familias
    if familia == "Todas":
        temp=df_sorted
    else:
        temp = df_sorted[df_sorted['Familia'] == familia]
    
    #Filter para refrigerantes
    if len(refrigerante)==1:
        temp = temp[temp['Refrigerant'] == refrigerante[0]]
    else:
        pass
    
    #Filter para lineas de producc
    if isinstance(linea_prod, list):
        if len(linea_prod) == 1:
            temp = temp[temp['Production_Line'] == linea_prod[0]]
        elif len(linea_prod) == 2:
            temp = temp[(temp['Production_Line'] == linea_prod[0])
                              | (temp['Production_Line'] == linea_prod[1])]
    else:
        pass
    
    #Filtro para plataforma
    if plataforma == "Todas":
        temp=temp
    else:
        temp = temp[temp['Platform'] == plataforma]
    
    #Filtro para proveedor
    if proveedor == "Todas":
        temp=temp
    else:
        temp = temp[temp['Supplier'] == proveedor]
    
    #Filtro fecha inicial y final
    temp = temp[(temp['Test_Date'] >= f_inicio) & (temp['Test_Date'] <= f_final)]
    
    #Inicio de creacion de gráficas y datos duros
    fig = px.box(temp, y=x_var)
    fig.update_layout(
    title=f'Box Plot de {x_var}',
    yaxis_title=x_var,
)
    return [dcc.Graph(id='boxplot1',figure=fig)]


# In[177]:


@app.callback(
    Output("tendencies_boxpt2", "children"),
    [Input("fam_dropdown", component_property="value"),
    Input("ref_checklist", component_property="value"), Input("lineaprod_checklist", component_property="value"),
    Input("plat_dropdown", component_property="value"), Input("prov_dropdown", component_property="value"),
    Input('dates-picker', 'start_date'), Input('dates-picker', 'end_date'), Input('y_dropdown', 'value')
    ]
)

def update_box1_tendencies(familia,refrigerante,linea_prod,plataforma,proveedor,f_inicio,f_final,y_var):
    df_sorted = df.sort_values(by='Test_Date')
    #Filter para posicion
#     if isinstance(posicion, list):
#         if len(posicion)==1 and posicion[0]== 1:
#             rc_y = ['RC1_Temp_P1','RC2_Temp_P1','RC3_Temp_P1']
#         elif len(posicion)==1 and posicion[0]== 2:
#             rc_y = ['RC1_Temp_P2','RC2_Temp_P2','RC3_Temp_P2']
#         else:
#             rc_y = ['RC1_Temp_P1','RC2_Temp_P1','RC3_Temp_P1','RC1_Temp_P2','RC2_Temp_P2','RC3_Temp_P2']
    #Filter para familias
    if familia == "Todas":
        temp=df_sorted
    else:
        temp = df_sorted[df_sorted['Familia'] == familia]
    
    #Filter para refrigerantes
    if len(refrigerante)==1:
        temp = temp[temp['Refrigerant'] == refrigerante[0]]
    else:
        pass
    
    #Filter para lineas de producc
    if isinstance(linea_prod, list):
        if len(linea_prod) == 1:
            temp = temp[temp['Production_Line'] == linea_prod[0]]
        elif len(linea_prod) == 2:
            temp = temp[(temp['Production_Line'] == linea_prod[0])
                              | (temp['Production_Line'] == linea_prod[1])]
    else:
        pass
    
    #Filtro para plataforma
    if plataforma == "Todas":
        temp=temp
    else:
        temp = temp[temp['Platform'] == plataforma]
    
    #Filtro para proveedor
    if proveedor == "Todas":
        temp=temp
    else:
        temp = temp[temp['Supplier'] == proveedor]
    
    #Filtro fecha inicial y final
    temp = temp[(temp['Test_Date'] >= f_inicio) & (temp['Test_Date'] <= f_final)]
    
    #Inicio de creacion de gráficas y datos duros
    fig = px.box(temp, y=y_var)
    fig.update_layout(
    title=f'Box Plot de {y_var}',
    yaxis_title=y_var,
)
    return [dcc.Graph(id='boxplot2',figure=fig)]


# In[178]:


if __name__=='__main__':
	app.run_server(debug=True, port=2223)

