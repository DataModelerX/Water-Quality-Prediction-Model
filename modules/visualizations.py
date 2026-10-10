"""
Interactive Plotly Data Visualizations for Water Quality Dashboard.
Includes correlation heatmaps, WQI gauge meters, radar comparisons, parity plots, and geographic charts.
"""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

COLOR_PALETTE = {
    "Excellent": "#10b981", # Emerald
    "Good": "#3b82f6",      # Blue
    "Poor": "#f59e0b",      # Amber
    "Very Poor": "#f97316", # Orange
    "Unsuitable": "#ef4444" # Red
}

def plot_wqi_gauge(wqi_value, title="Water Quality Index (WQI)"):
    """
    Creates an interactive speedometer gauge for WQI with standard Brown et al. scale.
    """
    wqi_disp = min(150.0, float(wqi_value)) if pd.notnull(wqi_value) else 0.0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=wqi_disp,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{title}</b>", 'font': {'size': 20, 'color': '#0f172a'}},
        gauge={
            'axis': {'range': [0, 150], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': "#1e293b", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 25], 'color': '#10b981'},    # Excellent
                {'range': [25, 50], 'color': '#3b82f6'},   # Good
                {'range': [50, 75], 'color': '#f59e0b'},   # Poor
                {'range': [75, 100], 'color': '#f97316'},  # Very Poor
                {'range': [100, 150], 'color': '#ef4444'}  # Unsuitable
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': wqi_disp
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=25, r=25, t=40, b=10),
        font={'family': 'Inter, sans-serif'}
    )
    return fig

def plot_correlation_heatmap(df, columns=None):
    """
    Plots interactive correlation heatmap across numeric parameters.
    """
    if columns is None:
        columns = ["Temperature", "DO", "pH", "Conductivity", "BOD", "Nitrate", "Total_Coliform", "WQI"]
    
    available_cols = [c for c in columns if c in df.columns]
    corr_matrix = df[available_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="<b>Pearson Correlation Matrix: Water Quality Parameters</b>"
    )
    fig.update_layout(
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif")
    )
    return fig

def plot_parameter_scatter(df, x_col, y_col, color_col="WQI_Category"):
    """
    Plots bivariate relationship with OLS trendline.
    """
    sub_df = df.dropna(subset=[x_col, y_col]).copy()
    
    # Cap upper outliers for cleaner chart readability
    q_x = sub_df[x_col].quantile(0.99)
    q_y = sub_df[y_col].quantile(0.99)
    sub_df = sub_df[(sub_df[x_col] <= q_x) & (sub_df[y_col] <= q_y)]
    
    fig = px.scatter(
        sub_df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in sub_df.columns else None,
        color_discrete_map=COLOR_PALETTE,
        trendline="ols",
        hover_data=["Station_Code", "River", "State"],
        title=f"<b>Bivariate Relationship: {y_col} vs {x_col}</b>",
        opacity=0.75
    )
    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif")
    )
    return fig

def plot_state_wqi_comparison(df, top_n=15):
    """
    Plots mean WQI by State.
    """
    state_stats = df.groupby("State").agg(
        Mean_WQI=("WQI", "mean"),
        Median_WQI=("WQI", "median"),
        Mean_DO=("DO", "mean"),
        Mean_BOD=("BOD", "mean"),
        Station_Count=("Station_Code", "count")
    ).reset_index()
    
    state_stats = state_stats[state_stats["Station_Count"] >= 5]
    state_stats = state_stats.sort_values(by="Median_WQI", ascending=True).head(top_n)
    
    fig = px.bar(
        state_stats,
        x="State",
        y="Median_WQI",
        color="Median_WQI",
        color_continuous_scale="Viridis_r",
        text="Median_WQI",
        title=f"<b>State Water Quality Comparison (Median WQI - Top {top_n} States)</b>",
        labels={"Median_WQI": "Median WQI (Lower = Cleaner)"}
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(
        template="plotly_white",
        height=460,
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=50, b=80),
        font=dict(family="Inter, sans-serif")
    )
    return fig

def plot_river_comparison(df, top_n=15):
    """
    Compares average DO and BOD across major Indian river basins.
    """
    river_stats = df.groupby("River").agg(
        Mean_DO=("DO", "mean"),
        Mean_BOD=("BOD", "mean"),
        Station_Count=("Station_Code", "count")
    ).reset_index()
    
    river_stats = river_stats[river_stats["Station_Count"] >= 10].sort_values(by="Mean_DO", ascending=False).head(top_n)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=river_stats["River"],
        y=river_stats["Mean_DO"],
        name="Dissolved Oxygen (DO mg/L)",
        marker_color="#0284c7"
    ))
    fig.add_trace(go.Bar(
        x=river_stats["River"],
        y=river_stats["Mean_BOD"],
        name="Biochemical Oxygen Demand (BOD mg/L)",
        marker_color="#f97316"
    ))
    fig.update_layout(
        barmode='group',
        title=f"<b>Major River Basins: Oxygen Dynamics (DO vs BOD)</b>",
        template="plotly_white",
        xaxis_tickangle=-45,
        height=450,
        margin=dict(l=40, r=40, t=50, b=80),
        font=dict(family="Inter, sans-serif")
    )
    return fig

def plot_feature_importance(importance_df, title="<b>Explainable AI: Feature Importances</b>"):
    """
    Plots horizontal bar chart of feature importances.
    """
    fig = px.bar(
        importance_df.sort_values(by="Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        text="Importance",
        title=title
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=40, t=40, b=30),
        font=dict(family="Inter, sans-serif")
    )
    return fig

def plot_actual_vs_predicted(y_true, y_pred, target_name="Target", r2=None, mae=None):
    """
    Plots parity plot: Actual vs Predicted with 45-degree reference line.
    """
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='#0284c7', opacity=0.6, size=6)
    ))
    # 45-degree line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Ideal Parity (1:1)',
        line=dict(color='#ef4444', dash='dash', width=2)
    ))
    
    title_text = f"<b>Model Parity Plot: Actual vs Predicted {target_name}</b>"
    if r2 is not None and mae is not None:
        title_text += f"<br><sub>R²: {r2:.3f} | MAE: {mae:.3f}</sub>"
        
    fig.update_layout(
        title=title_text,
        xaxis_title=f"Actual {target_name}",
        yaxis_title=f"Predicted {target_name}",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter, sans-serif")
    )
    return fig

def plot_confusion_matrix_chart(cm, classes):
    """
    Plots confusion matrix as interactive heatmap.
    """
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=classes,
        y=classes,
        text_auto=True,
        color_continuous_scale="Purples",
        title="<b>Classification Confusion Matrix</b>"
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif")
    )
    return fig

def plot_water_sample_radar(sample_dict):
    """
    Generates radar chart comparing user sample parameters to CPCB / BIS standards.
    """
    # Standards normalized to 100%
    categories = ["pH", "DO", "BOD", "Conductivity", "Nitrate"]
    standards = {"pH": 8.5, "DO": 5.0, "BOD": 3.0, "Conductivity": 1000.0, "Nitrate": 45.0}
    
    # Calculate % of standard limit
    sample_ratios = []
    for cat in categories:
        val = sample_dict.get(cat, 0.0)
        std = standards[cat]
        ratio = (val / std) * 100.0
        sample_ratios.append(min(200.0, ratio)) # Cap at 200% for radar aesthetic
        
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[100.0] * len(categories) + [100.0],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.15)',
        line=dict(color='#10b981', dash='dash'),
        name='CPCB Permissible Standard (100%)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=sample_ratios + [sample_ratios[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(2, 132, 199, 0.35)',
        line=dict(color='#0284c7', width=2),
        name='Current Sample (% of Standard)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 200]
            )
        ),
        title="<b>Water Sample Fingerprint vs Permissible Standards</b>",
        height=350,
        margin=dict(l=40, r=40, t=50, b=30),
        font=dict(family="Inter, sans-serif")
    )
    return fig
