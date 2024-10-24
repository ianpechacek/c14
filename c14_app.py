import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import math
from scipy import stats

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from iosacal import R, iplot, combine

# Parameters for normal distribution
def uncertainty_func(mu, variance):
    
    sigma = math.sqrt(variance)

    # Generate values for x and corresponding probability density function (PDF) values
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = stats.norm.pdf(x, mu, sigma)

    # Create a DataFrame
    df = pd.DataFrame({
        'Probability': y,
        'Years Uncalibrated': x
    })
    return df
    
def save_plot(years, error):
        r = R(years, error, 'P-769')
        cal_r = r.calibrate('intcal20')
        
        # Create the plot using iplot
        iplot(cal_r)
        
        # Save the current figure as a PNG file
        plot_path = os.path.join('static', 'calibrated_plot.png')
        plt.savefig(plot_path, format='png')
        # plt.close()  # Close the plot to free memory
        st.image(plot_path, caption='Calibrated Plot', use_column_width=True)



if "step" not in st.session_state:
    st.session_state.step = 1

if "info" not in st.session_state:
    st.session_state.info = {}
def go_to_step1():
    st.session_state.step = 1

def go_to_step2(measured_age, error):
    st.session_state.step = 2
    st.session_state.info['measured_age'] = measured_age
    st.session_state.info['error'] = error

def go_to_step3():
    st.session_state.step = 3

def go_to_step4():
    st.session_state.step = 4



st.title("📊 Interactive Radiocarbon Dating Analysis")
st.markdown("""
Explore radiocarbon dating with interactive visualizations using a simplified calibration curve.
Understand how measured radiocarbon ages translate to calendar dates.
    """)

if st.session_state.step == 1:

    # Input controls
    col1, col2 = st.columns(2)

    with col1:
        measured_age = st.number_input(
            "Measured Radiocarbon Age (BP)",
            min_value=0,
            max_value=50000,
            value=3000,
            step=10
        )

    with col2:
        error = st.number_input(
            "Measurement Error (±years)",
            min_value=10,
            max_value=500,
            value=30,
            step=10
        )
    st.button("Next", on_click=go_to_step2, args=(measured_age, error))

if st.session_state.step == 2:

    df = uncertainty_func(st.session_state.info['measured_age'], st.session_state.info['error'])

    st.subheader("Uncertainty")
    st.area_chart(df, x='Years Uncalibrated', y='Probability')

    st.write("Modern methods of calibration take the original normal distribution of radiocarbon age ranges and use it to generate a histogram showing the relative probabilities for calendar ages. This has to be done by numerical methods rather than by a formula because the calibration curve is not describable as a formula")
    st.write(f"This curve represents the probabilistic distribution function of normal distribution with mean **{st.session_state.info['measured_age']}**, and standard distribution **{st.session_state.info['error']}**")
    st.button("Back", on_click = go_to_step1)
    st.button("Next", on_click = go_to_step3)


# r = R(7505, 93, 'P-769')
# cal_r = r.calibrate('intcal20')

# fig, ax = plt.subplots()
# # sns.scatterplot(data=iris, x="sepal_length", y="sepal_width", hue="species", ax=ax)
# fig = iplot(cal_r)
# # ax = iplot(cal_r)
# st.pyplot(fig)

if st.session_state.step == 3:


    measured_age = st.session_state.info['measured_age']
    error = st.session_state.info['error']

    df = uncertainty_func(measured_age, error)

    st.subheader("Uncertainty")
    st.area_chart(df, x='Years Uncalibrated', y='Probability')

    r = R(measured_age, error, 'P-769')
    cal_r = r.calibrate('intcal20')
    window = 200
    filtered_curve = cal_r.calibration_curve[(cal_r.calibration_curve[:,1]>= measured_age - window) & (cal_r.calibration_curve[:,1]<= measured_age + window) ][:,0:2]

    # Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Calibrated Date Probability Distribution",
            "Calibration Curve"
        ),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.15
    )

    fig.update_layout(
    height=1200,  # Increase the height of the figure
    width=1000,  # Increase the width of the figure (optional)
    title_text="Radiocarbon Dating Analysis",  # Optional: Add a title
    showlegend=True  # Optional: Show legend
    )

    fig.add_trace(
        go.Scatter(
            x= df["Years Uncalibrated"],
            y=df["Probability"],
            # x='Years Uncalibrated', y='Probability'
            name = "Uncertainty Curve"
            # ,y_label = "Probability"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=filtered_curve[:,0],
            y=filtered_curve[:,1],
            name='Calibration Curve',
            line=dict(color='red', width=1),
        ),
        row=2, col=1
    )
    # fig.update_yaxes(autorange="reversed", row=2, col=1)
    # fig.update_xaxes(autorange="reversed", row=2, col=1)

    


    # # add boxplot to the second chart
    # fig.add_trace(
    #     go.Scatter(
    #         x=df['Probability'],
    #         y=df["Years Uncalibrated"],
    #         mode='markers',
    #         marker=dict(size=10, color='red'),
    #         name='Measured Date',
    #         error_y=dict(
    #             type='data',
    #             array=[error],
    #             visible=True,
    #             color='red'
    #         ),
    #     ),
    #     row=2, col=1
    # )
    

    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_yaxes(title_text="Uncalibrated Years BP", row=2, col=1)

    fig.update_xaxes(title_text="Calibrated Years BP", row=2, col=1)

    

    st.plotly_chart(fig, use_container_width=True)

    st.write("To produce a curve that can be used to relate calendar years to radiocarbon years, a sequence of securely-dated samples is needed, which can be tested to determine their radiocarbon age. Dendrochronology, or the study of tree rings, led to the first such sequence: tree rings from individual pieces of wood show characteristic sequences of rings that vary in thickness due to environmental factors such as the amount of rainfall in a given year. Those factors affect all trees in an area and so examining tree-ring sequences from old wood allows the identification of overlapping sequences. In that way, an uninterrupted sequence of tree rings can be extended far into the past.")
    


# Add calibration curve with uncertainty envelope
# mask = (cal_curve_df['CAL BP'] >= measured_age - window_size) & \
#         (cal_curve_df['CAL BP'] <= measured_age + window_size)
# visible_curve = cal_curve_df[mask]
    st.write("Click on the Next button to see the standard open source radiocarbon calibration software IOSACal for Python")
    st.button("Next", on_click=go_to_step4)



if st.session_state.step == 4:
    save_plot(st.session_state.info["measured_age"], st.session_state.info["error"])
    st.write(f"In the example CALIB output shown at left, the input data is {st.session_state.info["measured_age"]} BP, with a standard deviation of {st.session_state.info["error"]} radiocarbon years. The curve selected is the northern hemisphere INTCAL13 curve, part of which is shown in the output; the vertical width of the curve corresponds to the width of the standard error in the calibration curve at that point. A normal distribution is shown at left; this is the input data, in radiocarbon years.")
    link = "https://en.wikipedia.org/wiki/Radiocarbon_calibration"
    st.caption(f"_Source: {link}_")

    st.subheader("Thank you for trying out this brief application! And please press the button if you look forward to winter!")
    if st.button("Let it snow!"):
        st.snow()


    # measured_age = st.session_state.info['measured_age']
    # error = st.session_state.info['error']

        # save_plot(measured_age, error)
        # plot_path = os.path.join('static', 'calibrated_plot.png')
        # st.image(plot_path, caption='Calibrated Plot', use_column_width=True)
        # st.image(os.path.join(os.getcwd(), "static", "BG.jpg"), width=500)




# st.title("Streamlit Elements Demo")
# st.header("This is a header")
# st.subheader('Subheader')
# st.markdown("This is **_Markdown_**")
# st.caption("small text")
# code_example = """
# def greet(name):
#     print('hello', name)
# """
# st.code(code_example, language="python")
# st.divider()

