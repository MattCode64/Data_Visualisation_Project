# Author: Hatim CHAHDI | AXI Technologies
import streamlit as st
import matplotlib.pyplot as plt
from bokeh.plotting import figure
import plotly.figure_factory as ff
import numpy as np

st.title('Plotting data in streamlit the integrated way using external libraries !')
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)
st.write('Plotting a matplotlib histogram with st.pyplot :')
st.pyplot(fig)

x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

p = figure(
     title='simple line example',
     x_axis_label='x',
     y_axis_label='y')

p.line(x, y, legend_label='Trend', line_width=2)
st.write('Plotting a bokeh line figure with st.bokeh_chart :')
st.bokeh_chart(p, use_container_width=True)

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']

# Create distplot with custom bin_size
fig = ff.create_distplot(
         hist_data, group_labels, bin_size=[.1, .25, .5])

st.write('Plotting a distribution plot with st.plotly_chart :')
# Plot!
st.plotly_chart(fig, use_container_width=True)