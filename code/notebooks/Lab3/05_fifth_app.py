# Author: Hatim CHAHDI | AXI Technologies
import streamlit as st

st.title('Streamlit widgets')

st.write('Button usage')

if st.button('Say hi'):
    st.write('Hiiiiii')
else:
    st.write('Button not clicked')

st.write('Checkbox usage')
agree = st.checkbox('check to agree')
if agree:
    st.write('Great! you agreed ')

st.write('Select box usage')
option = st.selectbox(
     'How would you like to be contacted?',
     ('Email', 'Home phone', 'Mobile phone'))
st.write('You selected:', option)

st.write('Select slider usage')
color = st.select_slider(
     'Select a color of the rainbow',
     options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
st.write('My favorite color is', color)