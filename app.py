import streamlit as st

st.title("Video Analytics")

st.subheader('Example video after implemented Face Recognition and Re-Identification')
st.write("""In this video the label will be '(#1 Name confience)'. The number after '#' show that be the person1, person2, ...  which should be the same person along with video.
    The second tag is the name main person, and thrid show the confidence of prediction
         """)

st.video("example_video.mp4")