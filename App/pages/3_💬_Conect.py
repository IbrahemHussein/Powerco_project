import streamlit as st


st.header("About Me")
st.write("Welcome to my app! I'm a data scientist passionate about leveraging data to drive insights and solve complex problems.")
st.image("me.jpg", caption="ibrahem", use_column_width=False,width=200,)
st.subheader("Name")
st.write("Ibrahem elhosainy Gadalkreem")

st.subheader("Bio")
st.info("I'm a data scientist with expertise in machine learning, data analysis, and statistical modeling. I enjoy exploring datasets, building predictive models, and communicating insights.")
st.subheader("Skills")
skills = ["Machine Learning", "Data Analysis", "Statistical Modeling", "Python", "SQL"]
st.write(", ".join(skills))

st.subheader("Social Media")
st.write("[LinkedIn](https://www.linkedin.com/in/ibrahem-hussaein-a5891b218/)")
st.write("[GitHub](https://github.com/IbrahemHussein)")

st.title('Connect with Me')
st.sidebar.subheader('Personal Information')
name = st.sidebar.text_input('Name')
email = st.sidebar.text_input('Email')
phone = st.sidebar.text_input('Phone')

st.subheader('About Me')
about_me = st.text_area('Write something about yourself')

