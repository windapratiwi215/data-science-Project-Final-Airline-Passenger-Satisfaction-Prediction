import streamlit as st
import streamlit.components.v1 as stc

# import our app
from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#3872fb;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;"> Airline Passenger Satisfaction Prediction App </h1>
		    <h4 style="color:white;text-align:center;">The airline's management team and analysts </h4>
		    </div>
            """

desc_temp = """
            ### Airline Passenger Satisfaction App
            This app will be used by the airline's management team and analysts to predict passenger satisfaction or not
            #### Data Source
            - https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?select=train.csv
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning Section
            """


def main():
    # st.title("Main App")
    stc.html(html_temp)

    menu = ["Home", "Machine Learning"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        # st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning":
        st.subheader("Machine Learning Section")
        run_ml_app()


if __name__ == '__main__':
    main()
