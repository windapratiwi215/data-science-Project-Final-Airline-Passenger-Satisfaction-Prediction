import streamlit as st
import numpy as np

# Load ML package
import joblib
import os

custype = {"disloyal Customer": 0, "Loyal Customer": 1}
tot = {"Personal Travel": 0, "Business travel": 1}
clss = {"Eco": 0, "Eco Plus": 1, "Business": 2}

attribute_info = """
                 - Customer Type: Disloyal Customer, Loyal Customer 
                 - Age: 25-85  
                 - Type of Travel: Personal Travel, Business travel  
                 - Class: Eco, Eco Plus, Business 
                 - Flight Distance: 337-2475 
                 - Inflight wifi service: 0-5  
                 - Gate location: 0-5 
                 - Online boarding: 0-5  
                 - Seat comfort: 0-5 
                 - Inflight entertainment: 0-5 
                 - On-board service: 0-5 
                 - Baggage handling: 1-5 
                 - Checkin service: 0-5  
                 - Inflight service: 0-5
                 - Cleanliness: 0-5
                 """


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


@st.cache
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


def run_ml_app():
    st.subheader("ML Section")

    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    customer_type = st.radio("Customer Type", ["disloyal Customer", "Loyal Customer"])
    age = st.number_input("Age", 5, 90)
    type_of_travel = st.radio("Type of Travel", ["Personal Travel", "Business travel"])
    selected_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    flight_distance = st.number_input("Flight Distance", 20, 5000)
    inflight_wifi_service = st.number_input("Inflight wifi service", 0, 5)
    gate_location = st.number_input("Gate location", 0, 5)
    online_boarding = st.number_input("Online boarding", 0, 5)
    seat_comfort = st.number_input("Seat comfort", 0, 5)
    inflight_entertainment = st.number_input("Inflight entertainment", 0, 5)
    on_board_service = st.number_input("On-board service", 0, 5)
    baggage_handling = st.number_input("Baggage handling", 1, 5)
    checkin_service = st.number_input("Checkin service", 0, 5)
    inflight_service = st.number_input("Inflight service", 0, 5)
    cleanliness = st.number_input("Cleanliness", 0, 5)

    with st.expander("Your Selected Options"):
        result = {
            "Customer Type": customer_type,
            "Type of Travel": type_of_travel,
            "Class": selected_class,
            "Flight Distance": flight_distance,
            "Age": age,
            "Inflight wifi service": inflight_wifi_service,
            "Gate location": gate_location,
            "Online boarding": online_boarding,
            "Seat comfort": seat_comfort,
            "Inflight entertainment": inflight_entertainment,
            "On-board service": on_board_service,
            "Baggage handling": baggage_handling,
            "Checkin service": checkin_service,
            "Inflight service": inflight_service,
            "Cleanliness": cleanliness,
        }

    st.write(result)

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in ["disloyal Customer", "Loyal Customer"]:
            res = get_value(i, custype)
            encoded_result.append(res)
        elif i in ["Personal Travel", "Business travel"]:
            res = get_value(i, tot)
            encoded_result.append(res)
        elif i in ["Eco", "Eco Plus", "Business"]:
            res = get_value(i, clss)
            encoded_result.append(res)

    # st.write(encoded_result)

    # Prediction section
    st.subheader("Prediction Result")
    single_sample = np.array(encoded_result).reshape(1, -1)
    # st.write(single_sample)

    model = load_model("model_grad.pkl")

    prediction = model.predict(single_sample)
    pred_proba = model.predict_proba(single_sample)
    st.write(prediction)
    st.write(pred_proba)

    pred_probability_score = {
        "Satisfied": round(pred_proba[0][1] * 100, 4),
        "neutral or dissatisfied": round(pred_proba[0][0] * 100, 4),
    }

    if prediction == 1:
        st.success("Congratulations, You are Satisfied")

        st.write(pred_probability_score)
    else:
        st.warning("Always give your valuable feedback")
        st.write(pred_probability_score)


# Call the run_ml_app() function to run the Streamlit app
if __name__ == "__main__":
    run_ml_app()
