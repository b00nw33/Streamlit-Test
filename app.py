# https://medium.com/@abhisheky127/mastering-pycaret-and-streamlit-a-comprehensive-guide-to-building-and-deploying-machine-learning-71b28a22655e
!pip install pycaret -q
!pip install streamlit -q

from pycaret.classification import predict_model
import streamlit as st

model = load_model('diamond-pipeline')
st.title('Your Model Prediction App')

# Collect user input
input_1 = st.number_input('Input 1', min_value=0.0, max_value=10.0)
input_2 = st.number_input('Input 2', min_value=0.0, max_value=10.0)
input_3 = st.number_input('Input 3', min_value=0.0, max_value=10.0)
input_4 = st.number_input('Input 4', min_value=0.0, max_value=10.0)

# Predict the output
if st.button('Predict'):
    input_data = pd.DataFrame(
        [[input_1, input_2, input_3, input_4]],
        columns=['input_1', 'input_2', 'input_3', 'input_4'])
    prediction = predict_model(model, data=input_data)
    st.write(f"The predicted output is: {prediction['Label'].iloc[0]}")
