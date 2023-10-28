# https://medium.com/@abhisheky127/mastering-pycaret-and-streamlit-a-comprehensive-guide-to-building-and-deploying-machine-learning-71b28a22655e
from pycaret.classification import *
import streamlit as st
import pandas as pd
from pycaret.datasets import get_data

iris = get_data('iris')

print('\nSetup')
clf = setup(data=iris,
            target='species',
            session_id=123,
            normalize=True,
            transformation=True)

print('\nCreate Model')
model = create_model('lr')
                    #  feature_selection=True,
                    #  feature_interaction=True,
                    #  feature_ratio=True)

print('\nTune Model')
tuned_model = tune_model(model, n_iter=50, search_library='optuna')

# bagged_model = ensemble_model(model, method='Bagging')
# blended_model = blend_models(estimator_list=[model1, model2, model3])
# stacked_model = stack_models(estimator_list=[model1, model2],
#                              meta_model=model3)

# interpret_model(tuned_model, plot='summary')

evaluate_model(tuned_model)


def main():
    model = load_model('tuned_model')
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


if __name__ == "__main__":
    main()
