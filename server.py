import streamlit as st
import pandas as pd
import pickle
import gzip

with gzip.open('model.pkl.gz', 'rb') as f_in:
    # Read the compressed data and decompress it using pickle
    model = pickle.load(f_in)

# Define a function to make predictions on the input data
def predict(data):
    predictions = model.predict(data)
    return predictions

# Define the main function that runs the Streamlit app
def main():
    # Set the title and header of the app
    html_temp = """ <div style="background-color:teal;padding:10px;margin-bottom:30px;">
    <h3 style="color:white;text-align:center;">HR Analytics: Employee Promotion Prediction</h3>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
        
    # Create a file uploader component and read the uploaded file as a pandas dataframe
    uploaded_file = st.file_uploader("Upload a CSV file containing employee data to predict promotions", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Make predictions on the input data
        predictions = predict(data)
        
        # Display the prediction results for each employee
        st.write("Prediction results:")
        for i in range(len(data)):
            employee = data.iloc[i]
            can_be_promoted = "Yes" if predictions[i] == 1 else "No"
            st.write(f"Employee {i+1}: Can be promoted - {can_be_promoted}")

# Run the app
if __name__ == '__main__':
    main()
