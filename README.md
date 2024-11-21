# Customer-Churn-Prediction-ANN
Customer-Churn-Prediction-ANN is a Streamlit app that predicts customer churn using a trained ANN model. It takes customer data, processes it, and provides a prediction on whether a customer is likely to churn.

# Features:
* Data Preprocessing: The app processes customer data, including encoding categorical variables and scaling numerical features.
* ANN Model: A trained ANN model (saved as ann_churn_model.h5) is used for making predictions on customer churn.
* Streamlit Interface: A user-friendly interface for inputting customer details and viewing predictions.

# Files in the Repository:
* ANN.py: Python script for training the ANN model on the dataset.
* Churn_Modelling.csv: Dataset containing customer information used for training the model.
* ann_churn_model.h5: Trained ANN model saved in H5 format.
* app.py: Streamlit app that accepts user input and provides churn predictions.
* scalar.pkl: Pickled scaler used for feature scaling during prediction.

# How It Works:
* ***Data Preprocessing***:
    The customer data (Churn_Modelling.csv) is processed by encoding categorical features (city, gender, credit card ownership) and scaling numerical features (credit score, balance) using StandardScaler.

* ***Model Training***:
    An Artificial Neural Network (ANN) is built using TensorFlow to predict customer churn. The model is trained on the preprocessed data and saved as ann_churn_model.h5.

* ***Streamlit App***:
    Users input customer details through a web interface (e.g., age, city, credit score). The app processes the input, scales it, and passes it to the trained ANN model for prediction.

* ***Prediction***:
    The model predicts if a customer is likely to churn based on the input data. The result is displayed in the app:

* **Not Likely to Churn**: Success message.
* **Likely to Churn**: Error message.

# Installation and Setup:
* Clone the repository:
  ```bash
  git clone https://github.com/tanishqbololu/Customer-Churn-Prediction-ANN.git
  ```
* Navigate to the project folder:  
  ```bash
  cd Customer-Churn-Prediction-ANN
  ```

* Install Streamlit:
  ```bash
  pip install streamlit
  ```
* Install TensorFlow:
  ``` bash
  pip install tensorflow
  ```
* Install Pandas:
  ```bash
  pip install pandas
  ```
* Install Numpy:
  ```bash
  pip install numpy
  ```
* Install Scikit-Learn:
  ```bash
  pip install scikit-learn
  ```
* Download SpaCy model:
  ```bash
  python -m spacy download en_core_web_md
  ```
* Run the Streamlit app:
  ```bash
  streamlit run app.py
  ```
# To Test:
* Make sure you have all the required files (e.g., ann_churn_model.h5, scalar.pkl) and dependencies.
* Run each command one by one and check if they install successfully.
* After installation, running streamlit run app.py should open the Streamlit web interface in your browser.
