# Stock Predictor - Long Short Term Memory Networks

![deep-learning.jpg](Images/deep-learning.jpg)

Due to the volatility of cryptocurrency speculation, investors often try to incorporate sentiment from social media and news articles to guide their trading strategies. [Cyrpto Fear & Greed Index(FNG)](https://alternative.me/crypto/fear-and-greed-index/) is an indicator that attempts to use variety of data sources to produce a daily FNG value for cryptocurrency. 

Using deep learning recurrent neural networks, build and evaluate bitcoin closing prices to determine if the FNG indicator provides better signal for cryptocurrencies than the normal closing price data. One model will use the FNG indicators to provide the closing prices while the second model will use a window of closing prices to predict the nth closing price.

- - -

### Files

[Closing Prices Starter Notebook](Starter_Code/lstm_stock_predictor_closing.ipynb)

[FNG Starter Notebook](Starter_Code/lstm_stock_predictor_fng.ipynb)

- - -
<details>
<summary>Prepare the data for training and testing</summary>
    
<br>1. Use of Starter code as a guide to create a Jupyter Notebook for each RNN that contains a function to help window the data for each dataset.</br>
    
    
    ```python
        # This function accepts the column number for the features (X) and the target (y). It chunks the data up with a rolling window of Xt-n to predict Xt. It returns a numpy array of X any y
    
        def window_data(df, window, feature_col_number, target_col_number):
        X = []
        y = []
        for i in range(len(df) - window - 1):
            features = df.iloc[i:(i + window), feature_col_number]
            target = df.iloc[(i + window), target_col_number]
            X.append(features)
            y.append(target)
        return np.array(X), np.array(y).reshape(-1, 1)
    ```
    
   <br>2. For Fear & Greed Model, use FNG values to try and predict the closing price while for closing price model, use the previous closing prices to predict the next closing prices.</br>
   
   
   ```python
        # Predict Closing Prices using a 10 day window of previous closing prices. Try a window size anywhere from 1 to 10 and see how the model performance changes
        window_size = 1

        # Column index 1 is the `Close` column & Column index 0 is the 'FNG'
        feature_column = 1
        target_column = 1
        X, y = window_data(df, window_size, feature_column, target_column)
   ```
    

<br>3. Each model will need to use 70% of the data for training and 30% of the data for testing.</br>
   
   ```python
        split = int(0.7 * len(X))
        X_train = X[: split - 1]
        X_test = X[split:]
        y_train = y[: split - 1]
        y_test = y[split:]
   ```
    
<br>4. Apply a MinMaxScaler to the X and y values to scale the data for the model.</br>
   
   ```python
        # Use MinMaxScaler to scale the data between 0 and 1. Importing the MinMaxScaler from sklearn
        from sklearn.preprocessing import MinMaxScaler
        # Create a MinMaxScaler object
        scaler = MinMaxScaler()
        # Fit the MinMaxScaler object with the features data X
        scaler.fit(X)
        # Scale the features training and testing sets
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # Fit the MinMaxScaler object with the target data Y
        scaler.fit(y)
        # Scale the target training and testing sets
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)
   ```

<br>5. Finally, reshape the X_train and X_test values to fit the model's requirement of (samples, time steps, features).</br>
   
   ```python
        # Reshape the features for the model
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
   ```

</details>

<details>


<summary>Build and train custom LSTM RNNs</summary>
    
<br>1. Import the required library from tensorflow</br>
    
    ```python
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    ```
    
<br>2. Define the model architecture</br>
   
   ```python
        # Build the LSTM model. 
   
        model = Sequential()

        number_units = 30
        dropout_fraction = 0.2
        # The return sequences need to be set to True if you are adding additional LSTM layers, but don't have to do this for the final layer. 
        # Layer 1
        model.add(LSTM(
            units=number_units,
            return_sequences=True,
            input_shape=(X_train.shape[1], 1))
            )
        model.add(Dropout(dropout_fraction))
        # Layer 2
        model.add(LSTM(units=number_units, return_sequences=True))
        model.add(Dropout(dropout_fraction))
        # Layer 3
        model.add(LSTM(units=number_units))
        model.add(Dropout(dropout_fraction))
        # Output layer
        model.add(Dense(1))
   ```
    
<br>3. Compile the model</br>
   
   ```python
       model.compile(optimizer="adam", loss="mean_squared_error")
   ```
<br>4. Summarize the model</br>
   
   ![lstm_model_summary.png](Images/lstm_model_summary.png)
   
   <br>5. Fit the model and train the data</br>
   
   ```python
        # Train the model
        # Use at least 10 epochs
        # Do not shuffle the data
        # Experiement with the batch size, but a smaller batch size is recommended
        model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=5, verbose=1)
   ```  
    
 | Closing Price                       | FNG                              |
 | ----------------------------------- | ----------------------------------- |
 | <img src="Images/closing_price_model.png" width="400" />  | <img src="Images/fng_model.png" width="400" />  |

</details>



<details>
<summary>Evaluate the performance of each model</summary>
    
<br>1. Evaluate the model using the `X_test` and `y_test` data</br>

| **Closing Price**              | **FNG**                        |
| :--------------------------: | :-------------------------- |
| Loss = 0.0060              | FNG Loss : 0.07320         |

* Closing Price model has a lower loss

<br>2. Use `X_test` data to make the predictions</br>

<br>3. Recover the original prices instead of the scaled version</br>

    ```python
        predicted_prices = scaler.inverse_transform(predicted)
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    ```
<br>4. Create a DataFrame of Real (X) vs. Predicted Values</br>

| **Closing Price**                       | **FNG**                              |
| ----------------------------------- | ----------------------------------- |
| <img src="Images/closing_price_results.png" width="400" />  | <img src="Images/fng_results.png" width="400" />  |

* Closing Prices model tracks the actual values better over time.

<br>5. Plot the Real vs. Predicted values on a Line chart
 
| **Closing Price**                       | **FNG**                              |
| ----------------------------------- | ----------------------------------- |
| <img src="Images/closing_price_graph.png" width="500" />  | <img src="Images/fng_plot.png" width="500" />  |


* Window size 7 works best for the model.

</details>

- - -

### Resources

[Keras Sequential Model Guide](https://keras.io/getting-started/sequential-model-guide/)

[Illustrated Guide to LSTMs](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

[Stanford's RNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

- - -
