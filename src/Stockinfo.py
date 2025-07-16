# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb
# import yfinance as yf
# from keras.models import load_model  # Fixed typo
# import streamlit as st

# start = '2010-01-01'
# end = '2019-12-31'

# st.title('Stock Trend Prediction')
# user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# # Fetch stock data for AAPL and YHOO using yfinance
# ticker = user_input  # List of tickers

# # Error handling for invalid tickers
# try:
#     df = yf.download(ticker, start=start, end=end)
#     if df.empty:
#         st.write(f"No data found for {ticker}")

#     else:
#         # Select specific columns to match the screenshot
#         df = df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
#         df.columns = df.columns.droplevel(1)

#         # Describing data
#         st.subheader(f'Data from {start} to {end} for {ticker}')
#         st.write(df.describe())
#         # st.subheader('Data from 2010-2019')
#         # fig = plt.figure(figsize=(12,6))
#         # plt.plot(df.Close)
#         # st.pyplot(fig)

#         #Visualisations
#         st.subheader(f'Closing Price V/s Time Chart')
#         fig = plt.figure(figsize=(12,6))
#         plt.plot(df.Close,'darkcyan')
#         plt.xlabel('Time', fontweight='bold', fontsize=14) 
#         plt.ylabel('Price', fontweight='bold', fontsize=14)
#         st.pyplot(fig)

#         st.subheader(f'Closing Price V/s Time Chart With 100 Day Moving Average')
#         ma100 = df.Close.rolling(100).mean()
#         fig = plt.figure(figsize=(12,6))
#         plt.plot(ma100, 'black')
#         plt.plot(df.Close,'darkcyan')
#         plt.xlabel('Time', fontweight='bold', fontsize=14) 
#         plt.ylabel('Price', fontweight='bold', fontsize=14)
#         st.pyplot(fig)

#         st.subheader('Closing Price V/s Time Chart With 100 & 200 Day Moving Average')
#         ma100 = df.Close.rolling(100).mean()
#         ma200 = df.Close.rolling(200).mean()

#         fig = plt.figure(figsize=(12,6))
#         plt.plot(ma100,'black')
#         plt.plot(ma200, 'red')
#         plt.plot(df.Close, 'darkcyan')
#         plt.xlabel('Time', fontweight='bold', fontsize=14) 
#         plt.ylabel('Price', fontweight='bold', fontsize=14)
#         st.pyplot(fig)

#         #Splitting data into Training and Testing
#         data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
#         data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#         print(data_training.shape)
#         print(data_testing.shape)

#         from sklearn.preprocessing import MinMaxScaler
#         scaler = MinMaxScaler(feature_range=(0,1))

#         data_training_array = scaler.fit_transform(data_training)

#         #Load my model
#         model = load_model('keras_model.h5')

#         #Testing part
#         past_100_days = data_training.tail(100)
#         final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
#         input_data = scaler.fit_transform(final_df)
#         x_test = []
#         y_test = []

#         for i in range(100, input_data.shape[0]):
#             x_test.append(input_data[i-100: i])
#             y_test.append(input_data[i,0])

#         x_test, y_test = np.array(x_test), np.array(y_test)
#         y_predicted = model.predict(x_test)
#         scaler = scaler.scale_

#         scale_factor = 1/scaler[0]
#         y_predicted = y_predicted*scale_factor
#         y_test = y_test*scale_factor

#         #Final Graph
#         st.subheader('Predictions V/s Original')
#         fig2 = plt.figure(figsize=(12,6))
#         plt.plot(y_test, 'darkcyan', label='Original Price')
#         plt.plot(y_predicted, 'magenta', label='Predicted Price')
#         plt.xlabel('Time', fontweight='bold', fontsize=14) 
#         plt.ylabel('Price', fontweight='bold', fontsize=14)
#         plt.legend()
#         st.pyplot(fig2)
#         plt.show()

                
# except Exception as e:
#     st.write(f"Error fetching data: {e}")





#*Modified code for accepting ticker as user input
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb
# import yfinance as yf
# from keras.models import load_model
# import streamlit as st
# from sklearn.preprocessing import MinMaxScaler

# # Set the title of the Streamlit app
# st.title('Stock Trend Prediction')

# # Create a text input widget for the stock ticker
# ticker = st.text_input("Enter Stock Ticker", "")

# # Define the start and end dates
# start = '2010-01-01'
# end = '2019-12-31'

# # Check if the user has entered a ticker symbol
# if ticker:
#     # Error handling for invalid tickers
#     try:
#         # Fetch stock data for the user-inputted ticker
#         df = yf.download(ticker, start=start, end=end)
        
#         # Check if data is retrieved successfully
#         if not df.empty:
#             # Drop the second level of the column index if it exists
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = df.columns.droplevel(1)

#             # Select specific columns
#             df = df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]

#             # Basic descriptive statistics
#             st.subheader(f'Data from {start} to {end} for {ticker}')
#             st.write(df.describe())  # Display summary statistics only

#             # Visualizations
#             # Plot the closing prices over time
#             st.subheader(f"Closing Price for {ticker} from {start} to {end}")
#             st.line_chart(df['Close'])

#             # Closing Price with 100-day Moving Average
#             st.subheader(f'Closing Price V/s Time Chart With 100 Day Moving Average')
#             ma100 = df['Close'].rolling(100).mean()
#             fig = plt.figure(figsize=(12, 6))
#             plt.plot(df['Close'], 'darkcyan', label='Close')
#             plt.plot(ma100, 'black', label='100-Day MA')
#             plt.xlabel('Time', fontweight='bold', fontsize=14) 
#             plt.ylabel('Price', fontweight='bold', fontsize=14)
#             plt.legend()
#             st.pyplot(fig)

#             # Closing Price with 100 and 200-day Moving Average
#             st.subheader('Closing Price V/s Time Chart With 100 & 200 Day Moving Average')
#             ma200 = df['Close'].rolling(200).mean()
#             fig = plt.figure(figsize=(12, 6))
#             plt.plot(df['Close'], 'darkcyan', label='Close')
#             plt.plot(ma100, 'black', label='100-Day MA')
#             plt.plot(ma200, 'red', label='200-Day MA')
#             plt.xlabel('Time', fontweight='bold', fontsize=14) 
#             plt.ylabel('Price', fontweight='bold', fontsize=14)
#             plt.legend()
#             st.pyplot(fig)

#             # Splitting data into Training and Testing
#             data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
#             data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#             # Preprocessing and Scaling
#             scaler = MinMaxScaler(feature_range=(0, 1))
#             data_training_array = scaler.fit_transform(data_training)

#             # Load trained model
#             model = load_model('keras_model.h5')

#             # Prepare test data for prediction
#             past_100_days = data_training.tail(100)
#             final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
#             input_data = scaler.fit_transform(final_df)

#             x_test = []
#             y_test = []

#             for i in range(100, input_data.shape[0]):
#                 x_test.append(input_data[i-100:i])
#                 y_test.append(input_data[i, 0])

#             x_test, y_test = np.array(x_test), np.array(y_test)

#             # Model prediction
#             y_predicted = model.predict(x_test)
#             scaler = scaler.scale_

#             scale_factor = 1/scaler[0]
#             y_predicted = y_predicted * scale_factor
#             y_test = y_test * scale_factor

#             # Final Graph
#             st.subheader('Predictions V/s Original')
#             fig2 = plt.figure(figsize=(12, 6))
#             plt.plot(y_test, 'darkcyan', label='Original Price')
#             plt.plot(y_predicted, 'magenta', label='Predicted Price')
#             plt.xlabel('Time', fontweight='bold', fontsize=14) 
#             plt.ylabel('Price', fontweight='bold', fontsize=14)
#             plt.legend()
#             st.pyplot(fig2)

#         else:
#             st.error(f"No data found for ticker '{ticker}'")

#     except Exception as e:
#         st.error(f"Error fetching data for ticker '{ticker}': {e}")

# else:
#     st.write("Please enter a stock ticker to display data.")


#* Modified code for taking user input for the timespan of the data to be displayed 
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb
# import yfinance as yf
# from keras.models import load_model
# import streamlit as st
# from sklearn.preprocessing import MinMaxScaler
# from datetime import datetime
# import plotly.graph_objs as go
# import mplfinance as mpf

# # Set the title of the Streamlit app
# st.title('Stock Trend Prediction')

# # Create a text input widget for the stock ticker
# ticker = st.text_input("Enter Stock Ticker", "")

# # Check if the user has entered a ticker symbol
# if ticker:
#     # Ask the user to specify the date range for the data
#     start_date = st.date_input("Start date", value=datetime(2010, 1, 1))
#     end_date = st.date_input("End date", value=datetime(2019, 12, 31))

#     # Error handling for invalid tickers
#     try:
#         # Fetch stock data for the user-inputted ticker and specified date range
#         df = yf.download(ticker, start=start_date, end=end_date)
        
#         # Check if data is retrieved successfully
#         if not df.empty:
#             # Drop the second level of the column index if it exists
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = df.columns.droplevel(1)

#             # Select specific columns
#             df = df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]

#             # Basic descriptive statistics
#             st.subheader(f'Data from {start_date} to {end_date} for {ticker}')
#             st.write(df.describe())  # Display summary statistics only

#             #*Visualizations
#             # Candlestick Chart with Volume
#             st.subheader(f'Candlestick Chart with Volume for {ticker} from {start_date} to {end_date}')
#             df_for_mpf = df.copy()
#             df_for_mpf.index.name = 'Date'  # Set the index name to 'Date' for mplfinance

#             # Use mplfinance to plot the candlestick chart with volume
#             fig, axlist = mpf.plot(
#                 df_for_mpf,
#                 type='candle',
#                 volume=True,
#                 mav=(20, 50),  # You can choose different moving averages or remove them if not needed
#                 figsize=(12, 6),
#                 style='yahoo',
#                 returnfig=True
#             )
#             st.pyplot(fig)

#             # Closing Price with 100-day Moving Average
#             st.subheader(f'Closing Price V/s Time Chart With 100 Day Moving Average')
#             ma100 = df['Close'].rolling(100).mean()
#             fig = plt.figure(figsize=(12, 6))
#             plt.plot(df['Close'], 'darkcyan', label='Close')
#             plt.plot(ma100, 'black', label='100-Day MA')
#             plt.xlabel('Time', fontweight='bold', fontsize=14) 
#             plt.ylabel('Price', fontweight='bold', fontsize=14)
#             plt.legend()
#             st.pyplot(fig)

#             # Closing Price with 100 and 200-day Moving Average
#             st.subheader('Closing Price V/s Time Chart With 100 & 200 Day Moving Average')
#             ma200 = df['Close'].rolling(200).mean()
#             fig = plt.figure(figsize=(12, 6))
#             plt.plot(df['Close'], 'darkcyan', label='Close')
#             plt.plot(ma100, 'black', label='100-Day MA')
#             plt.plot(ma200, 'red', label='200-Day MA')
#             plt.xlabel('Time', fontweight='bold', fontsize=14) 
#             plt.ylabel('Price', fontweight='bold', fontsize=14)
#             plt.legend()
#             st.pyplot(fig)

#             # Splitting data into Training and Testing
#             data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
#             data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#             # Preprocessing and Scaling
#             scaler = MinMaxScaler(feature_range=(0, 1))
#             data_training_array = scaler.fit_transform(data_training)

#             # Load trained model
#             model = load_model('keras_model.h5')

#             # Prepare test data for prediction
#             past_100_days = data_training.tail(100)
#             final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
#             input_data = scaler.fit_transform(final_df)

#             x_test = []
#             y_test = []

#             for i in range(100, input_data.shape[0]):
#                 x_test.append(input_data[i-100:i])
#                 y_test.append(input_data[i, 0])

#             x_test, y_test = np.array(x_test), np.array(y_test)

#             # Model prediction
#             y_predicted = model.predict(x_test)
#             scaler = scaler.scale_

#             scale_factor = 1/scaler[0]
#             y_predicted = y_predicted * scale_factor
#             y_test = y_test * scale_factor

#             # Final Graph
#             st.subheader('Predictions V/s Original')
#             fig2 = plt.figure(figsize=(12, 6))
#             plt.plot(y_test, 'darkcyan', label='Original Price')
#             plt.plot(y_predicted, 'magenta', label='Predicted Price')
#             plt.xlabel('Time', fontweight='bold', fontsize=14) 
#             plt.ylabel('Price', fontweight='bold', fontsize=14)
#             plt.legend(loc='upper left')
#             st.pyplot(fig2)

#         else:
#             st.error(f"No data found for ticker '{ticker}' in the specified date range")

#     except Exception as e:
#         st.error(f"Error fetching data for ticker '{ticker}': {e}")

# else:
#     st.write("Please enter a stock ticker to display data.")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import altair as alt

# Set the title of the Streamlit app
st.title('Stock Trend Prediction')

# Create a text input widget for the stock ticker
ticker = st.text_input("Enter Stock Ticker", "")

# Check if the user has entered a ticker symbol
if ticker:
    # Ask the user to specify the date range for the data
    start_date = st.date_input("Start date", value=datetime(2010, 1, 1))
    end_date = st.date_input("End date", value=datetime(2019, 12, 31))

    # Error handling for invalid tickers
    try:
        # Fetch stock data for the user-inputted ticker and specified date range
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if data is retrieved successfully
        if not df.empty:
            # Drop the second level of the column index if it exists
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Select specific columns
            df = df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]

            # Reset the index to ensure the 'Date' column is available
            df.reset_index(inplace=True)

            # Make sure the "Date" column is of datetime type
            df['Date'] = pd.to_datetime(df['Date'])

            # Drop rows with any missing data in the required columns
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

             # Basic descriptive statistics
            st.subheader(f'Data from {start_date} to {end_date} for {ticker}')
            st.write(df.describe())  # Display summary statistics only

            # Plot the closing prices over time
            st.subheader(f"Closing Price for {ticker} from {start_date} to {end_date}")
            st.line_chart(df['Close'])

            # Calculate Heikin-Ashi values
            df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
            df['HA_Open'] = 0.0  # Initialize with zeros
            df['HA_High'] = 0.0  # Initialize with zeros
            df['HA_Low'] = 0.0   # Initialize with zeros
            
            # Fill the initial values for HA_Open using the first row's 'Open' value
            df.loc[0, 'HA_Open'] = (df.loc[0, 'Open'] + df.loc[0, 'Close']) / 2
            df.loc[0, 'HA_High'] = df.loc[0, ['High', 'HA_Open', 'HA_Close']].max()
            df.loc[0, 'HA_Low'] = df.loc[0, ['Low', 'HA_Open', 'HA_Close']].min()

            # Basic descriptive statistics
            st.subheader(f'Data from {start_date} to {end_date} for {ticker} with Heikin-ashi Candles')
            st.write(df.describe())  # Display summary statistics only

            # Plot the closing prices over time
            # st.subheader(f"Closing Price for {ticker} from {start_date} to {end_date}")
            # st.line_chart(df['Close'])

            # Calculate the Heikin-Ashi values for the rest of the dataframe
            for i in range(1, len(df)):
                df.loc[i, 'HA_Open'] = (df.loc[i - 1, 'HA_Open'] + df.loc[i - 1, 'HA_Close']) / 2
                df.loc[i, 'HA_High'] = df.loc[i, ['High', 'HA_Open', 'HA_Close']].max()
                df.loc[i, 'HA_Low'] = df.loc[i, ['Low', 'HA_Open', 'HA_Close']].min()

             # Customize the colors for bullish and bearish Heikin-Ashi candles
            bullish_color = "#236d18"  # Light teal for bullish candles
            bearish_color = "#EF5350"  # Light red for bearish candles

            # Customize the colors for volume bars
            bullish_volume_color = "#236d18"  # Lighter teal for bullish volume bars
            bearish_volume_color = "#D32F2F"  # Lighter red for bearish volume bars

            # Candlestick chart with Heikin-Ashi values
            st.subheader(f'Heikin-Ashi Candlestick Chart for {ticker} from {start_date} to {end_date}')
            heikin_ashi_candlestick = alt.Chart(df).mark_rule().encode(
                x='Date:T',
                y='HA_Low:Q',
                y2='HA_High:Q',
                color=alt.condition("datum.HA_Open < datum.HA_Close",  alt.value(bullish_color), alt.value(bearish_color))
            ).properties(
                width=700,
                height=400
            )

            bars = alt.Chart(df).mark_bar().encode(
                x='Date:T',
                y='Volume:Q',
                color=alt.condition("datum.Open < datum.Close", alt.value("green"), alt.value("red"))
            ).properties(
                width=700
            )

            ha_rule = alt.Chart(df).mark_bar(size=1).encode(
                x='Date:T',
                y='HA_Open:Q',
                y2='HA_Close:Q',
                color=alt.condition("datum.HA_Open < datum.HA_Close",  alt.value(bullish_color), alt.value(bearish_color))
            ).properties(
                width=700,
                height=400
            )

            # Customize the volume bar colors
            volume_bars = alt.Chart(df).mark_bar().encode(
                x='Date:T',
                y='Volume:Q',
                color=alt.condition("datum.HA_Open < datum.HA_Close", 
                                    alt.value(bullish_volume_color), alt.value(bearish_volume_color))
            ).properties(
                width=700
)

            # Combine Heikin-Ashi candles
            ha_combined_chart = alt.vconcat(
                heikin_ashi_candlestick + ha_rule,
                volume_bars
            ).resolve_scale(
                x='shared'
            ).interactive()

            st.altair_chart(ha_combined_chart, use_container_width=True)


            # # Basic descriptive statistics
            # st.subheader(f'Data from {start_date} to {end_date} for {ticker}')
            # st.write(df.describe())  # Display summary statistics only

            # # Plot the closing prices over time
            # st.subheader(f"Closing Price for {ticker} from {start_date} to {end_date}")
            # st.line_chart(df['Close'])

            # Closing Price with 100-day Moving Average
            st.subheader(f'Closing Price V/s Time Chart With 100 Day Moving Average')
            ma100 = df['Close'].rolling(100).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], 'darkcyan', label='Close')
            plt.plot(ma100, 'black', label='100-Day MA')
            plt.xlabel('Time', fontweight='bold', fontsize=14) 
            plt.ylabel('Price', fontweight='bold', fontsize=14)
            plt.legend()
            plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  # Enable gridlines
            st.pyplot(fig)

            # Closing Price with 100 and 200-day Moving Average
            st.subheader('Closing Price V/s Time Chart With 100 & 200 Day Moving Average')
            ma200 = df['Close'].rolling(200).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], 'darkcyan', label='Close')
            plt.plot(ma100, 'black', label='100-Day MA')
            plt.plot(ma200, 'red', label='200-Day MA')
            plt.xlabel('Time', fontweight='bold', fontsize=14) 
            plt.ylabel('Price', fontweight='bold', fontsize=14)
            plt.legend()
            plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  # Enable gridlines
            st.pyplot(fig)

            # # Candlestick with Volume Chart using Altair
            # st.subheader(f'Candlestick Chart with Volume for {ticker} from {start_date} to {end_date}')
            # candlestick = alt.Chart(df).mark_rule().encode(
            #     x='Date:T',
            #     y='Low:Q',
            #     y2='High:Q',
            #     color=alt.condition("datum.Open < datum.Close", alt.value("green"), alt.value("red"))
            # ).properties(
            #     width=700,
            #     height=400
            # )

            # bars = alt.Chart(df).mark_bar().encode(
            #     x='Date:T',
            #     y='Volume:Q',
            #     color=alt.condition("datum.Open < datum.Close", alt.value("green"), alt.value("red"))
            # ).properties(
            #     width=700
            # )

            # rule = alt.Chart(df).mark_bar(size=1).encode(
            #     x='Date:T',
            #     y='Open:Q',
            #     y2='Close:Q',
            #     color=alt.condition("datum.Open < datum.Close", alt.value("green"), alt.value("red"))
            # ).properties(
            #     width=700,
            #     height=400
            # )

            # # Combine candlestick and volume chart
            # combined_chart = alt.vconcat(
            #     candlestick + rule,
            #     bars
            # ).resolve_scale(
            #     x='shared'
            # ).interactive()

            # st.altair_chart(combined_chart, use_container_width=True)

            # Splitting data into Training and Testing
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

            # Preprocessing and Scaling
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            # Load trained model
            model = load_model('keras_model.h5')

            # Prepare test data for prediction
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            # Model prediction
            y_predicted = model.predict(x_test)
            scaler = scaler.scale_

            scale_factor = 1/scaler[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Final Graph for Predictions
            st.subheader('Predictions V/s Original')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'darkcyan', label='Original Price')
            plt.plot(y_predicted, 'magenta', label='Predicted Price')
            plt.xlabel('Time', fontweight='bold', fontsize=14) 
            plt.ylabel('Price', fontweight='bold', fontsize=14)
            plt.legend()
            plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  # Enable gridlines
            st.pyplot(fig2)

        else:
            st.error(f"No data found for ticker '{ticker}' in the specified date range")

    except Exception as e:
        st.error(f"Error fetching data for ticker '{ticker}': {e}")

else:
    st.write("Please enter a stock ticker to display data.")
