#Here is an Jupyter code for machine learning to predict ship stability during an inclining test using Python and scikit-learn library:

#python
#Copy code
# Import necessary libraries
import pandas as pd
import csv
import math
import base64
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define a dictionary to store the session state values
if 'button_pressed' not in st.session_state:
    st.session_state.button_pressed = False
    
# Load the dataset
@st.cache(allow_output_mutation=True)

        
def fetch_data():
        sheet_id ='d/1wLXZ4zRpTlixClfHejjNbqX9KyyTMHVFqHztn630hAs'
        xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/e/2PACX-1vSzJ2McdS3aIboBFt0MaFuwPxONxqOOr6wr3BPDoftmdAA7NR-nfqwdBNRzB8jpvmeBt5tfdJZzj4WU/pub?output=xlsx")
        data = pd.read_excel(xls , 'Usesheet' , header = 0)
        return data

data = fetch_data()



# Predict stability for a new inclining test
#make the interface
st.title("Ship inclining prediction Ver 1.7 (ANN)")
st.write("""### How to use: """)
st.write("""##### 1. this only applicable to monohull""")
st.write("""##### 2. only 4 and 6 weight method used on this inclining test  """)
st.write("""##### 3. each weight must have simmiliar (or close enough) weight """)
st.write("""##### 4. inclining weight must placed symetrical and no further than ship's half breadth """)
st.write("""##### 5. the result of this app is inclining angle during 4 or 6 weight method process """)
st.write("""##### 6. To prevent error, dont put "0" on ship dimension unless "Number Weight" also "0"  """)
st.write("""##### 7. Inclining test is based on BKI rules "Guidance for inclining test 2015" """)
st.markdown("For more information about BKI rules, [you can click on here](https://rules-api.bki.co.id/v1/publication?path=cHVibGljYXRpb25zL0d1aWRhbmNlL1BhcnQgNi4gU3RhdHV0b3J5LyggVm9sIEMgKSBQZXR1bmp1ayBQZW5ndWppYW4gS2VtaXJpbmdhbiBkYW4gUGVyaW9kZSBPbGVuZyBLYXBhbC8yMDE1Lzc3X1BldHVuanVrIFBlbmd1amlhbiBLZW1pcmluZ2FuIGRhbiBQZXJpb2RlIE9sZW5nIEthcGFsXzIwMTUtNl8zLnBkZg,,&act=view&app=252f31d48ff053e3a7bba35251ad78d1).")
st.write("""### We need some data to predict ship inclining angle""")

#input the new data here
#input some choose answer
beban = (
        "0",
        "4",
        "6",
        )
# input some wrrited answer

Lwl = st.number_input("Length Water Line (m)",min_value= 0.00, step =0.01)
Breadth = st.number_input("Breadth Water Line (m)", min_value= 0.00, step =0.01)
Depth = st.number_input("Depth  (m) ", min_value= 0.00, step =0.01)
Draft = st.number_input("Draft (m) ", min_value= 0.00, max_value= Depth, step =0.01)
Cb = st.number_input("Coefficient Block", min_value= 0.00, max_value= 1.00, step =0.01)
jumlah_beban = st.selectbox("Number Weight", beban)

##convert to anohter

if jumlah_beban == "4" :
        st.image('https://drive.google.com/uc?id=14WooQDOBkHLHqVkCy6wFwLfzb6Jxq8YS')
        st.write("""##### please make sure if every weight at one side, at least it inclined 1 degree""")
        bebanA = st.number_input("Weight 1 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanB = st.number_input("Weight 2 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanC = st.number_input("Weight 3 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanD = st.number_input("Weight 4 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanE = 0
        bebanF = 0
        

if jumlah_beban == "6" :
        st.image('https://drive.google.com/uc?id=1BqM-jtRUqNR5w9NNU2teF4R5qYJ2GI7D')
        st.write("""##### please make sure if every weight at one side, at least it inclined 1 degree""")
        bebanA = st.number_input("Weight 1 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanB = st.number_input("Weight 2 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanC = st.number_input("Weight 3 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanD = st.number_input("Weight 4 (Kg)",min_value= 0.0000,  step =0.0001)
        bebanE = st.number_input("Weight 5 (Kg)",min_value= 0.0000, step =0.0001)
        bebanF = st.number_input("Weight 6 (Kg)",min_value= 0.0000, step =0.0001)
        
        

ok = st.button("Calculate Incline")       
if ok:
    st.session_state.button_pressed = True

    #start machine learning process
    #Define mape
    def calculate_mape(actual, predicted):
        errors = np.abs(actual - predicted)
        denominator = np.abs(actual)
    
        # Handle cases where denominator is zero
        denominator[denominator == 0] = 0.01  # Convert zeros to other value to avoid division by zero

        # Calculate MAPE
        mape = np.nanmean(errors / denominator) * 100

        # Convert mape to a string
        mape_str = f"{mape:.2f}"

        return mape_str
    #start machine learning process
    # chnge some data into numeric
    

    #Select the features and target variable
    features = ['beban/disp', 'Cb' , 'cogm', 'B/T',]
    target = 'Inclinement'

    # Split the dataset into features (X) and target variable (y)
    X = data[features]
    y = data[target]
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize your data (important for neural networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(64, input_dim=len(features), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    # Now, all_predictions_best_model and all_predictions_model contain the predictions for all data points in your dataset
    all_predictions = model.predict(X)


    # Calculate MAPE
    def calculate_mape(y_true, y_pred):
        errors = np.abs(y_true - y_pred)
        denominator = np.abs(y_true)

        # Handle cases where denominator is zero
        denominator[denominator == 0] = 0.01  # Convert zeros to a small number to avoid division by zero

        # Calculate MAPE
        mape = np.mean(errors / denominator) * 100
        return mape
    
    y_true = data['inclinement']
    y_pred = data['all_predictions']
    
    # Calculate MAE, MAPE, and MSE
    mae = mean_absolute_error(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print('Mean Absolute Error (MAE):', mae)
    print('Mean Absolute Percentage Error (MAPE):', mape)
    print('Mean Squared Error (MSE):', mse)


    
if st.session_state.button_pressed:
        if jumlah_beban =="0" :
                
                # Print MAPE and else
                st.subheader(f"Mean squared error is {mse}  " )
                st.subheader(f"Mean Absolute Percentage Error is {mape}")
                # Create a download link
                # Get the values from the 'groups' column
                # Load the specific sheet
                xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/e/2PACX-1vSzJ2McdS3aIboBFt0MaFuwPxONxqOOr6wr3BPDoftmdAA7NR-nfqwdBNRzB8jpvmeBt5tfdJZzj4WU/pub?output=xlsx")
                worksheet_name = 'Usesheet'  # Replace with the actual name of your sheet
                worksheet = xls.parse(sheet_name=worksheet_name)
                groups = worksheet['groups'].tolist()

                # Create a DataFrame
                datap = {'Actual': y, 'Predicted': all_predictions_model}
                dg = pd.DataFrame(datap)
                predictions_dg = pd.DataFrame({'Group' : groups, 'Actual':y, 'Predicted':all_predictions_model})
                predictions_dg.to_csv( index=False, sep='|')
                st.success("Predictions saved to predictions.csv")
                
                def create_download_link(dg, filename="predictions.csv"):
                    csv_content = dg.to_csv(index=False, sep='|')  # Assuming '|' as separator
                    b64 = base64.b64encode(csv_content.encode()).decode()  # Encoding the CSV file
                    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
                    return href

                # Display the link
                st.markdown(create_download_link(predictions_dg), unsafe_allow_html=True)
            
        else:
                halfBreadth = Breadth/2
                #transfer weight
                beban_A = bebanA * halfBreadth
                beban_B = bebanB * halfBreadth
                beban_C = bebanC * halfBreadth
                beban_D = bebanD * halfBreadth
                beban_E = bebanE * halfBreadth
                beban_F = bebanF * halfBreadth
                totalB = (bebanA + bebanB + bebanC + bebanD + bebanE + bebanF)
        
                #calculate displacement
                displacement = (Lwl * Breadth * Draft * Cb)

                totdisp = (totalB / displacement)
                

                #calculate moment
                if jumlah_beban == "4" :
                # 4 pembebanan
                        kanan1 = (beban_B + beban_D)
                        kanan2 = (beban_D)
                        kanan3 = 0
                        kanan4 = (beban_A)
                        kanan5 = (beban_A + beban_C)
                        kanan6 = (beban_A + beban_B + beban_C)
                        kanan7 = (beban_A + beban_B + beban_C + beban_D)
                        kanan8 = (beban_B + beban_C + beban_D)
                        kanan9 = (beban_B + beban_D) 

                        kiri1 = (beban_A + beban_C)
                        kiri2 = (beban_A + beban_B + beban_C)
                        kiri3 = (beban_A + beban_B + beban_C + beban_D)
                        kiri4 = (beban_B + beban_C + beban_D)
                        kiri5 = (beban_B + beban_D)
                        kiri6 = (beban_D)
                        kiri7 = 0
                        kiri8 = (beban_A)
                        kiri9 = (beban_A + beban_C)
                        
                        
                
                if jumlah_beban == "6" :

                        kanan1 = (beban_A + beban_C + beban_E)
                        kanan2 = (beban_A + beban_B + beban_C + beban_E)
                        kanan3 = (beban_A + beban_B + beban_C + beban_D + beban_E + beban_F)
                        kanan4 = (beban_A + beban_B + beban_C + beban_D + beban_E )
                        kanan5 = (beban_A + beban_C + beban_E)
                        kanan6 = (beban_E)
                        kanan7 = 0
                        kanan8 = (beban_C + beban_E)
                        kanan9 = (beban_A + beban_C + beban_E)

                        kiri1 = (beban_B + beban_D + beban_F)
                        kiri2 = (beban_D + beban_F)
                        kiri3 = 0
                        kiri4 = (beban_F)
                        kiri5 = (beban_B + beban_D + beban_F)
                        kiri6 = (beban_A + beban_B + beban_C + beban_D + beban_F )
                        kiri7 = (beban_A + beban_B + beban_C + beban_D + beban_E + beban_F)
                        kiri8 = (beban_A + beban_B + beban_D + beban_F)
                        kiri9 = (beban_B + beban_D + beban_F)
                    
                     
                #finding ratio
                if Breadth == 0:
                        LB = 0
                else :
                        LB = (Lwl /Breadth) 

                if Draft == 0:
                        BT = 0
                        DT = 0
                else :
                        BT = (Breadth /Draft) 
                        DT = (Depth / Draft)

                    
                Mselisih1 =  (kiri1 - kanan1) 
                Mselisih2 =  (kiri2 - kanan2)  
                Mselisih3 =  (kiri3 - kanan3) 
                Mselisih4 =  (kiri4 - kanan4) 
                Mselisih5 =  (kiri5 - kanan5) 
                Mselisih6 =  (kiri6 - kanan6) 
                Mselisih7 =  (kiri7 - kanan7) 
                Mselisih8 =  (kiri8 - kanan8)
                Mselisih9 =  (kiri9 - kanan9)

                MD1 = (Mselisih1 / displacement)
                MD2 = (Mselisih2 / displacement)
                MD3 = (Mselisih3 / displacement)
                MD4 = (Mselisih4 / displacement)
                MD5 = (Mselisih5 / displacement)
                MD6 = (Mselisih6 / displacement)
                MD7 = (Mselisih7 / displacement)
                MD8 = (Mselisih8 / displacement)
                ND9 = (Mselisih9 / displacement)

                
                new_test1 = pd.DataFrame({'Moment': [Mselisih1], 'displacement': [displacement],'B/T' :[BT] , 'Cb': [Cb], })
                predicted_Incline1 = best_model.predict(new_test1)
        
                new_test2 = pd.DataFrame({'Moment': [Mselisih2], 'displacement': [displacement],'B/T' :[BT], 'Cb': [Cb], })
                predicted_Incline2 = best_model.predict(new_test2)
        
                new_test3 = pd.DataFrame({'Moment': [Mselisih3], 'displacement': [displacement],'B/T' :[BT], 'Cb': [Cb],})
                predicted_Incline3 = best_model.predict(new_test3)
        
                new_test4 = pd.DataFrame({'Moment': [Mselisih4], 'displacement': [displacement],'B/T' :[BT], 'Cb': [Cb], })
                predicted_Incline4 = best_model.predict(new_test4)
        
                new_test5 = pd.DataFrame({ 'Moment': [Mselisih5], 'displacement': [displacement],'B/T' :[BT], 'Cb': [Cb], })
                predicted_Incline5 = best_model.predict(new_test5)
        
                new_test6 = pd.DataFrame({'Moment': [Mselisih6], 'displacement': [displacement],'B/T' :[BT], 'Cb': [Cb],})
                predicted_Incline6 = best_model.predict(new_test6)
        
                new_test7 = pd.DataFrame({'Moment': [Mselisih7], 'displacement': [displacement],'B/T' :[BT], 'Cb': [Cb],})
                predicted_Incline7 = best_model.predict(new_test7)
        
                new_test8 = pd.DataFrame({'Moment': [Mselisih8], 'displacement': [displacement],'B/T' :[BT], 'Cb': [Cb],})
                predicted_Incline8 = best_model.predict(new_test8)

                new_test9 = pd.DataFrame({'Moment': [Mselisih9], 'displacement': [displacement],'B/T' :[BT], 'Cb': [Cb],})
                predicted_Incline9 = best_model.predict(new_test9)

                ##Convert into rad and tan θ
                radians1 = math.radians(predicted_Incline1[0])
                radians2 = math.radians(predicted_Incline2[0])
                radians3 = math.radians(predicted_Incline3[0])
                radians4 = math.radians(predicted_Incline4[0])
                radians5 = math.radians(predicted_Incline5[0])
                radians6 = math.radians(predicted_Incline6[0])
                radians7 = math.radians(predicted_Incline7[0])
                radians8 = math.radians(predicted_Incline8[0])
                radians9 = math.radians(predicted_Incline9[0])

                tantheta1 = math.tan(radians1)
                tantheta2 = math.tan(radians2)
                tantheta3 = math.tan(radians3)
                tantheta4 = math.tan(radians4)
                tantheta5 = math.tan(radians5)
                tantheta6 = math.tan(radians6)
                tantheta7 = math.tan(radians7)
                tantheta8 = math.tan(radians8)
                tantheta9 = math.tan(radians9)
                
                st.subheader(f"Mean squared error is {mse_best_model}  " )
                st.subheader(f"Mean Absolute Percentage Error is {mape_model}")
            
                dataS = pd.DataFrame({
                        'Moment Beban (Kg.m)': [Mselisih1, Mselisih2, Mselisih3, Mselisih4, Mselisih5, Mselisih6, Mselisih7, Mselisih8, Mselisih9],
                        'incline (degrees)': [predicted_Incline1[0], predicted_Incline2[0], predicted_Incline3[0], predicted_Incline4[0], predicted_Incline5[0], predicted_Incline6[0], predicted_Incline7[0], predicted_Incline8[0], predicted_Incline9[0]],
                        'incline (tan θ)' : [tantheta1, tantheta2, tantheta3, tantheta4, tantheta5, tantheta6, tantheta7, tantheta8, tantheta9]
                                   
                        })
                dataS_display = dataS.copy()
                dataS_display['Moment Beban (Kg.m)'] = dataS['Moment Beban (Kg.m)']
                dataS_display['incline (degrees)'] = dataS['incline (degrees)'].apply(lambda x: '{:.2f}'.format(x))
                dataS_display['incline (tan θ)'] = dataS['incline (tan θ)'].apply(lambda x: '{:.3f}'.format(x))

                st.table(dataS_display)
            
               
                # make graphics
                fig, ax = plt.subplots()

                # Create a scatter plot
                scatter = ax.scatter(dataS['Moment Beban (Kg.m)'], dataS['incline (tan θ)'], color='blue', label='Incliing result')
        
                # Set title, labels, and legend
                ax.set_title("Inclining graphic")
                ax.set_xlabel('Moment Beban (Kg.m)')
                ax.set_ylabel('incline (tan θ)')
                ax.legend()

                # Add annotations
                for i in range(len(dataS)):
                        ax.annotate(i, (dataS['Moment Beban (Kg.m)'].iloc[i], dataS['incline (tan θ)'].iloc[i])) # i+1 because Python's indexing starts at 0

                ax.set_xlabel('Moment Beban (Kg.m)')
                ax.set_ylabel('incline (tan θ)')

                # Customization: draw a vertical line (you can adjust this as per your requirement)
                threshold = dataS['Moment Beban (Kg.m)'].mean()  # example threshold using mean, adjust as needed
                ax.axvline(x=0, color='red', linestyle='--', label=" 0,0 coordinate")
                ax.legend()

                # Customization: draw a horizontal  line (you can adjust this as per your requirement)
                thresholds = dataS['incline (tan θ)'].mean()  # example threshold using mean, adjust as needed
                ax.axhline(y=0, color='red', linestyle='--', label="")
                ax.legend()

                # Display the plot in Streamlit
                st.pyplot(fig)

                # Plotting feature importances
                imp, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(importances)), importances[sorted_indices], align='center')
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels(np.array(features)[sorted_indices])
                ax.set_title("Feature Importances")
                ax.set_ylabel('Importance')
                ax.set_xlabel('Features')

            ###0.24554285714285714285714285714286

                st.pyplot(imp)  # Pass the figure object to st.pyplot()
    
