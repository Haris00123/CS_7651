import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score

class nn_support:

    def init(self):
        pass

    def load_hotel(self):
        '''Load Hotel Dataset
        
        Returns:
        X (np.array): X array
        Y (np.array): Y array
        col_index (dict): Dictionary containing the pairing for the column location and it's name'''
        
        #PLEASE CHANGE TO LOCATION OF YOUR HOTEL DATA
        df=pd.read_csv('Data/Hotel_Res/Hotel Reservations.csv')
        df=df.drop('arrival_year',axis=1) #Remove Year
        df['season']=df['arrival_month'].apply(lambda x:1 if (x>=4 and x<=9) else 0) #0-Summer, 1-Winter
        df.drop('arrival_month',inplace=True,axis=1)
        df['point_in_month']=df['arrival_date'].apply(lambda x:1 if (x<=15) else 0) #0-eralier in the month,1-later

        #Removing Booking ID as it not a usefull feature
        df=df.drop('Booking_ID',axis=1)

        #Creating the labels (Y)
        Y=np.array(df['booking_status'].apply(lambda x:0 if x=='Not_Canceled' else 1)) #0 if not canceled, 1 if canceled

        df.drop('booking_status',inplace=True,axis=1)

        #Creating the feature vector, X 
        label_columns=['booking_status']
        categorical_columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        non_categorical_variables=list(set(df.columns).difference(set(categorical_columns+label_columns)))
        X=np.array(df[non_categorical_variables])
        columns_categorized=non_categorical_variables

        #Now we need to one hot vectorize the type_of_meal_plan, room_type_reserved and market_segment_type
        label_dict={}
        for i in categorical_columns:
            label_dict[i]=OneHotEncoder()
            res=label_dict[i].fit_transform(np.array(df[i]).reshape(-1,1)).toarray()
            X=np.c_[X,res]
            columns_categorized=columns_categorized+[i+'%'+j for j in ['1','2','3','4','5','6','7'][:res.shape[-1]]]

            col_index={}
            results_corr={}
            for label,col in zip(columns_categorized,range(X.shape[-1])):
                corr=scipy.stats.pearsonr(X[:,col],Y)[0]
                results_corr[label]=corr
                col_index[label]=col
        return [X,Y]

    def load_heart(self):

        '''Load Heart Disease Dataset
        
        Returns:
        X (np.array): X array
        Y (np.array): Y array
        col_index (dict): Dictionary containing the pairing for the column location and it's name'''

        #PLEASE CHANGE TO LOCATION OF YOUR HEART DATA
        df=pd.read_csv('Data/Heart/heart.csv')
        Y=np.array(df['HeartDisease'])
        df.drop('HeartDisease',axis=1,inplace=True)
        
        label_columns=['HeartDisease']
        categorical_columns=['Sex', 'ChestPainType', 'RestingECG','ExerciseAngina','ST_Slope']

        non_categorical_variables=list(set(df.columns).difference(set(categorical_columns+label_columns)))
        X=np.array(df[non_categorical_variables])
        columns_categorized=non_categorical_variables

        #Now we need to one hot vectorize the type_of_meal_plan, room_type_reserved and market_segment_type
        label_dict={}
        for i in categorical_columns:
            label_dict[i]=OneHotEncoder()
            res=label_dict[i].fit_transform(np.array(df[i]).reshape(-1,1)).toarray()
            X=np.c_[X,res]
            columns_categorized=columns_categorized+[i+'%'+j for j in ['1','2','3','4','5','6','7'][:res.shape[-1]]]

            col_index={}
            results_corr={}
            for label,col in zip(columns_categorized,range(X.shape[-1])):
                corr=scipy.stats.pearsonr(X[:,col],Y)[0]
                results_corr[label]=corr
                col_index[label]=col
        return [X,Y]


    def split_data(self,X,Y,valid=True,standardize=False):

        '''
        Split the data between train, test and optional validation dataset

        Parameters:
        X (np.array): X features
        Y (np.rray): Labels
        valid (bool): Split into validation dataset 
        standardize (bool): Whether to standardize the data (introduces bias as Sklearn Standard Scaler is trained only on the train data)

        Returns:
        train (list): np.array list of train
        valid (list): optional np.array list of validation
        test (list): np.array list of test
        '''
        
        #Now let's split the data between test and train, we'll use the standard 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        
        if valid:
            #We'll also split the data between train and validation, we'll again use the standard 80/20 split
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,random_state=42)
            
            if standardize:
                sklr=StandardScaler()
                X_train=sklr.fit_transform(X_train)
                X_valid=sklr.transform(X_valid)
                X_test=sklr.transform(X_test)
            return [X_train,y_train],[X_valid,y_valid],[X_test,y_test]

        if standardize:
            sklr=StandardScaler()
            X_train=sklr.fit_transform(X_train)
            X_test=sklr.transform(X_test)
        return [X_train,y_train],[X_test,y_test]

    
    def standardize_data(self,train,test):
        sklr=StandardScaler()

        train_standardized=train.copy()
        test_standardized=test.copy()

        train_standardized[0]=sklr.fit_transform(train_standardized[0])
        test_standardized[0]=sklr.transform(test_standardized[0])

        return [train_standardized,test_standardized]