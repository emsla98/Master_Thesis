from sklearn import preprocessing, base, decomposition, linear_model

import pandas as pd
import numpy as np

from Scripts import my_functions as mf
from tqdm import tqdm

class Storage:

    def __init__(self):
        pass

    def load_data(dataframe):
        """
        Method for loading data into the Storage class.
        """
        # Rename:
        df = dataframe
        
        # Load all data and store in Storage class:
        Storage.z_data   = df.filter(regex="z_")
        Storage.u_data   = df.filter(regex="^u_")
        Storage.v_data   = df.filter(regex="^v_")
        Storage.bcn_data = df.filter(regex="bcn_")
        Storage.bcs_data = df.filter(regex="bcs_")
        Storage.wu_data  = df.filter(regex="wu_")
        Storage.wv_data  = df.filter(regex="wv_")
        
        Storage.n_samples = len(df)
    
    
    def get_active_columns(column_names, state_variables):
        """
        Method for extracting active columns.
        """
    
        # Allocate memory:
        active_cols = np.zeros_like(column_names, dtype="bool")

        # Loop over state variables:
        for var in state_variables:

            # (Boolean) addition of active columns for each state variable:
            active_cols += np.array([True if var in i else False for i in column_names])

        return active_cols
    
    
    def get_features(variables):
        """
        Method for extracting and concatenating features.
        """
        
        if len(variables) == 0:
            feats = None
        
        else:
        
            # Allocate memory for features:
            feats_list = []

            # Iterate over variables (In "correct" order):
            if "z" in variables:
                feats_list.append(Storage.z_data)
            if "u" in variables:
                feats_list.append(Storage.u_data)
            if "v" in variables:
                feats_list.append(Storage.v_data)
            if "bcn" in variables:
                feats_list.append(Storage.bcn_data)
            if "bcs" in variables:
                feats_list.append(Storage.bcs_data)
            if "wu" in variables:
                feats_list.append(Storage.wu_data)
            if "wv" in variables:
                feats_list.append(Storage.wv_data)

            # Concatenate features column-wise:
            feats  = pd.concat(feats_list,  axis=1)
        
        return feats
    
    
    def get_window_features(features, lag=None, lead=None):
        
        if (lag==None and lead==None):
            AssertionError("Either lag and/or lead must be positive")
        
        if lag == None:
            lag = 0
        
        if lead == None:
            lead = 0
        
        # Compute number of samples:
        n_samples, n_columns = features.shape
        
        # Allocate memory:
        window_df = pd.DataFrame(index=range(n_samples-lag-lead), 
                                 columns=range(n_columns*(lag+lead)))
        
        # Loop over samples:
        for i in range(n_samples):
            
            # Disregard lagged rows outside matrix:
            if i-lag < 0:
                pass
            
            # Disregard leading rows outside matrix:
            elif i+lead > n_samples-1:
                pass
            
            else:
                
                window_df.loc[i,:] = np.concatenate([features[k] for k in range(i-lag, i+lead)])
                
        
        return window_df
                    
        
    def get_regressor(regressor):
        
        match regressor:
            case "linear":
                reg = linear_model.LinearRegression()
            case "ridge_10":
                reg = linear_model.Ridge(alpha=10)
            case "ridge_2":
                reg = linear_model.Ridge(alpha=2)
            case "ridge_1":
                reg = linear_model.Ridge(alpha=1)
            case "ridge_01":
                reg = linear_model.Ridge(alpha=0.1)   
            case _:
                raise AssertionError("The chosen regressor was not found")
        
        return reg
    
class MyModels:

    def __init__(self, df, scaler, latent_space):
        """
        Upon initialization data is loaded and all scalers 
        and latent spaces are precomputed and stored such 
        that all submodels may utilize them if needed.
        
        Precomputation (fitting) of scaler and latent space
        are done using a subset of the whole data.
        """
        
        # Settings:
        n_every  = 10
        max_dim  = min(500, len(df) // n_every)
        bat_size = 512
        
        # Load data into Storage class:
        Storage.load_data(df)
        
        
        # Concatenate all state data:
        state_data = pd.concat([Storage.z_data, 
                                Storage.u_data, 
                                Storage.v_data],axis=1)
        
        # Append column names:
        Storage.columns = state_data.columns
        
        # Extract subset of state data:
        state_data_fit = state_data[::n_every]
        
        
        # Find matching scaler:
        match scaler:
            case "standard":
                scaler = preprocessing.StandardScaler()
            case _:
                scaler = None
        
        # Find matching latent space:
        match latent_space:
            case "pca":
                latent_space = decomposition.IncrementalPCA(
                                    n_components = max_dim,
                                    batch_size   = bat_size)
            case _:
                latent_space = None
        
        
        # Fit scaler:
        if scaler is not None:
            state_data_fit = scaler.fit_transform(state_data_fit)
        
        # Fit latent space:
        if latent_space is not None:
            latent_space.fit(state_data_fit)
            Storage.max_dim = max_dim
        
        
        # Append scaler and latent space to self object.
        Storage.scaler = scaler
        Storage.latent_space = latent_space
        
        print("Init was run.")
        
    
    class Template:
        """
        This is a basic model template.
        """

        def fit(self, X=None, y=None):
            """
            Placeholder for fitting the model.

            Args:
                X: Input features for training.
                y: Target labels for training.
            """
            raise NotImplementedError("Subclasses must implement the 'fit' method.")

        def predict(self, X=None, y=None):
            """
            Placeholder for making predictions.

            Args:
                X: Input features for making predictions.

            Returns:
                Predicted labels.
            """
            raise NotImplementedError("Subclasses must implement the 'predict' method.")
 
    
    class RegressionModel(Template):
        

        def __init__(self, regressor=None, inputs=None, outputs=None,
                     wind_conditions=False, boundary_conditions=False,
                     state_lag=1, extra_lag=0, extra_lead=1,
                     latent_dim=None, n_train=None, n_test=None):
           
           # Object initialization:
            self.inputs = inputs
            self.outputs = outputs
            self.wind_conditions = wind_conditions
            self.boundary_conditions = boundary_conditions
            self.state_lag = state_lag
            self.extra_lag = extra_lag
            self.extra_lead = extra_lead
            self.latent_dim = latent_dim
            self.n_train = n_train
            self.n_test = n_test

            scaler = Storage.scaler
            latent_space = Storage.latent_space

            # Check if either scaler or latent space is present:
            if not (scaler is None and latent_space is None):
                columns = Storage.columns

                # Get active columns if needed if needed:
                if inputs is not None:
                    active_cols = Storage.get_active_columns(columns, inputs)
                    self.active_cols_in = active_cols

                if outputs is not None:
                    active_cols = Storage.get_active_columns(columns, outputs)
                    self.active_cols_out = active_cols

            # Get scaler if needed:
            if scaler is not None:
                self.scaler_in = base.clone(scaler)
                self.scaler_out = base.clone(scaler)

                self.scaler_in.mean_ = scaler.mean_[self.active_cols_in].copy()
                self.scaler_in.scale_ = scaler.scale_[self.active_cols_in].copy()

                self.scaler_out.mean_ = scaler.mean_[self.active_cols_out].copy()
                self.scaler_out.scale_ = scaler.scale_[self.active_cols_out].copy()

            # Get latent space if needed:
            if latent_space is not None:

                self.latent_space_in = base.clone(latent_space)
                self.latent_space_out = base.clone(latent_space)

                self.latent_space_in.mean_ = latent_space.mean_[self.active_cols_in].copy()
                self.latent_space_in.components_ = latent_space.components_[:latent_dim, self.active_cols_in].copy()

                self.latent_space_out.mean_ = latent_space.mean_[self.active_cols_out].copy()
                self.latent_space_out.components_ = latent_space.components_[:latent_dim, self.active_cols_out].copy()

            # Get regressor:
            self.regressor = Storage.get_regressor(regressor)

            # Get input data:
            if inputs is not None:
                self.input_data = Storage.get_features(inputs)
                
            else:
                self.input_data = None

            # Get output data:
            if outputs is not None:
                self.output_data = Storage.get_features(outputs)
            
            # Get extra input data:
            if not (wind_conditions is False and boundary_conditions is False):
                
                self.extras = []

                if wind_conditions is True:
                    self.extras.extend(["wu", "wv"])
                
                if boundary_conditions is True:
                    self.extras.extend(["bcn", "bcs"])

                self.extra_data = Storage.get_features(self.extras)
            
            else:
                self.extra_data = None

            return

        def generate_X_y(self, X=None, y=None):
        
            # Check if input data is present:
            if X is not None:

                # Check if scaling is needed:
                if self.scaler_in is not None:
                    X = self.scaler_in.transform(X)

                # Check if projection is needed:
                if self.latent_space_in is not None:
                    X = self.latent_space_in.transform(X)
                
                # Check if lagged state features are needed:
                if self.state_lag >= 1:
                    X = Storage.get_window_features(X, self.state_lag)
            

            # Check if extra input data is present:
            if self.extra_data is not None:
                    
                # Check if lagged extra features are needed:
                if self.extra_lag >= 1 or self.extra_lead >= 1:
                    self.extra_data = Storage.get_window_features(
                        self.extra_data, self.extra_lag, self.extra_lead)
            
            # Combine input and extra input:
            if not (self.inputs is None and len(self.extras) == 0):
                X = pd.concat([X, self.extra_data], axis=1).dropna()
            

            # Check if output data is present:
            if y is not None:

                # Check if scaling is needed:
                if self.scaler_out is not None:
                    y = self.scaler_out.transform(y)

                # Check if projection is needed:
                if self.latent_space_out is not None:
                    y = self.latent_space_out.transform(y)

                y = pd.DataFrame(y).loc[X.index]
            
            return X, y


        def fit2(self, X=None, y=None):
            
            # Check if input data is present:
            if X is None:
                X = self.input_data
            
            # Check if output data is present:
            if y is None:
                y = self.output_data
            
            # Generate X and y:
            X, y = self.generate_X_y(X, y)

            # Check if n_train is None:
            if self.n_train is None:

                # Default training sample to 25% of data:
                self.n_train = Storage.n_samples // 4

            # Check if n_train is a fraction:
            elif self.n_train > 0 and self.n_train < 1:

                # Set training sample to fraction of data:
                self.n_train = Storage.n_samples * self.n_train

            # Raise error if n_train is incorrect:
            else:
                raise AssertionError("n_train must be a fraction in interval (0, 1)")
                
            # Convert n_train to integer:
            self.n_train = int(self.n_train)

            # Define training data:
            X_train = X.iloc[:self.n_train]
            y_train = y.iloc[:self.n_train]

            # Fit model:
            self.regressor.fit(X_train, y_train)

            # Attach data to object:
            self.X = X
            self.y = y

            return self

        def predict2(self, X=None, y=None, n_test=None):

            if X is None:
                X = self.input_data

            if y is None:
                y = self.output_data

            if n_test is None:
                n_test = len(y)

            # Allocate memory for predictions:
            y_pred = y.copy() * 0
            X_pred = X.copy() * 0

            # Get active columns:
            if self.inputs is not None:
                self.active_cols_in = Storage.get_active_columns(Storage.columns, self.inputs)
            else:
                self.active_cols_in = Storage.columns * False

            if self.outputs is not None:
                self.active_cols_out = Storage.get_active_columns(Storage.columns, self.outputs)
            else:
                self.active_cols_out = Storage.columns * False
            
            # Check if scaling is needed:
            if self.scaler_in is not None:
                scale_in = True

            if self.scaler_out is not None:
                scale_out = True

            if self.latent_space_in is not None:
                project_in = True
            
            if self.latent_space_out is not None:
                project_out = True

            

            # Create placeholder for values:
            test = np.zeros(shape=(Storage.columns.shape[0]))

            # Loop over samples:
            for i in tqdm(range(n_test)):
                
                if i < self.state_lag:
                    X_pred.iloc[i] = X.iloc[i]
                    continue 
                
                X_pred_i = X_pred.iloc[i-self.state_lag:i+1]

                # Fill place with in-going values:
                test[self.active_cols_in] = X_pred_i.values[-1]

                if i > self.state_lag:

                    # Overwrite placeholder with predictions:
                    test[self.active_cols_out] = y_pred.iloc[i-1]
                

                # Retrieve in-going values:
                X_pred_i.values[-1] = test[self.active_cols_in]
                
                # Generate X and y:
                X_pred_i, _ = self.generate_X_y(X_pred_i)

                # Make prediction:
                y_pred_i = self.regressor.predict(X_pred_i)
                

                # Check if projection is needed:
                if self.latent_space_out is not None:
                    y_pred_i = self.latent_space_out.inverse_transform(y_pred_i)

                # Check if scaling is needed:
                if self.scaler_out is not None:
                    y_pred_i = self.scaler_out.inverse_transform(y_pred_i)
                

                # Store prediction:
                y_pred.iloc[i,:] = y_pred_i

            # Store predictions:
            self.y_pred = pd.DataFrame(y_pred, index=y.index)

            return self



        def nothing():
            
            if 0:

                ### INPUT CHECKING: ###
                if 1:

                    # Check that a valid regressor was chosen:
                    if regressor is None:
                        raise AssertionError("Regressor is missing.")
                    
                    # Check that valid input exists:
                    if (len(inputs) == 0 and wind_conditions == False and
                        boundary_conditions == False):
                        raise AssertionError("Model input is missing. Choose between state variables, boundary- or wind conditions")

                    # Check that valid output exists:
                    if len(outputs) == 0:
                        raise AssertionError("Model target is missing. Choose between state variables.")
                    
                    # Check if lag of state features is reasonable:
                    if state_lag < 1 or state_lag > 10:
                        raise AssertionError("Number of lagged state features is either too low or too high. \
                                            Should be in interval [1, 10]")
                        
                    
                    # Check that latent space and dimensions are present simultaneously.
                    if (Storage.latent_space is None and latent_dim is None):
                        raise AssertionError("Mismatch between latent_dim and latent_space.")
                    
                    if Storage.max_dim < latent_dim:
                        raise AssertionError(f"Too many latent dimensions. Should be in interval [1, {Storage.max_dim}]") 
                    
                    # Ensure a reasonable number of training samples:
                    if n_train is None:
                        n_train = Storage.n_samples // 4
                    
                    # Check if n_train is a fraction:
                    if n_train > 0 and n_train < 1:
                        n_train = Storage.n_samples * n_train // 1
                        
                    if n_train >= Storage.n_samples - max(state_lag, extra_lag):
                        raise AssertionError("Too many training samples.")
                    
                
                ### ASSOCIATING INPUTS TO CLASS OBJECT: ###
                
                self.inputs = inputs
                self.outputs = outputs
                self.wind_conditions = wind_conditions
                self.boundary_conditions = boundary_conditions
                self.state_lag = state_lag
                self.extra_lag = extra_lag
                self.extra_lead = extra_lead
                self.latent_dim = latent_dim
                self.n_train = n_train
                self.n_test = n_test
                
                self.regressor = Storage.get_regressor(regressor)
                
            
                ### CONCATENATE STATE INPUT AND OUTPUT: ###
                
                # Check if input state variables are needed:
                if len(inputs) != 0:
                    
                    # Extract input state features:
                    self.input_data = Storage.get_features(inputs)
                    self.state_feats_in  = self.input_data.values.copy()
                else:
                    self.input_data = None
                    self.state_feats_in = None
                
                self.output_data = Storage.get_features(outputs)
                self.state_feats_out = self.output_data.values.copy()
                
                
                ### SCALE AND PROJECT STATE FEATURES: ###
                
                scaler       = Storage.scaler
                latent_space = Storage.latent_space
                columns      = Storage.columns
                
                # Check that input data exists:
                if self.input_data is not None:
                    
                    self.active_cols_in = Storage.get_active_columns(columns, inputs)
                    
                    if scaler is not None:
                        self.scaler_in = base.clone(scaler)
                        self.scaler_in.mean_ = scaler.mean_[self.active_cols_in]
                        self.scaler_in.scale_ = scaler.scale_[self.active_cols_in]
                        
                        self.state_feats_in = self.scaler_in.transform(self.state_feats_in)
                        
                    if (latent_space is not None and latent_dim is not None):
                        self.latent_space_in = base.clone(latent_space)
                        self.latent_space_in.mean_ = latent_space.mean_[self.active_cols_in].copy()
                        self.latent_space_in.components_ = latent_space.components_[:latent_dim, self.active_cols_in].copy()
                        
                        self.state_feats_in = self.latent_space_in.transform(self.state_feats_in)
                        
                    else:
                        raise AssertionError("Mismatch between latent_dim and latent_space.")
                

                self.active_cols_out = Storage.get_active_columns(columns, outputs)
                
                if scaler is not None:
                    self.scaler_out = base.clone(scaler)
                    self.scaler_out.mean_ = scaler.mean_[self.active_cols_out]
                    self.scaler_out.scale_ = scaler.scale_[self.active_cols_out]
                
                    self.state_feats_out = self.scaler_out.transform(self.state_feats_out)
                
                if latent_space is not None:
                    self.latent_space_out = base.clone(latent_space)
                    self.latent_space_out.mean_ = latent_space.mean_[self.active_cols_out].copy()
                    self.latent_space_out.components_ = latent_space.components_[:latent_dim, self.active_cols_out].copy()
                    
                    self.state_feats_out = self.latent_space_out.transform(self.state_feats_out)
                        
                
                ### LAG STATE FEATURES: ###
                
                # Check if lagged state features are needed:
                if len(inputs) != 0 and state_lag >= 1:
                    
                    self.state_feats_in = Storage.get_window_features(self.state_feats_in, state_lag)                    
                    
                    
                ### CONCATENATE EXTRA INPUT: ###

                
                # Allocate memory for extra input:
                self.input_extra_vars = []

                # Check if boundary conditions are needed:
                if boundary_conditions == True:
                    
                    self.input_extra_vars.extend(["bcn", "bcs"])
                    
                # Check if wind conditions are needed:
                if wind_conditions == True:
                    
                    self.input_extra_vars.extend(["wu", "wv"])

                # Check if any extra features were added:
                if len(self.input_extra_vars) != 0:

                    # Concatenate features column wise:
                    self.extra_feats_in = Storage.get_features(self.input_extra_vars)

                else:
                    self.extra_feats_in = None
                
                
                ### LAG/LEAD EXTRA INPUT: ###
                
                # Check if extra inputs are needed:
                if self.extra_feats_in is not None:
                    
                    self.extra_feats_in = Storage.get_window_features(self.extra_feats_in, extra_lag, extra_lead)
                    
                
                ### FIX INDICES: ###
                
                # Concatenate data
                self.feats_in = pd.concat([self.state_feats_in, self.extra_feats_in], axis=1).dropna()
                self.feats_out = pd.DataFrame(self.state_feats_out).loc[self.feats_in.index]
                
            return
        
        
        def fit(self, X=None, y=None):
            
            if n_train is None:
                n_train = self.n_train
            
            if (X is not None and y is None) or (X is None and y is not None):
                raise AssertionError("Either choose both X and y manually or let the program do it automatically.")
            
            if X is None and y is None:
                X = self.feats_in
                y = self.feats_out
            
            self.X = X
            self.y = y
            
            self.X_train = X.iloc[:n_train, :]
            self.y_train = y.iloc[:n_train, :]
            
            self.regressor.fit(self.X_train, self.y_train)
            
            return self
            
        
        def predict(self, X=None, n_test=None):
            
            scaler = Storage.scaler
            latent_space = Storage.latent_space
            
            if n_test is None:
                n_test = len(self.y)
            
            if X is None:
                X = self.X.values
            
            self.y_pred = self.y.copy()*0
            
            for i in range(n_test):
                self.y_pred.iloc[i,:] = self.regressor.predict(X[i].reshape(1,-1))

            
            if latent_space is not None:
                self.y_pred = self.latent_space_out.inverse_transform(self.y_pred)
            
            if scaler is not None:
                self.y_pred = self.scaler_out.inverse_transform(self.y_pred)
            
            self.y_id = self.feats_out.index
            
            self.y_pred = pd.DataFrame(self.y_pred,
                                       index=self.y_id)
            
            self.y_true = pd.DataFrame(self.output_data).loc[self.y_id]
            
            self.rmse = mf.rmse(self.y_true, self.y_pred, axis=1)
            self.mae = mf.mae(self.y_true, self.y_pred, axis=1)
            self.mape = mf.mape(self.y_true, self.y_pred, axis=1)
            
            
            