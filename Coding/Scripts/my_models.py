from sklearn import preprocessing, base, decomposition, linear_model, ensemble
from xgboost import XGBRegressor

from copy import copy, deepcopy

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
        
        # Dictionary w/ variables and regexes:
        my_var_dict = {"z"   : "z_", 
                       "u"   : "^u_",
                       "v"   : "^v_",
                       "bcn" : "bcn_",
                       "bcs" : "bcs_",
                       "wu"  : "wu_",
                       "wv"  : "wv_"}
        
        # Create data dictionary:
        Storage.data = {}
        
        # Loop over dictionary items:
        for key, regex in my_var_dict.items():
            
            # Fill data dictionary:
            Storage.data[key] = df.filter(regex=regex)
        
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
            active_cols += np.array([True if var in i else \
                                     False for i in column_names])

        return active_cols   

    def copy_dict_pd(dict_in, keys=None):

        dict_out = {}

        if dict_in is None:
            raise AssertionError("Input dictionary is empty.")

        if keys is None:
            keys = dict_in.keys()
        
        for key in keys:
            cols = dict_in[key].columns
            dict_out[key] = pd.DataFrame(dict_in[key], columns=cols)

        return dict_out

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

    def get_window_features_list(features, lag=None, lead=None):
        
        # Handling of lag and lead input:
        if (lag==None and lead==None):
            AssertionError("Either lag and/or lead must be positive")
        
        if lag == None:
            lag = 0
        
        if lead == None:
            lead = 0
        
        # List of features:
        features_list = []

        # Check if lagged features are needed:
        if lag > 0:

            # Iterate over lag:
            for i in range(1, lag+1)[::-1]:
                features_tmp = features.copy()
                features_tmp.columns = [f"{col}_lag{i}" for col in features_tmp.columns]
                features_list.append(features_tmp.shift(i))

        # Check if leading features are needed:
        if lead > 0:
                
            # Iterate over lead:
            for i in range(0, lead):
                features_tmp = features.copy()

                if i > 0:
                    features_tmp.columns = [f"{col}_lead{i}" for col in features_tmp.columns]
                features_list.append(features_tmp.shift(-i))
        
        return features_list
                         
    def get_regressor(regressor):
        
        match regressor:
            case "linear":
                reg = linear_model.LinearRegression()
            case "ridge_1":
                reg = linear_model.Ridge(alpha=1)
            case "XGBoost":
                #reg = ensemble.GradientBoostingRegressor(verbose=1)  
                reg = XGBRegressor()
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
        
        # Set name of Storage object:
        name = f""

        # State variables:
        state_vars = ["z", "u", "v"]
        
        # Find matching scalers:
        scalers = {}
        
        match scaler:
            
            case "standard":
                name += "scale: std, "
                for var in state_vars:
                    scalers[var] = preprocessing.StandardScaler()
            
            case _:
                scalers = None
        
        # Find matching latent space:
        latent_spaces = {}
        
        match latent_space:
            
            case "pca":
                name += "latent: pca, "
                for var in state_vars:
                    latent_spaces[var] = decomposition.IncrementalPCA(
                                            n_components = max_dim,
                                            batch_size   = bat_size)
            
            case _:
                latent_spaces = None
        
        # Check if fitting data is needed:
        if not (scalers is None and latent_spaces is None):
            
            # Create fit data:
            fit_data = {}
            
            # Fill fit data:
            for key in state_vars:
                fit_data[key] = Storage.data[key][::n_every]
        
            # Fit scaler:
            if scaler is not None:

                for key, obj in scalers.items():
                    fit_data[key] = obj.fit_transform(fit_data[key])

            # Fit latent space:
            if latent_spaces is not None:

                for key, obj in latent_spaces.items():
                    obj.fit(scalers[key].transform(fit_data[key]))

                Storage.max_dim = max_dim
        
        
        # Append scaler and latent space to self object.
        Storage.scalers = scalers
        Storage.latent_spaces = latent_spaces
        
        Storage.name = name
        Storage.name_set = "init"

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
 
        def load_data(self):
            
            # Get input data:
            if self.input is not None:
                
                self.input_data = Storage.data[self.input].copy()

                self.output_data = self.input_data.copy()
                
                return self
        
        def del_data(self):
            
            del self.output_data, self.output_preds, self.input_data
            
            return self
        
        
        def scale_n_proj(self, direction, X=None, y=None):
            """
            Method for scaling and projecting input and output data.

            Args:
                direction: "forward" or "inverse".
                X: Input data (pd.DataFrame).
                y: Output data (pd.DataFrame).
            
            Returns:
                X2: Scaled and projected input data. (pd.DataFrame)
                y2: Scaled and projected output data. (pd.DataFrame)

            """

            # Check if scaling is needed:
            if self.scalers is not None:
                scaler = self.scalers[self.input]
            
            else:
                scaler = None

            # Check if projection is needed:
            if self.latent_spaces is not None:
                latent_space = deepcopy((self.latent_spaces[self.input]))

                latent_space.components_ = latent_space.components_[:self.latent_dim, :]

            else:
                latent_space = None

            # Check if input data is present:
            if X is not None:
                
                # Copy input data:
                X2 = X.copy()

                X_index = X2.index

                # Forward method:
                if direction == "forward":

                    # Get column names from input data:
                    cols = self.input_data.columns

                    # Check if scaling is needed:
                    if scaler is not None:
                        X2 = scaler.transform(X2)
                        X2 = pd.DataFrame(X2,
                            columns=cols)

                    # Check if projection is needed:
                    if latent_space is not None:
                        X2 = latent_space.transform(X2)
                        X2 = pd.DataFrame(X2,
                            columns=cols[:self.latent_dim])

                # Inverse method:
                if direction == "inverse":

                    # Get column names from stored data:
                    cols = self.input_data.columns

                    # Check if projection is needed:
                    if latent_space is not None:
                        X2 = latent_space.inverse_transform(X2)
                        X2 = pd.DataFrame(X2,
                                    columns=cols)

                    # Check if scaling is needed:
                    if scaler is not None:
                        X2 = scaler.inverse_transform(X2)
                        X2 = pd.DataFrame(X2,
                                    columns=cols)

                # Set index:
                X2.index = X_index

            else:
                X2 = None


            # Check if output data is present:
            if y is not None:
                
                y2 = y.copy()
                
                y_index = y2.index

                # Forward method:
                if direction == "forward":

                    # Get column names:
                    cols = self.output_data.columns

                    # Check if scaling is needed:
                    if scaler is not None:
                        y2 = scaler.transform(y2)
                        y2 = pd.DataFrame(y2,
                                    columns=cols)
                        
                    # Check if projection is needed:
                    if latent_space is not None:
                        y2 = latent_space.transform(y2)
                        y2 = pd.DataFrame(y2,
                                    columns=cols[:self.latent_dim])
                        
                        y2 = y2.iloc[:,:self.latent_dim]

                # Inverse method:
                if direction == "inverse":
                    
                    # Get column names from stored data:
                    cols = self.output_data.columns

                    # Check if projection is needed:
                    if latent_space is not None:
                        y2 = latent_space.inverse_transform(y2)
                        y2 = pd.DataFrame(y2,
                                    columns=cols)

                    # Check if scaling is needed:
                    if scaler is not None:
                        y2 = scaler.inverse_transform(y2)
                        y2 = pd.DataFrame(y2,
                                    columns=cols)

                # Set index:
                y2.index = y_index

            else:
                y2 = None
            
            return X2, y2


        def generate_list_pd(self, X=None, y=None):
            
            # Check if input data is present:
            if X is not None:
                
                # Copy input data:
                X_mat = X.copy()
                
                # Check if lagged state features are needed:
                if self.state_lag >= 1:

                    # Return lagged state features as list:
                    X_mat = Storage.get_window_features_list(X_mat, self.state_lag)

                else:
                    # Convert dataframe to list:
                    X_mat = [X_mat]
            else:
                X_mat = []

            # Check if extra input data is present:
            if self.extra_data is not None:
                
                # Copy extra input data:
                extra_mat = self.extra_data.copy() 
            
                # Check if lagged extra features are needed:
                if self.extra_lag >= 1 or self.extra_lead >= 1:

                    # Return lagged extra features as list:
                    extra_mat = Storage.get_window_features_list(extra_mat, self.extra_lag, self.extra_lead)

                else:
                    # Convert dataframe to list:
                    extra_mat = [extra_mat]

            else:
                extra_mat = []

            # Check if output data is present:
            if y is not None:

                # Copy output data:
                y_mat = [y.copy()]

            else:

                y_mat = []

            return X_mat, extra_mat, y_mat


        def plot_errors(self, metric):

            pass


        def compute_errors(self, n_train=None):

            # Compute error metrics:
            data = self.output_data
            pred = self.output_preds

            rmse = mf.rmse(data, pred, axis=0).mean()
            me = mf.me(data, pred, axis=0).mean()
            mae = mf.mae(data, pred, axis=0).mean()
            minae = mf.minae(data, pred, axis=0).min()
            maxae = mf.maxae(data, pred, axis=0).max()

            error_metrics = pd.DataFrame(
                [[rmse, me, mae, minae, maxae]],
                columns=["Mean RMSE", "Mean ME", "Mean MAE", "Min Err", "Max Err"])

            if n_train is not None:
                data_train = data.iloc[:n_train]
                pred_train = pred.iloc[:n_train]
                data_test = data.iloc[n_train:]
                pred_test = pred.iloc[n_train:]
            
                rmse_train = mf.rmse(data_train, pred_train, axis=0).mean()
                me_train = mf.me(data_train, pred_train, axis=0).mean()
                mae_train = mf.mae(data_train, pred_train, axis=0).mean()
                minae_train = mf.minae(data_train, pred_train, axis=0).min()
                maxae_train = mf.maxae(data_train, pred_train, axis=0).max()

                rmse_test = mf.rmse(data_test, pred_test, axis=0).mean()
                me_test = mf.me(data_test, pred_test, axis=0).mean()
                mae_test = mf.mae(data_test, pred_test, axis=0).mean()
                minae_test = mf.minae(data_test, pred_test, axis=0).min()
                maxae_test = mf.maxae(data_test, pred_test, axis=0).max()

                error_metrics["Mean Train RMSE"] = rmse_train
                error_metrics["Mean Train ME"] = me_train
                error_metrics["Mean Train MAE"] = mae_train
                error_metrics["Mean Train Min Err"] = minae_train
                error_metrics["Mean Train Max Err"] = maxae_train

                error_metrics["Mean Test RMSE"] = rmse_test
                error_metrics["Mean Test ME"] = me_test
                error_metrics["Mean Test MAE"] = mae_test
                error_metrics["Mean Test Min Err"] = minae_test
                error_metrics["Mean Test Max Err"] = maxae_test

            self.error_metrics = error_metrics

            return self
                

    class BaselineModel(Template):
        """
        This is a baseline model that predicts the
        mean of the training data.
        """

        def __init__(self, var, model_type):
            """
            Args:
                var: Variable.
                model_type: Baseline or Coordinate.
            """

            # Object initialization:
            self.input = var
            self.output = var
            self.model_type = model_type
            self.model_tested = False

            # Naming scheme:
            if model_type == "Collective":
                self.name = "Collective Mean"
            
            if model_type == "Coordinate":
                self.name = "Coordinate Mean"

            # Get input data:
            if var is not None:
                self.load_data()

            self.name_set = Storage.name_set
        

        def fit(self, X=None, y=None, n_train=None):
            
            # Check if input data is present:
            if X is None:
                X = self.input_data.copy()
            
            # Check if output data is present:
            if y is None:
                y = self.output_data.copy()
            

            # Check if n_train is a fraction:
            if n_train is not None:

                if n_train > 0 and n_train < 1:   
                    
                    # Convert fraction to number of samples:
                    n_train = int(len(y) * n_train)

                self.n_train = n_train

            # Define training data:
            y_train = y.iloc[:self.n_train]

            if self.model_type == "Collective":
                
                # Get mean value:
                tmp = y_train.mean(axis=0)
                tmp_mean = tmp.mean()
                tmp = tmp * 0 + tmp_mean
                self.mean = tmp

            elif self.model_type == "Coordinate":

                # Get mean value:
                self.mean = y_train.mean(axis=0)

            return self


        def predict(self, X=None, y=None, n_test=None):
                
            # Check if input data is present:
            if X is None:
                X = self.input_data.copy()
            
            # Check if output data is present:
            if y is None:
                y = self.output_data.copy()
            
            # Allocate memory:
            self.output_preds = y.copy()
    
            # Get mean value:
            mean = self.mean

            if n_test is None:

                # Compute n_test:
                n_test = int(len(y) - self.n_train)

                self.n_test = n_test
                
                preds = self.output_preds.copy()

                # Loop over test range:
                for i in tqdm(range(len(preds))):
                    tmp = preds.iloc[i, :].values
                    tmp2 = tmp * 0 + mean
                    
                    preds.iloc[i, :] = tmp2
                
                self.output_preds = preds

            # Compute error metrics:
            self.compute_errors()
            
            self.model_tested = True
            
            return self


    class ReconModel(Template):
        """
        This is a reconstruction model that reconstructs the input and output data using scaler and latent space if available.
        """
        
        def __init__(self, var, latent_dim=1):
            """
            Args:
                input: Variable (string).
                latent_dim: Number of latent dimensions to use.
            """

            # Object initialization:
            self.input = var
            self.output = var
            self.latent_dim = latent_dim
            self.model_tested = False
            
            # Get input data:
            if var is not None:
                self.load_data()

            name = Storage.name

            self.latent_spaces = Storage.latent_spaces
            self.scalers = Storage.scalers

            if self.latent_spaces is not None:
                name = name[:-2]
                name += f"({self.latent_dim})-Recon" 

            # Set dimensions of latent spaces:
            if latent_dim is not None:

                if latent_dim > Storage.max_dim:
                    raise AssertionError(f"Too many latent dimensions. Should be in interval [1, {Storage.max_dim}]")
                
            self.name = name
        
        def fit(self, X=None, y=None, n_train=None):

            return self


        def predict(self, X=None, y=None):
                
            # Check if input data is present:
            if X is None:
                X = self.input_data.copy()
            
            # Check if output data is present:
            if y is None:
                y = self.output_data.copy()
            
            # Scale and project output:
            _, y1 = self.scale_n_proj("forward", X=None, y=y)

            # Rescale and project output:
            _, y2 = self.scale_n_proj("inverse", X=None, y=y1)

            # Store output:
            self.output_preds = y2

            # Compute error metrics:
            self.compute_errors()
            
            self.model_tested = True
            
            return self
        

    class RegressionModel(Template):
        """
        This is a regression model that predicts the
        output variables using the input variables.
        """

        def __init__(self, regressor=None, var=None,
                     wind=False, boundary=False,
                     state_lag=1, extra_lag=0, extra_lead=0,
                     latent_dim=1):
           
           # Object initialization:
            self.input = var
            self.output = var
            self.wind = wind
            self.boundary = boundary
            self.state_lag = state_lag
            self.extra_lag = extra_lag
            self.extra_lead = extra_lead
            self.latent_dim = latent_dim

            self.scalers = Storage.scalers
            self.latent_spaces = Storage.latent_spaces

            self.model_tested = False
            
            # Set dimensions of latent spaces:
            if latent_dim is not None:

                if latent_dim > Storage.max_dim:
                    raise AssertionError(f"Too many latent dimensions. Should be in interval [1, {Storage.max_dim}]")
                
            # Get regressor:
            self.regressor = Storage.get_regressor(regressor)

            # Get input data:
            if var is not None:
                self.load_data()
            
            # Get extra input data:
            if (wind or boundary):
                
                self.extras = []

                if boundary:
                    self.extras.extend(["bcn", "bcs"])

                if wind:
                    self.extras.extend(["wu", "wv"])                
                
                extra_data = pd.concat([Storage.data[extra] for extra in self.extras], axis=1)
                
            else:
                extra_data = None
            
            self.extra_data = extra_data

            self.input_cols_dict = None
            self.output_cols_dict = None

            self.name_set = Storage.name_set

            # Naming scheme:
            if 1:
                name = Storage.name

                if self.latent_spaces is not None:
                    name = name[:-2]
                    name += f"({self.latent_dim}), "

                if regressor is not None:
                    name += f"{regressor}("
                
                if var is not None:
                    in_str = ", ".join(var)
                
                if extra_lag > 0 or extra_lead > 0:

                    if boundary:
                        in_str += ", bcn, bcs"

                    if wind:
                        in_str += ", wu, wv"

                name += f"[{in_str}])->"

                if var is not None:
                    out_str = ", ".join(var)
                    name += f"([{out_str}]), "
                
                name += f"state_lag: {state_lag}"

                if (boundary or wind):

                    if extra_lag > 0:
                        name += f", extra_lag: {extra_lag}"
                    
                    if extra_lead > 0:
                        name += f", extra_lead: {extra_lead}"
        
                self.name = name
                self.name_set = "init"

            return

        
        def fit(self, X=None, y=None, n_train=None):
            
            # Check if input data is present:
            if X is None:
                X = self.input_data.copy()
            
            # Check if output data is present:
            if y is None:
                y = self.output_data.copy()

            # Scale and project input and output:
            X1, y1 = self.scale_n_proj("forward", X, y)

            [X_list, extra_X_list, y_list] = self.generate_list_pd(X1, y1)
            
            X_df = pd.concat(X_list+extra_X_list, axis=1).dropna()
            y_df = pd.concat(y_list, axis=1).loc[X_df.index]

            # Check if n_train is a fraction:
            if n_train is not None:

                if n_train > 0 and n_train < 1:   
                    
                    # Convert fraction to number of samples:
                    n_train = int(len(y_df) * n_train)

                self.n_train = n_train


            # Define training data:
            X_train = X_df.iloc[:self.n_train]
            y_train = y_df.iloc[:self.n_train]

            # Fit model:
            self.regressor.fit(X_train, y_train)
            
            return self


        def predict(self, X=None, y=None, n_test=None):
            
            # Check if input data is present:
            if X is None:
                X = self.input_data.copy()
            
            # Check if output data is present:
            if y is None:
                y = self.output_data.copy()

            y_pred = y.copy()

            # Scale and project input and output:
            X1, y1 = self.scale_n_proj("forward", X, y)

            [X_list, extra_X_list, y_list] = self.generate_list_pd(X1, y1)
            
            combined_df = pd.concat(X_list+extra_X_list, axis=1).dropna()

            if X_list == [] and extra_X_list == []:
                raise AssertionError("Problems with input data.")

            # Check if input data is present:
            if X_list != []:
                X_df = pd.concat(X_list, axis=1).loc[combined_df.index]
            
            # Check if extra input data is present:
            if extra_X_list != []:
                extra_df = pd.concat(extra_X_list, axis=1).loc[combined_df.index]
            
            y_df = pd.concat(y_list, axis=1).loc[X_df.index]

            y_df_pred = y_df.copy()
            y_df_pred.values[:] = np.NaN

            # Check if n_test is None:
            if n_test is None:
                
                # Set test sample to fraction of data:
                n_test = int(len(y_df) - self.n_train)

            else:

                # Check if n_test is a fraction:
                if n_test > 0 and n_test < 1:   
                    
                    # Convert fraction to number of samples:
                    n_test = int(len(y_df) * n_test)

                
                # Check if the sum of n_train and n_test is larger than the total number of samples:
                if n_test + self.n_train > len(y_df):

                    raise AssertionError("The sum of n_train and n_test is larger than the total number of samples.")

            self.n_test = n_test

            
            # Define test range:
            test_range = y_df.index[:self.n_train+self.n_test]

            self.test_range = test_range

            # Allocate memory and ensure correct dataframe shapes:
            if X_list != []:
                X_pred_i = pd.DataFrame(X_df.iloc[0].copy()).T
            else:
                X_pred_i = None

            if extra_X_list != []:
                extra_pred_i = pd.DataFrame(extra_df.iloc[0].copy()).T
            else:
                extra_pred_i = None
            
            y_pred_i = pd.DataFrame(y_df.iloc[0].copy()).T 


            # Find columns to replace with predictions:
            if X_pred_i is not None:
                
                # Allocate memory:
                preds_rep_arr = []

                # Loop over column names of model input:
                for col_in in X_pred_i:
                    
                    # Default to False:
                    rep_preds = False

                    # Check if lag1 is in column name:
                    if "lag1" in col_in:

                        # Loop over column names of model output:
                        for col_out in y_pred_i:
                            
                            # Check if input overlaps with output:
                            if col_out in col_in:
                                rep_preds=True

                    # Append result to list:
                    preds_rep_arr.append(rep_preds)
                
                preds_rep_arr = np.arange(len(preds_rep_arr))[preds_rep_arr] 

            # Set flag to False:
            preds_available = False

            # Number of columns to shift by:
            n_shift = int(len(X_pred_i.columns) / self.state_lag)

            # Loop over test range:
            for i in tqdm(test_range):

                # Fill input dataframes when predictions are not available:
                if not preds_available:

                    if X_pred_i is not None:
                        X_pred_i.values[:] = X_df.loc[i].values

                if extra_pred_i is not None:
                    extra_pred_i.values[:] = extra_df.loc[i].values
                
                # Overwrite with prediction if available:
                if preds_available:
                    
                    if X_pred_i is not None:
                        
                        # Shift columns:
                        X_pred_i = X_pred_i.shift(-n_shift, axis=1)

                        # Replace NaN columns with predictions:
                        X_pred_i.iloc[:, preds_rep_arr] = y_pred_i.values 

                # Concatenate input dataframes:
                if X_pred_i is not None and extra_pred_i is not None:
                    input_pred_i = pd.concat([X_pred_i, extra_pred_i], axis=1)

                elif X_pred_i is not None:
                    input_pred_i = X_pred_i

                elif extra_pred_i is not None:
                    input_pred_i = extra_pred_i

                else:
                    raise AssertionError("Something went wrong.")
                

                # Make prediction:
                y_pred_i.values[:] = self.regressor.predict(input_pred_i)

                # Set flag to True:
                preds_available = True

                # Store prediction:
                y_df_pred.loc[i,:] = y_pred_i.values

            # Rescale and project output:           
            _, self.output_preds = self.scale_n_proj("inverse", y=y_df_pred)
            
            # Compute error metrics:
            self.compute_errors()
            
            self.model_tested = True
            
            return self

