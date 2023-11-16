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
            active_cols += np.array([True if var in i else False for i in column_names])

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
 
        def plot_errors(self, metric):

            pass


    class BaselineModel(Template):
        """
        This is a baseline model that predicts the
        mean of the training data.
        """

        def __init__(self, inputs, model_type="Baseline", train_frac=0.25):
            """
            Args:
                inputs: List of variables.
                model_type: Baseline or Coordinate.
                train_frac: Fraction of data to use for training.
            """

            # Object initialization:
            self.inputs = inputs
            self.train_frac = train_frac
            self.model_type = model_type

            # Naming scheme:
            if model_type == "Collective":
                self.name = "Collective Mean"
            
            if model_type == "Coordinate":
                self.name = "Coordinate Mean"

            # Get input data:
            if inputs is not None:
                self.input_data = Storage.copy_dict_pd(dict_in=Storage.data,
                                                       keys=inputs)
                
                self.output_data = Storage.copy_dict_pd(dict_in=Storage.data,
                                                        keys=inputs)


        def fit(self, X=None, y=None):
            
            # Check if input data is present:
            if X is None:
                X = self.input_data.copy()
            
            # Check if output data is present:
            if y is None:
                y = self.output_data.copy()
            
            # Allocate memory:
            self.means = {}

            # Loop over output variables:
            for key in y:
                
                self.n_train = int(len(y[key]) * self.train_frac)

                # Get training data:
                y_train = y[key]
                y_train = y_train.iloc[:self.n_train]

                if self.model_type == "Collective":
                    
                    # Get mean value:
                    tmp = y_train.mean(axis=0)
                    tmp_mean = tmp.mean()
                    tmp = tmp * 0 + tmp_mean
                    self.means[key] = tmp

                elif self.model_type == "Coordinate":

                    # Get mean value:
                    self.means[key] = y_train.mean(axis=0)

            self.name += f", n_train: {self.n_train}"

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

            # Loop over output variables:
            for key in y:
                
                # Get mean value:
                mean = self.means[key]

                if n_test is None:

                    # Compute n_test:
                    n_test = int(len(y[key]) - self.n_train)

                self.n_test = n_test
                
                preds = self.output_preds[key].copy()

                # Loop over test range:
                for i in tqdm(range(len(preds))):
                    tmp = preds.iloc[i, :].values
                    tmp2 = tmp * 0 + mean
                    
                    preds.iloc[i, :] = tmp2
                
                self.output_preds[key] = preds


            self.name += f", n_test: {self.n_test}"

            return



    class RegressionModel(Template):
        

        def __init__(self, regressor=None, inputs=None, outputs=None,
                     wind=False, boundary=False,
                     state_lag=1, extra_lag=0, extra_lead=1,
                     latent_dim=1, train_frac=0, test_frac=0):
           
           # Object initialization:
            self.inputs = inputs
            self.outputs = outputs
            self.wind = wind
            self.boundary_conditions = boundary
            self.state_lag = state_lag
            self.extra_lag = extra_lag
            self.extra_lead = extra_lead
            self.latent_dim = latent_dim
            self.train_frac = train_frac
            
            if test_frac == 0:
                self.test_frac = 1-train_frac
            self.test_frac = test_frac

            self.scalers = Storage.scalers
            latent_spaces = Storage.latent_spaces

            # Set dimensions of latent spaces:
            if latent_dim is not None:

                if latent_dim > Storage.max_dim:
                    raise AssertionError("Too many latent dimensions. Should be in interval [1, {Storage.max_dim}]")
                
                for key, obj in latent_spaces.items():
                    obj.means_ = obj.mean_[:latent_dim]
                    obj.components_ = obj.components_[:latent_dim, :]

                self.latent_spaces = latent_spaces

            # Get regressor:
            self.regressor = Storage.get_regressor(regressor)

            # Get input data:
            if inputs is not None:
                
                input_data = Storage.copy_dict_pd(dict_in=Storage.data, keys=inputs)

            else:
                input_data = None
            
            # Get output data:
            if outputs is not None:
                
                output_data = Storage.copy_dict_pd(dict_in=Storage.data, keys=outputs)
            
            else:
                output_data = None
            
            # Get extra input data:
            if not (wind is False and boundary is False):
                
                self.extras = []

                if boundary is True:
                    self.extras.extend(["bcn", "bcs"])

                if wind is True:
                    self.extras.extend(["wu", "wv"])                

                extra_data = Storage.copy_dict_pd(dict_in=Storage.data, keys=self.extras)
            
            else:
                extra_data = None
            
            self.input_data  = input_data
            self.output_data = output_data
            self.extra_data = extra_data

            self.input_cols_dict = None
            self.output_cols_dict = None

            # Naming scheme:
            if 1:
                name = Storage.name

                if self.scalers is not None:
                    name = name[:-2]
                    name += f"({self.latent_dim}), "

                if regressor is not None:
                    name += f"{regressor}("
                
                if inputs is not None:
                    in_str = ", ".join(inputs)
                
                if boundary is True:
                    in_str += ", bcn, bcs"

                if wind is True:
                    in_str += ", wu, wv"

                name += f"[{in_str}])->"

                if outputs is not None:
                    out_str = ", ".join(outputs)
                    name += f"([{out_str}]), "
                
                name += f"state_lag: {state_lag}"

                if boundary is True or wind is True:

                    if extra_lag > 0:
                        name += f", extra_lag: {extra_lag}"
                    
                    if extra_lead > 0:
                        name += f", extra_lead: {extra_lead}"
        
                self.name = name

            return



        def scale_n_proj(self, direction, X=None, y=None):

            # Check if scaling is needed:
            if self.scalers is not None:
                scalers = self.scalers
            
            else:
                scalers = None

            # Check if projection is needed:
            if self.latent_spaces is not None:
                latent_spaces = self.latent_spaces
            else:
                latent_spaces = None

            # Initialize dictionaries if not already existing:
            if self.input_cols_dict is None:
                self.input_cols_dict = {}

            if self.output_cols_dict is None:
                self.output_cols_dict = {}

            # Check if input data is present:
            if X is not None:
                
                # Copy input data:
                X2 = X.copy()

                # Scale and transform each input:
                for key in X2:
                    
                    X_index = X2[key].index

                    # Forward method:
                    if direction == "forward":

                        # Get column names from input data:
                        cols = X2[key].columns
                    
                        # Store column names:
                        self.input_cols_dict[key] = cols

                        # Check if scaling is needed:
                        if scalers is not None:
                            X2[key] = pd.DataFrame(
                                            scalers[key].transform(X2[key]),
                                            columns=cols)

                        # Check if projection is needed:
                        if latent_spaces is not None:
                            X2[key] = pd.DataFrame(
                                        latent_spaces[key].transform(X2[key]),
                                        columns=cols[:self.latent_dim])

                    # Inverse method:
                    if direction == "inverse":
                        
                        # Get column names from stored data:
                        cols = self.input_cols_dict[key]

                        # Check if projection is needed:
                        if latent_spaces is not None:
                            X2[key] = pd.DataFrame(
                                        latent_spaces[key].inverse_transform(X2[key]),
                                        columns=cols)

                        # Check if scaling is needed:
                        if scalers is not None:
                            X2[key] = pd.DataFrame(
                                        scalers[key].inverse_transform(X2[key]),
                                        columns=cols)

                    # Set index:
                    X2[key].index = X_index

            else:
                X2 = None


            # Check if output data is present:
            if y is not None:
                
                y2 = y.copy()

                # Scale and transform each output:
                for key in y2:
                    
                    y_index = y2[key].index

                    # Forward method:
                    if direction == "forward":

                        # Get column names:
                        cols = y2[key].columns

                        # Store column names:
                        self.output_cols_dict[key] = cols

                        # Check if scaling is needed:
                        if scalers is not None:
                            y2[key] = pd.DataFrame(
                                        scalers[key].transform(y2[key]),
                                        columns=cols)
                            
                        # Check if projection is needed:
                        if latent_spaces is not None:
                            y2[key] = pd.DataFrame(
                                        latent_spaces[key].transform(y2[key]),
                                        columns=cols[:self.latent_dim])

                    # Inverse method:
                    if direction == "inverse":
                        
                        # Get column names from stored data:
                        cols = self.output_cols_dict[key]

                        # Check if projection is needed:
                        if latent_spaces is not None:
                            y2[key] = pd.DataFrame(
                                        latent_spaces[key].inverse_transform(y2[key]),
                                        columns=cols)

                        # Check if scaling is needed:
                        if scalers is not None:
                            y2[key] = pd.DataFrame(
                                        scalers[key].inverse_transform(y2[key]),
                                        columns=cols)

                    # Set index:
                    y2[key].index = y_index

            else:
                y2 = None
            
            return X2, y2


        def generate_list_pd(self, X=None, y=None):
            
            # Check if input data is present:
            if X is not None:
                
                # Copy input data:
                X2 = X.copy()

                # Allocate memory:
                X_mat = []

                # Concatenate input data:
                for key in X2:
                    cols = X2[key].columns
                    X_mat.append(pd.DataFrame(X2[key],
                                              columns=cols))

                # Convert list to dataframe:
                X_mat = pd.concat(X_mat, axis=1)
                
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
                extra_data2 = self.extra_data.copy() 
                
                # Allocate memory:
                extra_mat = []

                # Concatenate extra input data:
                for key in extra_data2:
                    cols = extra_data2[key].columns
                    extra_mat.append(pd.DataFrame(extra_data2[key],
                                                  columns=cols))

                # Convert list to dataframe:
                extra_mat = pd.concat(extra_mat, axis=1)


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
                y2 = y.copy()

                # Allocate memory:
                y_mat = []

                # Concatenate output data:
                for key in y2:
                    cols = y2[key].columns
                    y_mat.append(pd.DataFrame(y2[key],
                                              columns=cols))
                    
                y_mat = [pd.concat(y_mat, axis=1)]

            else:

                y_mat = []

            return X_mat, extra_mat, y_mat


        def fit(self, X=None, y=None):
            
            # Check if input data is present:
            if X is None:
                X = self.input_data.copy()
            
            # Check if output data is present:
            if y is None:
                y = self.output_data.copy()

            # Scale and project input and output:
            X1, y1 = self.scale_n_proj("forward", X, y)

            [X_list, 
             extra_X_list,
             y_list] = self.generate_list_pd(X1, y1)
            
            X_df = pd.concat(X_list+extra_X_list, axis=1).dropna()
            y_df = pd.concat(y_list, axis=1).loc[X_df.index]

            # Check if training fraction is None:
            if self.train_frac is None:

                # Default training sample to 25% of data:
                n_train = len(y_df) // 4

            # Check if n_train is a fraction:
            if self.train_frac > 0 and self.train_frac < 1:
                
                if self.train_frac + self.test_frac > 1:
                    raise AssertionError("test_frac and train_frac cannot sum to more than 1.")

                # Set training sample to fraction of data:
                n_train = int(len(y_df) * self.train_frac)

                self.n_train = n_train


            # Raise error if n_train is incorrect:
            else:
                raise AssertionError("n_train must be a fraction in interval (0, 1)")
                
            # Convert n_train to integer:
            self.n_train = n_train = int(n_train)

            # Define training data:
            X_train = X_df.iloc[:self.n_train]
            y_train = y_df.iloc[:self.n_train]

            # Fit model:
            self.regressor.fit(X_train, y_train)

            self.X_train = X_train
            self.y_train = y_train

            self.name += f", n_train: {n_train}"

            return self

        def predict(self, X=None, y=None, n_test=None):
            
            # Check if input data is present:
            if X is None:
                X = self.input_data.copy()
            
            # Check if output data is present:
            if y is None:
                y = self.output_data.copy()

            self.X = X
            self.y = y
            self.y_pred = y.copy()

            # Scale and project input and output:
            X1, y1 = self.scale_n_proj("forward", X, y)

            [X_list, 
             extra_X_list,
             y_list] = self.generate_list_pd(X1, y1)
            
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
                
                if self.test_frac is None:
                    raise AssertionError("n_test of predict and test_frac RegressionModel cannot both be None.")
                
                if self.test_frac + self.train_frac > 1:
                    raise AssertionError("test_frac and train_frac cannot sum to more than 1.")
                
                # Set test sample to fraction of data:
                n_test = int(len(y_df) - self.n_train)

            self.n_test = n_test

            

            # Define test range:
            test_range = y_df.index[:self.n_train+self.n_test+1]


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


            # input indices where predictions are available
            # and where original input data is needed:
            if X_pred_i is not None:
                
                # Allocate memory:
                preds_rep_arr = []
                X_rep_arr = []

                # Loop over column names of model input:
                for col_in in X_pred_i:
                    
                    # Default to False:
                    rep_preds = False
                    rep_X = False

                    # Check if lag1 is in column name:
                    if "lag1" in col_in:
                        
                        rep_X = True

                        # Loop over column names of model output:
                        for col_out in y_pred_i:
                            
                            # Check if input overlaps with output:
                            if col_out in col_in:
                                rep_preds=True
                                rep_X = False

                    # Append result to list:
                    preds_rep_arr.append(rep_preds)
                    X_rep_arr.append(rep_X)
                    
                
                preds_rep_arr = np.arange(len(preds_rep_arr))[preds_rep_arr] 
                X_rep_arr = np.arange(len(X_rep_arr))[X_rep_arr]

            # Set flag to False:
            preds_available = False

            # Number of columns to shift by:
            n_shift = int(len(X_pred_i.columns) / self.state_lag)

            # Loop over test range:
            for i in tqdm(test_range):

                # Fill input dataframes:
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

                        # Replace some NaN columns with predictions:
                        X_pred_i.iloc[:, preds_rep_arr] = y_pred_i.values

                        # Replace remaining NaN columns with input data:
                        X_pred_i.iloc[:, X_rep_arr] = \
                            X_df.loc[i].values[X_rep_arr]
                        self.X_pred_i = X_pred_i
                        

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

            # Allocate memory:
            output_preds = {}

            self.y_df_pred = y_df_pred

            # Loop over output variables:
            for key in self.outputs:

                # Get column names:
                cols = self.output_cols_dict[key]

                if self.latent_spaces is not None:
                    cols = cols[:self.latent_dim]

                # Get predictions:
                output_preds[key] = y_df_pred[cols]

            
            _, self.output_preds = self.scale_n_proj("inverse", y=output_preds)

            self.name += f", n_test: {n_test}"

            return 


            