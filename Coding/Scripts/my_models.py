
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LinearRegression, Ridge

import matplotlib.pyplot as plt

from tqdm import tqdm

def extract_features(df):
    
    # Surface elevation, U- and V-velocity:
    z_data = df.filter(regex="z_").values
    u_data = df.filter(regex="^u_").values
    v_data = df.filter(regex="^v_").values
    
    # North and south BC data:
    bcn_data = df.filter(regex="bcn_").values
    bcs_data = df.filter(regex="bcs_").values
    
    # U- and V- wind velocity data:
    wu_data = df.filter(regex="wu_").values
    wv_data = df.filter(regex="wv_").values
    
    data_list = {"z"  :   z_data, "u"  :   u_data, "v" : v_data,
                 "bcn": bcn_data, "bcs": bcs_data,
                 "wu" :  wu_data, "wv" :  wv_data}
    
    return data_list




# Super class:
class my_models():
    
    def __init__(self):
        pass
    
    # Template for models:
    class PredictionModel:
        """
        Ensures that the methods fit, predict, fit_predict and plot are 
        instantiated for all classes based on this template.
        """
        
        if 0:
            def fit(self):
                raise NotImplementedError("Fit method must be implemented in subclass.")

            def predict(self):
                raise NotImplementedError("Predict method must be implemented in subclass.")        

            def plot(self):
                raise NotImplementedError("Plot method must be implemented in subclass.")
    
        
        def plot_errors(self):
            
            df_train = self.df_train
            df_test  = self.df_test
            model    = self.model
            
            xs = range(len(df_train)+len(df_test))
            x_train = xs[:len(df_train)]
            x_test = xs[len(df_train):]
            
            train_errors = model["train_errors"]
            test_errors = model["test_errors"]
            
            plt.figure(figsize=(7,5), dpi=100)
            plt.title(f"Errors for model: {model['name']}", 
                      fontsize=16)
            
            plt.plot(x_train, train_errors, color="blue")
            plt.plot(x_test,  test_errors,  color="red")
            
            train_error_avg = train_errors.mean()
            test_error_avg = test_errors.mean()
            
            plt.legend([f"Train: ( $\mu$ = {train_error_avg:.4f} )",
                        f"Test:  ( $\mu$ = {test_error_avg:.4f} )"],
                        fontsize=11,
                        frameon=True, fancybox=True,
                        shadow=True, framealpha=1, facecolor="lightgrey")
            
            plt.xlabel("Time steps", fontsize=14)
            plt.ylabel("RMSE", fontsize=14)
            
            plt.ylim([0,1])
            
            plt.show()
            
            return
            

## Model 0:
    class PCAReconstruction(PredictionModel):
        """
        PCAReconstruction: 
        Projects data into PCA subspace and reconstructs back to original space.
        """
        
        def __init__(self, pca_bs=256, pca_comps=10):
            
            self.name = f"PCA-({pca_comps}) Reconstruction"
            self.pca = IncrementalPCA(batch_size=pca_bs,
                                      n_components=pca_comps)
            self.pca_bs = pca_bs
            
        def run(self, df_train, df_test):

            self.df_train = df_train
            self.df_test  = df_test

            # Extract features:
            train_feats = extract_features(df_train)
            test_feats = extract_features(df_test)

            # Extract training data:
            X_train = np.concatenate([train_feats[i] for i in ["z"]], axis=1)
            y_train = np.concatenate([train_feats[i] for i in ["z"]], axis=1)

            # Extract test data:
            X_test = np.concatenate([test_feats[i] for i in ["z"]], axis=1)
            y_test = np.concatenate([test_feats[i] for i in ["z"]], axis=1)

            # PCA and scaler:
            scaler = StandardScaler().fit(X_train)
            pca = self.pca
            pca_bs = self.pca_bs
            
            # Scale:
            X_train_scaled = scaler.transform(X_train)
            
            # Fit PCA:
            for i in tqdm(range(0, len(X_train_scaled), pca_bs)[:-2]):
                
                j = i + pca_bs
                pca.partial_fit(X_train_scaled[i:j])
            
            pca.partial_fit(X_train_scaled[j:])
             
            # PCA Transform:
            X_train_scaled_pca = pca.transform(X_train_scaled)
            
            # Reconstruct training data:
            y_train_scaled_pred = pca.inverse_transform(X_train_scaled_pca)
            
            # Rescale training data:
            y_train_pred = scaler.inverse_transform(y_train_scaled_pred)
            
            # Compute training error:
            train_errors = mf.rmse(y_train, y_train_pred, axis=1)
            
            # Predict test data:
            X_test_scaled = scaler.transform(X_test)
            
            # PCA Transform:
            X_test_scaled_pca = pca.transform(X_test_scaled)
            
            # Reconstruct test data:
            y_test_scaled_pred = pca.inverse_transform(X_test_scaled_pca)
            
            # Rescale test data:
            y_test_pred = scaler.inverse_transform(y_test_scaled_pred)
            
            # Compute test error:
            test_errors = mf.rmse(y_test, y_test_pred, axis=1)

            # Collect data:
            model = {"y_train_pred" : y_train_pred,
                     "y_test_pred"  : y_test_pred,
                     "train_errors" : train_errors,
                     "test_errors"  : test_errors,
                     "name"         : self.name}

            self.model = model

            return
            
## Model 1:
    class Baseline(PredictionModel):
        """
        Baseline: 
        Computes mean across all elements and time steps and predicts 
        the mean for all elements at all time steps.
        """
        
        def __init__(self):
            self.name = "Baseline"
            
        def run(self, df_train, df_test):
            
            self.df_train = df_train
            self.df_test  = df_test
            
            # Extract features:
            train_feats = extract_features(df_train)
            test_feats = extract_features(df_test)

            # Extract training data:
            X_train = train_feats["z"]
            y_train = train_feats["z"]

            # Extract test data:
            X_test = test_feats["z"]
            y_test = test_feats["z"]

            # Fit model:
            mean = X_train.mean()

            # Predict training data:
            y_train_pred = np.zeros_like(y_train) + mean

            # Compute training error:
            train_errors = mf.rmse(y_train, y_train_pred, axis=1)

            # Predict test data:
            y_test_pred = np.zeros_like(y_test) + mean

            # Compute test error:
            test_errors = mf.rmse(y_test, y_test_pred, axis=1)
            
            # Collect data:
            model = {"y_train_pred" : y_train_pred,
                     "y_test_pred"  : y_test_pred,
                     "train_errors" : train_errors,
                     "test_errors"  : test_errors,
                     "name"         : self.name}

            self.model = model
            
            return
        
## Model 2:        
    class Coordinate_Baseline(PredictionModel):
        """
        Coordinate_Baseline: 
        Computes mean for each element across all time steps and predicts 
        the element mean for each element at all time steps. 
        
        """
        
        def __init__(self):
            self.name = "Coordinate Baseline"
            
        def run(self, df_train, df_test):
            
            self.df_train = df_train
            self.df_test  = df_test
            
            # Extract features:
            train_feats = extract_features(df_train)
            test_feats = extract_features(df_test)

            # Extract training data:
            X_train = train_feats["z"]
            y_train = train_feats["z"]

            # Extract test data:
            X_test = test_feats["z"]
            y_test = test_feats["z"]

            # Fit model:
            mean = X_train.mean(axis=0)

            # Predict training data:
            y_train_pred = np.zeros_like(y_train) + mean

            # Compute training error:
            train_errors = mf.rmse(y_train, y_train_pred, axis=1)

            # Predict test data:
            y_test_pred = np.zeros_like(y_test) + mean

            # Compute test error:
            test_errors = mf.rmse(y_test, y_test_pred, axis=1)
            
            # Collect data:
            model = {"y_train_pred" : y_train_pred,
                     "y_test_pred"  : y_test_pred,
                     "train_errors" : train_errors,
                     "test_errors"  : test_errors,
                     "name"         : self.name}

            self.model = model
            
            return

    
## Model 5:    
    class PCA_Regression_Z_BC(PredictionModel):
        """
        PCA_Regression_Z_BC: 
        Compresses the state vector into PCA subspace and adds the raw boundary conditions before
        computing the least squares linear map between the state [x_(i-1), bc_(i)] and x_(i). 
        
        The linear map is used to predict the future state. 

        """

        def __init__(self, pca_bs=256, pca_comps=10):
            
            self.name = f"PCA-({pca_comps}) Regression with Z and BCs"
            self.pca = IncrementalPCA(batch_size=pca_bs,
                                      n_components=pca_comps)
            self.pca_bs = pca_bs
            
            
        def run(self, df_train, df_test):

            self.df_train = df_train
            self.df_test  = df_test

            # Extract features:
            train_feats = extract_features(df_train)
            test_feats = extract_features(df_test)

            # Extract training data:
            X_train = np.concatenate([train_feats[i] for i in ["z"]], axis=1)
            y_train = np.concatenate([train_feats[i] for i in ["z"]], axis=1)

            # Extract test data:
            X_test = np.concatenate([test_feats[i] for i in ["z"]], axis=1)
            y_test = np.concatenate([test_feats[i] for i in ["z"]], axis=1)

            # PCA and scaler:
            scaler = StandardScaler().fit(X_train)
            pca = self.pca
            pca_bs = self.pca_bs
            
            # Scale:
            X_train_scaled = scaler.transform(X_train)
            
            # Fit PCA:
            for i in tqdm(range(0, len(X_train_scaled), pca_bs)[:-2]):
                
                j = i + pca_bs
                pca.partial_fit(X_train_scaled[i:j])
            
            pca.partial_fit(X_train_scaled[j:])
             
            # PCA Transform:
            X_train_scaled_pca = pca.transform(X_train_scaled)
            
            # Create input and outputs for linear regressor:
            X_in = np.concatenate([X_train_scaled_pca[:-1],
                                  np.concatenate([train_feats[i][1:] for i in ["bcn", "bcs"]], axis=1)], axis=1)                    
            
            X_out = X_train_scaled_pca[1:]
            
            # Create linear regressor:
            LR = LinearRegression()
            
            # Fit linear regressor:
            LR.fit(X_in, X_out)
            
            # Predict training data:
            y_train_scaled_pca_pred = np.zeros_like(X_train_scaled_pca)
        
            # Loop for predicting training data:
            for i in tqdm(range(len(X_train))):
                if i == 0:
                    y_train_scaled_pca_pred[i, :] = X_train_scaled_pca[0, :]
                else:
                    X_in = np.concatenate([y_train_scaled_pca_pred[i-1, :],
                            np.concatenate([train_feats[feat][i,:] for feat in ["bcn", "bcs"]])], 
                                          ).reshape(1,-1)
                    
                    y_train_scaled_pca_pred[i, :] = LR.predict(X_in).reshape(-1)

            y_train_scaled_pred = pca.inverse_transform(
                                    y_train_scaled_pca_pred)
            
            y_train_pred = scaler.inverse_transform(y_train_scaled_pred)
            
            # Compute training error:
            train_errors = mf.rmse(y_train, y_train_pred, axis=1)
            
            # Predict test data:
            X_test_scaled = scaler.transform(X_test)
            
            X_test_scaled_pca = pca.transform(X_test_scaled)
            
            y_test_scaled_pca_pred = np.zeros_like(X_test_scaled_pca)
            
            # Loop for predicting testing data:
            for i in tqdm(range(len(X_test))):
                if i == 0:
                    X_in = np.concatenate([X_train_scaled_pca[-1],
                                           np.concatenate([test_feats[feat][0,:] for feat in ["bcn", "bcs"]])
                                          ]).reshape(1,-1)
                                      
                else:
                    X_in = np.concatenate([y_test_scaled_pca_pred[i-1, :],
                            np.concatenate([test_feats[feat][i,:] for feat in ["bcn", "bcs"]])
                                          ]).reshape(1,-1)
                                        
                y_test_scaled_pca_pred[i, :] = LR.predict(X_in).reshape(-1)
            
            y_test_scaled_pred = pca.inverse_transform(
                                   y_test_scaled_pca_pred)
            
            y_test_pred = scaler.inverse_transform(y_test_scaled_pred)
            
            
            # Compute test error:
            test_errors = mf.rmse(y_test, y_test_pred, axis=1)

            # Collect data:
            model = {"y_train_pred" : y_train_pred,
                     "y_test_pred"  : y_test_pred,
                     "train_errors" : train_errors,
                     "test_errors"  : test_errors,
                     "name"         : self.name}

            self.model = model

            return
        

## Model 3:
    class PCA_Multistep_Regression_Z(PredictionModel):
        """
        PCA_Multistep_Regression_Z: 
        Compresses the state vector into PCA subspace and computes the least squares linear map between 
        the previous state(s) and future state.
        
        The linear map is used to predict the future state. 
        """

        def __init__(self, pca_bs=256, pca_comps=10, ar=1):
            
            self.name = f"PCA-({pca_comps}) Regression with {ar}-previous Z"
            self.pca = IncrementalPCA(batch_size=pca_bs,
                                      n_components=pca_comps)
            self.pca_bs = pca_bs
            self.ar = ar
            
            
        def run(self, df_train, df_test):

            self.df_train = df_train
            self.df_test  = df_test

            # Extract features:
            train_feats = extract_features(df_train)
            test_feats = extract_features(df_test)

            # Extract training data:
            X_train = np.concatenate([train_feats[i] for i in ["z"]], axis=1)
            y_train = np.concatenate([train_feats[i] for i in ["z"]], axis=1)

            # Extract test data:
            X_test = np.concatenate([test_feats[i] for i in ["z"]], axis=1)
            y_test = np.concatenate([test_feats[i] for i in ["z"]], axis=1)

            # PCA and scaler:
            scaler = StandardScaler().fit(X_train)
            pca = self.pca
            pca_bs = self.pca_bs
            ar = self.ar
            
            # Scale:
            X_train_scaled = scaler.transform(X_train)
            
            # Fit PCA:
            for i in tqdm(range(0, len(X_train_scaled), pca_bs)[:-2]):
                
                j = i + pca_bs
                pca.partial_fit(X_train_scaled[i:j])
            
            pca.partial_fit(X_train_scaled[j:])
             
            # PCA Transform:
            X_train_scaled_pca = pca.transform(X_train_scaled)
            
            # Create input and output:
            n_in = len(X_train)
            X_in_arr = []
            
            for i in range(ar):
                X_in_arr.append(X_train_scaled_pca[i:(n_in - ar+i), :])
            
            X_in = np.concatenate(X_in_arr, axis=1)
            
            X_out = X_train_scaled_pca[ar:, :]
            
            
            # Create linear regressor:
            LR = LinearRegression()
            
            # Fit linear regressor:
            LR.fit(X_in, X_out)
            
            # Predict training data:
            y_train_scaled_pca_pred = np.zeros_like(X_train_scaled_pca)
        
        
            # Loop for predicting training data:
            for i in tqdm(range(len(X_train))):
                
                # If True: Set the value:
                if i < ar:
                    y_train_scaled_pca_pred[i, :] = X_train_scaled_pca[i, :]
               
                # If False: Predict the value:
                else:
                    X_in_arr = [y_train_scaled_pca_pred[i-ar+j] for j in range(ar)]
                    X_in = np.concatenate(X_in_arr).reshape(1,-1)
                    y_train_scaled_pca_pred[i, :] = LR.predict(X_in).reshape(-1)

            y_train_scaled_pred = pca.inverse_transform(
                                    y_train_scaled_pca_pred)
            
            y_train_pred = scaler.inverse_transform(y_train_scaled_pred)
            
            # Compute training error:
            train_errors = mf.rmse(y_train, y_train_pred, axis=1)
            
            # Predict test data:
            X_test_scaled = scaler.transform(X_test)
            
            X_test_scaled_pca = pca.transform(X_test_scaled)
            
            y_test_scaled_pca_pred = np.zeros_like(X_test_scaled_pca)
            
            
            # Loop for predicting testing data:
            for i in tqdm(range(len(X_test))):
                
                
                # If True: We need both X_train and X_test
                if i < ar:
                    
                    X_in_arr = []
                    
                    # Loop over preceding steps:
                    for j in range(ar):

                        k = i-ar+j
                        
                        # If True: Use X_train
                        if k < 0:
                            X_in_arr.append(X_train_scaled_pca[k])
                            
                        # If False: Use y_test_pred
                        else:
                            X_in_arr.append(y_test_scaled_pca_pred[k])
                        
                # If False: We need only y_test_pred:
                else:
                    X_in_arr = [y_test_scaled_pca_pred[i-ar+j] for j in range(ar)]
        
                X_in = np.concatenate(X_in_arr).reshape(1,-1)
                
                    
                y_test_scaled_pca_pred[i, :] = LR.predict(X_in).reshape(-1)
            
            y_test_scaled_pred = pca.inverse_transform(
                                   y_test_scaled_pca_pred)
            
            y_test_pred = scaler.inverse_transform(y_test_scaled_pred)
            
            
            # Compute test error:
            test_errors = mf.rmse(y_test, y_test_pred, axis=1)

            # Collect data:
            model = {"y_train_pred" : y_train_pred,
                     "y_test_pred"  : y_test_pred,
                     "train_errors" : train_errors,
                     "test_errors"  : test_errors,
                     "name"         : self.name}

            self.model = model

            return

        
## Model 4:
    class PCA_Multistep_Regression_BC(PredictionModel):
        """
        PCA_Multistep_Regression_BC: 
        Compresses the state vector into PCA subspace and
        computes the least squares linear map between previous BCs and the current state. 
        
        The linear map is used to predict the future state. 
        """

        def __init__(self, pca_bs=256, pca_comps=10, ar=1):
            
            self.name = f"PCA-({pca_comps}) Regression with {ar}-previous BCs"
            self.pca = IncrementalPCA(batch_size=pca_bs,
                                      n_components=pca_comps)
            self.pca_bs = pca_bs
            self.ar = ar
            
            
        def run(self, df_train, df_test):

            self.df_train = df_train
            self.df_test  = df_test

            # Extract features:
            train_feats = extract_features(df_train)
            test_feats = extract_features(df_test)

            # Extract training data:
            X_train = np.concatenate([train_feats[i] for i in ["bcn", "bcs"]], axis=1)
            y_train = np.concatenate([train_feats[i] for i in ["z"]], axis=1)

            # Extract test data:
            X_test = np.concatenate([test_feats[i] for i in ["bcn", "bcs"]], axis=1)
            y_test = np.concatenate([test_feats[i] for i in ["z"]], axis=1)

            # PCA and scaler:
            scaler = StandardScaler().fit(y_train)
            pca = self.pca
            pca_bs = self.pca_bs
            ar = self.ar
            
            # Scale:
            y_train_scaled = scaler.transform(y_train)
            
            # Fit PCA:
            for i in tqdm(range(0, len(y_train_scaled), pca_bs)[:-2]):
                
                j = i + pca_bs
                pca.partial_fit(y_train_scaled[i:j])
            
            pca.partial_fit(y_train_scaled[j:])
             
            # PCA Transform:
            y_train_scaled_pca = pca.transform(y_train_scaled)
            
            # Create input and output:
            n_in = len(X_train)
            
            X_in_arr = []
            
            for i in range(ar):
                X_in_arr.append(X_train[i:(n_in - ar+i), :])
            
            X_in = np.concatenate(X_in_arr, axis=1)
            
            y_out = y_train_scaled_pca[ar:, :]
            
            # Create linear regressor:
            LR = LinearRegression()
            
            # Fit linear regressor:
            LR.fit(X_in, y_out)
            
            # Predict training data:
            y_train_scaled_pca_pred = np.zeros_like(y_train_scaled_pca)
        
            # Loop for predicting training data:
            for i in tqdm(range(len(X_train))):
                
                # If True: Set the value:
                if i < ar:
                    y_train_scaled_pca_pred[i, :] = y_train_scaled_pca[i, :]
               
                # If False: Predict the value:
                else:
                    X_in_arr = [X_train[i-ar+j] for j in range(ar)]
                    X_in = np.concatenate(X_in_arr).reshape(1,-1)
                    y_train_scaled_pca_pred[i, :] = LR.predict(X_in).reshape(-1)

            y_train_scaled_pred = pca.inverse_transform(
                                    y_train_scaled_pca_pred)
            
            y_train_pred = scaler.inverse_transform(y_train_scaled_pred)
            
            # Compute training error:
            train_errors = mf.rmse(y_train, y_train_pred, axis=1)
            
            
            # Predict test data:
            y_test_scaled = scaler.transform(y_test)
            
            y_test_scaled_pca = pca.transform(y_test_scaled)
            
            y_test_scaled_pca_pred = np.zeros_like(y_test_scaled_pca)
            
            # Loop for predicting testing data:
            for i in tqdm(range(len(X_test))):
                
                # If True: We need both X_train and X_test
                if i < ar:
                    
                    X_in_arr = []
                    
                    # Loop over preceding steps:
                    for j in range(ar):

                        k = i-ar+j
                        
                        # If True: Use X_train
                        if k < 0:
                            X_in_arr.append(X_train[k])
                            
                        # If False: Use y_test_pred
                        else:
                            X_in_arr.append(X_test[k])
                        
                # If False: We need only y_test_pred:
                else:
                    X_in_arr = [X_test[i-ar+j] for j in range(ar)]
        
                X_in = np.concatenate(X_in_arr).reshape(1,-1)
                
                    
                y_test_scaled_pca_pred[i, :] = LR.predict(X_in).reshape(-1)
            
            y_test_scaled_pred = pca.inverse_transform(
                                   y_test_scaled_pca_pred)
            
            y_test_pred = scaler.inverse_transform(y_test_scaled_pred)
            
            
            # Compute test error:
            test_errors = mf.rmse(y_test, y_test_pred, axis=1)

            # Collect data:
            model = {"y_train_pred" : y_train_pred,
                     "y_test_pred"  : y_test_pred,
                     "train_errors" : train_errors,
                     "test_errors"  : test_errors,
                     "name"         : self.name}

            self.model = model

            return


## Model 5:    
    class PCA_Multistep_Regression_Z_BC(PredictionModel):
        """
        PCA_Multistep_Regression_Z_BC: 
        Compresses the state vector into PCA subspace and adds the raw boundary conditions before
        computing the least squares linear map between the (multiple) previous state [x_(i-1), bc_(i)] and x_(i). 
        
        The linear map is used to predict the future state. 

        """

        def __init__(self, pca_bs=256, pca_comps=10, ar=1):
            
            self.name = f"PCA-({pca_comps}) Regression with {ar}-previous Z and BCs"
            self.pca = IncrementalPCA(batch_size=pca_bs,
                                      n_components=pca_comps)
            self.pca_bs = pca_bs
            self.ar = ar            
            
        def run(self, df_train, df_test):

            self.df_train = df_train
            self.df_test  = df_test

            # Extract features:
            train_feats = extract_features(df_train)
            test_feats = extract_features(df_test)

            # Extract training data:
            X_train = np.concatenate([train_feats[i] for i in ["z"]], axis=1)
            bc_train = np.concatenate([train_feats[i] for i in ["bcn", "bcs"]], axis=1)
            
            y_train = np.concatenate([train_feats[i] for i in ["z"]], axis=1)
            
            
            # Extract test data:
            X_test = np.concatenate([test_feats[i] for i in ["z"]], axis=1)
            bc_test = np.concatenate([test_feats[i] for i in ["bcn", "bcs"]], axis=1)
            
            y_test = np.concatenate([test_feats[i] for i in ["z"]], axis=1)

            # PCA and scaler:
            scaler = StandardScaler().fit(X_train)
            pca = self.pca
            pca_bs = self.pca_bs
            ar = self.ar
            
            # Scale:
            X_train_scaled = scaler.transform(X_train)
            y_train_scaled = scaler.transform(y_train)
            
            # Fit PCA:
            for i in tqdm(range(0, len(X_train_scaled), pca_bs)[:-2]):
                
                j = i + pca_bs
                pca.partial_fit(X_train_scaled[i:j])
            
            pca.partial_fit(X_train_scaled[j:])
             
            # PCA Transform:
            X_train_scaled_pca = pca.transform(X_train_scaled)
            y_train_scaled_pca = pca.transform(y_train_scaled)
            
            # Create input and output:
            n_in = len(X_train)
            
            X_in_list = []
            bc_in_list = []
            
            for i in range(ar):
                j = n_in - ar + i
                X_in_list.append(X_train_scaled_pca[i:j, :])
                bc_in_list.append(bc_train[i+1:j+1, :])
            
            X_in_arr = np.concatenate(X_in_list, axis=1)
            bc_in_arr = np.concatenate(bc_in_list, axis=1)
            
            X_in = np.concatenate([X_in_arr, bc_in_arr], axis=1)
            
            y_out = y_train_scaled_pca[ar:, :]
            
            # Create linear regressor:
            LR = LinearRegression()
            
            # Fit linear regressor:
            LR.fit(X_in, y_out)
            
            # Predict training data:
            y_train_scaled_pca_pred = np.zeros_like(X_train_scaled_pca)
        
            # Loop for predicting training data:
            for i in tqdm(range(len(X_train))):
                
                # If True: Set the value:
                if i < ar:
                    y_train_scaled_pca_pred[i, :] = y_train_scaled_pca[i, :]
               
                # If False: Predict the value:
                else:
                    
                    X_in_list = []
                    bc_in_list = []

                    for j in range(ar):
                        
                        k = i-ar+j
                        
                        X_in_list.append(y_train_scaled_pca_pred[k, :])
                        bc_in_list.append(bc_train[k+1, :])
                    
                    X_in_arr = np.concatenate(X_in_list)
                    bc_in_arr = np.concatenate(bc_in_list)
                    
                    X_in = np.concatenate([X_in_arr, bc_in_arr]).reshape(1,-1)
                    y_train_scaled_pca_pred[i, :] = LR.predict(X_in).reshape(-1)

            y_train_scaled_pred = pca.inverse_transform(
                                    y_train_scaled_pca_pred)
            
            y_train_pred = scaler.inverse_transform(y_train_scaled_pred)
            
            # Compute training error:
            train_errors = mf.rmse(y_train, y_train_pred, axis=1)
               
        
            
            # Predict test data:
            X_test_scaled = scaler.transform(X_test)
            
            X_test_scaled_pca = pca.transform(X_test_scaled)
            
            y_test_scaled_pca_pred = np.zeros_like(X_test_scaled_pca)
           
            # Loop for predicting testing data:
            for i in tqdm(range(len(X_test))):
                
                
                # If True: We need both X_train and X_test
                if i < ar:
                                        
                    X_in_list = []
                    bc_in_list = []
                    
                    # Loop over preceding steps:
                    for j in range(ar):
                        
                        k = i-ar+j
                        
                        # If True: Use X_train
                        if k < 0:
                            X_in_list.append(X_train_scaled_pca[k])
                            
                        # If False: Use y_test_pred
                        else:
                            X_in_list.append(y_test_scaled_pca_pred[k])
                            
                        k += 1
                        
                        # If True: Use bc_train
                        if k < 0:
                            bc_in_list.append(bc_train[k])
                            
                        # If False: Use bc_test
                        else:
                            bc_in_list.append(bc_test[k])
                        
                        
                # If False: We need only y_test_pred:
                else:
                    
                    X_in_list = [y_test_scaled_pca_pred[i-ar+j, :] for j in range(ar)]
                    bc_in_list = [bc_test[i-ar+j+1, :] for j in range(ar)]
                
                
                X_in_arr = np.concatenate(X_in_list)
                bc_in_arr = np.concatenate(bc_in_list)
                
                X_in = np.concatenate([X_in_arr, bc_in_arr]).reshape(1,-1)
                
                y_test_scaled_pca_pred[i, :] = LR.predict(X_in).reshape(-1)
            
            y_test_scaled_pred = pca.inverse_transform(
                                   y_test_scaled_pca_pred)
            
            y_test_pred = scaler.inverse_transform(y_test_scaled_pred)
             
            
            # Compute test error:
            test_errors = mf.rmse(y_test, y_test_pred, axis=1)

            # Collect data:
            model = {"y_train_pred" : y_train_pred,
                     "y_test_pred"  : y_test_pred,
                     "train_errors" : train_errors,
                     "test_errors"  : test_errors,
                     "name"         : self.name}

            self.model = model

            return



        
        
        
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################





from sklearn import preprocessing, base, decomposition, linear_model

import pandas as pd
import numpy as np

from Scripts import my_functions as mf


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
            feats  = pd.concat(feats_list,  axis=1).values
        
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
            
            ### INPUT CHECKING: ###
            
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
            
            ### RENAME: ###
            
            input_state_vars = inputs
            output_state_vars = outputs
            
        
            ### CONCATENATE STATE INPUT AND OUTPUT: ###
            
            # Check if input state variables are needed:
            if len(input_state_vars) != 0:
                
                # Extract input state features:
                self.input_data = Storage.get_features(input_state_vars)
                self.state_feats_in  = self.input_data
            else:
                self.state_feats_in = None
            
            self.output_data = Storage.get_features(output_state_vars)
            self.state_feats_out = self.output_data
            
            
            ### SCALE AND PROJECT STATE FEATURES: ###
            
            scaler       = Storage.scaler
            latent_space = Storage.latent_space
            columns      = Storage.columns
            
            # Check if input state variables are needed:
            if len(input_state_vars) > 0:
                self.active_cols_in = Storage.get_active_columns(columns, input_state_vars)
                  
                if scaler is not None:
                    self.scaler_in = base.clone(scaler)
                    self.scaler_in.mean_ = scaler.mean_[self.active_cols_in]
                    self.scaler_in.scale_ = scaler.scale_[self.active_cols_in]
                    
                    self.state_feats_in = self.scaler_in.transform(self.state_feats_in)
                       
                if (latent_space is not None and latent_dim is not None):
                    self.latent_space_in = base.clone(latent_space)
                    self.latent_space_in.mean_ = latent_space.mean_[self.active_cols_in].copy()
                    self.latent_space_in.components_ = latent_space.components_[:latent_dim,
                                                                                self.active_cols_in].copy()
                    
                    self.state_feats_in = self.latent_space_in.transform(self.state_feats_in)
                    
                else:
                    raise AssertionError("Mismatch between latent_dim and latent_space.")
            
            # Check if output state variables are needed:
            if len(output_state_vars) > 0:
                
                self.active_cols_out = Storage.get_active_columns(columns, output_state_vars)
                
                if scaler is not None:
                    self.scaler_out = base.clone(scaler)
                    self.scaler_out.mean_ = scaler.mean_[self.active_cols_out]
                    self.scaler_out.scale_ = scaler.scale_[self.active_cols_out]
                
                    self.state_feats_out = self.scaler_out.transform(self.state_feats_out)
                
                if latent_space is not None:
                    self.latent_space_out = base.clone(latent_space)
                    self.latent_space_out.mean_ = latent_space.mean_[self.active_cols_out].copy()
                    self.latent_space_out.components_ = latent_space.components_[:latent_dim,
                                                                                self.active_cols_out].copy()
                    
                    self.state_feats_out = self.latent_space_out.transform(self.state_feats_out)
                    
            
            ### LAG STATE FEATURES: ###
            
            # Check if lagged state features are needed:
            if len(input_state_vars) != 0 and state_lag >= 1:
                
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
        
        
        def fit(self, X=None, y=None, n_train=None):
            
            
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
            
            
            
 
            
         