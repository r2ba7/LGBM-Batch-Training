
class DataGenerator:
    @classmethod
    def __set_seed(cls):
        """
        Set the random seed for reproducibility using a utility function from utils.Misc.

        This class method sets the random seed to ensure reproducibility in random operations.
        It utilizes the utility function from utils.Misc to set the seed value.

        Note:
        - This method is a class method, and it affects the randomness of the entire class.

        Returns:
        - None
        """
        utils.Misc.set_seed()

    def __init__(self, X, y, num_batches):
        """
        Initialize a DataGenerator instance.

        This constructor initializes a DataGenerator instance with input features, target values,
        and the desired number of batches for batch-wise data iteration.

        Args:
        - X (pandas.DataFrame): Input features for the data generator.
        - y (pandas.Series): Target values for the data generator.
        - num_batches (int): Number of batches to divide the data into.

        Modifies:
        - Initializes instance attributes including:
          - self.X: Input features.
          - self.y: Target values.
          - self.num_batches: Number of desired batches.
          - self.num_samples: Total number of samples in the data.
          - self.batch_size: Size of each batch.
          - self.indices: Indices representing the data samples.
          - self.current_batch: Current batch index.

        Returns:
        - None
        """
        self.__set_seed()  # Set random seed for reproducibility
        self.X = X
        self.y = y
        self.num_batches = num_batches
        self.num_samples = len(X)
        self.batch_size = (self.num_samples + self.num_batches - 1) // self.num_batches
        self.indices = np.arange(self.num_samples)
        self.current_batch = 0

    def __len__(self):
        """
        Get the total number of samples in the data generator.

        Returns:
        - int: Total number of samples.

        Returns the total number of samples in the data generator.
        """
        return self.num_samples

    def __iter__(self):
        """
        Initialize an iterator for the data generator.

        Returns:
        - self: The data generator instance.

        Initializes an iterator for the data generator.
        """
        return self

    def __getitem__(self, index):
        """
        Get a batch of data samples and targets.

        Args:
        - index (int): Index of the desired batch.

        Returns:
        - tuple: A tuple containing input features (X_batch) and target values (y_batch).

        Examples:
        Assuming an instance of the DataGenerator class:
        >>> data_gen = DataGenerator(X, y, num_batches)
        >>> data_gen[0]
        Retrieves the input features and target values for the first batch.
        """
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_samples)

        X_batch = self.X.iloc[start_idx:end_idx]
        y_batch = self.y.iloc[start_idx:end_idx]

        return X_batch, y_batch

    def __next__(self):
        """
        Get the next batch of data samples and targets.

        Returns:
        - tuple: A tuple containing input features (X_batch) and target values (y_batch).

        Raises:
        - StopIteration: When all batches have been processed.

        Retrieves the next batch of input features and target values.
        """
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        idx = self.current_batch  # Get the current batch index
        self.current_batch += 1

        return self.__getitem__(idx)

class BatchTrainer:
    @classmethod
    def __set_seed(cls):
        """
        Set the random seed for reproducibility using a utility function from utils.Misc.

        This class method sets the random seed to ensure reproducibility in random operations.
        It utilizes the utility function from utils.Misc to set the seed value.

        Note:
        - This method is a class method, and it affects the randomness of the entire class.

        Returns:
        - None
        """
        utils.Misc.set_seed()

    def __init__(self, file_names, files_directories, training_dict, num_batches=2,
                 save_flag=True, batch_train_flag=True):
        """
        Initialize a BatchTrainer instance.

        This constructor initializes a BatchTrainer instance with the provided parameters.
        
        Args:
        - file_names (dict): Dictionary containing names of various files.
        - files_directories (dict): Dictionary containing directories of different files.
        - training_dict (dict): Dictionary containing training-related information.
        - num_batches (int, optional): Number of batches to divide the training data into. Default is 2.
        - save_flag (bool, optional): Flag to indicate whether to save model checkpoints. Default is True.
        - batch_train_flag (bool, optional): Flag to indicate whether to perform batch-wise training. Default is True.

        Modifies:
        - Initializes instance attributes including:
          - self.batch_train_flag: Flag indicating batch-wise training.
          - self.save_flag: Flag indicating whether to save model checkpoints.
          - self.training_dict: Training-related information dictionary.
          - self.file_names: Dictionary containing file names.
          - self.files_directories: Dictionary containing file directories.
          - self.num_batches: Number of batches for training data division.

        Returns:
        - None
        """
        self.__set_seed()
        self.batch_train_flag = batch_train_flag
        self.save_flag = save_flag
        self.training_dict = training_dict
        self.file_names = file_names
        self.files_directories = files_directories
        self.num_batches = num_batches


    def __load_data4Training(self):
        """
        Load and prepare training data for batch training.

        This method loads various components required for training data preparation,
        such as parameters, predictors, target, transformer, and training data.
        It handles both raw and preprocessed data based on the provided information.

        Note:
        - This method assumes that the class instance has the following attributes:
          - self.file_names: Dictionary containing file names for different components.
          - self.files_directories: Dictionary containing directories of different files.
          - self.training_dict: Dictionary containing training-related information.

        Modifies:
        - Modifies the following instance attributes:
          - self.params: Loaded parameters as a JSON object.
          - self.predictors: List of predictor variables.
          - self.target: Target variable.
          - self.transformer: transformer information.
          - self.train: Training data (raw or preprocessed), if available.
          - self.holdout: Holdout data (raw or preprocessed), if available.

        Returns:
        - None
        """
        self.params = utils.Load.load_json(self.file_names["params_name"], self.files_directories['params_dir'])
        self.predictors = self.training_dict['predictors']
        self.target = self.training_dict['target']
        if self.training_dict['train'] is not None:
            self.train = self.training_dict['train']
            self.holdout = self.training_dict['holdout']
        
        if self.training_dict['processed_data'] is not None:
            self.train = self.training_dict['processed_data']
            self.holdout = None


    def __save_files(self):
        """
        Save model-related files and components.

        This method saves various model-related files and components using utility functions from utils.Save.
        It saves the trained model, transformer information, predictors, and target information.

        Note:
        - This method assumes that the class instance has the following attributes:
          - self.model: Trained machine learning model.
          - self.transformer: transformertransformer information.
          - self.predictors: List of predictor variables.
          - self.target: Target variable.
          - self.file_names: Dictionary containing file names for different components.
          - self.files_directories: Dictionary containing directories of different files.

        Returns:
        - None
        """        
        utils.Save.save_model(self.model, self.file_names["model_name"], self.files_directories['model_dir'])
        utils.Save.save_json(self.predictors, self.file_names["predictors_name"], self.files_directories['predictors_dir'])
        utils.Save.save_json(self.target, self.file_names["target_name"], self.files_directories['target_dir'])
        utils.Save.save_pickle(self.training_dict['transformer'], self.file_names["transformer_name"], self.files_directories['transformer_dir'])
        

    def batch_training(self, X=None, y=None):
        """
        Perform batch-wise training of the machine learning model.

        This method trains the model in a batch-wise manner, either using a data generator or given X and y data.
        It iterates through batches, incrementally updating the model's performance.

        Args:
        - X (numpy.ndarray or None, optional): Input features for training. None for data generator use. Default is None.
        - y (numpy.ndarray or None, optional): Target values for training. None for data generator use. Default is None.

        Modifies:
        - Updates the self.model attribute during batch-wise training.

        Returns:
        - None
        """
        if X is None and y is None:
            logger.info("Training on Data Generator, batch_train_flag is True")
            pbar = tqdm(self.data_generator, total=self.num_batches, desc="Training batches...")
            for idx, (X_batch, y_batch) in enumerate(pbar):
                if idx == 0:
                    self.model = LGBMRegressor(**self.params).fit(X_batch, y_batch)
                else:
                    train_dataset = lgbm.Dataset(X_batch, y_batch)
                    self.model = lgbm.train(self.params, train_dataset, num_boost_round=100, init_model=self.model)
                preds, targets = np.abs(np.round(self.model.predict(X_batch), decimals=0)).reshape(-1, 1), y_batch.to_numpy().reshape(-1, 1)
                pbar.set_postfix({"Metrics - Batch Data": utils.Metrics.combined_metrics_updated(targets, preds), "Shape": X_batch.shape})
                gc.collect()
            pbar.close()
        else:
            logger.info("Training on X and Y, batch_train_flag is False")
            self.model = LGBMRegressor(**self.params).fit(X, y)
            logger.info(f"Metrics - X: {utils.Metrics.combined_metrics_updated(y, np.abs(np.round(self.model.predict(X), decimals=0)).reshape(-1, 1))}")
        
    
    @utils.Decorators.calculate_execution_time
    def fit(self):
        """
        Execute the complete training pipeline.

        This method executes the complete training pipeline including loading data, preparing and optimizing data types,
        performing batch-wise training, saving model-related files, and computing metrics if holdout data is available.

        Modifies:
        - Modifies instance attributes and the self.model attribute based on the training process.

        Returns:
        - None
        """
        self.__load_data4Training()
        # self.train[self.predictors] = self.train[self.predictors].apply(pd.to_numeric, errors='coerce')

        X = self.train[self.predictors]
        y = self.train[self.target]
        if self.batch_train_flag:
            self.data_generator = DataGenerator(X, y, self.num_batches)
            self.batch_training()
        else:
            self.batch_training(X, y)

        if self.save_flag:
            self.__save_files()

        if self.holdout is not None:
            holdout_preds = np.abs(np.round(self.model.predict(self.holdout[self.predictors]), decimals=0).reshape(-1, 1))
            logger.info(f"Metrics - Holdout Data: {utils.Metrics.combined_metrics_updated(self.holdout[self.target], holdout_preds)}")


Example Usage:
trainer = BatchTrainer(train=train, holdout=None, predictors=predictors, params=params_lgbm, 
                       target=target, num_batches=3, batch_train_flag=True)
trainer.fit()
