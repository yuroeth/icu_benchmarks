import logging

import gin
import numpy as np
import tables
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm


@gin.configurable('ICUVariableLengthDataset')
class ICUVariableLengthDataset(Dataset):
    """torch.Dataset built around ICUVariableLengthLoaderTables """

    def __init__(self, source_path, split='train', maxlen=-1, scale_label=False):
        """
        Args:
            source_path (string): Path to the source h5 file.
            split (string): Either 'train','val' or 'test'.
            maxlen (int): Max size of the generated sequence. If -1, takes the max size existing in split.
            scale_label (bool): Whether or not to train a min_max scaler on labels (For regression stability).
        """
        self.h5_loader = ICUVariableLengthLoaderTables(source_path, batch_size=1, maxlen=maxlen, splits=[split])
        self.split = split
        self.maxlen = self.h5_loader.maxlen
        self.scale_label = scale_label
        if self.scale_label:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.get_labels().reshape(-1, 1))
        else:
            self.scaler = None

    def __len__(self):
        return self.h5_loader.num_samples[self.split]

    def __getitem__(self, idx):
        element = self.h5_loader.sample(None, self.split, idx)
        if len(element) == 3:
            data, label, pad_mask = element
            if self.scale_label:
                label = self.scaler.transform(label.reshape(-1, 1))[:, 0]
            return torch.from_numpy(data), torch.from_numpy(label), torch.from_numpy(pad_mask)
        elif len(element) == 4:
            data_num, data_cat, label, pad_mask = element
            if self.scale_label:
                label = self.scaler.transform(label.reshape(-1, 1))[:, 0]
            return torch.from_numpy(data_num), torch.from_numpy(data_cat), torch.from_numpy(label), torch.from_numpy(pad_mask)
        elif len(element) == 5:
            data_num, data_cat, label, pad_mask, impute_mask = element
            if self.scale_label:
                label = self.scaler.transform(label.reshape(-1, 1))[:, 0]
            return torch.from_numpy(data_num), torch.from_numpy(data_cat), torch.from_numpy(label), torch.from_numpy(pad_mask), torch.from_numpy(impute_mask)
        else:
            raise ValueError('Wrong number of elements in sample')

    def set_scaler(self, scaler):
        """Sets the scaler for labels in case of regression.

        Args:
            scaler: sklearn scaler instance

        """
        self.scaler = scaler

    def get_labels(self):
        return self.h5_loader.labels[self.split]

    def get_balance(self):
        """Return the weight balance for the split of interest.

        Returns: (list) Weights for each label.

        """
        labels = self.h5_loader.labels[self.split]
        _, counts = np.unique(labels[np.where(~np.isnan(labels))], return_counts=True)
        return list((1 / counts) * np.sum(counts) / counts.shape[0])

    def get_data_and_labels(self):
        """Function to return all the data and labels aligned at once.
        We use this function for the ML methods which don't require a iterator.

        Returns: (np.array, np.array) a tuple containing  data points and label for the split.

        """
        labels = []
        rep = []
        windows = self.h5_loader.patient_windows[self.split][:]
        resampling = self.h5_loader.label_resampling
        logging.info('Gathering the samples for split ' + self.split)
        for start, stop, id_ in tqdm(windows):
            label = self.h5_loader.labels[self.split][start:stop][::resampling][:self.maxlen]
            sample = self.h5_loader.lookup_table[self.split][start:stop][::resampling][:self.maxlen][~np.isnan(label)]
            if self.h5_loader.feature_table is not None:
                features = self.h5_loader.feature_table[self.split][start:stop, 1:][::resampling][:self.maxlen][
                    ~np.isnan(label)]
                sample = np.concatenate((sample, features), axis=-1)
            label = label[~np.isnan(label)]
            if label.shape[0] > 0:
                rep.append(sample)
                labels.append(label)
        rep = np.concatenate(rep, axis=0)
        labels = np.concatenate(labels)
        if self.scaler is not None:
            labels = self.scaler.transform(labels.reshape(-1, 1))[:, 0]
        return rep, labels

@gin.configurable('ICUVariableLengthDatasetPretrain')
class ICUVariableLengthDatasetPretrain(Dataset):
    """torch.Dataset built around ICUVariableLengthLoaderTables """

    def __init__(self, source_path, split='train'):
        """
        Args:
            source_path (string): Path to the source h5 file.
            split (string): Either 'train','val' or 'test'.
        """
        self.h5_loader = ICUVariableLengthLoaderTablesPretrain(source_path, batch_size=1, splits=[split])
        self.split = split
        self.scaler = None

    def __len__(self):
        return self.h5_loader.num_samples[self.split]

    def __getitem__(self, idx):
        x_num, x_cat, label_num, label_cat, mask_num, mask_cat, pred_idx_num, pred_idx_cat = self.h5_loader.sample(None, self.split, idx)
        x_num = torch.from_numpy(x_num)
        x_cat = torch.from_numpy(x_cat)
        label_num = torch.from_numpy(label_num)
        label_cat = torch.from_numpy(label_cat)
        mask_num = torch.from_numpy(mask_num)
        mask_cat = torch.from_numpy(mask_cat)
        pred_idx_num = torch.from_numpy(pred_idx_num)
        pred_idx_cat = torch.from_numpy(pred_idx_cat)
        return x_num, x_cat, label_num, label_cat, mask_num, mask_cat, pred_idx_num, pred_idx_cat
    
    def get_data_and_labels(self):
        """Function to return all the data and labels aligned at once.
        We use this function for the ML methods which don't require a iterator.

        Returns: (np.array, np.array) a tuple containing  data points and label for the split.

        """
        labels = []
        rep = []
        windows = self.h5_loader.patient_windows[self.split][:]
        resampling = self.h5_loader.label_resampling
        logging.info('Gathering the samples for split ' + self.split)
        for start, stop, id_ in tqdm(windows):
            label = self.h5_loader.labels[self.split][start:stop][::resampling][:self.maxlen]
            sample = self.h5_loader.lookup_table[self.split][start:stop][::resampling][:self.maxlen][~np.isnan(label)]
            if self.h5_loader.feature_table is not None:
                features = self.h5_loader.feature_table[self.split][start:stop, 1:][::resampling][:self.maxlen][
                    ~np.isnan(label)]
                sample = np.concatenate((sample, features), axis=-1)
            label = label[~np.isnan(label)]
            if label.shape[0] > 0:
                rep.append(sample)
                labels.append(label)
        rep = np.concatenate(rep, axis=0)
        labels = np.concatenate(labels)
        return rep, labels
    
    def set_scaler(self, scaler):
        """Sets the scaler for labels in case of regression.

        Args:
            scaler: sklearn scaler instance

        """
        self.scaler = scaler


@gin.configurable('ICUVariableLengthLoaderTables')
class ICUVariableLengthLoaderTables(object):
    """
    Data loader from h5 compressed files with tables to numpy for variable_size windows.
    """

    def __init__(self, data_path, on_RAM=True, shuffle=True, batch_size=1, splits=['train', 'val'], maxlen=-1, task=0,
                 data_resampling=1, label_resampling=1, use_feat=False, use_ms=True, data_frac=1.0):
        """
        Args:
            data_path (string): Path to the h5 data file which should have 3/4 subgroups :data, labels, patient_windows
            and optionally features. Here because arrays have variable length we can't stack them. Instead we
            concatenate them and keep track of the windows in a third file.
            on_RAM (boolean): Boolean whether to load data on RAM. If you don't have ram capacity set it to False.
            shuffle (boolean): Boolean to decide whether to shuffle data between two epochs when using self.iterate
            method. As we wrap this Loader in a torch Dataset this feature is not used.
            batch_size (int): Integer with size of the batch we return. As we wrap this Loader in a torch Dataset this
            is set to 1.
            splits (list): list of splits name . Default is ['train', 'val']
            maxlen (int): Integer with the maximum length of a sequence. If -1 take the maximum length in the data.
            task (int/string): Integer with the index of the task we want to train on in the labels. If string we find
            the matching tring in data_h5['tasks']
            data_resampling (int): Number of step at which we want to resample the data. Default to 1 (5min)
            label_resampling (int): Number of step at which we want to resample the labels (if they exists.
            Default to 1 (5min)
        """
        # We set sampling config
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data_h5 = tables.open_file(data_path, "r").root
        self.splits = splits
        self.maxlen = maxlen
        self.resampling = data_resampling
        self.label_resampling = label_resampling
        self.use_feat = use_feat
        self.use_ms = use_ms
        self.data_frac = data_frac

        self.columns = np.array([name.decode('utf-8') for name in self.data_h5['data']['columns'][:]])
        self.static_idx = None
        self.time_idx = None
        self.numerical_idx = []
        self.categorical_idx = []
        self.cat_cardinalities = []
        if len(self.columns) == 18:
            for i, col_name in enumerate(self.columns):
                if col_name == "Height":
                    self.static_idx = i
                if col_name == "Hours":
                    self.time_idx = i
                if len(col_name.split("_")) == 1: 
                    self.numerical_idx.append(i)
                else:
                    self.categorical_idx.append(i)
                    self.cat_cardinalities.append(int(col_name.split("_")[-1])) # Capillary refill rate_cat_2

        reindex_label = False
        tasks = np.array([name.decode('utf-8') for name in self.data_h5['labels']['tasks'][:]])
        if isinstance(task, str):
            self.task = task
            if self.task == 'Phenotyping_APACHEGroup':
                reindex_label = True

            self.task_idx = np.where(tasks == task)[0][0]
        else:
            self.task_idx = task
            self.task = tasks[task]

        self.on_RAM = on_RAM
        # Processing the data part
        if self.data_h5.__contains__('data'):
            if on_RAM:  # Faster but comsumes more RAM
                self.lookup_table = {split: self.data_h5['data'][split][:] for split in self.splits}
            else:
                self.lookup_table = {split: self.data_h5['data'][split] for split in self.splits}
        else:
            logging.warning('There is no data provided')
            self.lookup_table = None

        # impute missing values
        self.mask_table = None
        if self.lookup_table is not None and len(self.columns) == 18:
            if not self.use_ms:
                self.mask_table = {split: ~np.isnan(self.lookup_table[split]) for split in self.splits}
            all_categorical_data = np.concatenate([self.lookup_table[split][:, self.categorical_idx] for split in self.splits], axis=0) # (n_samples, n_categorical_features)
            for split in self.splits:
                # impute numerical features with mean
                temp = self.lookup_table[split][:, self.numerical_idx].copy()
                np.place(temp, mask=np.isnan(temp), vals=0.0)
                self.lookup_table[split][:, self.numerical_idx] = temp
                # impute categorical features with mode
                for i in range(len(self.categorical_idx)):
                    non_missing_values = all_categorical_data[:, i][~np.isnan(all_categorical_data[:, i])].astype(int) # (n_samples,)
                    mode = np.bincount(non_missing_values).argmax().astype(np.float32)
                    temp = self.lookup_table[split][:, self.categorical_idx[i]].copy() 
                    np.place(temp, mask=np.isnan(temp), vals=mode)
                    self.lookup_table[split][:, self.categorical_idx[i]] = temp

            # put categorical features in [0, cardinality - 1] range
            for split in self.splits:
                self.lookup_table[split][:, self.categorical_idx[1:]] -= 1 # exclude "Capillary refill rate_cat_2"

        # Processing the feature part
        if self.data_h5.__contains__('features') and self.use_feat:
            if on_RAM:  # Faster but comsumes more RAM
                self.feature_table = {split: self.data_h5['features'][split][:] for split in self.splits}
            else:
                self.feature_table = {split: self.data_h5['features'][split] for split in self.splits}
        else:
            self.feature_table = None

        # Processing the label part
        if self.data_h5.__contains__('labels'):
            self.labels = {split: self.data_h5['labels'][split][:, self.task_idx] for split in self.splits}

            # We reindex Apache groups to [0,15]
            if reindex_label:
                label_values = np.unique(self.labels[self.splits[0]][np.where(~np.isnan(self.labels[self.splits[0]]))])
                assert len(label_values) == 15

                for split in self.splits:
                    self.labels[split][np.where(~np.isnan(self.labels[split]))] = np.array(list(
                        map(lambda x: np.where(label_values == x)[0][0],
                            self.labels[split][np.where(~np.isnan(self.labels[split]))])))

            # Some steps might not be labeled so we use valid indexes to avoid them
            self.valid_indexes_labels = {split: np.argwhere(~np.isnan(self.labels[split][:])).T[0]
                                         for split in self.splits}

            self.num_labels = {split: len(self.valid_indexes_labels[split])
                               for split in self.splits}
        else:
            raise Exception('There is no labels provided')

        if self.data_h5.__contains__('patient_windows'):
            # Shape is N_stays x 3. Last dim contains [stay_start, stay_stop, patient_id]
            self.patient_windows = {split: self.data_h5['patient_windows'][split][:] for split in self.splits}
        else:
            raise Exception("patient_windows is necessary to split samples")

        # Some patient might have no labeled time points so we don't consider them in valid samples.
        self.valid_indexes_samples = {split: np.array([i for i, k in enumerate(self.patient_windows[split])
                                                       if np.any(~np.isnan(self.labels[split][k[0]:k[1]]))])
                                      for split in self.splits}
        self.num_samples = {split: len(self.valid_indexes_samples[split])
                            for split in self.splits}
        
        # limited labeled data case
        if self.data_frac < 1.0 and 'train' in self.splits:
            # randomize valid indexes samples:
            self.valid_indexes_samples['train'] = self.valid_indexes_samples['train'][np.random.permutation(self.num_samples['train'])]
            self.valid_indexes_samples['train'] = self.valid_indexes_samples['train'][:int(self.num_samples['train'] * self.data_frac)]
            self.num_samples['train'] = len(self.valid_indexes_samples['train'])
        # Iterate counters
        self.current_index_training = {'train': 0, 'test': 0, 'val': 0}

        if self.maxlen == -1:
            seq_lengths = [
                np.max(self.patient_windows[split][:, 1] - self.patient_windows[split][:, 0]) // self.resampling for
                split in
                self.splits]
            self.maxlen = np.max(seq_lengths)
        else:
            self.maxlen = self.maxlen // self.resampling

    def get_window(self, start, stop, split, pad_value=0.0):
        """Windowing function

        Args:
            start (int): Index of the first element.
            stop (int):  Index of the last element.
            split (string): Name of the split to get window from.
            pad_value (float): Value to pad with if stop - start < self.maxlen.

        Returns:
            window (np.array) : Array with data.
            pad_mask (np.array): 1D array with 0 if no labels are provided for the timestep.
            labels (np.array): 1D array with corresponding labels for each timestep.
        """
        # We resample data frequency
        window = np.copy(self.lookup_table[split][start:stop][::self.resampling])
        labels = np.copy(self.labels[split][start:stop][::self.resampling])
        if self.mask_table is not None:
            impute_mask = np.copy(self.mask_table[split][start:stop][::self.resampling])
        if self.feature_table is not None:
            feature = np.copy(self.feature_table[split][start:stop][::self.resampling])
            window = np.concatenate([window, feature], axis=-1)

        label_resampling_mask = np.zeros((stop - start,))
        label_resampling_mask[::self.label_resampling] = 1.0
        label_resampling_mask = label_resampling_mask[::self.resampling]
        length_diff = self.maxlen - window.shape[0]
        pad_mask = np.ones((window.shape[0],))

        if length_diff > 0:
            window = np.concatenate([window, np.ones((length_diff, window.shape[1])) * pad_value], axis=0)
            labels = np.concatenate([labels, np.ones((length_diff,)) * pad_value], axis=0)
            if self.mask_table is not None:
                impute_mask = np.concatenate([impute_mask, np.ones((length_diff, impute_mask.shape[1])) * pad_value], axis=0)
            pad_mask = np.concatenate([pad_mask, np.zeros((length_diff,))], axis=0)
            label_resampling_mask = np.concatenate([label_resampling_mask, np.zeros((length_diff,))], axis=0)

        elif length_diff < 0:
            window = window[:self.maxlen]
            labels = labels[:self.maxlen]
            if self.mask_table is not None:
                impute_mask = impute_mask[:self.maxlen]
            pad_mask = pad_mask[:self.maxlen]
            label_resampling_mask = label_resampling_mask[:self.maxlen]

        not_labeled = np.argwhere(np.isnan(labels))
        if len(not_labeled) > 0:
            labels[not_labeled] = -1
            pad_mask[not_labeled] = 0

        # We resample prediction frequency
        pad_mask = pad_mask * label_resampling_mask
        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        window = window.astype(np.float32)
        if self.mask_table is not None:
            impute_mask = impute_mask.astype(np.float32)
        return (window, labels, pad_mask) if self.mask_table is None else (window, labels, pad_mask, impute_mask)

    def sample(self, random_state, split='train', idx_patient=None):
        """Function to sample from the data split of choice.
        Args:
            random_state (np.random.RandomState): np.random.RandomState instance for the idx choice if idx_patient
            is None.
            split (string): String representing split to sample from, either 'train', 'val' or 'test'.
            idx_patient (int): (Optional) Possibility to sample a particular sample given a index.
        Returns:
            A sample from the desired distribution as tuple of numpy arrays (sample, label, mask).
        """

        assert split in self.splits

        if idx_patient is None:
            idx_patient = random_state.randint(self.num_samples[split], size=(self.batch_size,))
            state_idx = self.valid_indexes_samples[split][idx_patient]
        else:
            state_idx = self.valid_indexes_samples[split][idx_patient]

        patient_windows = self.patient_windows[split][state_idx]
        t_start = patient_windows[0]
        t_stop = patient_windows[1]
        if self.task == 'ihm':
            t_stop = t_start + 48
        if self.mask_table is not None:
            impute_mask = self.mask_table[split][state_idx]
        X = []
        y = []
        pad_masks = []
        if len(self.columns) != 18:
            X, y, pad_masks = self.get_window(t_start, t_stop, split)
            return X, y, pad_masks
        
        if self.mask_table is not None:
            X, y, pad_masks, impute_mask = self.get_window(t_start, t_stop, split)
        else:
            X, y, pad_masks = self.get_window(t_start, t_stop, split)
        X_num = X[:, self.numerical_idx]
        X_cat = X[:, self.categorical_idx]  
        return (X_num, X_cat, y, pad_masks) if self.mask_table is None else (
                X_num, X_cat, y, pad_masks, impute_mask)

    def iterate(self, random_state, split='train'):
        """Function to iterate over the data split of choice.
        This methods is further wrapped into a generator to build a tf.data.Dataset
        Args:
            random_state (np.random.RandomState): np.random.RandomState instance for the shuffling.
            split (string): String representing split to sample from, either 'train', 'val' or 'test'.
        Returns:
            A sample corresponding to the current_index from the desired distribution as tuple of numpy arrays.
        """
        if (self.current_index_training[split] == 0) and self.shuffle:
            random_state.shuffle(self.valid_indexes_samples[split])

        next_idx = list(range(self.current_index_training[split],
                              self.current_index_training[split] + self.batch_size))
        self.current_index_training[split] += self.batch_size

        if self.current_index_training[split] >= self.num_samples[split]:
            n_exceeding_samples = self.current_index_training[split] - self.num_samples[split]
            assert n_exceeding_samples <= self.batch_size
            next_idx = next_idx[:self.batch_size - n_exceeding_samples]
            self.current_index_training[split] = 0

        sample = self.sample(random_state, split, idx_patient=next_idx)

        return sample

@gin.configurable('ICUVariableLengthLoaderTablesPretrain')
class ICUVariableLengthLoaderTablesPretrain(object):
    """
    Data loader from h5 compressed files with tables to numpy for variable_size windows.
    """

    def __init__(self, data_path, on_RAM=True, shuffle=True, batch_size=1, splits=['train', 'val'], impute=True, drop_ms_rate=0.8, use_ms=False, use_prev_states=False):
        """
        Args:
            data_path (string): Path to the h5 data file which should have 3/4 subgroups :data, labels, patient_windows
            and optionally features. Here because arrays have variable length we can't stack them. Instead we
            concatenate them and keep track of the windows in a third file.
            on_RAM (boolean): Boolean whether to load data on RAM. If you don't have ram capacity set it to False.
            shuffle (boolean): Boolean to decide whether to shuffle data between two epochs when using self.iterate
            method. As we wrap this Loader in a torch Dataset this feature is not used.
            batch_size (int): Integer with size of the batch we return. As we wrap this Loader in a torch Dataset this
            is set to 1.
            splits (list): list of splits name . Default is ['train', 'val']
            impute (boolean): Boolean to decide whether to impute missing values with the mean for numerical features and the mode for categorical features.
            drop_ms_rate (float): Float between 0 and 1 to decide the percentage of missing values for each sample to drop.
            use_ms (boolean): Boolean to decide whether to use the imputed values to predict
            use_prev_states (boolean): Boolean to decide whether to use the previous states to predict
        """
        # We set sampling config
        # self.train_samples = 1000000 # for debugging
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data_h5 = tables.open_file(data_path, "r").root
        self.splits = splits
        self.impute = impute
        self.drop_ms_rate = drop_ms_rate
        self.use_ms = use_ms
        self.use_prev_states = use_prev_states

        self.columns = np.array([name.decode('utf-8') for name in self.data_h5['data']['columns'][:]])
        self.static_idx = None
        self.time_idx = None
        self.numerical_idx = []
        self.categorical_idx = []
        self.cat_cardinalities = []
        for i, col_name in enumerate(self.columns):
            if col_name == "Height":
                self.static_idx = i
            if col_name == "Hours":
                self.time_idx = i
            if len(col_name.split("_")) == 1: 
                self.numerical_idx.append(i)
            else:
                self.categorical_idx.append(i)
                self.cat_cardinalities.append(int(col_name.split("_")[-1])) # Capillary refill rate_cat_2

        self.on_RAM = on_RAM
        # Processing the data part
        if self.data_h5.__contains__('data'):
            if on_RAM:  # Faster but comsumes more RAM
                self.lookup_table = {split: self.data_h5['data'][split][:] for split in self.splits}
            else:
                self.lookup_table = {split: self.data_h5['data'][split] for split in self.splits}
        else:
            logging.warning('There is no data provided')
            raise ValueError('There is no data provided')
        
        if self.data_h5.__contains__('patient_windows') and self.use_prev_states:
            # Shape is N_stays x 3. Last dim contains [stay_start, stay_stop, patient_id]
            self.patient_windows = {split: self.data_h5['patient_windows'][split][:] for split in self.splits}
            self.start_steps = {split: self.patient_windows[split][:, 0] for split in self.splits}

        # generate missing value mask (0 for nan)
        self.mask_table = {split: ~np.isnan(self.lookup_table[split]) for split in self.splits}

        # impute missing values
        if self.impute is True:
            all_categorical_data = np.concatenate([self.lookup_table[split][:, self.categorical_idx] for split in self.splits], axis=0) # (n_samples, n_categorical_features)
            for split in self.splits:
                # impute numerical features with mean
                temp = self.lookup_table[split][:, self.numerical_idx].copy()
                np.place(temp, mask=np.isnan(temp), vals=0.0)
                self.lookup_table[split][:, self.numerical_idx] = temp
                # impute categorical features with mode
                for i in range(len(self.categorical_idx)):
                    non_missing_values = all_categorical_data[:, i][~np.isnan(all_categorical_data[:, i])].astype(int) # (n_samples,)
                    mode = np.bincount(non_missing_values).argmax().astype(np.float32)
                    temp = self.lookup_table[split][:, self.categorical_idx[i]].copy() 
                    np.place(temp, mask=np.isnan(temp), vals=mode)
                    self.lookup_table[split][:, self.categorical_idx[i]] = temp

        # put categorical features in [0, cardinality - 1] range
        for split in self.splits:
            self.lookup_table[split][:, self.categorical_idx[1:]] -= 1 # exclude "Capillary refill rate_cat_2"

        # drop samples with more than {self.drop_ms_rate * 100%} missing values
        self.valid_indexes_samples = {split: np.array([i for i, k in enumerate(self.mask_table[split])
                                                if (np.mean(k) >= 1 - self.drop_ms_rate) and 
                                                (np.sum(k[self.numerical_idx])>0) and 
                                                (np.sum(k[self.categorical_idx])>0)])
                                                for split in self.splits} 
        
        self.num_samples = {split: len(self.valid_indexes_samples[split]) for split in self.splits}

        # Iterate counters
        self.current_index_training = {'train': 0, 'test': 0, 'val': 0}

    def sample(self, random_state, split='train', idx_timestep=None):
        """Function to sample from the data split of choice.
        Args:
            random_state (np.random.RandomState): np.random.RandomState instance for the idx choice if idx_patient
            is None.
            split (string): String representing split to sample from, either 'train', 'val' or 'test'.
            idx_timestep (int): (Optional) Possibility to sample a particular sample given a index.
        Returns:
            A sample from the desired distribution as tuple of numpy arrays (sample, label, mask).
        """

        assert split in self.splits

        if idx_timestep is None:
            idx_timestep = random_state.randint(self.num_samples[split], size=(self.batch_size,))
            state_idx = self.valid_indexes_samples[split][idx_timestep]
        else:
            state_idx = self.valid_indexes_samples[split][idx_timestep]

        start_steps = self.start_steps[split]
        use_prev_states = self.use_prev_states

        timestep = self.lookup_table[split][state_idx]
        offset = 0
        if use_prev_states:
            prev_timestep = self.lookup_table[split][state_idx - 1]
            offset = prev_timestep.shape[0] 
            
        ms_mask = self.mask_table[split][state_idx]

        if self.batch_size == 1:
            timestep = timestep.reshape(1, -1)
            if use_prev_states:
                prev_timestep = prev_timestep.reshape(1, -1)
            ms_mask = ms_mask.reshape(1, -1)
        assert timestep.shape[0] == self.batch_size
        assert ms_mask.shape[0] == self.batch_size

        if use_prev_states: # use previous time step info to predict current states
            timestep = np.concatenate([prev_timestep, timestep], axis=1)

        # CBOW
        # valid_idx_num = [self.numerical_idx] * self.batch_size # (batch_size, n_numerical_features)
        # valid_idx_cat = [self.categorical_idx] * self.batch_size # (batch_size, n_categorical_features)
        valid_indices = [np.where(row)[0] for row in ms_mask]
        valid_idx_num = [list(set(valid_indices[i]) & set(self.numerical_idx)) 
                            for i in range(self.batch_size)] # (batch_size, n_numerical_features)
        valid_idx_cat = [list(set(valid_indices[i]) & set(self.categorical_idx))
                            for i in range(self.batch_size)] # (batch_size, n_categorical_features)
        
        pred_idx_num = [np.random.choice(valid_idx_num[i]) + offset for i in range(self.batch_size)]
        label_num = timestep[np.arange(self.batch_size), pred_idx_num]
        pred_idx_cat = [np.random.choice(valid_idx_cat[i]) + offset for i in range(self.batch_size)]
        label_cat = timestep[np.arange(self.batch_size), pred_idx_cat]
        if self.use_ms:
            mask_num = np.ones_like(timestep, dtype=bool)
            mask_cat = np.ones_like(timestep, dtype=bool)
        else: # deprecated
            mask_num = ms_mask.copy()
            mask_cat = ms_mask.copy()
        mask_num[np.arange(self.batch_size), pred_idx_num] = False
        mask_cat[np.arange(self.batch_size), pred_idx_cat] = False

        if state_idx in start_steps: # discard info from previous time step when at the beginning of a patient stay
            mask_num[:, :offset] = False
            mask_cat[:, :offset] = False

        if use_prev_states:
            x_num = np.concatenate((timestep[:, self.numerical_idx], timestep[:, [item + offset for item in self.numerical_idx]]), axis=1) # (batch_size, n_numerical_features * 2)
            x_cat = np.concatenate((timestep[:, self.categorical_idx], timestep[:, [item + offset for item in self.categorical_idx]]), axis=1) # (batch_size, n_categorical_features * 2)
        else:
            x_num = timestep[:, self.numerical_idx] # (batch_size, n_numerical_features)
            x_cat = timestep[:, self.categorical_idx] # (batch_size, n_categorical_features)

        pred_idx_num = np.array([item - offset for item in pred_idx_num], dtype=np.int32)
        pred_idx_cat = np.array([item - offset for item in pred_idx_cat], dtype=np.int32)

        if self.batch_size == 1:
            x_num = x_num.reshape(-1)
            x_cat = x_cat.reshape(-1)
            label_num = label_num.reshape(-1)
            label_cat = label_cat.reshape(-1)
            mask_num = mask_num.reshape(-1)
            mask_cat = mask_cat.reshape(-1)
            pred_idx_num = pred_idx_num.reshape(-1)
            pred_idx_cat = pred_idx_cat.reshape(-1)

        return x_num, x_cat, label_num, label_cat, mask_num, mask_cat, pred_idx_num, pred_idx_cat

    def iterate(self, random_state, split='train'):
        """Function to iterate over the data split of choice.
        This methods is further wrapped into a generator to build a tf.data.Dataset
        Args:
            random_state (np.random.RandomState): np.random.RandomState instance for the shuffling.
            split (string): String representing split to sample from, either 'train', 'val' or 'test'.
        Returns:
            A sample corresponding to the current_index from the desired distribution as tuple of numpy arrays.
        """
        if (self.current_index_training[split] == 0) and self.shuffle:
            random_state.shuffle(self.valid_indexes_samples[split])

        next_idx = list(range(self.current_index_training[split],
                              self.current_index_training[split] + self.batch_size))
        self.current_index_training[split] += self.batch_size

        if self.current_index_training[split] >= self.num_samples[split]:
            n_exceeding_samples = self.current_index_training[split] - self.num_samples[split]
            assert n_exceeding_samples <= self.batch_size
            next_idx = next_idx[:self.batch_size - n_exceeding_samples]
            self.current_index_training[split] = 0

        sample = self.sample(random_state, split, idx_patient=next_idx)

        return sample