import os
import random
import h5py
import tables
import re

import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import torch
from datetime import timedelta


def impute_sample_1h(measures, stay_time=None):
    """Forward imputes with 1h resolution any stay to duration stay_time.
    Args:
        measures: Array-like matrix with succesives measurement.
        stay_time: (Optional) Time until which we want to impute.
    Returns:
        Imputed time-series.
    """
    forward_filled_sample = impute_sample(measures)
    # forward_filled_sample = measures
    imputed_sample = [np.array(forward_filled_sample[0])]
    imputed_sample[0][0] = 0
    if not stay_time:
        max_time = int(np.ceil(measures[-1, 0]))
    else:
        max_time = int(np.ceil(stay_time))
    for k in range(1, max_time + 1):
        diff_to_k = forward_filled_sample[:, 0].astype(float) - k
        if np.argwhere(diff_to_k <= 0).shape[0] > 0:
            idx_for_k = np.argwhere(diff_to_k <= 0)[-1][0]
            time_k = np.array(forward_filled_sample[idx_for_k])
            time_k[0] = k
            imputed_sample.append(time_k)
        else:
            time_k = np.array(imputed_sample[-1])
            time_k[0] = k
            imputed_sample.append(time_k)
    imputed_sample = np.stack(imputed_sample, axis=0)

    return imputed_sample


def impute_sample(measures_t):
    """ Used under impute_sample_1h to forward impute without re-defining the resolution.
    """
    measures = np.array(measures_t)
    imputed_sample = [measures[0]]
    for k in range(1, len(measures)):
        r_t = measures[k]
        r_t_m_1 = np.array(imputed_sample[-1])
        idx_to_impute = np.argwhere(r_t == '')
        r_t[idx_to_impute] = r_t_m_1[idx_to_impute]
        imputed_sample.append(np.array(r_t))
    imputed_sample = np.stack(imputed_sample, axis=0)
    return imputed_sample


def remove_strings_col(data, col, channel_to_id, matching_dict):
    """Replaces the string arguments existing in the MIMIC-III data to category index.
    """
    transfo_data = {}
    for split in ['train', 'test', 'val']:
        current_data = np.copy(data[split])
        for channel in col:
            if channel in list(matching_dict.keys()):
                m_dict = matching_dict[channel]
                m_dict[''] = np.nan
                idx_channel = channel_to_id[channel]
                data_channel = current_data[:, idx_channel]
                r = list(map(lambda x: m_dict[x], data_channel))
                current_data[:, idx_channel] = np.array(r)
            else:
                idx_channel = channel_to_id[channel]
                data_channel = current_data[:, idx_channel]
                data_channel[np.where(data_channel == '')] = np.nan
                current_data[:, idx_channel] = data_channel.astype(float)
        transfo_data[split] = current_data.astype(float)
    return transfo_data


class Reader(object):
    """Reader class derived from https://github.com/YerevaNN/mimic3-benchmarks to read their data.
    """

    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class MultitaskReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader class derived from https://github.com/YerevaNN/mimic3-benchmarks to read their data.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]

        def process_ihm(x):
            return list(map(int, x.split(';')))

        def process_los(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x) // 2])), list(map(float, x[len(x) // 2:])))

        def process_ph(x):
            return list(map(int, x.split(';')))

        def process_decomp(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x) // 2])), list(map(int, x[len(x) // 2:])))

        self._data = [(fname, float(t), process_ihm(ihm), process_los(los),
                       process_ph(pheno), process_decomp(decomp))
                      for fname, t, ihm, los, pheno, decomp in self._data]

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.
        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Return dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            ihm : array
                Array of 3 integers: [pos, mask, label].
            los : array
                Array of 2 arrays: [masks, labels].
            pheno : array
                Array of 25 binary integers (phenotype labels).
            decomp : array
                Array of 2 arrays: [masks, labels].
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": self._data[index][1],
                "ihm": self._data[index][2],
                "los": self._data[index][3],
                "pheno": self._data[index][4],
                "decomp": self._data[index][5],
                "header": header,
                "name": name}


class DataAndTextReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader class derived from https://github.com/YerevaNN/mimic3-benchmarks to read their data.
        """
        Reader.__init__(self, dataset_dir, listfile)
        # self._text_dir = text_dir
        self._data = [line.split(',') for line in self._data[:]]

        def process_ihm(x):
            return list(map(int, x.split(';')))

        def process_los(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x) // 2])), list(map(float, x[len(x) // 2:])))

        def process_ph(x):
            return list(map(int, x.split(';')))

        def process_decomp(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x) // 2])), list(map(int, x[len(x) // 2:])))

        self._data = [(fname, float(t), process_ihm(ihm), process_los(los),
                       process_ph(pheno), process_decomp(decomp))
                      for fname, t, ihm, los, pheno, decomp in self._data]

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    # def _read_textseries(self, ts_filename):
    #     ret = []
    #     csv_file = os.path.join(self._text_dir, ts_filename)
    #     df = pd.read_csv(csv_file)
    #     header = list(df.columns)

    #     assert header == ['Hours', 'CATEGORY', 'TEXT']
    #     df["CATEGORY_TEXT"] = df["CATEGORY"] + " \t " + df["TEXT"]
    #     df = df[["Hours", "CATEGORY_TEXT"]]
    #     df["Hours"] = df["Hours"].apply(lambda x: np.ceil(np.clip(x, 0.0, 1e6)))
    #     value = df.values
    #     header = list(df.columns)

    #     return (value, header)

    def read_example(self, index):
        """ Reads the example with given index.
        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Return dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            ihm : array
                Array of 3 integers: [pos, mask, label].
            los : array
                Array of 2 arrays: [masks, labels].
            pheno : array
                Array of 25 binary integers (phenotype labels).
            decomp : array
                Array of 2 arrays: [masks, labels].
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")
        name = self._data[index][0]
        # text_name = name.replace('timeseries', 'textseries')

        (X, header) = self._read_timeseries(name)
        # (T, T_header) = self._read_textseries(text_name)
        # if len(T) == 0:  # Handling case where no text is provided
        #     return None

        return {"X": X,
                # "T": T,
                "t": self._data[index][1],
                "ihm": self._data[index][2],
                "los": self._data[index][3],
                "pheno": self._data[index][4],
                "decomp": self._data[index][5],
                "header": header,
                "name": name}
                # "text_header": T_header


def extract_raw_data(base_path, text_path):
    """Wrapper around MultitaskReader to extract MIMIC-III benchmark data to our h5 format.
    Args:
        base_path: Path to source data 'data/multitask'.
        You obtain it with this command 'python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/'.
    Returns:
        data_d: Dict with data array concatenating all stays for each split.
        labels_d: Dict with labels array associate data_d for each split.
        patient_windows_d : Containing the (start, stop, patient_id) for each stay in each split.
        col: Name of the columns in data
        tasks: Name of the columns in labels
    """
    data_d = {}
    labels_d = {}
    patient_window_d = {}
    # text_d = {}
    # text_window_d = {}

    for split in ['train', 'test', 'val']:
        print('Extracting Data for {} Split.'.format(split))
        if split in ['train', 'val']:
            folder = os.path.join(base_path, 'train')
            # folder_text = os.path.join(text_path, 'train_text')
        else:
            folder = os.path.join(base_path, 'test')
            # folder_text = os.path.join(text_path, 'test_text')

        file = os.path.join(base_path, split + '_listfile.csv')
        # sample_reader = DataAndTextReader(folder, folder_text, file)
        sample_reader = DataAndTextReader(folder, file)
        num_samples = sample_reader.get_number_of_examples()
        lookup_table = []
        # text_table = []
        # text_start_stop_id = []
        start_stop_id = []
        labels_split = []
        current_idx = 0
        # current_idx_text = 0
        col = []

        for idx in tqdm(range(num_samples)):
            patient_sample = sample_reader.read_example(idx)
            if patient_sample is None:
                continue
            elif not col:  # setting col only once as it is always the same
                col = list(patient_sample['header'])
            d = patient_sample['X']
            # texts = patient_sample['T']
            imputed_d = impute_sample_1h(d, float(patient_sample['t']))
            patient_id = int(patient_sample['name'].split('_')[0])
            episode_nb = int(re.search('[0-9]+', patient_sample['name'].split('_')[1]).group(0))
            stay_id = episode_nb * 1000000 + patient_id  # We avoid confusing different episodes
            label_los = patient_sample['los']
            label_decomp = patient_sample['decomp'] # (masks;labels)
            label_ihm = patient_sample['ihm']  # (pos;mask;label)
            label_pheno = patient_sample['pheno']

            n_step = int(np.ceil(patient_sample['t']))

            # Handling of samples where LOS and Decomp masks are not same shape
            if len(patient_sample['los'][0]) > n_step:
                label_los = (patient_sample['los'][0][:n_step], patient_sample['los'][1][:n_step])
            elif len(patient_sample['los'][0]) < n_step:
                raise Exception()
            if len(patient_sample['decomp'][0]) > n_step:
                label_decomp = (patient_sample['decomp'][0][:n_step], patient_sample['decomp'][1][:n_step])
            if len(label_decomp[0]) - len(label_los[0]) != 0:
                adding = [0 for k in range(abs(len(label_decomp[0]) - len(label_los[0])))]
                new_mask = label_decomp[0] + adding
                new_labels = label_decomp[1] + adding
                label_decomp = (new_mask, new_labels)
                assert len(label_decomp[0]) - len(label_los[0]) == 0

            # We build labels in our format witih np.nan when we don't have a label
            mask_decomp, label_decomp = label_decomp
            mask_los, label_los = label_los
            mask_decomp = np.array(mask_decomp).astype(float)
            mask_los = np.array(mask_los).astype(float)
            mask_decomp[np.argwhere(mask_decomp == 0)] = np.nan
            mask_los[np.argwhere(mask_los == 0)] = np.nan
            masked_labels_los = np.concatenate([[np.nan], mask_los * np.array(label_los)], axis=0)
            masked_labels_decomp = np.concatenate([[np.nan], mask_decomp * np.array(label_decomp).astype(float)],
                                                  axis=0)
            assert imputed_d.shape[0] == masked_labels_los.shape[-1]

            masked_labels_ihm = masked_labels_los.copy()
            masked_labels_ihm[:] = np.nan
            if label_ihm[1] == 1:  # if mask == 1, set label at position to 1
                masked_labels_ihm[label_ihm[0]] = label_ihm[2]

            masked_labels_pheno = masked_labels_los.copy()
            masked_labels_pheno[:] = np.nan
            masked_labels_pheno = masked_labels_pheno[np.newaxis, :]
            masked_labels_pheno = np.repeat(masked_labels_pheno, len(label_pheno), axis=0)
            masked_labels_pheno[:, -1] = np.array(label_pheno)


            # Data
            lookup_table.append(imputed_d)
            start_stop_id.append([current_idx, current_idx + len(imputed_d), stay_id])
            current_idx = current_idx + len(imputed_d)
            labels_split.append(np.concatenate([
                masked_labels_los.reshape((1, -1)),
                masked_labels_decomp.reshape((1, -1)),
                masked_labels_ihm.reshape((1, -1)),
                masked_labels_pheno
            ], axis=0))

            # Text
            # text_table.append(texts)
            # text_start_stop_id.append([current_idx_text, current_idx_text + len(texts), stay_id])
            # current_idx_text = current_idx_text + len(texts)

        data_d[split] = np.concatenate(lookup_table, axis=0)
        # text_d[split] = np.concatenate(text_table, axis=0)

        labels_d[split] = np.concatenate(labels_split, axis=1).T
        patient_window_d[split] = np.array(start_stop_id)
        # text_window_d[split] = np.array(text_start_stop_id)
        tasks = ['los', 'decomp', 'ihm'] + [f'pheno_{i}' for i in range(len(label_pheno))]
        text_d = None
        text_window_d = None
    return data_d, labels_d, patient_window_d, text_d, text_window_d, col, tasks


def put_static_first(data, col, static_col):
    """Simple function putting the static columns first in the data.
    Args:
        data: Dict with a data array for each split.
        col: Ordered list of the columns in the data.
        static_col: List of static columns names.
    Returns:
        data_inverted : Analog  to data with columns reordered in each split.
        col_inverted : Analog to col woth columns names reordered.
    """
    static_index = list(np.where(np.isin(np.array(col), static_col))[0])
    n_col = len(col)
    non_static_index = [k for k in range(n_col) if k not in static_index]
    new_idx = static_index + non_static_index
    data_inverted = {}
    for split in ['train', 'test', 'val']:
        data_inverted[split] = data[split][:, new_idx]
    col_inverted = list(np.array(col)[new_idx])
    return data_inverted, col_inverted


def clip_dataset(var_range, data, columns):
    """Set each values outside of predefined range to NaN.
    Args:
        var_range: Dict with associated range [min,max] to each variable name.
        data: Dict with a data array for each split.
        columns: Ordered list of the columns in the data.
    Returns:
        new_data : Data with no longer any value outside of the range.
    """
    new_data = {}
    for split in ['train', 'test', 'val']:
        clipped_data = data[split][:]
        for i, col in enumerate(columns):
            if var_range.get(col):
                idx = np.sort(np.concatenate([np.argwhere(clipped_data[:, i] > var_range[col][1]),
                                              np.argwhere(clipped_data[:, i] < var_range[col][0])])[:, 0])
                clipped_data[idx, i] = np.nan
        new_data[split] = clipped_data
    return new_data


def finding_cat_features(rep_data, threshold):
    """
    Extracts the index and names of categorical in a pre-built dataset.
    Args:
        rep_data: Pre-built dataset as a h5py.File(...., 'r').
        threshold: Number of uniqur value below which we consider a variable as categorical if it's an integer
    Returns:
        categorical: List of names containing categorical features.
        categorical_idx: List of matching column indexes.
    """
    columns = np.array([name.decode('utf-8') for name in rep_data['data']['columns'][:]])

    categorical = []

    for i, c in enumerate(columns):
        values = rep_data['data']['train'][:, i]
        values = values[~np.isnan(values)]
        nb_values = len(np.unique(values))

        if nb_values <= threshold and np.all(values == values.astype(int)):
            categorical.append(c)

    categorical_idx = np.sort([np.argwhere(columns == feat)[0, 0] for feat in categorical])

    return categorical, categorical_idx


def finding_cat_features_fom_file(rep_data, info_df):
    """
    Extracts the index and names of categorical in a pre-built dataset.
    Args:
        rep_data: Pre-built dataset as a h5py.File(...., 'r').
        info_df: Dataframe with information on each variable.
    Returns:
        categorical: List of names containing categorical features.
        categorical_idx: List of matching column indexes.
    """
    columns = np.array([name.decode('utf-8') for name in rep_data['data']['columns'][:]])
    categorical = []

    for i, c in enumerate(columns):
        if c.split('_')[0] != 'plain':
            pass
        else:
            if info_df[info_df['VariableID'] == c.split('_')[-1]]['Datatype'].values == 'Categorical':
                categorical.append(c)
    categorical_idx = np.sort([np.argwhere(columns == feat)[0, 0] for feat in categorical])
    return categorical, categorical_idx


def get_one_hot(rep_data, cat_names, cat_idx):
    """
    One-hots the categorical features in a given pre-built dataset.
    Args:
        rep_data: Pre-built dataset as a h5py.File(...., 'r').
        cat_names: List of names containing categorical features.
        cat_idx: List of matching column indexes.
    Returns:
        all_categorical_data: Dict with each split one-hotted categorical column as a big array.
        col_name: List of name of the matching columns
    """
    all_categorical_data = np.concatenate([rep_data['data']['train'][:, cat_idx],
                                           rep_data['data']['test'][:, cat_idx],
                                           rep_data['data']['val'][:, cat_idx]], axis=0)
    cat_dict = {}
    col_name = []
    for i, cat in enumerate(cat_idx):
        dum = np.array(pd.get_dummies(all_categorical_data[:, i]))
        if dum.shape[-1] <= 2:
            dum = dum[:, -1:]
            col_name += [cat_names[i].split('_')[-1] + '_cat']
        else:
            col_name += [cat_names[i].split('_')[-1] + '_cat_' + str(k) for k in range(dum.shape[-1])]
        cat_dict[cat] = dum

    all_categorical_data_one_h = np.concatenate(list(cat_dict.values()), axis=1)

    all_categorical_data = {}
    all_categorical_data['train'] = all_categorical_data_one_h[:rep_data['data']['train'].shape[0]]
    all_categorical_data['test'] = all_categorical_data_one_h[
                                   rep_data['data']['train'].shape[0]:rep_data['data']['train'].shape[0] +
                                                                      rep_data['data']['test'].shape[0]]
    all_categorical_data['val'] = all_categorical_data_one_h[-rep_data['data']['val'].shape[0]:]

    return all_categorical_data, col_name

def get_multi_cat(rep_data, cat_names, cat_idx):
    all_categorical_data = {}
    col_name = []
    all_categorical_data['train'] = rep_data['data']['train'][:, cat_idx]
    all_categorical_data['test'] = rep_data['data']['test'][:, cat_idx]
    all_categorical_data['val'] = rep_data['data']['val'][:, cat_idx]
    all_data = np.concatenate([all_categorical_data['train'], all_categorical_data['test'], all_categorical_data['val']], axis=0)
    for i in range(len(cat_idx)):
        dum = np.array(pd.get_dummies(all_data[:, i]))
        col_name += [cat_names[i].split('_')[-1] + '_cat_' + str(dum.shape[-1])]
    return all_categorical_data, col_name


def scaling_data_common(data_path, threshold=25, scaler=StandardScaler(), static_idx=None, df_ref=None):
    """
    Wrapper which one-hot and scales the a pre-built dataset.
    Args:
        data_path: String with the path to the pre-built non scaled dataset
        threshold: Int below which we consider a variable as categorical
        scaler: sklearn Scaler to use, default is StandardScaler.
        static_idx: List of indexes containing static columns.
        df_ref: Reference dataset containing supplementary information on the columns.
    Returns:
        data_dic: dict with each split as a big array.
        label_dic: dict with each split and and labels array in same order as lookup_table.
        patient_dic: dict containing a array for each split such that each row of the array is of the type
        [start_index, stop_index, patient_id].
        col: list of the variables names corresponding to each column.
        labels_name: list of the tasks name corresponding to labels columns.
    """
    rep_data = tables.open_file(data_path, "r").root
    columns = np.array([name.decode('utf-8') for name in rep_data['data']['columns'][:]])
    train_data = rep_data['data']['train'][:]
    test_data = rep_data['data']['test'][:]
    val_data = rep_data['data']['val'][:]

    time_idx = np.where(columns == 'Hours')[0]  # We scale time with a MinMaxScaler

    # We just extract tasks name to propagate
    if rep_data.__contains__('labels'):
        labels_name = np.array([name.decode('utf-8') for name in rep_data['labels']['tasks'][:]])
    else:
        labels_name = None
    # We treat np.inf and np.nan as the same
    np.place(train_data, mask=np.isinf(train_data), vals=np.nan)
    np.place(test_data, mask=np.isinf(test_data), vals=np.nan)
    np.place(val_data, mask=np.isinf(val_data), vals=np.nan)
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    # We pad after scaling, Thus zero is equivalent to padding with the mean value across patient
    np.place(train_data_scaled, mask=np.isnan(train_data_scaled), vals=0.0)
    np.place(test_data_scaled, mask=np.isnan(test_data_scaled), vals=0.0)
    np.place(val_data_scaled, mask=np.isnan(val_data_scaled), vals=0.0)

    # If we have static values we take one per patient stay
    if static_idx:
        train_static_values = train_data[rep_data['patient_windows']['train'][:][:, 0]][:, static_idx]
        static_scaler = StandardScaler()
        static_scaler.fit(train_static_values)

        # Scale all entries
        train_data_static_scaled = static_scaler.transform(train_data[:, static_idx])
        val_data_static_scaled = static_scaler.transform(val_data[:, static_idx])
        test_data_static_scaled = static_scaler.transform(test_data[:, static_idx])
        # Replace NaNs
        np.place(train_data_static_scaled, mask=np.isnan(train_data_static_scaled), vals=0.0)
        np.place(val_data_static_scaled, mask=np.isnan(val_data_static_scaled), vals=0.0)
        np.place(test_data_static_scaled, mask=np.isnan(test_data_static_scaled), vals=0.0)

        # Insert in the scaled dataset
        train_data_scaled[:, static_idx] = train_data_static_scaled
        test_data_scaled[:, static_idx] = test_data_static_scaled
        val_data_scaled[:, static_idx] = val_data_static_scaled

    if time_idx:
        train_time_values = train_data[:, time_idx]
        time_scaler = MinMaxScaler()
        time_scaler.fit(train_time_values)

        # Scale all entries
        train_data_time_scaled = time_scaler.transform(train_data[:, time_idx])
        val_data_time_scaled = time_scaler.transform(val_data[:, time_idx])
        test_data_time_scaled = time_scaler.transform(test_data[:, time_idx])

        # Insert in the scaled dataset
        train_data_scaled[:, time_idx] = train_data_time_scaled
        test_data_scaled[:, time_idx] = test_data_time_scaled
        val_data_scaled[:, time_idx] = val_data_time_scaled

    # We deal with the categorical features.
    if df_ref is None:
        cat_names, cat_idx = finding_cat_features(rep_data, threshold)
    else:
        cat_names, cat_idx = finding_cat_features_fom_file(rep_data, df_ref)

    # We check for columns that are both categorical and static
    if static_idx:
        common_idx = [idx for idx in cat_idx if idx in static_idx]
        if common_idx:
            common_name = columns[common_idx]
        else:
            common_name = None

    if len(cat_names) > 0:
        # We one-hot categorical features with more than two possible values
        all_categorical_data, oh_cat_name = get_one_hot(rep_data, cat_names, cat_idx)
        # all_categorical_data, oh_cat_name = get_multi_cat(rep_data, cat_names, cat_idx) 
        if common_name is not None:
            common_cat_name = [c for c in oh_cat_name if c.split('_')[0] in common_name]

        # We replace them at the end of the features
        train_data_scaled = np.concatenate([np.delete(train_data_scaled, cat_idx, axis=1),
                                            all_categorical_data['train']], axis=-1)
        test_data_scaled = np.concatenate([np.delete(test_data_scaled, cat_idx, axis=1),
                                           all_categorical_data['test']], axis=-1)
        val_data_scaled = np.concatenate([np.delete(val_data_scaled, cat_idx, axis=1),
                                          all_categorical_data['val']], axis=-1)
        columns = np.concatenate([np.delete(columns, cat_idx, axis=0), oh_cat_name], axis=0)

        # We ensure that static categorical features are also among the first features with other static ones.
        if common_name is not None:
            common_current_idx = [i for i, n in enumerate(columns) if n in common_cat_name]
            new_idx = common_current_idx + [k for k in range(len(columns)) if k not in common_current_idx]
            columns = columns[new_idx]
            train_data_scaled = train_data_scaled[:, new_idx]
            test_data_scaled = test_data_scaled[:, new_idx]
            val_data_scaled = val_data_scaled[:, new_idx]

    data_dic = {'train': train_data_scaled,
                'test': test_data_scaled,
                'val': val_data_scaled}

    if rep_data.__contains__('labels'):
        label_dic = {split: rep_data['labels'][split][:] for split in data_dic.keys()}
    else:
        label_dic = None

    if rep_data.__contains__('patient_windows'):

        patient_dic = {split: rep_data['patient_windows'][split][:] for split in data_dic.keys()}
    else:
        patient_dic = None

    return data_dic, label_dic, patient_dic, columns, labels_name, time_scaler


def scale_text(h5_path, time_scaler, nlp_model, note_embedding):
    rep_data = h5py.File(h5_path, "r")
    note_types_path = os.path.join(os.path.dirname(h5_path), 'note_types.pkl')
    columns = np.array([name for name in rep_data['data'].attrs['columns'][:]])
    train_data = rep_data['data']['train'][:]
    test_data = rep_data['data']['test'][:]
    val_data = rep_data['data']['val'][:]

    time_idx = np.where(columns == 'Hours')[0]  # We scale time with a MinMaxScaler
    train_data[:, time_idx] = time_scaler.transform(train_data[:, time_idx])
    test_data[:, time_idx] = time_scaler.transform(test_data[:, time_idx])
    val_data[:, time_idx] = time_scaler.transform(val_data[:, time_idx])

    patient_windows = {'train': rep_data['patient_windows']['train'][:],
                       'test': rep_data['patient_windows']['test'][:],
                       'val': rep_data['patient_windows']['val'][:]}

    if nlp_model:
        train_data, patient_windows['train'] = get_tokenized_data_and_window(train_data, patient_windows['train'],
                                                                             time_idx, nlp_model, note_embedding,
                                                                             note_types_path, 'train')
        test_data, patient_windows['test'] = get_tokenized_data_and_window(test_data, patient_windows['test'],
                                                                           time_idx, nlp_model, note_embedding,
                                                                           note_types_path, 'test')
        val_data, patient_windows['val'] = get_tokenized_data_and_window(val_data, patient_windows['val'],
                                                                         time_idx, nlp_model, note_embedding,
                                                                         note_types_path, 'val')

    data_dic = {'train': train_data,
                'test': test_data,
                'val': val_data}

    return data_dic, patient_windows


def get_tokenized_data_and_window(data, data_windows, time_idx, nlp_model, note_embedding, note_types_path, split):
    def update_windows(wins, reps):
        train_window_stops = np.array([
            [sum(reps[:stop])]
            for _, stop, _ in wins
        ])
        train_window_starts = train_window_stops[:-1]  # shape: (n_starts, 1)
        train_window_starts = np.concatenate([np.array([[0]]), train_window_starts])
        pids = wins[:, [2]]
        updated_windows = np.concatenate(
            [train_window_starts, train_window_stops, pids],
            axis=-1,
            dtype=np.int32
        )
        return updated_windows

    def update_note_types_file(reps):
        stops = reps.cumsum()
        starts = stops[:-1]
        starts = np.concatenate([np.array([0]), starts])
        type_windows = np.stack([starts, stops], axis=-1)

        # read current note_types
        with open(note_types_path, 'rb') as f:
            notes = pickle.load(f)

        updated_note_types_split = {}
        for note_type in notes.keys():
            idxs_collector = []
            # for 'Case Management' we try to access 'train' but only exists for 'val' for certain batch
            if split not in notes[note_type].keys():
                continue
            for idx in notes[note_type][split]:
                start, stop = type_windows[idx]
                idxs_collector.append(np.arange(start, stop))
            if not idxs_collector:
                idxs_collector.append(np.array([]))
            updated_note_types_split[note_type] = np.concatenate(idxs_collector)

        # update notes
        for note_type in updated_note_types_split.keys():
            notes[note_type][split] = updated_note_types_split[note_type]

        with open(note_types_path, 'wb') as f:
            pickle.dump(notes, f)

    text = np.delete(data, time_idx, axis=-1)
    time = data[:, time_idx]
    if note_embedding in ['chunks', 'sentences']:  # len(tokenize_text())==2
        tokens, time_repetitions = tokenize_text(text, nlp_model, chunk_size=15)
        token_times = np.repeat(time, time_repetitions, axis=0)
        update_note_types_file(time_repetitions)
        windows = update_windows(data_windows, time_repetitions)
    elif note_embedding == 'single':
        tokens, _ = tokenize_text(text, nlp_model)
        token_times = time
        windows = data_windows
    else:
        raise ValueError("Only 'chunks' and 'single' are valid note_embedding arguments")
    return np.concatenate([token_times, tokens], axis=-1), windows


def tokenize_text(text_data, embedding_model, chunk_size=25):
    embedded_data = []
    splits_per_note = []
    k = 0
    n_chunks = np.ceil(text_data.shape[0] / chunk_size)
    tqdm_split = tqdm(np.array_split(text_data, n_chunks, axis=0), position=0, leave=False)
    for encoded_text_chunk in tqdm_split:
        text_patient = [t[0].decode('utf-8') for t in encoded_text_chunk]
        with torch.no_grad():
            out = embedding_model(text_patient)
            if len(out) == 2:
                embeddings, n_splits = out[0], out[1]
                embedded_data.append(embeddings.cpu().numpy())
                splits_per_note.append(n_splits)
            else:
                embedded_data.append(out.cpu().numpy())
        if k % 10 == 0:
            torch.cuda.empty_cache()
        k += 1
    if splits_per_note:
        return np.concatenate(embedded_data, axis=0), np.concatenate(splits_per_note, axis=0)
    else:
        return np.concatenate(embedded_data, axis=0)


def _write_data_to_hdf(data, dataset_name, node, f, first_write, nr_cols, expectedrows=1000000):
    filters = tables.Filters(complevel=5, complib='blosc:lz4')

    if first_write:
        ea = f.create_earray(node, dataset_name,
                             atom=tables.Atom.from_dtype(data.dtype),
                             expectedrows=expectedrows,
                             shape=(0, nr_cols),
                             filters=filters)
        if len(data) > 0:
            ea.append(data)
    elif len(data) > 0:
        node[dataset_name].append(data)


def save_to_h5_with_tasks(save_path, col_names, task_names, data_dict, label_dict,
                          patient_windows_dict):
    """
    Save a dataset with the desired format as h5.
    Args:
        save_path: Path to save the dataset to.
        col_names: List of names the variables in the dataset.
        data_dict: Dict with an array for each split of the data
        label_dict: (Optional) Dict with each split and and labels array in same order as lookup_table.
        patient_windows_dict: Dict containing a array for each split such that each row of the array is of the type
        [start_index, stop_index, patient_id].
    Returns:
    """

    # data labels windows

    first_write = not save_path.exists()
    mode = 'w' if first_write else 'a'

    with tables.open_file(save_path, mode) as f:
        if first_write:
            n_data = f.create_group("/", 'data', 'Dataset')
            f.create_array(n_data, 'columns', obj=[str(k).encode('utf-8') for k in col_names])
        else:
            n_data = f.get_node('/data')

        splits = ['train', 'val', 'test']
        for split in splits:
            _write_data_to_hdf(data_dict[split].astype(float), split, n_data, f, first_write,
                               data_dict['train'].shape[1])

        if label_dict is not None:
            if first_write:
                labels = f.create_group("/", 'labels', 'Labels')
                f.create_array(labels, 'tasks', obj=[str(k).encode('utf-8') for k in task_names])
            else:
                labels = f.get_node('/labels')

            for split in splits:
                _write_data_to_hdf(label_dict[split].astype(float), split, labels, f, first_write,
                                   label_dict['train'].shape[1])

        if patient_windows_dict is not None:
            if first_write:
                p_windows = f.create_group("/", 'patient_windows', 'Windows')
            else:
                p_windows = f.get_node('/patient_windows')

            for split in splits:
                _write_data_to_hdf(patient_windows_dict[split].astype(int), split, p_windows, f, first_write,
                                   patient_windows_dict['train'].shape[1])

        if not len(col_names) == data_dict['train'].shape[-1]:
            raise Exception(
                "We saved to data but the number of columns ({}) didn't match the number of features {} ".format(
                    len(col_names), data_dict['train'].shape[-1]))


def save_to_h5_with_tasks_old(save_path, col_names, task_names, data_dict, label_dict, patient_windows_dict):
    """
    Save a dataset with the desired format as h5.
    Args:
        save_path: Path to save the dataset to.
        col_names: List of names the variables in the dataset.
        data_dict: Dict with an array for each split of the data
        label_dict: (Optional) Dict with each split and and labels array in same order as lookup_table.
        patient_windows_dict: Dict containing a array for each split such that each row of the array is of the type
        [start_index, stop_index, patient_id].
    Returns:
    """
    with h5py.File(save_path, "w") as f:
        n_data = f.create_group('data')
        n_data.create_dataset('train', data=data_dict['train'].astype(float), dtype=np.float32)
        n_data.create_dataset('test', data=data_dict['test'].astype(float), dtype=np.float32)
        n_data.create_dataset('val', data=data_dict['val'].astype(float), dtype=np.float32)
        n_data.attrs['columns'] = list(col_names)

        if label_dict is not None:
            labels = f.create_group('labels')
            labels.create_dataset('train', data=label_dict['train'], dtype=np.float32)
            labels.create_dataset('test', data=label_dict['test'], dtype=np.float32)
            labels.create_dataset('val', data=label_dict['val'], dtype=np.float32)
            labels.attrs['tasks'] = list(task_names)

        if patient_windows_dict is not None:
            p_windows = f.create_group('patient_windows')
            p_windows.create_dataset('train', data=patient_windows_dict['train'], dtype=np.int32)
            p_windows.create_dataset('test', data=patient_windows_dict['test'], dtype=np.int32)
            p_windows.create_dataset('val', data=patient_windows_dict['val'], dtype=np.int32)

        if not len(col_names) == data_dict['train'].shape[-1]:
            raise Exception(
                "We saved to data but the number of columns ({}) didn't match the number of features {} ".format(
                    len(col_names), data_dict['train'].shape[-1]))


def save_to_text_to_h5(save_path, col_names, data_dict, patient_windows_dict):
    """
    Save a dataset with the desired format as h5.
    Args:
        save_path: Path to save the dataset to.
        col_names: List of names the variables in the dataset.
        data_dict: Dict with an array for each split of the data
        label_dict: (Optional) Dict with each split and and labels array in same order as lookup_table.
        patient_windows_dict: Dict containing a array for each split such that each row of the array is of the type
        [start_index, stop_index, patient_id].
    Returns:
    """
    dt = h5py.string_dtype('utf-8')

    def encode(string):
        if isinstance(string, bytes):
            return string
        else:
            return str(string).encode('utf-8')

    with h5py.File(save_path, "w") as f:
        n_data = f.create_group('data')
        train_data = [[encode(k) for k in row] for row in data_dict['train']]
        n_data.create_dataset('train', data=[[encode(k) for k in row] for row in data_dict['train']],
                              dtype=dt)
        n_data.create_dataset('test', data=[[encode(k) for k in row] for row in data_dict['test']],
                              dtype=dt)
        n_data.create_dataset('val', data=[[encode(k) for k in row] for row in data_dict['val']], dtype=dt)
        n_data.attrs['columns'] = list(col_names)

        if patient_windows_dict is not None:
            p_windows = f.create_group('patient_windows')
            p_windows.create_dataset('train', data=patient_windows_dict['train'], dtype=np.int32)
            p_windows.create_dataset('test', data=patient_windows_dict['test'], dtype=np.int32)
            p_windows.create_dataset('val', data=patient_windows_dict['val'], dtype=np.int32)


def benchmark_to_h5(base_path, text_path, channel_to_id, matching_dict, var_range, static_col=['Height']):
    """Wrapper around the full pre-process
    """
    data_d, labels_d, patient_window_d, text_d, text_window_d, col, tasks = extract_raw_data(base_path, text_path)

    no_string_data = remove_strings_col(data_d, col, channel_to_id, matching_dict)

    clipped_data = clip_dataset(var_range, no_string_data, col)

    data_inverted, col_inverted = put_static_first(clipped_data, col, static_col)

    return data_inverted, labels_d, patient_window_d, text_d, text_window_d, col_inverted, tasks


def create_note_type_per_split_table(save_path):
    """
    Saves lookup table with note type per split and corresponding index from texts.h5
    Args:
        save_path: Path to where texts.h5 is stored and note_types.pkl file should be saved to.
    """
    texts_path = os.path.join(save_path, 'texts.h5')
    data_h5 = h5py.File(texts_path, "r")

    note_types = {}
    for split in data_h5['data']:
        for i, [time, text] in enumerate(data_h5['data'][split][:]):
            n_t = text.decode('utf-8').split(' \t ', maxsplit=1)[0].strip()
            if n_t not in note_types.keys():
                note_types[n_t] = {}
            if split not in note_types[n_t].keys():
                note_types[n_t][split] = []
            note_types[n_t][split].append(i)

    for n_t in note_types:
        for split in note_types[n_t]:
            note_types[n_t][split] = np.array(note_types[n_t][split])

    note_types_path = os.path.join(save_path, 'note_types.pkl')
    with open(note_types_path, 'wb') as f:
        pickle.dump(note_types, f)


"""Functions to extract text for mimic dataset"""


def add_hours_elapsed_to_events(events, dt, remove_charttime=True):
    """
    Adds elapsed hours since a given datetime to the dataframe
    Args:
        events: Events dataframe to whcih hours elapsed since datetime should be added as column 'Hours'.
        dt: Datetime relative to which hours elapsed should be calculated to.
        remove_charttime: Boolean based on which 'CHARTTIME' column is removed from events dataframe.
    """
    events = events.copy()
    events['HOURS'] = (pd.to_datetime(events.CHARTTIME) - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60. / 60
    if remove_charttime:
        del events['CHARTTIME']
    return events


def read_stays(subject_path):
    """
    Reads the stays.csv file for the single subject's given path and converts timestamps to datetime
    Args:
        subject_path: Path to single subject ID for which the stays.csv file should be read.
    """
    stays = pd.read_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.DOB = pd.to_datetime(stays.DOB)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays


def add_no_charttime_prestring(events):
    """
    Prepends 'NCT_' to category of given patient dataframe.
    :param events: Patient dataframe.
    :return: Patient dataframe with category names prepended with 'NCT_'.
    """
    events = events.copy()
    events['CATEGORY'] = 'NCT_' + events['CATEGORY']
    return events


def add_chartdate_end_of_day_as_charttime(events):
    """
    Adds end of day of chartdate as charttime to given patient dataframe.
    :param events: Patient dataframe.
    :return: Patient dataframe with added charttime.
    """
    if pd.isna(events['CHARTDATE']).any():
        raise Exception('Given events have notes with no CHARTDATE which is required')

    events = events.copy()
    events['CHARTTIME'] = pd.to_datetime(events.CHARTDATE) + timedelta(days=1) - timedelta(seconds=1)
    return events


def extract_text_data(subjects_root_path, out_path, noteevents_path):
    """
    Extracts mimic base text data into texts per patients. For patients without CHARTTIME, end of day of CHARTDATE is used.
    Args:
        subjects_root_path: Path to basic data (train or test).
        out_path: Path to where extracted data should be saved to (train_text or test_text).
        noteevents_path: Path to where mimic source NOTEEVENTS.csv.
    """
    pat = pd.read_csv(noteevents_path)
    for subject_dir in tqdm(os.listdir(subjects_root_path)):
        try:
            stays = read_stays(os.path.join(subjects_root_path, subject_dir))

            for i in range(stays.shape[0]):
                hadm_id = stays.HADM_ID.iloc[i]
                pid = stays.SUBJECT_ID.iloc[i]
                intime = stays.INTIME.iloc[i]

                pat_df = pat[pat.HADM_ID == hadm_id].copy()  # get around pandas SettingWithCopyWarning with copying

                # prepare static texts (i.e texts without charttime)
                pat_df.loc[pd.isna(pat_df.CHARTTIME)] = add_no_charttime_prestring(
                    pat_df.loc[pd.isna(pat_df.CHARTTIME)])
                pat_df.loc[pd.isna(pat_df.CHARTTIME)] = add_chartdate_end_of_day_as_charttime(
                    pat_df.loc[pd.isna(pat_df.CHARTTIME)])

                pat_df = add_hours_elapsed_to_events(pat_df, intime).set_index('HOURS').sort_index(axis=0)
                pat_selected = pat_df[['CATEGORY', 'TEXT']]
                pat_selected.to_csv(os.path.join(out_path, '{}_episode{}_textseries.csv'.format(pid, i + 1)),
                                    index_label='Hours')

        except:
            continue
