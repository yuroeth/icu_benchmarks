import argparse
import os
import pickle
from pathlib import Path
from icu_benchmarks.data.preprocessing.utils import benchmark_to_h5, save_to_h5_with_tasks, scaling_data_common, save_to_text_to_h5, \
    scale_text, extract_text_data, create_note_type_per_split_table
from sklearn.preprocessing import StandardScaler
# from data import text_embeddings
# import torch


def build_mimic3_h5(load_path, text_path, save_path, channel_to_id, matching_dict, var_range, static_col, save_file):
    """Wrapper to build MIMIC-III benchmark dataset in the desired format for our loader.
    Args:
        load_path: String with path to source data from https://github.com/YerevaNN/mimic3-benchmarks.
        save_path: String with path where to save the final h5 file.
        channel_to_id: Dict obtained from mimic_resources/ that matches variables to id.
        matching_dict: Dict obtained from mimic_resources/ that matches string to categories.
        var_range: Dict obtained from mimic_resources/ with ranges for certain variables to remove false entries.
        static_col: Name of the static columns, should be only Height.
    Returns:
    """
    data, labels, windows, texts, text_windows, col, tasks = benchmark_to_h5(load_path, text_path, channel_to_id,
                                                                             matching_dict, var_range, static_col)
    # save_file = Path(os.path.join(save_path, 'non_scaled.h5'))
    # texts_file = Path(os.path.join(save_path, 'texts.h5'))
    # save_to_text_to_h5(texts_file, ['Hours', 'Text'], texts, text_windows)
    save_to_h5_with_tasks(Path(save_file), col, tasks, data, labels, windows)


def scale_dataset(non_scaled_path, save_path, static_col, note_embedding, nlp_model, scale_file):
    """Scales dataset with StandardScaling on the non-categorical variables.
    Args:
        non_scaled_path: String with path to the non scaled dataset h5 file.
        save_path: String with path where to save the final h5 file.
        static_col: Name of the static columns.
    Returns:
    """
    static_idx = [k for k in range(len(static_col))]
    data, labels, windows, col, tasks, time_scaler = scaling_data_common(non_scaled_path, threshold=25,
                                                                         scaler=StandardScaler(),
                                                                         static_idx=static_idx, df_ref=None)
    # save_file = Path(os.path.join(save_path, 'scaled.h5'))
    save_to_h5_with_tasks(Path(scale_file), col, tasks, data, labels, windows)
    del data
    del labels

    # texts_path = Path(os.path.join(save_path, 'texts.h5'))
    # texts_path_scaled = Path(os.path.join(save_path, 'texts_scaled.h5'))
    # text_scaled, text_windows = scale_text(texts_path, time_scaler, nlp_model, note_embedding)
    # if nlp_model is not None:
    #     text_columns = ['Hours'] + ['embedding_dim_{}'.format(k) for k in range(text_scaled['train'].shape[-1] - 1)]
    #     save_to_h5_with_tasks(texts_path_scaled, text_columns, None, text_scaled, None, text_windows)

    # else:
    #     text_columns = ['Hours', 'Text']
    #     save_to_text_to_h5(texts_path_scaled, text_columns, text_scaled, text_windows)


if __name__ == "__main__":

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of the dataset to build, either 'mimic3' or 'physionet2019' ")
    parser.add_argument("--load_path", type=Path, help="Path to folder containing MIMIC-III Benchmark source")
    parser.add_argument("--embedding_path", type=Path, default=None, help="Path to NLP embedding model")
    parser.add_argument("--noteevents_path", type=Path, default=None,
                        help="Path to folder containing MIMIC-III NOTEEVENTS.csv source")
    parser.add_argument("--text_path", type=Path, default=None, help="Path to folder containing MIMIC-III text source")
    parser.add_argument("--save_path", type=Path, help="Path to folder where we save the extracted dataset")
    parser.add_argument("--resource_path", type=Path, help="Path to folder with the resources pickles files ")
    parser.add_argument("--static_columns", nargs='+', help="List of static columns names ")
    parser.add_argument("--scale", default=True, type=boolean_string, help="Whether or not to save a scaled copy")
    parser.add_argument("--overwrite", default=False, type=boolean_string, help="Whether to overwrite existing files")
    parser.add_argument("--note_embedding", default='single', type=str,
                        help="Note representation: 'single', 'chunks', 'sentences'")

    arg = parser.parse_args()

    # if arg.dataset == 'mimic3':
    #     # build text source dataset
    #     # if not os.path.exists(os.path.join(arg.text_path, 'train_text')) and \
    #     #         not os.path.exists(os.path.join(arg.text_path, 'test_text')):
    #     #     os.mkdir(os.path.join(arg.text_path, 'train_text'))
    #     #     os.mkdir(os.path.join(arg.text_path, 'test_text'))

    #         # for split in ['train', 'test', 'val']:  # val is probably not needed here
    #         #     print('Building Text Data for {} Split.'.format(split))
    #         #     if split in ['train', 'val']:
    #         #         folder = os.path.join(arg.text_path, 'train')
    #         #         folder_text = os.path.join(arg.text_path, 'train_text')
    #         #     else:
    #         #         folder = os.path.join(arg.text_path, 'test')
    #         #         folder_text = os.path.join(arg.text_path, 'test_text')
    #         #     extract_text_data(folder, folder_text, arg.noteevents_path)

    #     with open(os.path.join(arg.resource_path, 'channel_to_id_m3.pkl'), 'rb') as f:
    #         channel_to_id = pickle.load(f)
    #     with open(os.path.join(arg.resource_path, 'matching_dict_m3.pkl'), 'rb') as f:
    #         matching_dict = pickle.load(f)
    #     with open(os.path.join(arg.resource_path, 'var_range_m3.pkl'), 'rb') as f:
    #         var_range = pickle.load(f)

    save_file = os.path.join(arg.save_path, 'non_scaled_42.h5')
    #     if (not os.path.isfile(save_file)) or arg.overwrite:
    #         os.makedirs(arg.save_path, exist_ok=True)
    #         build_mimic3_h5(arg.load_path, arg.text_path,
    #                         arg.save_path, channel_to_id, matching_dict, var_range, arg.static_columns, save_file)

    #         # build note_types.pkl
    #         # create_note_type_per_split_table(arg.save_path)

    #     else:
    #         print('Non scaled data already exist')

    # else:
    #     raise Exception("arg.dataset has to be either 'mimic3' or 'physionet2019' ")

    scale_file = os.path.join(arg.save_path, 'scaled_42.h5')
    # if not os.path.isfile(scale_file) and arg.scale:
        # if arg.embedding_path:
        #     nlp_model = text_embeddings.BertEmbeddings(arg.embedding_path, note_embedding=arg.note_embedding)
        #     if torch.cuda.is_available():
        #         device = torch.device('cuda')
        #     else:
        #         device = torch.device('cpu')
        #     nlp_model.to(device)
        # else:
        #     nlp_model = None
    nlp_model = None
    scale_dataset(save_file, arg.save_path, arg.static_columns, arg.note_embedding, nlp_model, scale_file)
