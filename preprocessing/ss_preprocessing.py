import preprocessing
import numpy as np
from preprocessing import *
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

def build_ss_task_streams(exercise_number, path, shuffle, batch_size=32, num_subjects = 27):
    task_stream = []

    for subject in range(num_subjects):
        base_path = f"{path}/s{subject + 1}/S{subject + 1}_A1_E{exercise_number}.mat"
        data = preprocessing.load_data(base_path)

        x_train, y_train, x_test, y_test, _cw = preprocessing.preprocessing_internals(data)

        x_train = np.transpose(x_train, (0, 2, 1))
        x_test = np.transpose(x_test, (0, 2, 1))

        train_loader = DataLoader(
            preprocessing.NinaProDataset(x_train, y_train), 
            batch_size=batch_size, 
            shuffle=shuffle, # Shuffle differentiates stateless vs stateful training
            collate_fn=padding
        )
            
        test_loader = DataLoader(
            preprocessing.NinaProDataset(x_test, y_test), 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=padding
        )
        
        # Append as a distinct continual learning boundary
        task_stream.append({
            'task_id': f'subject_{subject+1}',
            'train': train_loader,
            'test': test_loader
        })
    return task_stream
        

def padding(batch):
    # ensures all sequences are the same size
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    sequences = [torch.tensor(seq, dtype=torch.float32) if not isinstance(seq, torch.Tensor) else seq for seq in sequences]
    labels = torch.tensor(labels, dtype=torch.long)

    padded_batch = pad_sequence(sequences, batch_first=True, padding_value=0)
    attention_mask = (padded_batch[..., 0] != 0).long()
    return padded_batch, labels, attention_mask

# def build_dataset(ex_num, data_path, batch_size, num_subjects):
#     x_train, y_train, x_test, y_test, __ = preprocessing.multi_preprocess(ex_num, data_path, num_subjects)

#     train_data = preprocessing.NinaProDataset(x_train, y_train)
#     test_data = preprocessing.NinaProDataset(x_test, y_test)
#     train_loader = DataLoader(
#         train_data,
#         batch_size,
#         collate_fn = padding
#     )
#     test_loader = DataLoader(
#         test_data,
#         batch_size,
#         collate_fn = padding
#     )
#     return train_loader, test_loader
