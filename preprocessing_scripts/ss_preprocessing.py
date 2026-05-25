from . import preprocessing
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

def build_ss_task_streams(exercise_number, path, shuffle, batch_size=32, num_subjects = 27):
    task_stream = []

    for subject in range(num_subjects):
        base_path = f"{path}/s{subject + 1}/S{subject + 1}_A1_E{exercise_number}.mat"
        data = preprocessing.load_data(base_path)

        x_train, y_train, x_test, y_test, _cw = preprocessing.preprocessing_internals(data)
        
        train_loader = DataLoader(
            preprocessing.NinaProDataset(x_train, y_train), 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=padding,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
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
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Track the actual lengths before padding
    lengths = torch.tensor([len(seq) for seq in sequences])

    sequences = [torch.tensor(seq, dtype=torch.float32) if not isinstance(seq, torch.Tensor) else seq for seq in sequences]
    labels = torch.tensor(labels, dtype=torch.long)

    padded_batch = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Safely generate a mask based on lengths, not values
    batch_size, max_len, _ = padded_batch.shape
    attention_mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    attention_mask = attention_mask.long()
    
    return padded_batch, labels, attention_mask
