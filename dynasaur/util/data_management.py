import os
import numpy as np
import torch
import zipfile
import tempfile

def save_data_compressed(data, path):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    if isinstance(data, np.ndarray):
        np.savez_compressed(path if path.endswith('.npz') else path + '.npz', data=data)
    elif isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()):
        np.savez_compressed(path if path.endswith('.npz') else path + '.npz', **data)
    else:
        # Use PyTorch with zip compression
        temp_path = tempfile.mktemp(suffix=".pt")
        torch.save(data, temp_path)
        zip_path = path if path.endswith('.pt.zip') else path + '.pt.zip'
        print(zip_path)
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            print(f"Actual path: {os.path.abspath(zipf.filename)}")
            zipf.write(temp_path, arcname=os.path.basename(temp_path))
        os.remove(temp_path)

def load_data_compressed(path):
    if path.endswith('.npz'):
        data = np.load(path)
        if 'data' in data and len(data.files) == 1:
            return data['data']
        else:
            return {k: data[k] for k in data.files}
    elif path.endswith('.pt.zip'):
        with zipfile.ZipFile(path, 'r') as zipf:
            pt_files = zipf.namelist()
            if len(pt_files) != 1:
                raise RuntimeError("Unexpected multiple files in .pt.zip archive")
            with zipf.open(pt_files[0]) as f:
                return torch.load(f)
    else:
        raise ValueError("Unsupported compressed file extension. Use '.npz' or '.pt.zip'.")