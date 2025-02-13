## sequences

This subdirectory is ...



Merging multiple ``.pt`` files, for exmaple from one data source or subset:
  ```python
  import torch
  import os

  root_dir = "sequences"  # e.g., your "sequences" directory
  file_list = [
      "EcoCyc_Release_28_SmartTable_All_Promoters.pt",
      "Kosuri_et_al_2013_promoters.pt",
      # ... etc.
  ]

  merged_data = []
  for file_name in file_list:
      path = os.path.join(root_dir, file_name)
      current_data = torch.load(path)
      merged_data.extend(current_data)

  torch.save(merged_data, "merged_all_sequences.pt")
  ```

Then, to load:
```python
loaded_data = torch.load("my_sequences.pt")
```



Subset Selection
If you only need a subset of sequences, you can:

Load the entire .pt file, then filter entries by their metadata (e.g., meta_type == "promoter").
Save specialized subsets as separate .pt files (like “train”, “test”, etc.) if you expect to reuse them frequently.

## Subdirectory Structure


sequences/
├── EcoCyc/
│   ├── ecocyc_28_all_promoters.pt
│   ├── ecocyc_28_all_tfbs.pt
│   └── ...
├── RegulonDB/
│   ├── regulondb_13_promoters.pt
│   ├── regulondb_13_tfbs.pt
│   └── ...
├── Kosuri/
│   └── kosuri_2013_promoters.pt
├── Urtecho/
│   └── urtecho_2019_promoters.pt
├── ...
└── merged/
    └── all_sequences_2025-01-23.pt





## Purpose



---

**Author:** Eric J. South 
**Date:** January 19, 2025