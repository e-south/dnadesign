## sequences

  **sequences** serves as the central storage location for nucleotide sequences within the project, organizing outputs from sibling pipelines into a standardized data structure.

  Each sequence file is structured as a list of dictionaries, following the below format:  
   ```python
   example_sequence_entry = [

      {
         "id": "90b4e54f-b5f9-48ef-882a-8763653ae826",
         "meta_date_accessed": "2025-02-19T12:01:30.602571",
         "meta_source": "deg2tfbs_all_DEG_sets",
         "sequence": "gtactgCTGCAAGATAGTGTGAATGACGTTCAATATAATGGCTGATCTTATTTCCAGGAAACCGTTGCCACA",
         "meta_type": "dense-array",
         "evo2_logits_mean_pooled": tensor([[[-10.3750,  10.3750, ..., 10.3750,  10.3750]]], dtype=torch.bfloat16),
         "evo2_logits_shape": [1, 512]
      },
      # Additional dictionary entries extend the list
   ]
   ```

### Gitignore considerations

- **Ignored Folders:**  
  Folders matching the patterns `**/seqbatch*/` and `**/densebatch*/` have been added to the project's `.gitignore` file, preventing large binary files from being tracked.

- **File Size Warnings:**  
  When adding Evo 2-derived keys to sequence entries, the resulting `.pt` files may exceed GitHub's 100 MB limit, causing your push to be rejected.  
  **Workaround:** Instead of pushing these large files, compress the sequences and transfer them to your local machine for downstream analysis.

### Transferring the sequences directory

1. Compress the **sequences** folder on the remote project instance

    Run the following command in a directory where you have write permission (e.g., your home directory):

    ```bash
    cd ~
    tar -czvf sequences_archived.tar.gz absolute/path/to/your/sequences
    ```

2. Check that the archive was created successfully by listing its details:

    ```bash
    ls -lh sequences_archived.tar.gz
    ```

    (Optional) Verify the archive’s integrity with a checksum:

    ```bash
    md5sum sequences_archived.tar.gz
    ```

3. Transfer sequences to your local machine

    Navigate to the folder where you want to save the archive. Then use `rsync` for a robust transfer that can resume interrupted downloads:

    ```bash
    rsync -avP username@remote_host:/path/to/sequences_archived.tar.gz .
    ```

4. After transfer (and waiting a little), confirm the file was copied correctly:

    ```bash
    ls -lh sequences_archived.tar.gz
    ```

    You can check its integrity with a checksum (comparing it to the checksum from the remote):

    ```bash
    md5sum sequences_archived.tar.gz
    ```

5. Cleanup

    Once you have successfully transferred the sequences, you can free up space on the remote machine by deleting the compressed file:

    ```bash
    rm ~/sequences_archived.tar.gz
    ```
