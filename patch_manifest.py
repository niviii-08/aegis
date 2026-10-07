import csv

input_file = 'data/processed/audio/manifest.csv'
output_file = 'data/processed/audio/manifest.csv.tmp'

expected_cols = [
    "clip_id", "file_path", "dataset", "modality", "label", 
    "generator", "speaker_id", "duration_sec", "sample_rate", 
    "file_size", "file_hash", "split", "preprocessing_version"
]

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8', newline='') as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=expected_cols)
    writer.writeheader()
    
    for row in reader:
        new_row = {}
        for col in expected_cols:
            if col == 'file_path':
                new_row[col] = row.get('path', '')
            elif col == 'duration_sec':
                new_row[col] = ''
            elif col == 'sample_rate':
                new_row[col] = ''
            elif col == 'file_size':
                new_row[col] = '0'
            elif col == 'preprocessing_version':
                new_row[col] = '1.0.0'
            else:
                new_row[col] = row.get(col, '')
        writer.writerow(new_row)

import os
import shutil
shutil.move(output_file, input_file)
print('Fixed audio manifest.')
