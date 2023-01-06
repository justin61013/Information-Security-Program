import os
import random
import shutil
import tqdm

source = r"C:\Thesis\Datasets\original_sequences\actors\c23\videos"
dest = r"C:\Thesis\Kittiwat\dataset\real"
files = os.listdir(source)
no_of_files = 200

for file_name in tqdm.tqdm(random.sample(files, no_of_files)):
    shutil.copy(os.path.join(source, file_name), dest)
