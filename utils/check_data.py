import glob
import os
def list_files_in_repo(repo_path):
    print(repo_path)
    files = glob.glob(os.path.join(repo_path, '**'), recursive=True)
    return [f for f in files if os.path.isfile(f)]

repo_path = '/home/team5/project/data/alameda/images_4'
all_files = list_files_in_repo(repo_path)
print(len(all_files))