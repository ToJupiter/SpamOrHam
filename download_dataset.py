import kagglehub
import os

path = kagglehub.dataset_download("venky73/spam-mails-dataset")

print("Path to dataset files:", path)

copy_command = f"cp -r {path}/* {os.getcwd()}"
exit_code = os.system(copy_command)
if exit_code == 0:
    print("Dataset successfully copied to the current directory.")
else:
    print("There was an error copying the dataset.")