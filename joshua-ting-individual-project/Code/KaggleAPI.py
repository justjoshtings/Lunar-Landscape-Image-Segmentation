"""
KaggleAPI.py
Object to handle connection to Kaggle API to upload and download files.

author: @saharae, @justjoshtings
created: 11/11/2022
"""
from LunarModules.Logger import MyLogger
from datetime import datetime
import subprocess
import json

class KaggleAPI:
    '''
    Object to handle connection to Kaggle API to upload and download files.
    '''
    def __init__(self, kaggle_username, log_file=None):
        '''
        Params:
            self: instance of object
            kaggle_username (str): kaggle username
            log_file (str): default is None to not have logging, otherwise, specify logging path ../filepath/log.log

        https://adityashrm21.github.io/Setting-Up-Kaggle/
        https://github.com/Kaggle/kaggle-api#api-credentials

        '''
        self.kaggle_username = kaggle_username

        self.LOG_FILENAME = log_file
        if self.LOG_FILENAME:
            # Set up a specific logger with our desired output level
            self.mylogger = MyLogger(self.LOG_FILENAME)
            # global MY_LOGGER
            self.MY_LOGGER = self.mylogger.get_mylogger()
            self.MY_LOGGER.info(f"{datetime.now()} -- [KaggleAPI]...")

    def create_dataset(self, path_to_data, data_url_end_point, data_title, message=None):
        '''
        Create Kaggle dataset from command line prompts

        Params:
            self: instance of object
            path_to_data (str): path to dataset
            data_url_end_point (str): url endpoint for your dataset on kaggle
            data_title (str): title for your dataset on kaggle
            message (str): optional message for updating existing datasets on kaggle
        '''
        if self.LOG_FILENAME:
            self.MY_LOGGER.info(f"{datetime.now()} -- [KaggleAPI] Creating dataset from {path_to_data}")

        kaggle_dataset_commands = {'init_metadata': f"kaggle datasets init -p {path_to_data}", 
                                    'create_dataset':f"kaggle datasets create -u -p {path_to_data} --dir-mode zip",
                                    'update_dataset':f"kaggle datasets version -p {path_to_data} --dir-mode zip -m"
        }

        # Init dataset metadata
        command = kaggle_dataset_commands['init_metadata'].split()
        self.command_line_execution(command)

        # Edit dataset-metadata.json
        with open(f'{path_to_data}dataset-metadata.json', 'r+') as f:
            data = json.load(f)
            data['id'] = f"{self.kaggle_username}/{data_url_end_point}"
            data['title'] = f"{data_title}"
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        
        # If update or create new
        if message:
            # Update dataset
            command = kaggle_dataset_commands['update_dataset'].split()
            command.append(f'{message}')
            self.command_line_execution(command)
        else:
            # Create new dataset
            command = kaggle_dataset_commands['create_dataset'].split()
            self.command_line_execution(command)

    def download_dataset(self, owner, data_url_end_point, path_to_data):
        '''
        Download Kaggle dataset from command line prompts

        Params:
            self: instance of object
            owner (str): kaggle owner of dataset
            data_url_end_point (str): url endpoint for your dataset on kaggle
            path_to_data (str): path to save dataset
        '''

        download_command = f'kaggle datasets download -d {owner}/{data_url_end_point} -p {path_to_data} --unzip'

        # Download dsataset
        command = download_command.split()
        self.command_line_execution(command)

    def command_line_execution(self, command):
        '''
        Execute bash command line prompts

        Params:
            self: instance of object
            command (str): bash commands to execute
        '''
        if self.LOG_FILENAME:
            self.MY_LOGGER.info(f"{datetime.now()} -- [KaggleAPI] 'Executing command:'{' '.join(command)}")
        
        print('Executing command:', ' '.join(command))
        results = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
        print('Returned results:', results)

        if self.LOG_FILENAME:
            self.MY_LOGGER.info(f"{datetime.now()} -- [KaggleAPI] Returned results: {results}")

    def check_dataset_status(self, owner, data_url_end_point):
        '''
        Execute bash command line prompts

        Params:
            self: instance of object
            owner (str): kaggle owner of dataset
            data_url_end_point (str): url endpoint for your dataset on kaggle
        '''
        command = f'kaggle datasets status {owner}/{data_url_end_point}'

        # Download dsataset
        command = command.split()
        self.command_line_execution(command)
        
        
if __name__ == "__main__":
    print("Executing KaggleAPI.py")
else:
    print("Importing KaggleAPI")