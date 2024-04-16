import subprocess
from annotations import get_annotations
import os
from pathlib import Path


def install_packages():
    packages = [
        'torch',
        'torchvision',
        'torchaudio',
        'torchrl',
        'tqdm',
        'matplotlib',
        'numpy',
        'scipy',
        'librosa',
        'sounddevice',
        'gymnasium',
        'stable-baselines3',
        'gdown'
        # Add more packages here
    ]

    try:
        subprocess.check_call(['python3', '-m', 'pip', 'install', '--upgrade', 'pip'])
        print(f'Pip is up-to date')
    except subprocess.CalledProcessError:
        print(f'Failed to update pip')

    for package in packages:
        try:
            subprocess.check_call(['pip3', 'install', package])
            print(f'Successfully installed {package}')
        except subprocess.CalledProcessError:
            print(f'Failed to install {package}')

def download_dataset():
    import gdown
    id = "1zvM8xA4M9W0Z2p-ZLQNxStLhRvzelijI"
    output = "soundfiles/SDDS_segmented_Allfiles.zip"
    gdown.download(id=id, output=output, postprocess=gdown.extractall)

def filter_dataset():
    df = get_annotations('soundfiles/SDDS_segmented_Allfiles')
    
    df = df[df.Position.isin(['BTM', 'TP']) == False]

    for index, row in df.iterrows():
        os.remove(Path('soundfiles/SDDS_segmented_Allfiles', row['FileName']))

if __name__ == '__main__':
    install_packages()