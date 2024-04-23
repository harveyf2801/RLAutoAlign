import subprocess
import os
from pathlib import Path

output = "soundfiles/SDDS_segmented_Allfiles"

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
        'gdown',
        'pandas'
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
    if not os.path.exists(output):
        gdown.download(id=id, output=output+'.zip')
        gdown.extractall(output+'.zip')
        os.remove(output+'.zip')

def filter_dataset():
    from annotations import get_annotations

    df = get_annotations(output)
    
    df = df[df.Position.isin(['BTM', 'TP']) == False]

    for index, row in df.iterrows():
        os.remove(Path(output, row['FileName']))


if __name__ == '__main__':
    install_packages()