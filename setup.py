import subprocess

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

if __name__ == '__main__':
    install_packages()