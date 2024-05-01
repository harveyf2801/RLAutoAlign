import os
import pandas as pd
from pathlib import Path
import random

def get_annotations(audio_dir, csv_path='annotations.csv'):
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f'{audio_dir} does not exist')
    
    if not(os.path.exists(Path(csv_path))):
        data = []

        # | FileName | Snare | Dampening | Brand | Model | Position | Strike | Peak |

        for (dirpath, dirnames, filenames) in os.walk(audio_dir):
            for filename in filenames:
                if (filename[0] == '.') or (filename[-4:] != '.wav'):
                    continue
                else:
                    names = filename.split('_')
                    names.insert(0, filename)
                    names[-1] = names[-1].replace('.wav', '')
                    names[7] = int(names[7])
                    names[9] = float(names[9])
                    names[3 : 5] = [' '.join(names[3 : 5])]
                    data.append(names)
            break

        # Creating a pandas dataframe to hold this data
        df = pd.DataFrame(data, columns = ['FileName', 'Snare', 'Dampening', 'Mic',
                                        'Position', '_Segment', 'Strike', '_Peak', 'Peak'])
        df = df.drop('_Segment', axis=1)
        df = df.drop('_Peak', axis=1)
        del data

        # Create class ID numbers from the class labels
        # Grouping the class labels for each snare, dampening type, microphone and strike
        # x10 Snares
        # x5 Dampening Types
        # x16 Mic Brands
        # x31 Mic Models
        # x6 Mic Positions (reduced to top and bottom now)
        # x102 Strikes

        # Get unique combinations of snare, dampening, and strike values
        # Creating a class ID for rows where the snare, dampening type and strike are the same
        # i.e. the same recorded hit but with varying position / microphone
        df['ClassID'] = df.groupby(['Snare', 'Dampening', 'Strike']).ngroup()
        df.reset_index(drop=True, inplace=True)

        # Save the dataframe to a CSV file
        df.to_csv(path_or_buf=csv_path, sep=',', encoding='utf-8', index=False)
    else:
        df = pd.read_csv(csv_path)
    
    return df


if __name__ == '__main__':
    df = get_annotations('soundfiles/SDDS_segmented_Allfiles')

    # Select a unique top and bottom pair of positions for each class ID
    import numpy as np
    np.random.seed(0)
    max_class_id = max(df.ClassID)

    for i in range(0, 10):
        class_id = np.random.randint(0, max_class_id)
        target = df.query(f'(ClassID == {class_id}) & (Position == "SHL")').sample(n=1)
        input = df.query(f'(ClassID == {class_id}) & (Position == "BTM")').sample(n=1)
        print(target)
        print(input)
        print('-'*10)