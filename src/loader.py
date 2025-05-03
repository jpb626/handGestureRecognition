import os
from glob import glob

# Map the exact folder names to labels 0–4
CLASS_FOLDERS = {
    '01_palm':  0,
    '05_thumb': 1,
    '06_index': 2,
    '07_ok': 3,
    '09_c': 4,
}

def make_data_tuples(root_dir):
    """
    Walks this tree:
      root_dir/
        00/
          01_palm/
            frame_00_01_0001.png
            …
          … (other gesture folders)
        01/
          01_palm/
            …
          …
        …
    Returns list of (filepath, label, subject_id).
    """
    data = []
    for subject in sorted(os.listdir(root_dir)):
        subj_path = os.path.join(root_dir, subject)
        # skip if not a directory
        if not os.path.isdir(subj_path):
            continue

        # Get each class folder (e.g. 01_palm, 05_thumb, etc.)
        for class_name, label in CLASS_FOLDERS.items():
            class_path = os.path.join(subj_path, class_name)
            if not os.path.isdir(class_path):
                continue

            # grab every image in that folder
            for img_path in glob(os.path.join(class_path, '*.png')):
                data.append((img_path, label, int(subject)))
    print(data[3200])
    return data

if __name__ == '__main__':
    import os

    # compute the data root relative to this file
    here     = os.path.dirname(__file__)                           
    root_dir = os.path.abspath(os.path.join(here, '..',          
                                            'segmented-data',
                                            'leapGestRecog'))
    print("Loading from:", root_dir)

    data_tuples = make_data_tuples(root_dir)
    print(f"Total samples: {len(data_tuples)}")
    from collections import Counter
    print("Per-class counts:", Counter(label for _,label,_ in data_tuples))
    print("Subjects present:", sorted({s for _,_,s in data_tuples}))

