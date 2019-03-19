import os
import sys
from tqdm import tqdm
from PIL import Image

root_dir = "/media/kopoden/5C1E16301E1603A4/Datasets/synth90k"
#root_dir = sys.argv[1]
annotations = ['annotation_val', 'annotation_train', 'annotation_test']

for name in annotations:
    f_old = open(os.path.join(root_dir, name+'.txt'), 'r')
    f_new = open(os.path.join(root_dir, name+'_new.txt'), 'w')
    f_broken = open(os.path.join(root_dir, name+'_broken.txt'), 'w')
    broken_buffer = []
    new_buffer = []
    n_lines = sum([1 for line in f_old])
    f_old.seek(0)

    for line in tqdm(f_old, total=n_lines):
        try:
            img = Image.open(os.path.join(root_dir, line.split(" ")[0]))
        except:
            print(line.split(" ")[0])
            broken_buffer.append(line)
        else:
            new_buffer.append(line)

    f_new.writelines(new_buffer)
    f_broken.writelines(broken_buffer)

    f_old.close()
    f_new.close()
    f_broken.close()