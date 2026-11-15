from pathlib import Path
from PIL import Image
import glob
image_list = []
for filename in sorted(glob.glob('test/*.png')):
    im = Image.open(filename)
    image_list.append(im)
print(image_list)