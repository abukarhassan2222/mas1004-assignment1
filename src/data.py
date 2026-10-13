"""Turn a folder of images into arrays. YOU write the bodies of these functions.

Your images are all different sizes, different shapes, and some of them are not
even really images. These functions are where that mess becomes clean arrays
that the model can eat.

Run `pytest tests/test_data.py` after you fill them in.
"""

import numpy as np
from pathlib import Path
from PIL import Image

# Files with any other extension should be ignored.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# How every photo was prepared when ResNet18 was trained on ImageNet. The model
# you start from learned to read photos prepared exactly like this, so yours
# have to be prepared the same way. Do not change these numbers. The export
# script copies them into docs/model.json, and the web page reads them there.
RESIZE = 256                    # the shorter side is resized to this
CROP = 224                      # then the square in the middle is cut out
MEAN = (0.485, 0.456, 0.406)    # red, green, blue
STD = (0.229, 0.224, 0.225)     # red, green, blue


def prepare_image(image):
    """Turn one picture into the numbers ResNet18 expects.

    Argument
        image  a PIL.Image, in any mode and of any size

    Returns np.float32 of shape (3, CROP, CROP), that is (3, 224, 224)

    Do these six steps, in this order:
        1. Convert to RGB with image.convert("RGB"). Some of your files are
           grayscale, and some are PNGs with transparency.
        2. Resize so that the shorter side is RESIZE pixels and the picture
           keeps its shape, with bilinear interpolation. A 1000 x 500 photo
           becomes 512 x 256.
        3. Cut out the CROP x CROP square in the middle. Whatever sticks out
           on the long side is thrown away.
        4. Divide by 255, so that every value is between 0.0 and 1.0.
        5. For each channel c, subtract MEAN[c] and then divide by STD[c].
        6. Put the channels first: shape (3, 224, 224), not (224, 224, 3).
           Row 0 is red, row 1 is green, row 2 is blue.

    torchvision.transforms can do steps 2 to 6 for you (Resize, CenterCrop,
    ToTensor, Normalize). If you use it, check that the result matches this
    list, and return a numpy array, not a torch tensor.

    The web page does these same six steps in JavaScript. If your version is
    different, the self test badge at the top of your page turns red.
    """
    rgb = image.convert("RGB")

    from torchvision import transforms

    pipeline = transforms.Compose([
        transforms.Resize(RESIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(CROP),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return pipeline(rgb).numpy()


def load_folder(root):
    """Read every image under `root` and return them as one array.

    `root` is a folder that holds one sub-folder per class, like this:

        data/clean/
            espresso_cup/  0001.jpg 0002.jpg ...
            wine_glass/    0001.jpg 0002.jpg ...
            paper_cup/     0001.jpg 0002.jpg ...

    Returns a tuple (X, y, class_names, paths)
        X            np.float32, shape (N, 3, 224, 224). Row i is
                     prepare_image of image i.
        y            np.int64, shape (N,), the class index of each row
        class_names  list of str, the sub-folder names sorted alphabetically.
                     class_names[y[i]] is the name of the class of row i.
        paths        list of Path, length N, where each row came from. Keep the
                     order the same as X and y.

    Things that will happen to you
        Only files whose extension is in IMAGE_SUFFIXES are images. Ignore
        everything else, in any letter case (.JPG counts).
        Some files are broken and raise an exception when you open them. Skip
        them instead of crashing.
        The order of files on disk is not guaranteed. Sort them so that you get
        the same result every time you run.

    Memory: every image becomes 3 x 224 x 224 numbers of 4 bytes, about 0.6 MB.
    750 images is about 450 MB. That fits on Colab and on most laptops.
    """
    root = Path(root)
    class_names = sorted(d.name for d in root.iterdir() if d.is_dir())

    X, y, paths = [], [], []
    for index, name in enumerate(class_names):
        folder = root / name
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in files:
            try:
                array = prepare_image(Image.open(path))
            except Exception:
                continue  # a broken file is not an image
            X.append(array)
            y.append(index)
            paths.append(path)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        class_names,
        paths,
    )


def split_train_test(X, y, paths, test_ratio=0.2, seed=0):
    """Split the rows into a training part and a test part.

    Arguments
        X, y, paths  exactly what load_folder returned
        test_ratio   float between 0 and 1, the share that goes to the test side
        seed         int, so that you get the same split every time you run

    Returns (X_train, y_train, paths_train, X_test, y_test, paths_test)

    Two rules the tests check
        No image may appear on both sides. Every path belongs to exactly one.
        Every class must appear on both sides. If you shuffle badly you can end
        up with a class that has no test images at all, and then your accuracy
        number means nothing.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    paths = list(paths)
    rng = np.random.default_rng(seed)

    train_idx, test_idx = [], []
    for label in sorted(np.unique(y)):
        rows = np.nonzero(y == label)[0]
        shuffled = rows[rng.permutation(len(rows))]
        # Every class must appear on both sides, so keep at least one on each.
        n_test = max(1, round(len(rows) * test_ratio))
        n_test = min(n_test, len(rows) - 1)
        test_idx.extend(shuffled[:n_test].tolist())
        train_idx.extend(shuffled[n_test:].tolist())

    X_train = X[np.asarray(train_idx)]
    y_train = y[np.asarray(train_idx)]
    paths_train = [paths[i] for i in train_idx]
    X_test = X[np.asarray(test_idx)]
    y_test = y[np.asarray(test_idx)]
    paths_test = [paths[i] for i in test_idx]

    return X_train, y_train, paths_train, X_test, y_test, paths_test
