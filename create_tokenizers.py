import os
import cv2
import string
import pickle
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer


def hash_array_to_hash_hex(hash_array):

    hash_array = np.array(
        hash_array,
        dtype=np.uint8
    )

    hash_str = ''.join(
        str(i)
        for i in hash_array.flatten()
    )

    return hex(
        int(hash_str, 2)
    )


def getHash(name):

    img = cv2.imread(name)

    img = cv2.resize(
        img,
        (64, 64)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    img = np.array(
        img,
        dtype=np.float32
    )

    dct = cv2.dct(img)

    dct_block = dct[:8, :8]

    dct_average = (
        dct_block.mean()
        * dct_block.size
        - dct_block[0, 0]
    ) / (dct_block.size - 1)

    dct_block[
        dct_block < dct_average
    ] = 0.0

    dct_block[
        dct_block != 0
    ] = 1.0

    hashing = hash_array_to_hash_hex(
        dct_block.flatten()
    )

    return hashing.strip()


image_hash = []
image_label = []

seen = []


with open(
    "model/captions.txt",
    "r"
) as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        arr = line.split(",")

        if len(arr) < 2:
            continue

        filename = arr[0]

        if filename == "image":
            continue

        if filename in seen:
            continue

        if len(image_hash) > 130:
            break

        seen.append(filename)

        caption = arr[1].strip()

        image_path = (
            "Dataset/Images/"
            + filename
        )

        image = cv2.imread(
            image_path
        )

        if image is None:
            print(
                "Image not found:",
                image_path
            )
            continue

        hash_value = getHash(
            image_path
        )

        image_hash.append(
            hash_value
        )

        image_label.append(
            caption
        )


print(
    "Images processed:",
    len(image_hash)
)

print(
    "Labels processed:",
    len(image_label)
)


image_tokenizer = Tokenizer()

image_tokenizer.fit_on_texts(
    image_hash
)


label_tokenizer = Tokenizer()

label_tokenizer.fit_on_texts(
    image_label
)


os.makedirs(
    "model",
    exist_ok=True
)


with open(
    "model/image_tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        image_tokenizer,
        file
    )


with open(
    "model/label_tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        label_tokenizer,
        file
    )


print(
    "Tokenizers saved successfully!"
)

print(
    "Image vocabulary:",
    len(image_tokenizer.word_index)
)

print(
    "Label vocabulary:",
    len(label_tokenizer.word_index)
)