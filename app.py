import streamlit as st
import cv2
import numpy as np
import string

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json


st.set_page_config(
    page_title="Multi-Label Image Retrieval",
    page_icon="🖼️",
    layout="centered"
)


st.title("🖼️ Multi-Label Image Retrieval")
st.write("Upload an image to retrieve its predicted labels.")


# --------------------------------------------------
# HASH FUNCTIONS
# --------------------------------------------------

def hash_array_to_hash_hex(hash_array):
    hash_array = np.array(hash_array, dtype=np.uint8)

    hash_str = ''.join(
        str(i) for i in hash_array.flatten()
    )

    return hex(int(hash_str, 2))


def get_hash_from_image(image):

    image = cv2.resize(image, (64, 64))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    dct = cv2.dct(image)

    dct_block = dct[:8, :8]

    dct_average = (
        dct_block.mean() * dct_block.size
        - dct_block[0, 0]
    ) / (dct_block.size - 1)

    dct_block[dct_block < dct_average] = 0.0

    dct_block[dct_block != 0] = 1.0

    hashing = hash_array_to_hash_hex(
        dct_block.flatten()
    )

    return hashing.strip()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache(allow_output_mutation=True)
def load_model():

    with open("model/model.json", "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)

    model.load_weights(
        "model/model_weights.h5"
    )

    return model


# --------------------------------------------------
# PREPARE TOKENIZERS
# --------------------------------------------------

@st.cache(allow_output_mutation=True)
def prepare_tokenizers():

    image_hash = []
    image_label = []

    seen = []

    with open("model/captions.txt", "r") as file:

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

            image_path = "Dataset/Images/" + filename

            image = cv2.imread(image_path)

            if image is None:
                continue

            image_hash_value = get_hash_from_image(
                image
            )

            image_hash.append(
                image_hash_value
            )

            image_label.append(
                caption
            )


    # Image tokenizer
    image_tokenizer = Tokenizer()

    image_tokenizer.fit_on_texts(
        image_hash
    )


    # Label tokenizer
    label_tokenizer = Tokenizer()

    label_tokenizer.fit_on_texts(
        image_label
    )


    # Find maximum image sequence length
    image_sequences = image_tokenizer.texts_to_sequences(
        image_hash
    )

    max_image_len = int(
        len(max(image_sequences, key=len))
    )


    return (
        image_tokenizer,
        label_tokenizer,
        max_image_len
    )


# --------------------------------------------------
# PREDICT LABEL
# --------------------------------------------------

def predict_label(logits, tokenizer):

    index_to_words = {
        index: word
        for word, index in tokenizer.word_index.items()
    }

    index_to_words[0] = ""

    predictions = np.argmax(
        logits,
        axis=1
    )

    result = []

    for prediction in predictions:

        word = index_to_words.get(
            prediction,
            ""
        )

        if word:
            result.append(word)

    return " ".join(result)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    # Display image
    st.image(
        image,
        caption="Uploaded Image"
    )


    # Predict button
    if st.button("Predict Labels"):

        with st.spinner(
            "Processing image..."
        ):

            try:

                # Load trained model
                model = load_model()


                # Load tokenizers
                (
                    image_tokenizer,
                    label_tokenizer,
                    max_image_len
                ) = prepare_tokenizers()


                # Generate hash for uploaded image
                image_hash = get_hash_from_image(
                    image
                )


                # Convert hash into sequence
                image_sequence = (
                    image_tokenizer
                    .texts_to_sequences(
                        [image_hash]
                    )
                )


                # Padding
                image_sequence = pad_sequences(
                    image_sequence,
                    maxlen=max_image_len,
                    padding="post"
                )


                # Model prediction
                prediction = model.predict(
                    image_sequence,
                    verbose=0
                )


                # Convert prediction into labels
                predicted_labels = predict_label(
                    prediction[0],
                    label_tokenizer
                )


                st.success(
                    "Prediction completed successfully!"
                )


                st.subheader(
                    "Predicted Labels"
                )


                st.write(
                    predicted_labels
                )


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )