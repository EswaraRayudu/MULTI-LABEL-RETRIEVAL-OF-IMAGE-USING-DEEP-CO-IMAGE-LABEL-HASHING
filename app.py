import streamlit as st
import cv2
import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-Label Image Retrieval",
    page_icon="🖼️",
    layout="centered"
)


st.title("🖼️ Multi-Label Image Retrieval")
st.write(
    "Upload an image to retrieve its predicted labels."
)


# --------------------------------------------------
# HASH FUNCTIONS
# --------------------------------------------------

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


def get_hash_from_image(image):

    image = cv2.resize(
        image,
        (64, 64)
    )

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


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache(allow_output_mutation=True)
def load_model():

    with open(
        "model/model.json",
        "r"
    ) as json_file:

        model_json = json_file.read()

    model = model_from_json(
        model_json
    )

    model.load_weights(
        "model/model_weights.h5"
    )

    return model


# --------------------------------------------------
# LOAD TOKENIZERS
# --------------------------------------------------

@st.cache(allow_output_mutation=True)
def prepare_tokenizers():

    # Load image tokenizer
    with open(
        "model/image_tokenizer.pkl",
        "rb"
    ) as file:

        image_tokenizer = pickle.load(
            file
        )


    # Load label tokenizer
    with open(
        "model/label_tokenizer.pkl",
        "rb"
    ) as file:

        label_tokenizer = pickle.load(
            file
        )


    # The trained model expects
    # one image token as input
    max_image_len = 1


    return (
        image_tokenizer,
        label_tokenizer,
        max_image_len
    )


# --------------------------------------------------
# PREDICT LABEL
# --------------------------------------------------

def predict_label(
    logits,
    tokenizer
):

    index_to_words = {
        index: word
        for word, index
        in tokenizer.word_index.items()
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
            result.append(
                word
            )


    return " ".join(result)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    # ----------------------------------------------
    # READ IMAGE
    # ----------------------------------------------

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    # ----------------------------------------------
    # CHECK IMAGE
    # ----------------------------------------------

    if image is None:

        st.error(
            "Unable to read the uploaded image."
        )

    else:

        # ------------------------------------------
        # DISPLAY IMAGE
        # ------------------------------------------

        st.image(
            image,
            caption="Uploaded Image"
        )


        # ------------------------------------------
        # PREDICT BUTTON
        # ------------------------------------------

        if st.button(
            "Predict Labels"
        ):

            with st.spinner(
                "Processing image..."
            ):

                try:

                    # ----------------------------------
                    # LOAD MODEL
                    # ----------------------------------

                    model = load_model()


                    # ----------------------------------
                    # LOAD TOKENIZERS
                    # ----------------------------------

                    (
                        image_tokenizer,
                        label_tokenizer,
                        max_image_len
                    ) = prepare_tokenizers()


                    # ----------------------------------
                    # GENERATE IMAGE HASH
                    # ----------------------------------

                    image_hash = (
                        get_hash_from_image(
                            image
                        )
                    )


                    # ----------------------------------
                    # CONVERT HASH TO TOKEN
                    # ----------------------------------

                    image_sequence = (
                        image_tokenizer
                        .texts_to_sequences(
                            [image_hash]
                        )
                    )


                    # ----------------------------------
                    # CHECK UNKNOWN IMAGE HASH
                    # ----------------------------------

                    if (
                        not image_sequence
                        or not image_sequence[0]
                    ):

                        st.warning(
                            "This image is not present "
                            "in the trained image vocabulary."
                        )

                        st.info(
                            "Please upload one of the "
                            "images used during training."
                        )

                        st.stop()


                    # ----------------------------------
                    # PADDING
                    # ----------------------------------

                    image_sequence = (
                        pad_sequences(
                            image_sequence,
                            maxlen=max_image_len,
                            padding="post"
                        )
                    )


                    # ----------------------------------
                    # MODEL PREDICTION
                    # ----------------------------------

                    prediction = model.predict(
                        image_sequence,
                        verbose=0
                    )


                    # ----------------------------------
                    # CONVERT PREDICTION TO LABELS
                    # ----------------------------------

                    predicted_labels = (
                        predict_label(
                            prediction[0],
                            label_tokenizer
                        )
                    )


                    # ----------------------------------
                    # DISPLAY RESULT
                    # ----------------------------------

                    st.success(
                        "Prediction completed successfully!"
                    )

                    st.subheader(
                        "Predicted Labels"
                    )

                    if predicted_labels:

                        st.write(
                            predicted_labels
                        )

                    else:

                        st.info(
                            "No labels were predicted."
                        )


                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )