import streamlit as st
import cv2
from keras_facenet import FaceNet

# Load the FaceNet model and the SVM classifier
facenet_model = FaceNet()
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_embeddings, y_train)  # Ensure the model is trained

# Function to get the embedding of a face image using FaceNet
def get_embedding(model, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    samples = np.expand_dims(face, axis=0)
    yhat = model.embeddings(samples)
    return yhat[0]

# Streamlit application
st.title('Face Recognition Application')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract the face from the image
    face = extract_face(image_rgb)
    if face is not None:
        st.image(face, caption='Detected Face', use_column_width=True)
        
        # Get embedding for the face
        embedding = get_embedding(facenet_model, face)
        embedding = np.expand_dims(embedding, axis=0)
        
        # Predict the label
        prediction = svm_model.predict(embedding)
        prediction_proba = svm_model.predict_proba(embedding)
        
        st.write(f'Prediction: {prediction[0]}')
        st.write(f'Prediction Probability: {prediction_proba[0]}')
    else:
        st.write("No face detected in the image.")
