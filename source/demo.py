import subprocess
import streamlit as st
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    return output, error

@st.cache_resource
def load_models_Yolo():
    model_YOLOv8n = YOLO('yolov8n.pt')
    model_YOLOv8x = YOLO('yolov8x.pt')
    return model_YOLOv8n, model_YOLOv8x


model_YOLOv8n, model_YOLOv8x = load_models_Yolo()

# Giao diện Streamlit
st.title("Model Inference Comparison")
st.write("Upload an image and set the threshold.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

# Set threshold
threshold = st.slider("Set threshold", 0.0, 1.0, 0.5, 0.01)

# Run inference
if st.button("Run Inference") and uploaded_file is not None:
    # Save uploaded image locally
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Initialize timing variables
    start_time_faster_rcnn = 0
    end_time_faster_rcnn = 0
    start_time_yolov8n = 0
    end_time_yolov8n = 0
    start_time_yolov8x = 0
    end_time_yolov8x = 0
    # Run Faster RCNN inference
    start_time_faster_rcnn = time.time()

    # Construct the Faster RCNN command
    command_faster_rcnn = f"python fastercnn-pytorch-training-pipeline/inference.py --input {image_path} --weights best_model.pth --threshold {threshold}"
    output_faster_rcnn, error_faster_rcnn = run_command(command_faster_rcnn)

    # Calculate Faster RCNN inference time
    end_time_faster_rcnn = time.time()
    inference_time_faster_rcnn = end_time_faster_rcnn - start_time_faster_rcnn

    # Run YOLOv8 inference
    start_time_yolov8n = time.time()
    result1 = model_YOLOv8n.predict(image_path, save=True, conf=threshold)  # results list

    # Calculate YOLOv8 inference time
    end_time_yolov8n = time.time()
    inference_time_yolov8n = end_time_yolov8n - start_time_yolov8n

    # Run YOLOv8x inference
    start_time_yolov8x = time.time()
    result2 = model_YOLOv8x.predict(image_path, save=True, conf=threshold)  # results list

    # Calculate YOLOv8 inference time
    end_time_yolov8x = time.time()
    inference_time_yolov8x = end_time_yolov8x - start_time_yolov8x

    # Show the results side by side
    col1, col2 = st.columns(2)

    # Display Faster RCNN output
    with col1:
        st.write("Output for Faster RCNN:")
        image_faster_rcnn = plt.imread('outputs/result_faster_rcnn.jpg')
        plt.imshow(image_faster_rcnn)
        plt.axis('off')
        st.pyplot(plt)
        st.write(f"Inference Times - Faster RCNN: {inference_time_faster_rcnn:.2f} seconds")

    # Display YOLOv8 output
    with col2:
        st.write("Output of YOLOv8n:")
        im_array = result1[0].plot()  # plot a BGR numpy array of predictions
        im_yolov8 = im_array[..., ::-1]  # convert BGR to RGB
        plt.imshow(im_yolov8)  # show image
        plt.axis('off')
        st.pyplot(plt)
        st.write(f"Inference Times - YOLOv8n: {inference_time_yolov8n:.2f} seconds")
    with col1:
        st.write("Output of YOLOv8x:")
        im_array = result2[0].plot()  # plot a BGR numpy array of predictions
        im_yolov8 = im_array[..., ::-1]  # convert BGR to RGB
        plt.imshow(im_yolov8)  # show image
        plt.axis('off')
        st.pyplot(plt)
        st.write(f"Inference Times - YOLOv8x: {inference_time_yolov8x:.2f} seconds")
