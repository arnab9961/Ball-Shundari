import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np
import tempfile
import os
import cv2
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="eN2hqyjRB1E7vCewJrAD"
)

# Load Llama 2 model and tokenizer
model_name = "TheBloke/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q2_K.bin"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

st.set_page_config(layout="wide")

# Sidebar options
st.sidebar.title("Ball Shundari Apple Plum Object Detection")
mode = st.sidebar.radio("Choose Mode", ["Static Image Detection", "Real-Time Detection"])

if mode == "Static Image Detection":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    min_confidence = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.5)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image.save(temp_file, format='JPEG')
            temp_file_path = temp_file.name

        # Perform inference
        result = CLIENT.infer(temp_file_path, model_id="alpa/1")

        # Draw bounding boxes on the image
        image_np = np.array(image)
        descriptions = []
        for detection in result['predictions']:
            if detection['confidence'] >= min_confidence:
                x = int(detection['x'] - detection['width'] / 2)
                y = int(detection['y'] - detection['height'] / 2)
                width = int(detection['width'])
                height = int(detection['height'])
                confidence = detection['confidence']
                class_name = detection['class']
                descriptions.append(f"{class_name} with confidence {confidence:.2f}")

                # Draw the bounding box
                cv2.rectangle(image_np, (x, y), (x + width, y + height), (255, 0, 0), 2)
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(image_np, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert back to PIL image
        annotated_image = Image.fromarray(image_np)

        # Display the annotated image
        st.image(annotated_image, caption='Annotated Image.', use_column_width=True)

        # Clean up the temporary file
        os.remove(temp_file_path)

        # Generate description using Llama 2
        if descriptions:
            description_text = ". ".join(descriptions)
            prompt = f"""Analyze the following objects detected in an image: {description_text}

            Please provide a detailed description of the image based on these detections. Include:
            1. The types of objects present
            2. Their approximate quantities
            3. Any notable characteristics or arrangements
            4. A brief interpretation of what the image might represent (e.g., a fruit basket, an orchard scene, etc.)

            Description:"""

            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=True
                    )
                
                description = tokenizer.decode(outputs[0], skip_special_tokens=True)
                description = description.split("Description:")[1].strip()  # Extract only the generated description
                
                st.write("Description of the image:")
                st.write(description)
            except Exception as e:
                st.error(f"An error occurred while generating the description: {str(e)}")
        else:
            st.write("No objects were detected in the image.")

else:
    st.markdown("<h2>Real-Time Detection</h2>", unsafe_allow_html=True)
    components.html("""
    <!DOCTYPE html>
    <html>

    <head>
        <title>Roboflow Demo</title>
        <meta name="viewport" content="width=640, user-scalable=no" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js"
            integrity="sha512-bLT0Qm9VnAYZDflyKcBaQ2gg0hSYNQrJ8RilYldYQ1FxQYoCLtUjuuRuZo+fjqhx/qtq/1itJ0C2ejDxltZVFg=="
            crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.20/lodash.min.js"
            integrity="sha512-90vH1Z83AJY9DmlWa8WkjkV79yfS2n2Oxhsi2dZbIv0nC4E6m5AbH8Nh156kkM7JePmqD6tcZsfad1ueoaovww=="
            crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/async/3.2.0/async.min.js"
            integrity="sha512-6K6+H87tLdCWvY5ml9ZQXLRlPlDEt8uXmtELhuJRgFyEDv6JvndWHg3jadJuBVGPEhhA2AAt+ROMC2V7EvTIWw=="
            crossorigin="anonymous"></script>
        <script src="https://cdn.roboflow.com/0.2.26/roboflow.js"></script>
        <style>
            body.loading {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100vh;
            }
            video, canvas {
                position: absolute;
                display: block;
            }
        </style>
    </head>

    <body class="loading">
        <video id="video" autoplay muted playsinline></video>
        <div id="fps" style="position: fixed; top: 10px; left: 10px; color: white; font-size: 20px;"></div>
        <script>
            /*jshint esversion:6*/
            $(function () {
                const video = $("video")[0];
                var model;
                var cameraMode = "environment"; // or "user"
                const startVideoStreamPromise = navigator.mediaDevices
                    .getUserMedia({
                        audio: false,
                        video: {
                            facingMode: cameraMode
                        }
                    })
                    .then(function (stream) {
                        return new Promise(function (resolve) {
                            video.srcObject = stream;
                            video.onloadeddata = function () {
                                video.play();
                                resolve();
                            };
                        });
                    });

                var publishable_key = "rf_3DcgkrMt7VdPabCHl6dFb2XgntG3";
                var toLoad = {
                    model: "alpa",
                    version: 1
                };

                const loadModelPromise = new Promise(function (resolve, reject) {
                    roboflow
                        .auth({
                            publishable_key: publishable_key
                        })
                        .load(toLoad)
                        .then(function (m) {
                            model = m;
                            resolve();
                        });
                });

                Promise.all([startVideoStreamPromise, loadModelPromise]).then(function () {
                    $("body").removeClass("loading");
                    resizeCanvas();
                    detectFrame();
                });

                var canvas, ctx;
                const font = "16px sans-serif";

                function videoDimensions(video) {
                    var videoRatio = video.videoWidth / video.videoHeight;
                    var width = video.offsetWidth,
                        height = video.offsetHeight;
                    var elementRatio = width / height;

                    if (elementRatio > videoRatio) {
                        width = height * videoRatio;
                    } else {
                        height = width / videoRatio;
                    }

                    return {
                        width: width,
                        height: height
                    };
                }

                $(window).resize(function () {
                    resizeCanvas();
                });

                const resizeCanvas = function () {
                    $("canvas").remove();
                    canvas = $("<canvas/>");
                    ctx = canvas[0].getContext("2d");
                    var dimensions = videoDimensions(video);
                    console.log(
                        video.videoWidth,
                        video.videoHeight,
                        video.offsetWidth,
                        video.offsetHeight,
                        dimensions
                    );
                    canvas[0].width = video.videoWidth;
                    canvas[0].height = video.videoHeight;
                    canvas.css({
                        width: dimensions.width,
                        height: dimensions.height,
                        left: ($(window).width() - dimensions.width) / 2,
                        top: ($(window).height - dimensions.height) / 2
                    });
                    $("body").append(canvas);
                };

                const renderPredictions = function (predictions) {
                    var dimensions = videoDimensions(video);
                    var scale = 1;
                    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
                    predictions.forEach(function (prediction) {
                        const x = prediction.bbox.x;
                        const y = prediction.bbox.y;
                        const width = prediction.bbox.width;
                        const height = prediction.bbox.height;
                        ctx.strokeStyle = prediction.color;
                        ctx.lineWidth = 4;
                        ctx.strokeRect(
                            (x - width / 2) / scale,
                            (y - height / 2) / scale,
                            width / scale,
                            height / scale
                        );
                        ctx.fillStyle = prediction.color;
                        const textWidth = ctx.measureText(prediction.class).width;
                        const textHeight = parseInt(font, 10);
                        ctx.fillRect(
                            (x - width / 2) / scale,
                            (y - height / 2) / scale,
                            textWidth + 8,
                            textHeight + 4
                        );
                    });

                    predictions.forEach(function (prediction) {
                        const x = prediction.bbox.x;
                        const y = prediction.bbox.y;
                        const width = prediction.bbox.width;
                        const height = prediction.bbox.height;
                        ctx.font = font;
                        ctx.textBaseline = "top";
                        ctx.fillStyle = "#000000";
                        ctx.fillText(
                            prediction.class,
                            (x - width / 2) / scale + 4,
                            (y - height / 2) / scale + 1
                        );
                    });
                };

                var prevTime;
                var pastFrameTimes = [];
                const detectFrame = function () {
                    if (!model) return requestAnimationFrame(detectFrame);
                    model
                        .detect(video)
                        .then(function (predictions) {
                            requestAnimationFrame(detectFrame);
                            renderPredictions(predictions);
                            if (prevTime) {
                                pastFrameTimes.push(Date.now() - prevTime);
                                if (pastFrameTimes.length > 30) pastFrameTimes.shift();
                                var total = 0;
                                _.each(pastFrameTimes, function (t) {
                                    total += t / 1000;
                                });
                                var fps = pastFrameTimes.length / total;
                                $("#fps").text(Math.round(fps));
                            }
                            prevTime = Date.now();
                        })
                        .catch(function (e) {
                            console.log("CAUGHT", e);
                            requestAnimationFrame(detectFrame);
                        });
                };
            });
        </script>
    </body>
    </html>
    """, height=720)
