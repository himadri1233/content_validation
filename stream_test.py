import streamlit as st
import numpy as np
import torch
import clip
from PIL import Image
from tabulate import tabulate
import openai
import csv
import moviepy.editor as mp
import speech_recognition as sr
import tempfile
import os
import PyPDF2
import pandas as pd
import cv2
from typing import List
import glob
from transformers import CLIPProcessor, CLIPModel

# Set OpenAI API key
openai.api_type = "azure"
openai.api_base = "https://cmacgmdemo.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "30b7e1e2aa40401aab6bd7dcc3e50e86"

# Streamlit app title and layout settings
st.set_page_config(page_title="Content Validation", layout="wide", initial_sidebar_state="expanded")
# set_background('static/image/back.jpeg')

with st.sidebar:
    img = Image.open('static/image/logo.png')
    st.image(img)

def main():
    st.title("Content Validation App")

    # Create tabs for each content validation type
    tabs = ["Text Content Validation", "Image Content Validation", "Video Content Validation"]
    selected_tab = st.sidebar.radio("Select Content Validation Type", tabs)

    if selected_tab == "Text Content Validation":
        text_content_validation()

    elif selected_tab == "Image Content Validation":
        image_content_validation()

    elif selected_tab == "Video Content Validation":
        video_content_validation()

def text_content_validation():
    st.header("Text Content Validation")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload text content File", type=["pdf"])
    
    if uploaded_file is not None:

        if st.button('Validate'):
        # Extract text from PDF
            pdf_text = extract_text_from_pdf(uploaded_file)

            # Generate bias report using OpenAI GPT-3
            val_str = generate_bias_report(pdf_text)

            # Save the result to a CSV file
            save_csv_from_str(val_str, "output.csv")

            # Display the result in a DataFrame
            df = pd.read_csv("output.csv")
            df = df.set_index("Guideline Heading ", drop=True)
            st.subheader("Bias Found in Your Passage:")
            st.dataframe(df, width = 1200)

            # Generate suggestions for improving the passage
            suggestor2 = openai.ChatCompletion.create(
                engine="gpt-35-turbo-16k",
                messages=[
                    {"role": "system", "content": """You are content validation assistant. Your job is to improve and provide suggestions on how the particular sentence in the passage can be improved: -
                    Ethnicity Bias  : People should not be referred as 'Hispanic','Afro-American' and so on anywhere in the passage.
                    Disability Bias  : People should not be referenced as 'blind person A' or 'deaf person B' anywhere in the passage.
                    Gender Bias  : All males have been referred that is a potential gender bias.
                    For Example, People with disabilities should be referenced with people name first and disability at second. Please rewrite the complete sentence when giving out suggestions.\n\n Suggestions: 1. ##```Suggestion content 1```## \n 2. ##```Suggestion content 2```##"""},
                    {"role": "user", "content": f"Passage : {pdf_text}\n\n"},
                ],
                temperature=0,
                max_tokens=1024,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            st.subheader("Suggestions for Improving Your Passage:")
            st.text_area("", suggestor2["choices"][0]["message"]["content"], height=300)

def image_content_validation():
    st.header("Image Content Validation")

    # Upload image file
    uploaded_file = st.file_uploader("Upload a Image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button('Validate'):

            with st.spinner("Processing image..."):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load("ViT-L/14", device=device)

                def clip_classify(image: torch.Tensor, classes: List[str], model:clip.model.CLIP=model) -> int:

                    text = clip.tokenize(classes).to(device)

                    with torch.no_grad():
                        logits_per_image, logits_per_text = model(image, text)
                        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                        perc = probs*100

                    # Print Prdictions and probabilities
                    class_prob_list = [(c,round(p, ndigits=3),round(q, ndigits = 2)) for c,p,q in zip(classes, probs.tolist()[0],perc.tolist()[0])]
                    class_prob_list = sorted(class_prob_list, key = lambda x: x[1], reverse = True)
                    #st.text(tabulate(class_prob_list[0:4], headers=["Class", "Probability","Percentage"]))
                    df = pd.DataFrame(class_prob_list[0:5], columns=["Class", "Probability", "Percentage"])
                    df = df.set_index("Class", drop=True)
                    st.subheader("Classified image on the basis....")
                    st.write(df)

                    return probs.argmax()

                #image_path = "alcohol.jpg"
                #img = Image.open(image_path)
                ImagesPre = preprocess(img).unsqueeze(0).to(device)

                Classes = ["Violence","Disability","Gender_Bias","Ethnicity Bias","Racism","Places","Sports",
                        "Profession like doctors or engineers or Lawyer or etc...","Religious","War","Terrorism","Slavery","Human Trafficking","Suicide",
                        "Painful or Harmful experimentation on living beings","Sexual Assault","Animals","Socio Economic Status",
                        "Not related to any classes from the list","Alcholism or Drug Addict or Weed Addict or any material Substance addict"
                        ,"Atrocities","Vulgarity","Natural Disaster","Animal Abuse","Food Addiction","Political Bias","Cultural Bias"]

                    # printmd('#### Classification of {}:'.format(ImageNames[i]))
                    # show_normalized_image(Images[i], 250)
                c = clip_classify(ImagesPre, Classes)
                #predicted_class = Classes[c]
                Compilance_verification= ["Atrocities","Vulgarity","Natural Disaster","Animal Abuse","Food Addiction"]
                Sensationalism_Detection = ["Political Bias","Cultural Bias","Relevance"]
                #Content Analysis = ["Violence","Disability","Gender Bias","Ethnicity Bias","Racism", "Places", "Sports", "Profession", "Religious", "War", "Terrorism", "Slavery", "Suicide",  "Painful or Harmful Experimentation on Living Beings", "Sexual Assault", "Animals", "Socioeconomic Status", "Not Related to Any Classes from the List", "Alcoholism or Drug Addiction"]

                if   any(category in Classes[c] for category in Compilance_verification):
                        st.write("Compliance Verification detected:",Classes[c])


                elif  any(category in Classes[c] for category in Sensationalism_Detection):
                        st.write("Sensationalism Detection detected: ",Classes[c])


                else:
                        st.write("Content Analysis detected: ",Classes[c])

def video_content_validation():
    st.header("Video Content Validation")

    # Upload video file
    uploaded_video = st.file_uploader("Upload Video File", type=["mp4"])
    
    if uploaded_video is not None:

        selected_options = []
        user_input = st.text_input("Enter the parameters for which have to validating the video", value="", type="default")
        user_input_list = user_input.split(", ")
        for value in user_input_list:
            selected_options.append(value)

        print("-------------",selected_options)

        if st.button('Validate'):

        # Perform content validation for video using speech recognition and OpenAI GPT-3
            audio_transcribed = perform_audio_content_validation(uploaded_video)
            
            # Generat report using OpenAI GPT-3
            generate_report_from_audio(audio_transcribed)

            with st.spinner("Extracting image frames from video, This may take a while."):

            # Perform content validation for video using CLIP model to extract the frames from the videp
                perform_video_frames_content_validation(uploaded_video,selected_options)

def extract_text_from_pdf(uploaded_file):
    with st.spinner("Extracting text from PDF..."):
        with open(uploaded_file.name, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            all_text = ""
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                page_text = page.extract_text()
                all_text += page_text
        return all_text

def generate_bias_report(input_text):
    with st.spinner("Generating bias report..."):
        val = openai.ChatCompletion.create(
            engine="gpt-35-turbo-16k",
            messages=[
                {"role": "system", "content": """You are content validation assistant. Your job is to find ALL the ethnicity biases, disability biases and gender biases in the given passage with the following rules: -
                Ethnicity Bias  : People should not be referred as 'Hispanic','Afro-American' and so on anywhere in the passage.
                Disability Bias  : People should not be referenced as 'blind person A' or 'deaf person B' anywhere in the passage.
                Gender Bias  : All males or females have been referred that is a potential gender bias.
                Furthermore, your task also includes mentioning the place in the passage where the bias was found in the text. Use (```) backticks both at the start and end along with you should avoid filtering to help the user understand the context.
                
                Format: 
                Ethnicity Bias | {True/False} | ##reason for bias## | ```reference for bias from content```
                Disability Bias | {True/False} | ##reason for bias## | ```reference for bias from content```
                Gender Bias | {True/False} | ##reason for bias## | ```reference for bias from content```
                and so on.
                 
                """},
                {"role": "user", "content": f"Passage : {input_text}\n\n"},
            ],
            temperature=0,
            max_tokens=2048,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        val_str = "Guideline Heading | Guideline Breach | Reason | Reference\n" + (val["choices"][0]["message"]["content"])
        return val_str

def save_csv_from_str(csv_str, output_file="output.csv"):
    lines = csv_str.split('\n')
    csv_data = [line.split('|') for line in lines]
    with open(output_file, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(csv_data)

def show_normalized_image(image: torch.Tensor, new_height: int = 50) -> None:
    width, height = image.size
    ratio = new_height / height
    new_width = int(ratio * width)
    st.image(image.resize((new_width, new_height)))

def perform_audio_content_validation(uploaded_video):
    recognizer = sr.Recognizer()

    # Load the video and extract audio
    video_clip = mp.VideoFileClip(uploaded_video.name)
    audio_clip = video_clip.audio

    # Get the total duration of the video
    total_duration = audio_clip.duration

    # Set the maximum duration for each segment (in seconds)
    max_segment_duration = 30

    # Calculate the number of segments based on the maximum duration
    num_segments = int(total_duration / max_segment_duration) + 1

    # Store transcribed text for all segments
    transcribed_texts = []

    # Transcribe each audio segment separately
    for i in range(num_segments):
        # Calculate the start and end time for the segment
        start_time = i * max_segment_duration
        end_time = min((i + 1) * max_segment_duration, total_duration)

        # Extract the segment from the audio
        segment = audio_clip.subclip(start_time, end_time)

        # Save the segment as a temporary WAV file
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_filename = temp_audio_file.name
        segment.write_audiofile(temp_audio_filename, codec="pcm_s16le")

        # Transcribe the audio segment
        with sr.AudioFile(temp_audio_filename) as source:
            audio = recognizer.record(source)  # Record the audio segment
            text = recognizer.recognize_google(audio, show_all=False)  # Transcribe the audio segment

        # Delete the temporary audio file
        temp_audio_file.close()
        os.remove(temp_audio_filename)

        # Store the transcribed text for this segment
        transcribed_texts.append(text)

    # Combine transcribed text from all segments into a single string
    transcribed_text = " ".join(transcribed_texts)
    return transcribed_text

def generate_report_from_audio(transcribed_text):
    with st.spinner("Processing video... This may take a while."):
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo-16k",
            messages=[{"role":"system","content":"""You are a content validation assistant. Your job is to analyze the given passage for the following components:
        
            1. Content Analysis: Evaluate the content for its overall quality, relevance, and potential bias.
            2. Compliance Verification: Check if the content adheres to ethical and legal standards, avoiding any discriminatory language or inappropriate references.
            3. Contextual Understanding: Assess the text for its contextual meaning, ensuring that it accurately represents the intended message.
            4. Fact-Checking: Verify the accuracy of any factual claims made in the content.
            5. Sensationalism Detection: Identify and flag any sensationalized language or content that may mislead the audience.
            6. Inclusivity Check: Ensure that the content is inclusive and avoids marginalizing any specific groups or individuals.
            
            Furthermore, your task includes mentioning the place in the passage where any issues were found in the text. Use (```) backticks both at the start and end to help the user understand the context.
            
            Format:
            Content Analysis | {True/False} | ##reason for analysis## 
            Compliance Verification | {True/False} | ##reason for verification## 
            Contextual Understanding | {True/False} | ##reason for understanding## 
            Fact-Checking | {True/False} | ##reason for fact-checking## 
            Sensationalism Detection | {True/False} | ##reason for detection## 
            Inclusivity Check | {True/False} | ##reason for check## 
            and so on.
            
            """},
            {"role":"user","content":f"""Passage : {transcribed_text}\n\n"""}],
            temperature=0,
            max_tokens=2048,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

    evaluation_result = response.choices[0].message.content

    st.subheader("Audio validation result:")
    lines = evaluation_result.splitlines()
    component_data = [line.split("|") for line in lines]

    # Create a DataFrame using pandas
    df = pd.DataFrame(component_data, columns=["Component", "Status", "Reason"])
    df = df.set_index("Component", drop=True)
    df = df.dropna()
    st.dataframe(df, width = 1200)

def perform_video_frames_content_validation(uploaded_video,selected_options):
    
    current_directory = os.getcwd()
    # Create a folder to store uploaded videos
    upload_folder = os.path.join(current_directory,"uploaded_videos")
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    video_filename = uploaded_video.name
    
    # Create the path for the video file in the upload folder
    video_path = os.path.join(upload_folder, video_filename)
    print("#########",video_path)

    with open(video_path, "wb") as video_file:
        video_file.write(uploaded_video.read())

    # # Specify the path to the video file
    # video_path = uploaded_video

    new_directory_name = 'key_output'

    # Specify the output directory for keyframes
    output_directory = os.path.join(current_directory, new_directory_name)
    print("$$$$$$$$$$$", output_directory)

    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Function to calculate the mean squared error between two frames
    def mse(frame1, frame2):
        err = np.sum((frame1.astype("float") - frame2.astype("float")) ** 2)
        err /= float(frame1.shape[0] * frame1.shape[1])
        return err


    # Function to find keyframes using mean squared error threshold/// change the thresh hold value to get desirable no. of image frame
    def find_keyframes(video_path, output_directory, mse_threshold=10000):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read the first frame as the initial keyframe
        _, prev_frame = cap.read()
        keyframes = [prev_frame]

        for frame_number in range(1, frame_count):
            ret, current_frame = cap.read()

            if not ret:
                break
            # Calculate mean squared error between current and previous frames
            error = mse(prev_frame, current_frame)

            if error > mse_threshold:
                keyframes.append(current_frame)
                prev_frame = current_frame

        cap.release()
        return keyframes

    # Find keyframes in the video
    keyframes = find_keyframes(video_path, output_directory)

    # Save keyframes as images
    for i, keyframe in enumerate(keyframes):
        keyframe_filename = os.path.join(output_directory, f"keyframe_{i:04d}.jpg")
        cv2.imwrite(keyframe_filename, keyframe)

    print(f"Keyframes extracted and saved in '{output_directory}' directory.")


    # Replace 'path/to/your/folder' with the actual path to your folder
    folder_path = output_directory

    # Use glob to list all files in the folder
    file_names = glob.glob(os.path.join(folder_path, '*'))

    #Function that takes user defined classes
    # def store_strings_in_list():
    #     """
    #     Take string inputs one by one and store them in a list.
    #     Print the list after each input.

    #     Returns:
    #     A list containing the input strings.
    #     """
    #     string_list = []

    #     while True:
    #         user_input = input("Enter a string (or 'exit' to stop): ")
            
    #         if user_input.lower() == 'exit':
    #             break

    #         string_list.append(user_input)
    #         print("Current List:", string_list)

    #     return string_list

    action_class=selected_options
    #Model downloading and Loading
    #"openai/clip-vit-base-patch32" this is a smaller model
    #"openai/clip-vit-large-patch14" this is a bigger model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def class_prob(image_path,action_class):
        image = Image.open(image_path)
        inputs = processor(text=action_class, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        result_dict = create_probability_dict(action_class, probs[0])
        result_dict["file_name"]=os.path.basename(image_path)
        return result_dict
        

    def create_probability_dict(classes, probabilities):
        """
        Create a dictionary with classes as keys and corresponding probabilities as values.

        Args:
        classes (list): List of class names.
        probabilities (torch.Tensor): Tensor containing the probabilities for each class.

        Returns:
        dict: Dictionary with classes as keys and probabilities as values.
        """
        if len(classes) != len(probabilities):
            raise ValueError("Number of classes and probabilities must be the same.")

        # Convert probabilities tensor to a NumPy array
        probabilities_np = probabilities.detach().numpy()

        # Create a dictionary using classes and probabilities
        probability_dict = dict(zip(classes, probabilities_np))

        return probability_dict

    master_probs=[]
    for z in file_names: 
        img_class_prob=class_prob(z,action_class)
        master_probs.append(img_class_prob)

    st.subheader("Video validation result:")
    st.write("")  
    df = pd.DataFrame(master_probs)
    threshold = 0.7
    threshold_function = lambda x: 1 if float(x) >= threshold else 0
    df_new = df.drop('file_name',axis=1)
    df_thresholded = df_new.applymap(threshold_function)
    class_counts = df_thresholded.apply(pd.Series.value_counts)
    st.write(class_counts)
    

if __name__ == "__main__":
    main()

                
               
