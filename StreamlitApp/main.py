#!/usr/bin/env python
import os,re
from twikit import Client
import subprocess
import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
## -------------------------------------------------------------
# Enter your account information
USERNAME = "..."
EMAIL = "..."
PASSWORD = "..."

client = Client('en-US')
client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD
    )

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)


def translate_to_english(text):
    """translate to english"""
    # Define the shell command
    command = 'echo "translate to english: ' + text + '" | ollama run mistral'

    # Execute the command and capture the output
    try:
        result = subprocess.check_output(command, shell=True, text=True)
        # print("Output:", result)
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", e)
    
    return result

def remove_urls(text_):
    # Use a regular expression to find all URLs in the string
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text_)
    if len(urls) != 0:
        # Replace each URL in the string with an empty string
        for url in urls:
            text = text_.replace(url, '')
    else:
        text = text_
    return text

def remove_special_characters(string):
    return ''.join(e for e in string if e.isalnum() or e.isspace() or e in ".',;:!?-")

def get_text_single_tweet(tweet_id): 
    tweet = client.get_tweet_by_id(tweet_id)
    # get language and text:
    orig_ = tweet.text
    lang_ = tweet.lang
    id_ = tweet.user
    timestamp_ = tweet.created_at
    text_ = re.sub(r'\s+', ' ', re.sub(r'[\|\"]', '',orig_))
    text_ = remove_urls(text_)
    text_ = remove_special_characters(text_)
    
    if lang_ == 'it':
        # translate to english
        text_ = translate_to_english(text_)
        print(text_)
    elif lang_ == 'en':
        text_ = text_
    else:
        print("Language not yet supported/Lingua non ancora supportata")
    
    return text_, orig_, lang_, id_, timestamp_

def analyze_text(input_text: str, tweet_id, orig_, lang_, id_, timestamp_) -> dict:
    scores = classifier(text_)
    print(scores[0])
    return [
        tweet_id,
        # "User": id_,
        # "Text": orig_,
        lang_,
        # "Created at": timestamp_,
        next((d['score'] for d in scores[0] if d['label'] == 'anger'), None),
        next((d['score'] for d in scores[0] if d['label'] == 'disgust'), None),
        next((d['score'] for d in scores[0] if d['label'] == 'fear'), None),
        next((d['score'] for d in scores[0] if d['label'] == 'joy'), None),
        next((d['score'] for d in scores[0] if d['label'] == 'neutral'), None),
        next((d['score'] for d in scores[0] if d['label'] == 'sadness'), None),
        next((d['score'] for d in scores[0] if d['label'] == 'surprise'), None),
    ]

def load_or_create_file(filename):
    """
    This function checks if a file exists and loads its contents if it does.
    Otherwise, it creates a new empty file.

    Args:
        filename: The name of the file to load or create.

    Returns:
        The contents of the file as a string (if it existed) or an empty string (if created).
    """
    # Define the header columns for the DataFrame
    header_columns = [
        "Tweet ID",
        # "User",
        # "Text",
        "Language",
        # "Created at",
        "Rabbia",
        "Disgusto",
        "Paura",
        "Gioia",
        "Neutrale",
        "Tristezza",
        "Sorpresa",
    ]

    # Check if the file "local_dataset.csv" exists
    if os.path.exists("local_dataset.csv"):
        # Load existing data from the file
        existing_df = pd.read_csv("local_dataset.csv", sep='|')
    else:
        # Create an empty DataFrame with header columns
        dataframe = pd.DataFrame(columns=header_columns)
        # Save it to "local_dataset.csv"
        dataframe.to_csv("local_dataset.csv", sep='|', index=False)
        # print("Created an empty DataFrame and saved it to local_dataset.csv")
        existing_df = pd.read_csv("local_dataset.csv", sep='|')

    return existing_df

def draw_pie_chart(row_):
    plt.style.use('dark_background')
    x = np.char.array(["Anger/Rabbia", "Disgust/Disgusto", "Fear/Paura", "Joy/Gioia", "Neutral/Neutrale", "Sadness/Tristezza", "Surprise/Sorpresa"])
    y = row_
    colors = ['yellowgreen','red','gold','lightskyblue','lightcoral','grey', 'darkgreen'] #,'grey','violet','magenta','cyan
    porcent = 100.*y/y.sum()
    fig = plt.figure()
    patches, texts = plt.pie(y, colors=colors, startangle=90, radius=1.2)
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

    sort_legend = False
    if sort_legend:
        patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                            key=lambda x: x[2],
                                            reverse=True))

    plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.1, 1.),
            fontsize=8)
    plt.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)

## -----------------------------------------------------------------
header_columns = ["Tweet ID", "Language", "Rabbia", "Disgusto", "Paura", "Gioia", "Neutrale", "Tristezza", "Sorpresa"]
# st.set_page_config(layout="wide")
st.title("Twitter Sentiment Analysis")
tweet_id= st.text_input('Insert a TWEET ID', '1779813966006423851')
st.write('The current TWEET ID is', tweet_id)

if st.button('Analyse Tweet', type="primary"):
    df = load_or_create_file('local_dataset.csv')
    df = df.drop_duplicates(subset=['Tweet ID'], keep='first')
    if df["Tweet ID"].astype(str).str.contains(tweet_id).any():
        text_, orig_, lang_, id_, timestamp_ = get_text_single_tweet(tweet_id)
        st.write("-"*50)
        st.write("Tweet's text:")
        st.write(orig_)
        st.write("-"*50)
        # Convert tweet_id to integer for comparison
        tweet_id_int = int(tweet_id)
        # Get the row that contains the Tweet ID
        row = df.loc[df["Tweet ID"] == tweet_id_int]
        draw_pie_chart(np.asarray(row.iloc[0,2:]))
             
    else:
        text_, orig_, lang_, id_, timestamp_ = get_text_single_tweet(tweet_id)
        st.write("-"*50)
        st.write("Tweet's text:")
        st.write(orig_)
        st.write("-"*50)
        new_row= analyze_text(text_, tweet_id, orig_, lang_, id_, timestamp_)      
        draw_pie_chart(np.asarray(new_row[2:]))
        new_row = pd.DataFrame([new_row], columns=header_columns)
        df_up= pd.concat([df, new_row])
        df_up.to_csv("local_dataset.csv", sep="|", mode="a", index=False, header=None)

