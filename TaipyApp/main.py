#!/usr/bin/env python
import os,sys 
import re,csv
from twikit import Client
import subprocess
from transformers import pipeline
import numpy as np
import pandas as pd
from taipy.gui import Gui, notify

text = "1779813966006423851"

page = """
# **Sentiment Analysis**{: .color-primary} **Twitter**{: .color-primary}

<|layout|columns=1 1|
<|
**Insert TWEET ID for example:** <|{text}|>

**Enter a tweet ID:**
<|{text}|input|>
<|Analyze|button|on_action=local_callback|>
|>


<|Table|expandable|
<|{dataframe}|table|width=100%|number_format=%.2f|>
|>
|>


<|{dataframe}|chart|type=bar|x=Text|y[1]=Rabbia|y[2]=Disgusto|y[3]=Paura|y[4]=Gioia|y[5]=Neutrale|y[6]=Tristezza|y[7]=Sorpresa|color[1]=green|color[2]=grey|color[3]=red|color[4]=yellow|color[5]=magenta|color[6]=brown|color[7]=violet|>
"""

## for sentiment analysis:
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

dataframe = pd.DataFrame(
    {   
        "Tweet ID":[""],
        "Text":[""],
        "Rabbia":[0.14],
        "Disgusto":[0.14],
        "Paura":[0.14],
        "Gioia":[0.14],
        "Neutrale":[0.14],
        "Tristezza":[0.14],
        "Sorpresa":[0.14]
    }
)

dataframe2 = dataframe.copy()

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

def translate_to_english(text):
    """translate to english"""
    # Define the shell command
    command = 'echo "translate to english: ' + text + '" | ollama run tinydolphin'

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

    # Replace each URL in the string with an empty string
    for url in urls:
        text = text_.replace(url, '')
    return text

def remove_special_characters(string):
    return ''.join(e for e in string if e.isalnum() or e.isspace() or e in ".',;:!?-")

def get_text_single_tweet(tweet_id): 
    tweet = client.get_tweet_by_id(tweet_id)
    # get language and text:
    orig_ = tweet.text
    lang_ = tweet.lang
    text_ = re.sub(r'\s+', ' ', re.sub(r'[\|\"]', '',tweet.text))
    text_ = remove_urls(text_)
    text_ = remove_special_characters(text_)
    
    print(text_, lang_)
    if lang_ == 'it':
        # translate to english
        text_ = translate_to_english(text_)
        print(text_)
    elif lang_ == 'en':
        text_ = text_
    else:
        print("Language not yet supported/Lingua non ancora supportata")
    
    return text_, orig_

def analyze_text(input_text: str) -> dict:
    text_, orig_ = get_text_single_tweet(input_text)
    scores = classifier(text_)
    print(scores[0])
    return {
        "Tweet ID": input_text,
        "Text": orig_[:50]+"...",
        "Rabbia": next((d['score'] for d in scores[0] if d['label'] == 'anger'), None),
        "Disgusto": next((d['score'] for d in scores[0] if d['label'] == 'disgust'), None),
        "Paura": next((d['score'] for d in scores[0] if d['label'] == 'fear'), None),
        "Gioia": next((d['score'] for d in scores[0] if d['label'] == 'joy'), None),
        "Neutrale": next((d['score'] for d in scores[0] if d['label'] == 'neutral'), None),
        "Tristezza": next((d['score'] for d in scores[0] if d['label'] == 'sadness'), None),
        "Sorpresa": next((d['score'] for d in scores[0] if d['label'] == 'surprise'), None),
    }


def local_callback(state) -> None:
    """
    Analyze the text and updates the dataframe

    Args:
        - state: state of the Taipy App
    """
    notify(state, "Info", f"The text is: {state.text}", True)
    temp = state.dataframe.copy()
    scores = analyze_text(state.text)
    temp.loc[len(temp)] = scores
    state.dataframe = temp
    state.text = ""


pages = {  
    "Tweet": page
}
Gui(pages=pages).run(title="Sentiment Analysis")

