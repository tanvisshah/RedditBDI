from flask import Flask, request
from flask_cors import CORS

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from transformers import BertConfig, BertTokenizerFast

from cloud_predict import predict_json

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})


def compile_model():
    model_name = 'bert-base-uncased'
    max_length = 128

    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False

    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
    
    model = None

    return model, tokenizer, max_length


def get_predictions(subject, writings):
    # testing_users, model, tokenizer, max_length
    text = []
    responses = []
    t = ""

    for dic in writings:
        t = dic["TITLE"] + " " + dic["TEXT"]
        text.append(t)

    nparr = [[]]
    for i in range(len(text)):
        arr = np.asarray([[text[i]]])
        arr = np.append( arr, [])
        nparr.append(arr)

    df = pd.DataFrame(nparr, columns=['Text'])
    df = df.drop(df.index[0])

    test_x = tokenizer(
        text = df['Text'].to_list(),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=True, 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = False,
        verbose = True)

    # Temporary fix to avoid error for users with few posts
    padded = np.hstack([test_x["input_ids"], np.zeros([test_x["input_ids"].shape[0], 128-test_x["input_ids"].shape[1]])])
    test_input = padded.astype(int).tolist()

    i = 0
    preds = []
    while i<len(test_input):
        temp_pred = predict_json("reddit-bdi-313122","tfmodel1",test_input[i:i+50],"version2")
        preds += temp_pred
        i += 50

    d = preds[0]

    for k in preds[0].keys():
        d[k] = [d[k]]
        d[k]+= list(dictionaries[k] for dictionaries in preds[1:])
    
    preds = d

    count = 0
    y_preds = np.zeros([1,21])
    for i, key in enumerate([subject]):
        for k in range(1,22):
            q = preds["q"+str(k)][count : count+len(writings)]
            q = np.asarray(q)
            q = np.mean(q, axis = 0)

            y_preds[i][k-1] = np.argmax(q)
        count += len(writings)


    y_preds = [[str(i) for i in item] for item in y_preds]

    for i in range(1):
        for j in range(21):
            val = y_preds[i][j]
            val = int(float(val))
            choice = str(val)
            if j==15 or j==17:
                idx = int(val/2+0.5)
                variant = 'a'
                if val % 2 == 0:
                    variant = 'b'
                if val == 0:
                    variant = ''
                choice = str(idx)+variant
            y_preds[i][j] = choice

    predictions = y_preds[0]

    return predictions


@app.route("/post", methods=['GET', 'POST', 'OPTIONS'])
def post():
    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'X-Requested-With, Content-Type, Origin, Accept',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    if request.method == 'POST':
        request_json = request.get_json(silent=True)

        if request_json and 'subject' in request_json:
            subject = request_json['subject']
        
        if request_json and 'responses' in request_json:
            responses = request_json['responses']

            if os.path.isdir("../ftp/task3/responses/"):
                with open("../ftp/task3/responses/Depression Questionnaires.txt", "a") as f:
                    date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    response_id = f'reddit_{subject}_{date}'
                    f.write(f"{response_id} {' '.join(responses)}\n")

            data = "Thank you for contributing your responses, {}!".format(subject)
            return (data, 200, headers)

        if request_json and 'writings' in request_json:
            writings = request_json['writings']

            responses = get_predictions(subject, writings)
            
            data = json.dumps({"subject": subject, "responses": responses})
            return (data, 200, headers)
        
        return ("Failure", 500, headers)

    return ("GET method", 500, headers)

model, tokenizer, max_length = compile_model()