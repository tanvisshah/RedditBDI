import xml.etree.ElementTree as ET
import glob
import re

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.layers import Input, Dropout, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical


def get_xml_tag_contents(xml_str, tag):
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    open_tag_end = xml_str.find(open_tag)+len(open_tag)
    close_tag_start = xml_str.find(close_tag, open_tag_end)
    return xml_str[open_tag_end:close_tag_start], xml_str[close_tag_start+len(close_tag):]

def parse_subject_xml(filename):
    with open(filename, "r+", encoding="utf8") as f:
        xml_str = f.read()
    xml_str, _ = get_xml_tag_contents(xml_str, "INDIVIDUAL")
    id_str, xml_str = get_xml_tag_contents(xml_str, "ID")
    subjectId = id_str.strip()
    writings = []
    while (xml_str.strip() != ""):
        writing = {}
        writing_str, xml_str = get_xml_tag_contents(xml_str, "WRITING")
        
        title_str, writing_str = get_xml_tag_contents(writing_str, "TITLE")
        writing["TITLE"] = title_str.strip()
        date_str, writing_str = get_xml_tag_contents(writing_str, "DATE")
        writing["DATE"] = date_str.strip()
        info_str, writing_str = get_xml_tag_contents(writing_str, "INFO")
        writing["INFO"] = info_str.strip()
        text_str, writing_str = get_xml_tag_contents(writing_str, "TEXT")
        writing["TEXT"] = text_str.strip()

        writings.append(writing)
    return subjectId, writings

def load_users():
  training_users = {}
  # Add training subjects & posts
  filepath = f"./ftp/task3/training/*/subject*.xml"
  filenames = glob.glob(filepath)
  for filename in filenames:
      subjectId, writings = parse_subject_xml(filename)
      if subjectId == "" or writings == []:
          continue
      year = re.findall("./ftp/task3/training/([0-9]{4})/subject[0-9]+.xml", filename)[0]
      subjectId = f"{year}-{subjectId}"
      training_users[subjectId] = {"WRITINGS": writings}
  # Add training subject responses
  filepath = f"./ftp/task3/training/*/Depression Questionnaires_anon.txt"
  filenames = glob.glob(filepath)
  for filename in filenames:
      year = re.findall("./ftp/task3/training/([0-9]{4})/Depression Questionnaires_anon.txt", filename)[0]
      with open(filename, "r+") as f:
          user_responses = [line.strip().split("\t") for line in f.readlines() if line.strip != ""]
      for user_response in user_responses:
          subjectId = f"{year}-{user_response[0]}"
          responses = user_response[1:]
          training_users[subjectId]['RESPONSES'] = responses
  print(f"Loaded {len(training_users)} training users")

  # Add testing subjects & posts
  testing_users = {}
  filepath = f"./ftp/task3/data/2021/erisk2021-T3_Subject*.xml"
  filenames = glob.glob(filepath)
  for filename in filenames:
      subjectId, writings = parse_subject_xml(filename)
      if subjectId == "" or writings == []:
          continue
      testing_users[subjectId] = {"WRITINGS": writings, "RESPONSES": []}
  print(f"Loaded {len(testing_users)} testing users")

  return training_users, testing_users


def split_train_test(training_users):
  text = []
  responses = []
  t = ""

  for key in training_users:
    key_response = []
    for i, val in enumerate(training_users[key]["RESPONSES"]):
      one_hot = [0]*4
      idx = int(val[0])
      variant = val[1:]
      if i==15 or i==17:
        one_hot = [0]*7
        idx = idx*2
        if variant == 'a':
          idx -= 1
      one_hot[idx] = 1

      key_response.append(np.asarray(one_hot))

    for dic in training_users[key]["WRITINGS"]:
      t = dic["TITLE"] + dic["TEXT"]
      text.append(t)
      responses.append(key_response)

  nparr=[[]]
  for i in range(len(text)):
    arr = np.asarray([[text[i]]])
    arr = np.append( arr, [responses[i]])
    nparr.append(arr)
  
  df = pd.DataFrame(nparr, columns=['Text'] + [f'Q{i+1}' for i in range(21)])
  df = df.drop(df.index[0])

  # df, df_test = train_test_split(df, train_size = 46503, shuffle=False)
  df, df_test = train_test_split(df, train_size = 10, shuffle=False)

  Q = [np.asarray([j for j in df[f"Q{i+1}"]]) for i in range(21)]

  Q_test = [np.asarray([j for j in df_test[f"Q{i+1}"]]) for i in range(21)]

  return df, df_test, Q, Q_test


def compile_model():
  # for i in range(1,22):
  #     df['Q'+str(i)+'_label'] = pd.Categorical(df['Q'+str(i)])
  #     df['Q'+str(i)] = df['Q'+str(i)+'_label'].cat.codes

  model_name = 'bert-base-uncased'
  max_length = 128

  config = BertConfig.from_pretrained(model_name)
  config.output_hidden_states = False

  tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

  transformer_model = TFBertModel.from_pretrained(model_name, config = config)


  bert = transformer_model.layers[0]


  input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
  inputs = {'input_ids': input_ids}

  bert_model = bert(inputs)[1]
  dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
  pooled_output = dropout(bert_model, training=False)

  q = []
  for i in range(21):
    units = 4
    if i==15 or i==17:
      units = 7
    q.append(Dense(units=units, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name=f'q{i+1}', activation='softmax')(pooled_output))

  outputs = {f'q{i+1}': q[i] for i in range(21)}


  model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')
  model.summary()

  # Set an optimizer
  optimizer = Adam(
      learning_rate=5e-5,
      epsilon=1e-08,
      decay=0.01,
      clipnorm=1.0)

  # Set loss and metrics
  loss = {f'q{i+1}': CategoricalCrossentropy(from_logits = True) for i in range(21)}

  metric = {f'q{i+1}': CategoricalAccuracy('accuracy') for i in range(21)}

  # Compile the model
  model.compile(optimizer = optimizer, loss = loss, metrics = metric)
  model._layers = [
      layer for layer in model._layers if isinstance(layer, Layer)
  ]

  return model, tokenizer, max_length


def train(df, Q, model, tokenizer, max_length):
  # Ready output data for the model
  y_q = [Q[i] for i in range(21)]

  # Tokenize the input (takes some time)
  x = tokenizer(
      text=df['Text'].to_list(),
      add_special_tokens=True,
      max_length = max_length,
      truncation=True,
      padding = True, 
      return_tensors='tf',
      return_token_type_ids = False,
      return_attention_mask = False,
      verbose = True)

  history = model.fit(
      x={'input_ids': x['input_ids']},
      y={f'q{i+1}': y_q[i] for i in range(21)},
      validation_split=0.2,
      batch_size=64,
      epochs=1)

  model.save("./tiny_model")

  print("Saved model")

  return model


def test(df_test, Q_test, model, tokenizer, max_length, training_users):
  test_y_q = [Q_test[i] for i in range(21)]

  test_x = tokenizer(
      text = df_test['Text'].to_list(),
      add_special_tokens=True,
      max_length=max_length,
      truncation=True,
      padding=True, 
      return_tensors='tf',
      return_token_type_ids = False,
      return_attention_mask = False,
      verbose = True)

  # Run evaluation
  model_eval = model.evaluate(
      x={'input_ids': test_x['input_ids']},
      y={f'q{i+1}': test_y_q[i] for i in range(21)}
  )

  preds = model.predict(test_x["input_ids"])

  keys_test=[]
  for i,key in enumerate(training_users):
    if i>=75:
      keys_test.append(key)
  test_users = {key: training_users[key] for key in keys_test}

  count = 0
  y_preds = np.zeros([15,21])
  for i,key in enumerate(test_users):
      for k in range(1,22):
        q = preds["q"+str(k)][count : count+len(test_users[key]["WRITINGS"])]
        q = np.asarray(q)
        q = np.mean(q, axis = 0)

        y_preds[i][k-1] = np.argmax(q)
      count += len(test_users[key]["WRITINGS"])


  y_preds = [[str(i) for i in item] for item in y_preds]

  for i in range(15):
    for j in range(21):
      val = y_preds[i][j]
      choice = int(val)
      if j==15 or j==17:
        idx = int((int(val)+1)/2)
        variant = 'a'
        if int(val) % 2 == 0:
          variant = 'b'
        if int(val) == 0:
          variant = ''
        choice = str(idx)+variant
      y_preds[i][j] = choice

  correct_count = []
  depression_level_predicted =[]
  depression_level_truth = []

  for i, key in enumerate(test_users):
    arr = test_users[key]["RESPONSES"]
    count=0
    depr_pred=0
    depr_truth=0
    for j in range(21):
      depr_truth = depr_truth + int(arr[j][0])
      depr_pred = depr_pred + int(y_preds[i][j][0])

      if arr[j] == y_preds[i][j]:
        count+=1

    correct_count.append(count)
    depression_level_truth.append(depr_truth)
    depression_level_predicted.append(depr_pred)

  true_cat = []
  predicted_cat = []
  for i in range(15):
    val1 = depression_level_truth[i]
    val2 = depression_level_predicted[i]

    if val1>=0 and val1<=9:
      true_cat.append("minimal")
    elif val1>=10 and val1<=18:
      true_cat.append("mild")
    elif val1>=19 and val1<=29:
      true_cat.append("moderate")
    else:
      true_cat.append("severe")

    if val2>=0 and val2<=9:
      predicted_cat.append("minimal")
    elif val2>=10 and val2<=18:
      predicted_cat.append("mild")
    elif val2>=19 and val2<=29:
      predicted_cat.append("moderate")
    else:
      predicted_cat.append("severe")

  
  print(correct_count)
  print("average % of correct responses: ", 100*np.mean(np.asarray(correct_count))/21.)

  abs_depression = np.absolute(np.array(depression_level_truth) - np.array(depression_level_predicted))
  print("absolute difference in deprresion level ranging (0,63) : ",abs_depression)
  print("average absolute difference: ", np.mean(abs_depression))

  # for i in y_preds:
  #   print(i)
  # print()  
  # for key in test_users:
  #   print(test_users[key]["RESPONSES"])


def predict(testing_users, model, tokenizer, max_length):
  text = []
  responses = []
  t = ""

  for key in testing_users:
    for dic in testing_users[key]["WRITINGS"]:
      t = dic["TITLE"] + dic["TEXT"]
      text.append(t)

  nparr = [[]]
  for i in range(len(text)):
    arr = np.asarray([[text[i]]])
    arr = np.append( arr, [])
    nparr.append(arr)

  df = pd.DataFrame(nparr, columns=['Text'])
  df = df.drop(df.index[0])

  print(df['Text'].head)

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
  
  print(test_x["input_ids"])

  preds = model.predict(test_x["input_ids"])

  count = 0
  y_preds = np.zeros([80,21])
  for i,key in enumerate(testing_users):
      for k in range(1,22):
        q = preds["q"+str(k)][count : count+len(testing_users[key]["WRITINGS"])]
        q = np.asarray(q)
        q = np.mean(q, axis = 0)

        y_preds[i][k-1] = np.argmax(q)
      count += len(testing_users[key]["WRITINGS"])


  y_preds = [[str(i) for i in item] for item in y_preds]

  for i in range(80):
    for j in range(21):
      val = y_preds[i][j]
      choice = int(val)
      if j==15 or j==17:
        idx = int((int(val)+1)/2)
        variant = 'a'
        if int(val) % 2 == 0:
          variant = 'b'
        if int(val) == 0:
          variant = ''
        choice = str(idx)+variant
      y_preds[i][j] = choice

  i = 0
  for key in testing_users:
    testing_users[key]["RESPONSES"] = y_preds[i]
    i += 1


  filepath = f"./ftp/task3/results/Depression Questionnaires_anon.txt"
  with open(filepath, "w+") as f:
      f.write('\n'.join([user + ' ' + ' '.join(testing_users[user]['RESPONSES']) for user in testing_users]))


training_users, testing_users = load_users()
df, df_test, Q, Q_test = split_train_test(training_users)
base_model, tokenizer, max_length = compile_model()
model = train(df, Q, base_model, tokenizer, max_length)
# test(df_test, Q_test, model, tokenizer, max_length, training_users)
predict(testing_users, model, tokenizer, max_length)
