import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
from oauth2client import service_account
# from google.oauth2 import service_account
# https://cloud.google.com/ai-platform/prediction/docs/regional-endpoints#python
# service = googleapiclient.discovery.build('ml','v1')

# import socket    
# socket.setdefaulttimeout(200)

import httplib2

# credentials = service_account.Credentials.from_service_account_file(
#     'reddit-bdi-313122-5198c6eead0b.json', scopes=["https://www.googleapis.com/auth/cloud-platform"]
# )

credentials = service_account.ServiceAccountCredentials.from_json_keyfile_name('reddit-bdi-313122-5198c6eead0b.json', scopes=["https://www.googleapis.com/auth/cloud-platform"])

http = httplib2.Http(timeout=300)
http = credentials.authorize(http)

endpoint = 'https://us-central1-ml.googleapis.com'

client_options = ClientOptions(api_endpoint=endpoint)
# service = googleapiclient.discovery.build('ml', 'v1', client_options=client_options,credentials=credentials)
service = googleapiclient.discovery.build('ml', 'v1', client_options=client_options,http=http)

def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the AI Platform Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


# instances = [[101,1045,2074,2066,1037,8744,14960,1997,8370,1011,5451,5530,1006,7960,3485,1007,1012,2059,1045,2064,4339,1999,2151,6179,5906,2066,3289,1013,4638,7201,1010,7058,2933,1010,1998,3964,1010,2004,1045,2342,2068,1012,2296,2051,1045,1005,2310,2018,1037,24555,2007,3653,1011,6267,2838,1010,1045,2203,2039,2025,2478,2070,1013,2035,1997,2068,1998,1996,2686,2003,13842,1012,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

# response = predict_json("reddit-bdi-313122","tfmodel1",instances,"version2")



