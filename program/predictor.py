# This is the file that implements a flask server to do inferences.


import os
import json
import flask
import numpy
import rl2048player as rl


prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')


class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            load_model_path = os.path.join(model_path, 'agent.pkl')
            cls.model = rl.agents.load_agent(load_model_path)
        return cls.model

    @classmethod
    def predict(cls, state):
        """For the input state, return confidence for each action
        Args:
            state: Game state to predict next action"""
        confidence = cls.get_model().getActionConfidence(state)
        response = {'Left':confidence[rl.game.ACTION_LEFT], 'Up':confidence[rl.game.ACTION_UP],
                    'Right':confidence[rl.game.ACTION_RIGHT], 'Down':confidence[rl.game.ACTION_DOWN]}
        return json.dumps(response)


app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    if flask.request.is_json:
        data = numpy.array(flask.request.get_json())
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')
    prediction = ScoringService.predict(data)
    return flask.Response(response=prediction, status=200, mimetype='application/json')
