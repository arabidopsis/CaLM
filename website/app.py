from flask import Flask, request, jsonify, abort
from calm.fasta import RandomFasta
from calm import CaLM
import torch

model = CaLM()

app = Flask(__name__)

# run with: `FLASK_APP=website.app flask run`

@app.route("/", methods=["POST"])
def convert():

    # curl -F file=@tf.fasta http://127.0.0.1:5000
    # json.loads(requests.post(" http://127.0.0.1:5000", files=dict(file=open('tf.fasta','rb'))).text)
    if not "file" in request.files:
        abort(404)
    fasta = RandomFasta(request.files["file"].read())

    def tolist(t: torch.Tensor) -> list:
        return torch.round(t, decimals=4).tolist()

    ret = {f.id: tolist(model.embed_sequence(f.seq)) for f in fasta}

    return jsonify(ret)
