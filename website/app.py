from flask import Flask, request, jsonify, abort
from calm.fasta import RandomFasta
from calm import CaLM
from calm.utils import batched
import torch


def get_embeddings(
    url: str,
    fasta_file: str,
    start: int | None = None,
    end: int | None = None,
    timeout: int = 20,
) -> dict[str, torch.Tensor]:
    import requests
    import json

    with open(fasta_file, "rb") as fp:
        ret = json.loads(
            requests.post(
                url,
                files=dict(file=fp),
                data=dict(start=start, end=end),
                timeout=timeout,
            ).text
        )
    return {sid: torch.tensor(arr) for sid, arr in ret.items()}


model = CaLM()


app = Flask(__name__)

# run with: `FLASK_APP=website.app flask run`

# see also https://github.com/ShannonAI/service-streamer/tree/master


@app.route("/", methods=["POST"])
def convert():

    # curl -F file=@tf.fasta http://127.0.0.1:5000
    if not "file" in request.files:
        abort(404)
    fasta = RandomFasta(request.files["file"].read())

    start = int(request.values.get("start", 0))
    end = int(request.values.get("end", len(fasta)))

    def tolist(t: torch.Tensor) -> list:
        # tolist just stuffs up the rounding....
        # return torch.round(t, decimals=4).tolist()
        return t.tolist()

    fbatch = fasta[start:end]
    ret = {}
    for batch in batched(fbatch, 5):
        result = model.embed_sequences([f.seq for f in batch])

        ret.update({f.id: tolist(r) for f, r in zip(batch, result)})

    return jsonify(ret)
