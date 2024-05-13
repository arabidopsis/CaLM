from flask import Flask, request, jsonify, abort, Response
from calm.fasta import RandomFasta
from calm import CaLM
from calm.utils import batched
import torch


def get_embeddings(
    fasta_file: str,
    url: str = 'http://127.0.0.1:5000/fasta',
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

def get_embeddings_json(
    data: dict[str,str], # id, sequence
    url: str = 'http://127.0.0.1:5000/json',
    timeout: int = 20,
) -> dict[str, torch.Tensor]:
    import requests
    import json


    ret = json.loads(
        requests.post(
            url,
            json=data,
            timeout=timeout,
        ).text
    )
    return {sid: torch.tensor(arr) for sid, arr in ret.items()}

model = CaLM()


app = Flask(__name__)

# run with: `FLASK_APP=website.app flask run`

# see also https://github.com/ShannonAI/service-streamer/tree/master


@app.route("/fasta", methods=["POST"])
def convert():

    # curl -F file=@tf.fasta http://127.0.0.1:5000/fasta
    if not "file" in request.files:
        abort(404)
    fasta = RandomFasta(request.files["file"])  # .read())

    start = int(request.values.get("start", 0))
    end = int(request.values.get("end", len(fasta))
    if end - start > 100:
        abort(413)  # Content Too Large

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


@app.route("/json", methods=["POST"])
def from_json() -> Response:
    # requests.post(url, json=data)
    data: dict[str, str] = request.json # type: ignore

    def tolist(t: torch.Tensor) -> list:
        # tolist just stuffs up the rounding....
        # return torch.round(t, decimals=4).tolist()
        return t.tolist()

    ret = {}
    for batch in batched(data.items(), 5):
        result = model.embed_sequences([seq for _id, seq in batch])

        ret.update({id: tolist(r) for (id, _), r in zip(batch, result)})

    return jsonify(ret)
