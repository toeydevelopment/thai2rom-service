from flask import Flask, request
import re
from flask_cors import CORS
import thai2rom.model as m

app = Flask(__name__)
CORS(app)


@app.route("/thai2rom")
def thai2rom():
    _THAI_TO_ROM = m.ThaiTransliterator()
    thai = request.args.get('thai')
    x = re.search("[a-zA-Z]", thai)
    if x:
        return {
            "status": "failure",
            "message": "thai must be thai alphabets"
        }
    else:
        result = _THAI_TO_ROM.romanize(thai)
        return {
            "status": "ok",
            "message": "converted successfully",
            "result": result
        }
if __name__ == "__main__":
    app.run("0.0.0.0","5000")
