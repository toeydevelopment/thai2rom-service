from flask import Flask, request
import re
from flask_cors import CORS
from pythainlp.transliterate import romanize

app = Flask(__name__)
CORS(app)

# roman = _THAI_TO_ROM.romanize
romanize("ทดสอบ","thai2rom")

@app.route("/thai2rom")
def thai2rom():
    thai = request.args.get('thai')
    x = re.search("[a-zA-Z]", thai)
    if x or thai == None or thai == "":
        return {
            "status": "failure",
            "message": "thai must be thai alphabets"
        }
    else:
        result = romanize(thai,"thai2rom")
        return {
            "status": "ok",
            "message": "converted successfully",
            "result": result
        }


if __name__ == "__main__":
    app.run("0.0.0.0", "5000")
