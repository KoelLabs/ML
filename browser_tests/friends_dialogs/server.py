from flask import send_from_directory, Flask, send_file

app = Flask(__name__)


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/clips/<path:path>")
def send_report(path):
    return send_from_directory("clips", path)


# flask --app server run
