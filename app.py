from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    input_text = request.form['input_text']
    # Run Docker container to execute summarize_text.py
    docker_command = f'docker run --rm text-summarizer python summarize_text.py "{input_text}"'
    result = subprocess.check_output(docker_command, shell=True).decode('utf-8').strip()
    return jsonify({'summary': result})

if __name__ == '__main__':
    app.run(debug=True)
