import argparse
import os

from flask import Flask, make_response, render_template, send_from_directory

parser = argparse.ArgumentParser(description='Serve HTML files with navigation')
parser.add_argument('html_folder', help='Path to the folder containing HTML files')
args = parser.parse_args()

args.html_folder = os.path.abspath(args.html_folder)

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    # Check and list files
    try:
        files = sorted(
            [f for f in os.listdir(args.html_folder) if f.endswith('.html')],
            key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort numerically
    except Exception as e:
        return f'Error accessing directory: {e}', 500
    return render_template('index.html', files=files)


@app.route('/html/<path:filename>')
def html(filename):
    safe_path = os.path.join(args.html_folder, filename)
    if os.path.exists(safe_path) and os.path.isfile(safe_path):
        response = make_response(send_from_directory(args.html_folder, filename, as_attachment=False))
        response.headers['Cache-Control'] = 'public, max-age=300'  # Cache for 5 minute
        return response
    return f'File not found: {safe_path}', 404


if __name__ == '__main__':
    print(f'Serving files from: {args.html_folder}')
    app.run(debug=True)
