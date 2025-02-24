from flask import Flask, render_template, request, jsonify
import os

from apps import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)