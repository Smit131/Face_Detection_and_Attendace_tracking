#!/usr/bin/env python
# coding: utf-8

# In[21]:

import numpy as np
from flask import Flask, request, jsonify, render_template
import model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/Start', methods=("POST", "GET"))
def Start():
    df = model.model()
    return render_template('index.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)


if __name__ == "__main__":
    app.run(debug=True)

