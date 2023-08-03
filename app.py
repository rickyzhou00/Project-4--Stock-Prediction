from flask import Flask, redirect, render_template, request, url_for
from model import perform_stock_prediction

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_stock():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        return redirect(url_for('show_result', stock_symbol=stock_symbol))
    return render_template('index.html', prediction=None, stock_symbol=None)

@app.route('/result/<stock_symbol>')
def show_result(stock_symbol):
    prediction, error = perform_stock_prediction(stock_symbol)
    if error:
        return render_template('error.html', error_message=error)
    return render_template('result.html', prediction=prediction[0][0], stock_symbol=stock_symbol.upper())


if __name__ == '__main__':
    app.run(debug=True)
