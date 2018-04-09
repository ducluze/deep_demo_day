from flask import Flask, render_template, request
from get_text_from_url_demo import *
from feedforward_cbow_prints import *
import time
app = Flask(__name__)

PRETRAINED_VOCAB_PATH = "./glove.6B/glove.6B.100d.txt"


@app.route('/index')
@app.route('/', methods=['GET'])
def index():
    print request.form
    return render_template('template.html')



@app.route('/new_prediction', methods=['POST'])
def new_prediction():
    t0 = time.time()
    article = request.form['choosed_url']
    #article = get_text(url)
    all_x, lengths = cbow_glove_v2(article, dict1, size_of_vocab, "output_pickled")
    prediction = do_predict_v2(all_x, "model/", "model.ckpt")
    return render_template('prediction.html', time=str(time.time() - t0), pred=str(prediction[0][0][0]))


@app.route('/show_article', methods=['POST'])
def show_article():
    url = request.form['choosed_url']
    print url
    article = get_text(url)
    article.encode(encoding='UTF-8')
    return render_template('article.html', article=article, url=url)




if __name__ == '__main__':
    print("Starting to load cbow")
    t0 = time.time()
    dict1, size_of_vocab = vocab_dict_glove(PRETRAINED_VOCAB_PATH)
    print("It tooks --->", str(time.time() - t0))
    print("Starting to load web page")
    app.run(debug=False)

