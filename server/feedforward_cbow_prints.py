# -*- coding: utf-8 -*-

import sys
import pickle
import os
import time
import datetime
import getopt
import numpy as np
from collections import OrderedDict
import tensorflow as tf
import math

from word2int_cbow_for_testing import *
#from get_text_from_url import *

class Config:
    max_length = 600  # text size
    embed_size = 100
    batch_size = 256
    n_classes = 1
    hidden_size = 100
    n_epochs = 100
    lr = 0.0001
    dropout = 0.0

    def __init__(self):
        pass

6
class FeedforwardModel():
    '''
    This is the Feedforward Model class. It takes in a 100 dimensional BOW vector per document example and
     makes a binary prediction (high or low quality news)
    '''

    def __init__(self, data, config, labels):
        '''
        :param data: tuple of data (train, dev, test) with corresponding lengths of documents
        :param config: configuration for our current model initialization
        :param labels: tuple of labels (train, dev, test) corresponding with data
        Initialize class variables
        '''
        self.documents  = np.array(data[0][0]).T       #### TRAIN DATA (embed size x m examples)
        assert self.documents.shape[0] == config.embed_size, "Embedding size not matching the configuration indicated"
        self.labels = np.array(labels[0]).reshape(1,len(labels[0]))    #### TRAIN LABELS (1 x m examples)
        assert self.labels.shape[1] == self.documents.shape[1], "Different number of labels and documents in training"
        self.dev_examples  = np.array(data[1][0]).T
        self.dev_labels = np.array(labels[1]).reshape(1,len(labels[1]))
        self.test_examples  = np.array(data[2][0]).T
        self.test_labels = np.array(labels[2]).reshape(1,len(labels[2]))
        self.new_test_examples  = np.array(data[3][0]).T
        self.new_test_labels = np.array(labels[3]).reshape(1,len(labels[3]))
        
        self.config = config
        self.num_batch = int(self.documents.shape[1]/config.batch_size)     #### number of train batches (take floor)
        print("number of batches: " + str(self.num_batch))
        print("each batch contains: " + str(self.config.batch_size))
        print("size of train set: " + str((self.documents.shape[1])))
        print("size of dev set: " + str((self.dev_examples).shape[1]))
        print("size of test set: " + str((self.test_examples).shape[1]))
        print("size of new test set: " + str((self.new_test_examples).shape[1]))

        ### initialize placeholders ###
        self.input_placeholder = tf.placeholder(shape=[self.config.embed_size, None], dtype=tf.float64, name="input_placeholder")  #(600,m)
        self.labels_placeholder = tf.placeholder(shape=[1, None], dtype=tf.float64, name="labels_placeholder")   # (1,m)

        ### initialize weights ###
        xavier = tf.contrib.layers.xavier_initializer()
        self.W_1 = tf.get_variable("W_1", initializer = xavier, shape =[self.config.hidden_size, self.config.embed_size], dtype=tf.float64)
        self.b_1 = tf.get_variable("b_1", initializer= tf.zeros_initializer, shape =[self.config.hidden_size, 1], dtype=tf.float64)
        self.W_2 = tf.get_variable("W_2", initializer = xavier, shape =[self.config.n_classes, self.config.hidden_size], dtype=tf.float64)
        self.b_2 = tf.get_variable("b_2", initializer=tf.zeros_initializer, shape =[self.config.n_classes, 1], dtype = tf.float64)

        ### forward prop ###
        # two-layer feedforward model (relu activation)
        # A1 = relu(W_1*x + b_1)
        # Z2 = W_2 x A1 + b_2

        self.A1 = tf.nn.relu(tf.matmul(self.W_1,self.input_placeholder) + self.b_1)

        self.Z2 = tf.matmul(self.W_2,self.A1) + self.b_2

        ### sigmoid cross entropy loss averaged over m examples ###
        self.labels1 = tf.transpose(self.labels_placeholder)
        self.logits = tf.transpose(self.Z2, name='logits')

        self.sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels1,logits=self.logits)
        self.loss = tf.reduce_sum(self.sigmoid_cross_entropy, name="loss")

        #### optimizer (Adam) ####
        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)

        #### log acc/loss lists ###
        self.print_train_loss_list = []
        self.print_dev_acc_list = []
        self.print_test_loss_list = []
        self.print_dev_loss_list = []
        self.print_train_acc_list = []
        self.print_new_test_loss_list = []
        self.print_new_test_acc_list = []

    def print_helper(self, write_file):
        '''
        :param write_file: path to open file
        Files Details of the model to given write_file
        '''

        write_file.write("max_length: " + str(self.config.max_length) + "\n")
        write_file.write("embed_size:" + str(self.config.embed_size) + "\n")
        write_file.write("classes: " + str(self.config.n_classes) + "\n")
        write_file.write("hidden_size: " + str(self.config.hidden_size) + "\n")
        write_file.write("n_epochs: " + str(self.config.n_epochs) + "\n")
        write_file.write("learn_rate: "+ str(self.config.lr) + "\n")
        write_file.write("batch_size: " + str(self.config.batch_size)+ "\n")
        write_file.write("layers: " + str(1) + "\n")
        write_file.write("num_buckets: " + str(self.num_batch) + "\n")
        write_file.write("batch_size: " + str(self.config.batch_size) + "\n")
        write_file.write("size_test_set: " + str(len(self.test_examples))  + "\n")
        write_file.write("size_dev_set: " + str(len(self.dev_examples)) + "\n")
        write_file.write("size_train_set: " + str(len(self.documents)) + "\n")
        write_file.flush()

    def random_mini_batches(self, X, Y, mini_batch_size):
        '''
        Creates a list of random minibatches from (X, Y)
        :param X: input data (n_x , m)
        :param Y: "true" labels (1, m)
        :param mini_batch_size: size of mini-batches (integer)
        :return: mini_batches: list of synchronous (mini_batches_X, mini_batches_Y)
        '''

        m = X.shape[1]                  # number of training examples
        mini_batches = []
        # np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        #shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
        shuffled_Y = Y[:, permutation].reshape((1,m))
        
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, self.num_batch):
            mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def sigmoid(self, z):
        s = 1.0 / (1.0 + np.exp(-1.0 * z))
        return s

    def train(self, file_print):

        '''
        Trains the model with the data/labels/config fed during initialization

        :param file_print: path to log file
        '''
        self.print_helper(file_print)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.config.n_epochs):   #iterate through epochs
                # average_loss_over_epoch = 0
                print ("-----epoch no." + str(i) + "------")
                epoch_cost_batches = 0
                epoch_preds = []
                epoch_labels = []
                num_minibatches = int(self.documents.shape[1] / self.config.batch_size)
                minibatches = self.random_mini_batches(X = self.documents,Y = self.labels, mini_batch_size = self.config.batch_size)
                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _ , minibatch_cost = sess.run([self.train_op, self.loss], feed_dict={self.input_placeholder: minibatch_X, self.labels_placeholder: minibatch_Y})
                
                    epoch_cost_batches += minibatch_cost / num_minibatches
                    self.print_train_loss_list.append(minibatch_cost)

                print("epoch cost is %s" % str(epoch_cost_batches))
                Z2_preds = sess.run(self.Z2, feed_dict={self.input_placeholder: self.documents})
                train_preds = Z2_preds > 0
                assert(len(train_preds[0]) == len(self.labels[0]))
                train_epoch_accuracy = sum([(train_preds[0][_] == self.labels[0][_]) for _ in range(len(self.labels[0]))])/float(len(self.labels[0]))
                print ("train accuracy in current epoch %s" % str(train_epoch_accuracy))
                self.print_train_acc_list.append(train_epoch_accuracy)


                #### calculate dev accuracy ###
                Z2_preds_new, loss_new = sess.run([self.Z2, self.loss], feed_dict={self.input_placeholder: self.new_test_examples, self.labels_placeholder: self.new_test_labels})
                new_preds = Z2_preds_new > 0
                print("New preds -->",new_preds)
                print('New_test_labels -->',self.new_test_labels)
                print("LEN New preds -->",len(new_preds[0]))
                print('LEN New_test_labels -->',len(self.new_test_labels))
                assert(len(new_preds[0]) == len(self.new_test_labels[0]))
                self.print_new_test_loss_list.append(loss_new)
                new_accuracy = sum([(new_preds[0][_] == self.new_test_labels[0][_]) for _ in range(len(self.new_test_labels[0]))])/float(len(self.new_test_labels[0]))
                self.print_new_test_acc_list.append(new_accuracy)
                print ("new_test accuracy in current epoch %s" % str(new_accuracy))

                Z2_preds_dev, loss_dev = sess.run([self.Z2, self.loss], feed_dict={self.input_placeholder: self.dev_examples, self.labels_placeholder: self.dev_labels})
                dev_preds = Z2_preds_dev > 0
                assert(len(dev_preds[0]) == len(self.dev_labels[0]))
                self.print_dev_loss_list.append(loss_dev)
                dev_accuracy = sum([(dev_preds[0][_] == self.dev_labels[0][_]) for _ in range(len(self.dev_labels[0]))])/float(len(self.dev_labels[0]))
                self.print_dev_acc_list.append(dev_accuracy)
                print ("dev accuracy in current epoch %s" % str(dev_accuracy))
                save_path = saver.save(sess, "C:/Users/victor/Documents/GitHub/deepnews_final/tmp/model.ckpt")
                print("Model saved for epoch ->{} in path {}".format(i, save_path))
                
                #### print results ####
            file_print.write("train_loss_per_batch: "+ str(self.print_train_loss_list)+"\n")
            file_print.write("train_acc_per_epoch: " + str(self.print_train_acc_list) + "\n")
            file_print.write("dev_loss: "+ str(self.print_dev_loss_list)+ "\n")
            file_print.write("dev_acc: " + str(self.print_dev_acc_list) + "\n")
            file_print.write("new_test_loss: "+ str(self.print_new_test_loss_list)+ "\n")
            file_print.write("new_test_acc: " + str(self.print_new_test_acc_list) + "\n")



def do_train(input_path):

    try:
        train_path = open(input_path+"train", "rb")
        dev_path = open(input_path+"dev", "rb")
        test_path = open(input_path+"test", "rb")
        new_test_path = open(input_path+"new_test", "rb")

    except IOError as e:
        print(e)
        print ("Could not open file from " + input_path)
        sys.exit()

    dev = pickle.load(dev_path)              #list of list (len3); 1st is list of list of ints; 2nd length of docs; 3rd Glove dictionary
    test = pickle.load(test_path)
    train = pickle.load(train_path)
    #print("Train's shape ",train.shape())
    #print("Train 1 --> ",train[1])
    new_test = pickle.load(new_test_path)
    y_labels = []
    y_labels.append(list(map(int,train[2])))
    y_labels.append(list(map(int,dev[2])))
    y_labels.append(list(map(int,test[2])))
    y_labels.append(list(map(int,new_test[2])))
    dev = dev[0:2] # dev = [ [list of lists having tokens of documents] , [lenghts of docs]]
    test = test[0:2]
    train = train[0:2]
    new_test = new_test[0:2]
    assert len(train[0]) == len(y_labels[0])
    assert len(dev[0]) == len(y_labels[1])

    print_files = "./print_files"
    if not os.path.isdir(print_files):
        os.makedirs(print_files)

    output_dir = print_files + "/{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    file_print = open(output_dir + "/run_result.txt", "w")
    all_data = (train, dev, test, new_test)

    start = time.time()

    our_model = FeedforwardModel(data=all_data, config=Config(),
                                 labels=y_labels)
    elapsed = (time.time() - start)
    print ("BUILDING THE MODEL TOOK " + str(elapsed) + "SECS")
    our_model.train(file_print)



def do_predict_v2(data, model_path, name_model):
    test_example  = np.array(data).T
    print("Data shape  -->",len(data))

    print("Loading model from path -->", model_path)
    PATH = os.path.join(model_path, name_model)
    print("Luaunching Saver")
    saver = tf.train.import_meta_graph(PATH + ".meta")
    print("saver launched")
    graph = tf.get_default_graph()
    print("Graph obtained ")

    with tf.Session() as sess:
        saver.restore(sess, PATH)
        print("Session restored")
        ops = [n.name for n in graph.get_operations()]

        global_step_tensor = graph.get_tensor_by_name('logits'+':0')
        print("Global step ready .. running predictions")
        predictions = sess.run([global_step_tensor], feed_dict={"input_placeholder:0": test_example})
    print("Predictions ready, writting them..")
    print("Predictions --> ", predictions)
    list_neg = [pred[0] for pred in predictions[0] if pred[0] < 0]
    list_pos = [pred[0] for pred in predictions[0] if pred[0] > 0]
    if len(list_neg) > 0:
        print("Nombre de neg -->", len(list_neg))
        print('Mean neg -->', np.mean(list_neg))
    if len(list_pos) > 0:
        print("Nombre de pos -->", len(list_pos))
        print('Mean pos -->', np.mean(list_pos))
    return predictions

def do_predict(input_path,model_path, name_model):
    try:
        data_path = open(input_path+"data_test", "rb")
    except IOError:
        print ("Could not open file from " + input_path)
        sys.exit()

    data = pickle.load(data_path)
    data = data[0:2]

    test_examples  = np.array(data[0]).T
    print("Data shape  -->",len(data))
    print("Loading model from path -->",model_path)
    PATH = os.path.join(model_path,name_model)
    print("Luaunching Saver")
    saver = tf.train.import_meta_graph(PATH+".meta")
    print("saver launched")
    graph = tf.get_default_graph()
    print("Graph obtained ")
    with tf.Session() as sess:
        saver.restore(sess, PATH)
        print("Session restored")
        ops = [n.name for n in graph.get_operations()]

        global_step_tensor = graph.get_tensor_by_name('logits'+':0')
        print("Global step ready .. running predictions")
        predictions = sess.run([global_step_tensor], feed_dict={"input_placeholder:0": test_examples})
    print("Predictions ready, writting them..")
    print("Predictions --> ",predictions)
    list_neg = [pred[0] for pred in predictions[0] if pred[0] < 0]
    list_pos = [pred[0] for pred in predictions[0] if pred[0] > 0]
    if len(list_neg) > 0:
        print("Nombre de neg -->", len(list_neg))
        print('Mean neg -->', np.mean(list_neg))
    if len(list_pos) > 0:
        print("Nombre de pos -->", len(list_pos))
        print('Mean pos -->', np.mean(list_pos))
    file = codecs.open("results_predictions.txt","w", encoding="utf8")
    for pred in predictions[0]:
        file.write(str(pred[0])+"\n")
    try:
        os.remove("output_pickled/data_test")
    except:
        pass

    return predictions



def main(argv):
    global input_path
    input_path = ""
    try:
        opts, args = getopt.getopt(argv,"u:p",["url","mode"])
    except getopt.GetoptError as e:
        print(e)
        print ('test.py -i')
        sys.exit(2)
    print(opts)
    mode = "predict"
    for opt, arg in opts:
        if opt in ("-u", "--url"):
            url_article = arg
            print(opt)
        if opt in ("-m", "--mode"):
            mode = arg
            print(opt)

    if url_article == "":
        print ("Must enter the url of the article  ")
        sys.exit()
    if mode == "":
        print("Must enter a mode for the model ")
        sys.exit()

    # if mode == "train":
    #     do_train(input_path)
    # if mode == "predict":
    #     PRETRAINED_VOCAB_PATH = "../../model-data-deepnews/glove.6B/glove.6B.100d.txt"
    #     input_path = get_text(url_article) # enregistre un fichier texte au path retourné
    #     print("Starting cbow calculation..")
    #     cbow_glove(input_path, PRETRAINED_VOCAB_PATH, "output_pickled")
    #
    #     do_predict("output_pickled/","model/","model.ckpt")

if __name__ == "__main__":
   main(sys.argv[1:])
