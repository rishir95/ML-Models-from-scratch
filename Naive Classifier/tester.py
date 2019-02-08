
# coding: utf-8

import argparse
import os
from naiveclassifier import NaiveBayesClassifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 1'
                                                 'to Implement Gaussian Naive Bayes in Python.')
    parser.add_argument("-t","--Train", metavar="Question", help="path of the training csv file")
    parser.add_argument("-s","--Test", metavar="Responses", help="path of the test csv file")
    parser.add_argument("-n","--Txt", metavar="Document", help="path of the new txt file to be created with the output")

    args = parser.parse_args()


    NaiveBayesClassifier(args.Train,args.Test,args.Txt)