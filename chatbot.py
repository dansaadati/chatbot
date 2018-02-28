#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2018
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)
######################################################################
import csv
import math
import numpy as np
import re
from movielens import ratings
from random import randint
from deps.PorterStemmer import PorterStemmer

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      self.name = 'moviebot'
      self.is_turbo = is_turbo
      self.alphanum = re.compile('[^a-zA-Z0-9]')
      self.p = PorterStemmer()
      self.read_raw_data()
      self.read_data()



    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = 'How can I help you?'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = 'Have a nice day!'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return goodbye_message


    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################

    def read_raw_data(self):
        title_pattern = re.compile('(.*) \d+\.txt')

        # make sure we're only getting the files we actually want
        filename = 'data/sentiment.txt'
        contents = []
        f = open(filename)
        of = open('data/stemmed.txt', 'w')
        
        for line in f:
            # make sure everything is lower case
            line = line.lower()

            pair = line.split(',')

            # stem words
            line = self.p.stem(pair[0]) + ',' + pair[1][:-2]

            # add to the document's conents
            contents.extend(line)
            if len(line) > 0:
                of.write("".join(line))
                of.write('\n')
        f.close()
        of.close()
        pass

    def evaluateSentiment(self, line):
      posScore = 0
      negScore = 0
      lam = 1.0

      # Remove the title from the sentence
      removed = re.sub(r'[\"](.*?)[\"]', '', line)
      negation = ['no', 'not', 'none', 'never', 'hardly', 'scarcely',
        'n\'t', 'didn\'t',
       'barely', 'doesn\'t', 'isn\'t', 'wasn\'t', 'shouldn\'t',
        'couldn\'t', 'won\'t', 'can\'t', 'don\'t']
      punctuation = set(',.?!;()')

      # Lemmetize the sentence
      line = self.lemonizeLine(removed)

      # Set negation state and calculate score
      negating = False
      for word in line.split(' '):
        if word in negation:
          negating = not negating

        if word in self.sentiment:
          if(self.sentiment[word] == 'pos'):
            if negating:
              negScore = negScore + 1
            else:
              posScore = posScore + 1
          if(self.sentiment[word] == 'neg'):
            if negating:
              posScore = posScore + 1
            else:
              negScore = negScore + 1

        # if we see punctuation in the word, reset negation flag
        if set(word) & punctuation:
          negating = False
      
      if(negScore == 0):
        posNegRatio = lam
      else:
        posNegRatio = float(posScore) / float(negScore)
      
      if(posNegRatio >= lam):
        return 'pos'
      else:
        return 'neg'

    def grabAndValidateMovieTitle(self, line):
      titleReg = re.compile('[\"](.*?)[\"]')
      results = re.findall(titleReg, line)

      if(len(results) > 1):
        return "" #TODO : HANDLE ERROR
      if(len(results) == 0):
        return ""

      for pair in self.titles:
          if results[0] in pair[0]:
            return results[0]
      else:
        return "couldn't find" #TODO : Handle ERROR

    def lemonizeLine(self, input):
      processInput = []
      for word in input.split(' '):
        processInput.append(self.p.stem(word))


      return ' '.join(processInput)


    def process(self, input):
      """Takes the input string from the REPL and call delegated functions
      that
        1) extract the relevant information and
        2) transform the information into a response to the user
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method, possibly#
      # calling other functions. Although modular code is not graded, it is       #
      # highly recommended                                                        #
      #############################################################################
      print(self.ratings)

      if self.is_turbo == True:
        response = 'processed %s in creative mode!!' % input
      else:
        response = 'processed %s in starter mode' % input



      title = self.grabAndValidateMovieTitle(input)
      sentiment = self.evaluateSentiment(input)
      

      if title != "":
        response = response + " Also, the movie title is " + title

      return response


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    # from piazza - "If you're going to use the Porter Stemmer,
    # we recommend stemming the words in the sentiment lexicon
    # as well as the words in the input sentence."

    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)


    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""

      pass


    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # TODO: Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure

      pass


    def recommend(self, u):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # TODO: Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot

      pass


    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, input):
      """Returns debug information as a string for the input string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self): #INITIALIZATION

      return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """


    #############################################################################
    # Auxiliary methods for the chatbot.                                        #
    #                                                                           #
    # DO NOT CHANGE THE CODE BELOW!                                             #
    #                                                                           #
    #############################################################################

    def bot_name(self):
      return self.name


if __name__ == '__main__':
    Chatbot()
