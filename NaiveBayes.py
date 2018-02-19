import sys
import getopt
import os
import math
import operator
import collections
import string

class NaiveBayes:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
    self.BEST_MODEL = False
    self.stopList = set(self.readFile('./deps/english.stop'))
    self.positiveList = set(self.readFile('./deps/english.positive'))
    self.negativeList = set(self.readFile('./deps/english.negative'))
    self.negationList = set(self.readFile('./deps/negation'))
    self.numFolds = 10
    self.klassWordCount = {'pos': collections.defaultdict(lambda: 0), 
                           'neg': collections.defaultdict(lambda:0)}
    self.klassVocabCount = collections.defaultdict(lambda:0)
    self.klassDocumentCount = collections.defaultdict(lambda:0)
    self.totalDocumentCount = 0
    self.allWords = set()
    # TODO: Add any data structure initialization code here


  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  # If the BEST_MODEL flag is true, include your new features and/or heuristics that
  # you believe would be best performing on train and test sets. 
  #
  # If any one of the FILTER_STOP_WORDS, BOOLEAN_NB and BEST_MODEL flags is on, the 
  # other two are meant to be off. That said, if you want to include stopword removal
  # or binarization in your best model, write the code accordingl
  # 
  # Hint: Use filterStopWords(words) defined below
  def classify(self, words):
    # STOP WORD FILTERING
    if self.FILTER_STOP_WORDS:
      words = self.filterStopWords(words)

    # a list of probabilities corresponding to each klass
    probs = {}
    for klass in self.klassWordCount.keys():
      # compute prior prob 
      probs[klass] = math.log(self.klassDocumentCount[klass]) - math.log(self.totalDocumentCount)
      negation = False # negation flag
      wordsDone = set() # binarization set

      for word in words:

        ## NEGATION HANDLING FOR BEST MODEL
        negWord = word
        if self.BEST_MODEL:
          if negation is True:
            negWord = 'NOT_' + word
          if word in self.negationList:
            negation = True
          elif word in string.punctuation:
            negation = False
        
        # IF WORD NOT SEEN IN VOCAB, SKIP
        if negWord not in self.allWords: 
          continue
        
        # BINARIZATION - if we have already computed this word, then skip
        if self.BOOLEAN_NB or self.BEST_MODEL:
          if negWord in wordsDone:
            continue

        # compute probability, add one smoothing
        probs[klass] += math.log(self.klassWordCount[klass][negWord] + 1)
        probs[klass] -= math.log(self.klassVocabCount[klass] + len(self.allWords))

        # BINARIZATION - keep track of words already seen
        if self.BOOLEAN_NB or self.BEST_MODEL:
          wordsDone.add(negWord)

    # Calculate the most likely klass
    mostLikelyKlass = ''
    highestProb = -float('inf')
    for klass in probs:
      if probs[klass] > highestProb:
        highestProb = probs[klass]
        mostLikelyKlass = klass

    return mostLikelyKlass


  def addExample(self, klass, words):
    # STOP WORD FILTERING
    if self.FILTER_STOP_WORDS:
      words = self.filterStopWords(words)

    negation = False # negation flag
    wordsDone = set() # binarization set
    for w in words:
      ## NEGATION HANDLING FOR BEST MODEL
      if self.BEST_MODEL:
        # if we are in a negation state, pre-append NOT token
        if negation is True:
          negW = 'NOT_' + w

          # KEEP binarization - do not double count negW if we already have it
          if negW not in wordsDone:
            self.klassWordCount[klass][negW] += 1
            self.klassVocabCount[klass] += 1
            self.allWords.add(negW)
            wordsDone.add(negW)

        # update negation flag as necessary
        if w in self.negationList:
          negation = True
        elif w in string.punctuation:
          negation = False

      ## BINARIZATION - skip word if we have already seen it
      if self.BOOLEAN_NB or self.BEST_MODEL:
        if w in wordsDone:
          continue

      self.klassWordCount[klass][w] += 1
      self.klassVocabCount[klass] += 1
      self.allWords.add(w)
      wordsDone.add(w)


      ## WEIGHTING ADJUSTMENT - in total, triple count words in pos/neg lists
      if self.BEST_MODEL:
        if klass == 'pos':
          if w in self.positiveList:
            self.klassWordCount[klass][w] += 2
            self.klassVocabCount[klass] += 2
        elif klass == 'neg':
          if w in self.negativeList:
            self.klassWordCount[klass][w] += 2  
            self.klassVocabCount[klass] += 2    

    self.klassDocumentCount[klass] += 1
    self.totalDocumentCount += 1
      

  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      self.addExample(example.klass, words)


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      yield split

  def test(self, split):
    """Returns a list of labels for split.test."""
    labels = []
    for example in split.test:
      words = example.words
      guess = self.classify(words)
      labels.append(guess)
    return labels
  
  def buildSplits(self, args):
    """Builds the splits for training/testing"""
    trainData = [] 
    testData = []
    splits = []
    trainDir = args[0]
    if len(args) == 1: 
      print '[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir)

      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fold in range(0, self.numFolds):
        split = self.TrainSplit()
        for fileName in posTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
          example.klass = 'pos'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        for fileName in negTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
          example.klass = 'neg'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        splits.append(split)
    elif len(args) == 2:
      split = self.TrainSplit()
      testDir = args[1]
      print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        split.train.append(example)

      posTestFileNames = os.listdir('%s/pos/' % testDir)
      negTestFileNames = os.listdir('%s/neg/' % testDir)
      for fileName in posTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (testDir, fileName)) 
        example.klass = 'pos'
        split.test.append(example)
      for fileName in negTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (testDir, fileName)) 
        example.klass = 'neg'
        split.test.append(example)
      splits.append(split)
    return splits
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL):
  nb = NaiveBayes()
  splits = nb.buildSplits(args)
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    classifier.BEST_MODEL = BEST_MODEL
    accuracy = 0.0
    for example in split.train:
      words = example.words
      classifier.addExample(example.klass, words)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
    
    
def classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, trainDir, testFilePath):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  classifier.BOOLEAN_NB = BOOLEAN_NB
  classifier.BEST_MODEL = BEST_MODEL
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testFile = classifier.readFile(testFilePath)
  print classifier.classify(testFile)
    
def main():
  FILTER_STOP_WORDS = False
  BOOLEAN_NB = False
  BEST_MODEL = False
  (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  elif ('-b','') in options:
    BOOLEAN_NB = True
  elif ('-m','') in options:
    BEST_MODEL = True
  
  if len(args) == 2 and os.path.isfile(args[1]):
    classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, args[0], args[1])
  else:
    test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL)

if __name__ == "__main__":
    main()
