import sys
sys.path.append("..")
import nltk, unicodedata
import regex, sys
import pickle
import pandas as pd


def filt(x):
    return x.label()=='NP'


def ngroupChunkerCount(essaySet):
    """
    Count the ngroup(to distinguish it from single NN as a ngroup, we filter by len(ngroup) >=3 ) in given essay set.
    :param essaySet: a list of essays.
    :return: a list of numbers, representing the number of long ngroup used in each essay.
    """
    grammar = r"""
              NP:  {<DT>*(<NN.*>|<JJ.*>)*<NN.*>}    # Chunk sequences of DT, JJ, NN
              PP: {<IN><NP>} # Chunk prepositions followed by NP
              VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
              """
    cp = nltk.RegexpParser(grammar)

    ngroupCount = []

    for essay in essaySet:
        try:
            essay = essay.lower()
            sentences = filter(None, regex.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', essay))
        
            count = 0
            for s in sentences:
                s = s.decode('utf-8', 'ignore')
                s = unicodedata.normalize('NFKD', s).encode('ascii','ignore')
                tree = cp.parse(filter(None, nltk.pos_tag(s.split())))
                for subtree in tree.subtrees(filter =  filt):
                    if len(subtree) >= 3:
                        count += 1
        
            ngroupCount.append(count)
            
        except Exception:
            print "Cannot write word_list into file due to the exception:", sys.exc_info()[0]
            
    return ngroupCount


def main():    
    train = pd.read_csv('Data/training_set_rel3.tsv', sep = '\t', index_col = 'essay_id')
    validate = pd.read_csv('Data/valid_set.tsv', sep = '\t', index_col = 'essay_id')

    # Chunking ngroup in both training and test data set can take hours. So you could try the following to test the correctness of the code.
    #ngroupCountTrain = ngroupChunkerCount(train['essay'][:20])
    #ngroupCountValidate = ngroupChunkerCount(validate['essay'][:20])
    ngroupCountTrain = ngroupChunkerCount(train['essay'])
    ngroupChunkerCount(validate['essay'])
    
    try:
        print 'Save ngroupCount for training data in ngroupCountTrain.pkl'
        ngroupCountTrain_pkl = open('FeatureData/ngroupCountTrain.pkl', 'wb')
        pickle.dump(ngroupCountTrain, ngroupCountTrain_pkl)
        ngroupCountTrain_pkl.close()
        
        print 'Save ngroupCount for validate data in ngroupCountValidate.pkl'
        ngroupCountValidate_pkl = open('FeatureData/ngroupCountValidate.pkl', 'wb')
        pickle.dump(ngroupCountValidate, ngroupCountValidate_pkl)
        ngroupCountValidate_pkl.close()

    except Exception:
        print "Cannot write word_list into file due to the exception:", sys.exc_info()[0]
        raise 
		
if __name__ == '__main__':
    main()
