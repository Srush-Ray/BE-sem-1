import nltk
nltk.download('state_union')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer,sent_tokenize,word_tokenize

train_text = state_union.raw("2005-GWBush.txt")
sample_text = "A lion was once sleeping in the jungle when a mouse started running up and down his body just for fun."

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"Chunk: {<RB.?><VB.?><NNP>+<NN>?}"
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()
            
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)
    except Exception as e:
        print(str(e))
process_content()
