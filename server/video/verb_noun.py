import os
import pickle


def get_verb_noun_classes():
    verb_set = set()
    noun_set = set()
    for root, dirs, files in os.walk('static'):
    	for filename in [os.path.join(root, name) for name in files]:
            verb_id = filename.split('.')[0].split('_')[-2]
            noun_id = filename.split('.')[0].split('_')[-1]

            if verb_id in verb_set:
            	continue
            else:
            	verb_set.add(verb_id)
            if noun_id in noun_set:
            	continue
            else:
            	noun_set.add(noun_id)
    file = open('verb_noun.pkl', 'wb')
    pickle.dump([list(verb_set), list(noun_set)], file)
    file.close()

if __name__=='__main__':
	get_verb_noun_classes()
	with open('verb_noun.pkl', 'rb') as f:
		verb_list, noun_list = pickle.load(f)
	print(len(verb_list))
	print((noun_list))
