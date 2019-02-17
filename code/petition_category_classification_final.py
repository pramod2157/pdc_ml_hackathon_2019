import pandas as pd
import numpy as np
import re
#data directory
data_dir = '/home/pramod/pramod_work/pdc_hackathon/data/pdc_ml_hackathon_2019-master/data/'

#read_dataset
train_data = pd.read_json(data_dir+'train.json', encoding='ascii')
test_data = pd.read_json(data_dir+'validation.json', encoding='ascii')


def classify_record(record):
    highlighted_text = record['highlight_ask']
    petition_id = record['petition_id']
    tags  = np.unique([match.lower() for match in re.findall(r'<mark>.*?</mark>', highlighted_text)])
    #print ('tags: ', tags)
    tag = ' '.join([re.sub(r'</?mark>', ' ', sub_string).strip() for sub_string in tags])
    if tag =='tax':
        return [petition_id, 'tax']
    elif tag == 'education':
        return [petition_id, 'education']
    elif tag in ['health', 'care', 'care health', 'health care']:
        return [petition_id, 'health care']
    elif tag == 'infrastructure':
        return [petition_id, 'infrastructure']
    elif tag in ['environment', 'issue', 'environment issue', 'issue environment']:
        return [petition_id, 'environment issue']
    else:
        raise ValueError('Unknown information')

def test(record):
    return record['highlight_ask']

y_pred = train_data.apply(lambda x: classify_record(x), axis=1)

y_pred = test_data.apply(lambda x: classify_record(x), axis=1)
final_submission = pd.DataFrame(y_pred.tolist(), columns=['petition_id', 'predicted_petition_category'])
final_submission.to_csv('final_submission_petition_category.csv', index=False)
