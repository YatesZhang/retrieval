"""
    modify from official code: 
        https://github.com/allenai/aokvqa
"""

import argparse
import pathlib
import json
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import os
import json


def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    """ 
        load json
    """
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, "aokvqa_{version}_{split}.json".format(version=version, split=split))
    ))
    return dataset


def map_to_choices(dataset, predictions, device='cpu'):
    """ 
        meta: 
            split: str = 'train',
            image_id: int = 39446,
            question_id: str = 'hashcode'
            instance_id: str = '1'
        data:
            question: str: 'where is the man?' 
            choices: list = ['gt_category', ...]
            correct_choice_idx: int = 0
            direct_answers: list = ['key word as gt', ..., ]
            difficult_direct_answer: bool = False
            rationals: list = ['proof1','proof2', ... ]
    """
    if isinstance(dataset, list):
        dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }
    """
        prediction: dict 
            {   
                'value_of_question_id': 'your_prediction'
            }
    """
    if all([p in dataset[q]['choices'] for q, p in predictions.items()]):
        return predictions
    
    """
        use cos similarity:
    """
    model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
    model.to(device)
    for q in tqdm(predictions.keys()):
        choices = dataset[q]['choices']
        if predictions[q] not in choices:
            choice_embeddings = model.encode([predictions[q]] + choices, convert_to_tensor=True)
            a_idx = cos_sim(choice_embeddings[0], choice_embeddings[1:]).argmax().item()
            predictions[q] = choices[a_idx]
    return predictions


def eval_aokvqa(dataset, preds, multiple_choice=False, strict=True):
    """ 
        evaluate
    """
    if isinstance(dataset, list):
        dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }

    if multiple_choice is False:
        dataset = {k:v for k,v in dataset.items() if v['difficult_direct_answer'] is False}

    if strict:
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids)

    # dataset = q_id (str) : dataset element (dict)
    # preds = q_id (str) : prediction (str)

    acc = []

    for q in dataset.keys():
        if q not in preds.keys():
            acc.append(0.0)
            continue

        pred = preds[q]
        choices = dataset[q]['choices']
        direct_answers = dataset[q]['direct_answers']

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
            correct_choice_idx = dataset[q]['correct_choice_idx']
            acc.append( float(pred == choices[correct_choice_idx]) )
        ## Direct Answer setting
        else:
            num_match = sum([pred == da for da in direct_answers])
            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

    acc = sum(acc) / len(acc) * 100

    return acc


class AOKVQAEval(object):
    def __init__(self):
        """ 
            init Evaluator
        """
        # ground truth: 
        self.ground_truth = {}
        self.prediction = {}
        pass

    def load_pred(self, pred):
        """
            load prediction
                pred: str 
                
        """
        if isinstance(pred, str):
            with open(pred, 'r') as f:
                pred = json.load(f)
        assert isinstance(pred, dict)
        self.prediction = pred 

    def load_gt(self, aokvqa_dir, split, version='v1p0'):
        """ 
            load A-OKVQA
        """
        self.gt = load_aokvqa(aokvqa_dir=aokvqa_dir, split=split, version=version)
        return 
    
    def evaluate(self):
        """ 
            eval:
                1) map choices (use cos similarity)
                2) call evaluator 
        """
        self.prediction = map_to_choices(dataset=self.ground_truth, predictions=self.prediction)

    for prediction_file in glob.glob(args.prediction_files):
        predictions = json.load(open(prediction_file, 'r'))

        # Multiple choice

        mc_predictions = {}

        for q in predictions.keys():
            if 'multiple_choice' in predictions[q].keys():
                mc_predictions[q] = predictions[q]['multiple_choice']

        if mc_predictions != {}:
            mc_acc = eval_aokvqa(
                dataset,
                mc_predictions,
                multiple_choice=True,
                strict=False
            )
            print(prediction_file, 'MC', mc_acc)

        # Direct Answer

        da_predictions = {}

        for q in predictions.keys():
            if 'direct_answer' in predictions[q].keys():
                da_predictions[q] = predictions[q]['direct_answer']

        if da_predictions != {}:
            da_acc = eval_aokvqa(
                dataset,
                da_predictions,
                multiple_choice=False,
                strict=False
            )
            print(prediction_file, 'DA', da_acc)