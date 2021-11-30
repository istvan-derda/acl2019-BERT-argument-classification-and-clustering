"""
Runs a pre-trained BERT model for argument classification.

You can download pre-trained models here: https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_classification_ukp_all_data.zip

The model 'bert_output/ukp/bert-base-topic-sentence/all_ukp_data/' was trained on all eight topics (abortion, cloning, death penalty, gun control, marijuana legalization, minimum wage, nuclear energy, school uniforms) from the Stab et al. corpus  (UKP Sentential Argument
Mining Corpus)

Usage: python inference.py

"""

from dataclasses import dataclass
from pprint import pprint
from typing import List

import numpy as np
import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from train import InputExample, convert_examples_to_features

num_labels = 3
default_model_path = 'bert_output/argument_classification_ukp_all_data/'
label_list = ["NoArgument", "Argument_against", "Argument_for"]
max_seq_length = 64
eval_batch_size = 8

# Input examples. The model 'bert_output/ukp/bert-base-topic-sentence/all_topics/' expects text_a to be the topic
# and text_b to be the sentence. label is an optional value, only used when we print the output in this script.

default_input_examples = [
    InputExample(text_a='zoo',
                 text_b='A zoo is a facility in which all animals are housed within enclosures, displayed to the '
                        'public, and in which they may also breed. ',
                 label='NoArgument'),
    InputExample(text_a='zoo',
                 text_b='Zoos produce helpful scientific research. ',
                 label='Argument_for'),
    InputExample(text_a='zoo',
                 text_b='Zoos save species from extinction and other dangers.',
                 label='Argument_for'),
    InputExample(text_a='zoo',
                 text_b='Zoo confinement is psychologically damaging to animals.',
                 label='Argument_against'),
    InputExample(text_a='zoo',
                 text_b='Zoos are detrimental to animals\' physical health.',
                 label='Argument_against'),
    InputExample(text_a='autonomous cars',
                 text_b='Zoos are detrimental to animals\' physical health.',
                 label='NoArgument'),
    InputExample(text_a='autonomous cars',
                 text_b='The carbon-foorefactor extract to functiontprint is worse than combustion-engines.',
                 label='Argument_against'),
    InputExample(text_a='eating vegetables',
                 text_b='The carbon-footprint is worse than combustion-engines.',
                 label='NoArgument'),
    InputExample(text_a='combustion-engines',
                 text_b='The carbon-footprint of electric vehicles is worse than combustion-engines.',
                 label='Argument_for'),  # fails
    InputExample(text_a='electric vehicles',
                 text_b='The carbon-footprint is worse than combustion-engines.',
                 label='Argument_against'),
    InputExample(text_a='electric vehicles',
                 text_b='The carbon-footprint is smaller than combustion-engines.',
                 label='Argument_for'),
    InputExample(text_a='electric vehicles',
                 text_b='The carbon-footprint is bigger than combustion-engines.',
                 label='Argument_against'),  # fails
    InputExample(text_a='sex education in schools',
                 text_b='It is important to teach children about STDs',
                 label='Argument_for'),
    InputExample(text_a='sex education in schools',
                 text_b='It makes children homosexual.',
                 label='Argument_against'),
    InputExample(text_a='sex education in schools',
                 text_b='Sex is fun.',
                 label='NoArgument'),  # Argument_for
    InputExample(text_a='sex education in schools',
                 text_b='Sex education is about teaching children safe sex-practices.',
                 label='NoArgument'),  # Argument_for
    InputExample(text_a='sex education in schools',
                 text_b='I had sex education in school.',
                 label='NoArgument'),
    InputExample(
        text_a="Saddam Hussein is gone and Iraq is now functioning as one of very few democracies in the Middle East",
        text_b="It's important to be clear that this debate is looking at the results of the Iraq war and, by any definition Iraq is in a much more stable and secure position than it was in 2003 when American, British and other international troops arrived in the country.",
        label='NoArgument'),  # ok?
    InputExample(text_a="Punishing objectively harmful conduct",
                 text_b="They said that they would come back and kill my parents if I didn’t do as they said.”[i] Once inducted into the army, children are vulnerable to abuse and exploitation.",
                 label='NoArgument'),  # ok?
    InputExample(text_a="Multiple vaccines cause no harm",
                 text_b="No evidence exists that there is a link between MMR or any multiple vaccine and the development of autism.",
                 label='Argument_for'),  # ok?
    # sentences from args.me with query
    InputExample(text_a="Should stem cell research be expanded?",
                 text_b="There is also stem cell research which is not directly related to stem cell research which is also being banned in the United States.",
                 label='NoArgument'),
    InputExample(text_a="Should stem cell research be expanded?",
                 text_b="He says, “My opponent has argued that abortion is a good idea not that stem cell embryonic stem cell research should be expanded.” However, in my last post, I clearly said, “Embryonic stem cell research needs to expand because it can save more lives!” My opponent also states, “This is extremely irrelevant to debate and as much as the child could be a criminal, they could equally be the next Shakespeare or Mozart.” Two things are wrong with this claim.",
                 label='Argument_for'),
    InputExample(text_a="Should stem cell research be expanded?",
                 text_b="My opponent has argued that abortion is a good idea not that stem cell embryonic stem cell research should be expanded.",
                 label='NoArgument'),
    InputExample(text_a="Do we need sex education in schools?",
                 text_b='I believe that education is necessary, but we do not need "schools" for it.',
                 label='Argument_against'),
    InputExample(text_a="Do we need sex education in schools?",
                 text_b='Pro has argued that we need to do away with schools.',
                 label='NoArgument'),
    InputExample(text_a="Do we need sex education in schools?",
                 text_b='Thus, we need to uphold my standard of "monitoring education level in public high schools", because we NEED a standard to uphold education by.',
                 label='NoArgument'),
    # sentences from args.me with topic
    InputExample(text_a="Do we need sex education in schools?",
                 text_b='I believe that education is necessary, but we do not need "schools" for it.',
                 label='Argument_against'),
    InputExample(text_a="Do we need sex education in schools?",
                 text_b='Pro has argued that we need to do away with schools.',
                 label='NoArgument'),
    InputExample(text_a="Do we need sex education in schools?",
                 text_b='Thus, we need to uphold my standard of "monitoring education level in public high schools", because we NEED a standard to uphold education by.',
                 label='NoArgument'),
]


@dataclass()
class ArgumentClassificationResult:
    topic: str
    sentence: str
    predicted_label: str

    def __repr__(self):
        return f"ArgumentClassificationResult(\n" \
               f"|   topic: {self.topic}\n" \
               f"|   \"{self.sentence}\"\n" \
               f"|   predicted label: {self.predicted_label})"


@dataclass()
class ArgumentClassificationInput:
    topic: str
    sentence: str


class BertArgumentClassifier:
    def __init__(self, model_path=default_model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _inference(self, sentence_pairs: List[ArgumentClassificationInput]) -> List[ArgumentClassificationResult]:

        # conversion so I don't have to touch internal code
        internal_argument_input_repr = [
            InputExample(text_a=sentence_pair.topic, text_b=sentence_pair.sentence, label="NoArgument") for
            sentence_pair in sentence_pairs]
        eval_features = convert_examples_to_features(internal_argument_input_repr, label_list, max_seq_length,
                                                     self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

        predicted_labels = []
        with torch.no_grad():
            for input_ids, input_mask, segment_ids in eval_dataloader:
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)

                logits = self.model(input_ids, segment_ids, input_mask)
                logits = logits.detach().cpu().numpy()

                for prediction in np.argmax(logits, axis=1):
                    predicted_labels.append(label_list[prediction])

        return [ArgumentClassificationResult(sentence_pair.topic, sentence_pair.sentence, predicted_label) for
                sentence_pair, predicted_label in zip(sentence_pairs, predicted_labels)]

    def classify(self, topic: str, sentence: str) -> ArgumentClassificationResult:
        input_list = [ArgumentClassificationInput(topic, sentence)]
        result_list = self._inference(input_list)
        return result_list[0]

    def classify_batch(self, input_list: List[ArgumentClassificationInput]) -> List[ArgumentClassificationResult]:
        return self._inference(input_list)


def main():
    classifier = BertArgumentClassifier()
    classification_inputs = [ArgumentClassificationInput(topic=example.text_a, sentence=example.text_b)
                             for example in default_input_examples]
    result = classifier.classify_batch(classification_inputs)
    pprint(result)


if __name__ == "__main__":
    main()
