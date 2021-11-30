import csv
import time

from inference import ArgumentClassificationInput, BertArgumentClassifier

IN_PATH = "args_processed.csv"
LINES_COUNT = 365409
OUT_PATH = "args_with_bert_stance.csv"


def main():
    classificator = BertArgumentClassifier()
    start_time = time.localtime()

    with open(IN_PATH) as in_file:
        with open(OUT_PATH, 'w') as out_file:
            reader = csv.DictReader(in_file)
            writer = csv.DictWriter(out_file, fieldnames=['sent_id', 'topic_text', 'predicted_stance', 'sent_text'])
            writer.writeheader()

            for line_numer, line in enumerate(reader):

                sentences_dict = eval(line['sentences'])
                topic = line['conclusion']
                print(f"topic: {topic}")

                sentence_texts = [sentence_obj['sent_text'] for sentence_obj in sentences_dict]
                sentence_ids = [sentence_obj['sent_id'] for sentence_obj in sentences_dict]
                classification_inputs = [ArgumentClassificationInput(topic, sentence_text) for sentence_text in
                                         sentence_texts]
                results = classificator.classify_batch(classification_inputs)
                # pprint(results)

                for sent_id, result in zip(sentence_ids, results):
                    writer.writerow({
                        'sent_id': sent_id,
                        'topic_text': result.topic,
                        'predicted_stance': result.predicted_label,
                        'sent_text': result.sentence
                    })

                # progress indication
                print(f"starttime: {time.strftime('%H:%M', start_time)}")
                print(f"progress: {(line_numer * (100 / LINES_COUNT)):3.3f}%")
                print(f"last processed: line {line_numer}")
                print(f"--------------------------------------------")


if __name__ == '__main__':
    main()
