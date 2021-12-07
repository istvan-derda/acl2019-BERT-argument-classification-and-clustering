import csv
import sys
from datetime import datetime

from inference import ArgumentClassificationInput, BertArgumentClassifier

IN_PATH = "args_processed.csv"
LINES_COUNT = 365409
OUT_PATH = "args_with_bert_stance.csv"


def main():
    classificator = BertArgumentClassifier()
    start_time = datetime.now()
    csv.field_size_limit(sys.maxsize)

    with open(IN_PATH) as in_file:
        with open(OUT_PATH, 'w') as out_file:
            reader = csv.DictReader(in_file)
            writer = csv.DictWriter(out_file, fieldnames=['sent_id', 'topic_text', 'predicted_stance', 'sent_text'])
            writer.writeheader()

            for line_number, line in enumerate(reader):
                if line_number < 40_000:
                    continue

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

                if line_number % 1000 == 0:
                    # progress indication
                    progress_percentage = (line_number + 1) * (100 / LINES_COUNT)
                    time_running = datetime.now() - start_time
                    predicted_full_runtime = (100 / progress_percentage) * time_running
                    print(f"starttime: {start_time:%H:%M}\n"
                          f"progress: {(progress_percentage):3.3f}%\n"
                          f"running since: {time_running}\n"
                          f"predicted full runtime: {predicted_full_runtime}\n"
                          f"last processed: line {line_number}\n"
                          f"topic: {topic}\n"
                          f"--------------------------------------------\n")


if __name__ == '__main__':
    main()
