import sys

from model import Model
from preprocessing import PreProcessing


# final-testData-no-label-Obama-tweets
# final-testData-no-label-Romney-tweets

def write_to_file(twt_id, p_labels, who):
    t_len = len(twt_id)
    p_len = len(p_labels)

    if t_len != p_len:
        raise Exception('The tweet id and the predicated labels does not match')

    with open(f'{who}.txt', 'w') as output:

        line = '54\n'
        output.write(line)

        for index, val in enumerate(twt_id):
            line = str(t_id[index]) + ';;' + str(p_labels[index]) + ('\n' if index < p_len - 1 else "")
            # print(line)
            output.write(line)


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('Usage: python main.py <excel file>')
        exit(-1)

    training_data_file_path = sys.argv[1]
    test_data_file_path = sys.argv[2]
    who = sys.argv[3]

    print(f"Running model for {who} with test file {test_data_file_path}")

    if who == 'obama':
        # obama
        sheet_index = 0
    else:
        # romney
        sheet_index = 1

    # Tweet, Class label
    use_cols = [3, 4]

    # always last column should be the class label
    column_names = ['Tweet', 'Class']
    skip_rows = [0, 1]
    labels = [-1, 0, 1]

    processing = PreProcessing()

    [training_tweet, class_label], training_tweet_id = processing.preprocess(training_data_file_path, False,
                                                                             sheet_index, use_cols, column_names,
                                                                             skip_rows, labels, False)

    # create the model class with training data, test data and labels
    model = Model(training_tweet, class_label, labels)

    models = [
        'SVC_RBF',
        # 'SVC_SIGMOID',
        # 'SVC_LINEAR',
        # 'SVM',
        # 'LOGISTIC_REGRESSION',
        # 'NEAREST_CENTROID',
        # 'KNN',
        # 'DECISION_TREE',
        # 'RANDOM_FOREST',
    ]
    resp = model.model_accuracy(models)

    sheet_index = 0

    # ID, Tweet
    use_cols = [0, 1]
    column_names = ['id', 'Tweet']
    skip_rows = []

    # in case of test data the classes are ignored
    [tweet, _], t_id = processing.preprocess(test_data_file_path, True, sheet_index, use_cols, column_names, skip_rows,
                                             None, True)

    predicted_labels = model.prediction_with_model_name('SVC_RBF', tweet)

    write_to_file(t_id, predicted_labels, who)
