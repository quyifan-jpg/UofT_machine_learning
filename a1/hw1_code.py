import random
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import math

def load_data(filename1, filename2):
    fake = []
    real = []
    label = []
    f1 = open(filename1, "r")
    f2 = open(filename2, "r")
    for line1 in f1:
        real.append(line1)
    for line2 in f2:
        fake.append(line2)
    for _ in range(len(real)):
        label.append(1)
    for _ in range(len(fake)):
        label.append(fake)

    vt = TfidfVectorizer()
    total = real + fake
    cv_fit = vt.fit_transform(total)

    training_x, rest_x, training_y, rest_y = train_test_split(cv_fit, label, test_size=0.3,
                                                              random_state=41, shuffle=True)
    testing_x, validation_x, testing_y, validation_y = train_test_split(rest_x, rest_y, test_size=0.5,
                                                                        random_state=41, shuffle=True)
    training_x_array = training_x.toarray()
    training_y_array = np.array(training_y)
    testing_x_array = testing_x.toarray()
    testing_y_array = np.array(testing_y)
    validation_x_array = validation_x.toarray()
    validation_y_array = np.array(validation_y)
    return [training_x_array,testing_x_array,validation_x_array],[training_y_array, testing_y_array, validation_y_array], vt

def select_model(dataset):
    training = dataset[0][0]
    training_label = dataset[1][0]
    validate = dataset[0][1]
    validate_label = dataset[1][1]
    targets = training_label
    scores = []
    for type1 in ['gini', 'entropy']:
        for depth in range(3, 12, 2):
            clf = DecisionTreeClassifier(criterion=type1, max_depth=depth)
            clf.fit(training, targets)
            res_pre = clf.predict(validate)
            scores.append(accuracy_score(validate_label, res_pre))
    plot_model(scores)
    max_index = scores.index(max(scores))
    range1 = [3, 5, 7, 9, 11]
    result = []
    if max_index < 5:
        result = 'gini', range1[max_index]
    else:
        result = 'entropy', range1[max_index - 5]
    clf = DecisionTreeClassifier(criterion=result[0], max_depth=result[1])
    clf.fit(training, targets)
    tree.plot_tree(clf, feature_names=dataset[2].get_feature_names(), class_names=['real', 'fake'], filled=True,
                   fontsize=8, node_ids=True)
    plt.show()
    return clf


def draw_tree(type1, depth, ):
    pass


def plot_model(scores):
    plt.figure(figsize=(20, 8), dpi=80)
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy (in %)")
    plt.grid(alpha=0.3)
    plt.title("2b")

    x_axis = [3, 5, 7, 9, 11]
    plt.plot(x_axis, scores[:5], color="green", label="gini")
    plt.plot(x_axis, scores[5:], color="purple", label="entropy")
    plt.legend(loc=0)
    plt.savefig("./q2b.png")
    # ###plt.show()


def H(p):
    if p == 1 or p == 0:
        return 0
    return -(1 - p) * math.log(1 - p, 2) - p * math.log(p, 2)


def compute_information_gain(my_tree1, keyword, vt):
    node_index = 0
    feature_set = vt.get_feature_names()
    threshold = my_tree1.tree_.threshold[0]
    for i in range(len(my_tree1.tree_.feature)):
        if feature_set[my_tree1.tree_.feature[i]] == keyword:
            node_index = i
    if node_index == -1:
        print("not the keyword in featrue_set")
        return 0
    p_node = (my_tree1.tree_.value[node_index].item(0) / my_tree1.tree_.n_node_samples[node_index])
    index_left = my_tree1.tree_.children_left[node_index]
    index_right = my_tree1.tree_.children_right[node_index]

    en_root = H(p_node)
    en_left = H(my_tree1.tree_.value[index_left].item(0) / my_tree1.tree_.n_node_samples[index_left])
    en_right = H(my_tree1.tree_.value[index_right].item(0) / my_tree1.tree_.n_node_samples[index_right])
    left = (my_tree1.tree_.n_node_samples[index_left] / my_tree1.tree_.n_node_samples[node_index]) * en_left
    right = (my_tree1.tree_.n_node_samples[index_right] / my_tree1.tree_.n_node_samples[node_index]) * en_right
    ig = en_root - left - right
    print(f"information gain for word {feature_set[my_tree1.tree_.feature[node_index]]}: {ig}")
    return ig




if __name__ == '__main__':
    my_data = load_data('clean_real.txt', 'clean_fake.txt')
    my_tree = select_model(my_data)
    compute_information_gain(my_tree, 'trump', my_data[2])
    # trump just for example, keyword could be others
    # if keyword not in classifier tree, do nothin return 0
