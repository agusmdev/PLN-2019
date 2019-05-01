"""Model analysis tools."""
def prettify_names(str_):
    return str_.replace("vect__", "").replace("vect2__", "")

def print_maxent_features(vect, clf, n=5, prettify=False):
    """
    Most relevant features for each class (logistic regression).

    vect -- vectorizer (count or tf-idf)
    clf -- LogisticRegression classifier
    n -- number of features to show
    """
    C = clf.coef_
    A = clf.coef_.argsort()
    features = vect.get_feature_names()
    if prettify:
        features = list(map(prettify_names, features))
        
    for i, label in enumerate(clf.classes_):
        print('{}:'.format(label))
        print('\t{} ({})'.format(
            ' '.join([features[j] for j in A[i, :n]]),
            C[i, A[i, :n]]))
        print('\t{} ({})'.format(
            ' '.join([features[j] for j in A[i, -n:]]),
            C[i, A[i, -n:]]))


def print_feature_weights_for_item(vect, clf, x):
    """
    Print active features and their weight for a specific item.

    vect -- text vectorizer (count or tf-idf)
    clf -- LogisticRegression classifier
    """
    features = vect.get_feature_names()
    features = list(map(prettify_names, features))
    x2 = vect.transform([x])
    col = x2.tocoo().col
    for i in col:
        print(features[i], clf.coef_[:,i])
