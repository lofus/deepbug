import numpy as np
from build_dataset import chronological_cv
from keras.callbacks import EarlyStopping, ModelCheckpoint
from plot_loss import PlotLosses
from deepbug_models import *

def run_deepbug_with_cv(
        dataset_name,
        min_train_samples_per_class,
        num_cv,
        merged_wordvec_model=False,
):

    if min_train_samples_per_class not in [0, 5, 10, 20]:
        print("Wrong min train samples per class")
        return

    if num_cv < 1:
        print("Wrong number of chronological cross validation (num_cv)")
        return

    if dataset_name not in ["google_chromium", "mozilla_firefox", "jira"]:
        print("Wrong dataset name")
        return

    print("Deployed dataset", dataset_name)

    # Word2vec parameters
    embed_size_word2vec = 200

    # Classifier hyperparameters
    max_sentence_len = 50
    rank_k = 10
    batch_size = 1024

    print("Loading dataset ...")

    slices = chronological_cv(
        dataset_name, min_train_samples_per_class, num_cv, merged_wordvec_model
    )

    slice_results = {}
    top_rank_k_accuracies = []
    for i, (X_train, y_train, X_test, y_test, classes) in enumerate(slices):
        print("X_train.shape", X_train.shape)
        print("y_train.shape", y_train.shape)
        print("X_test.shape", X_test.shape)
        print("y_test.shape", y_test.shape)
        print("classes:", len(classes))

        # Need merge?
        M_X = np.zeros((X_train.shape[0]+X_test.shape[0], X_train.shape[1], X_train.shape[2]))
        M_y = np.zeros((y_train.shape[0]+y_test.shape[0], y_train.shape[1]))
        M_X[:X_train.shape[0]] = X_train
        M_X[X_train.shape[0]:] = X_test
        M_y[:y_train.shape[0]] = y_train
        M_y[y_train.shape[0]:] = y_test
        X_train = M_X
        y_train = M_y

        #reshape
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], -1))
        print("X_train.shape", X_train.shape)
        print("y_train.shape", y_train.shape)

        #create the model
        model = deepbug_cnn_model(
            (max_sentence_len, embed_size_word2vec, 1), len(classes)
        )

        # Train the deep learning model and test using the classifier
        early_stopping = EarlyStopping(monitor="val_loss", patience=8)
        plot_losses = PlotLosses()
        checkpt = ModelCheckpoint('deepbug.model.h5', monitor='val_acc', save_best_only=True, save_weights_only=True)

        print("model summary:")
        model.summary()
        hist = model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            batch_size=batch_size,
            epochs=250,
            callbacks=[early_stopping, plot_losses]
        )

        prediction = model.predict(X_test)
        accuracy = topk_accuracy(prediction, y_test, classes, rank_k=rank_k)
        print("CV{}, top1 - ... - top{} accuracy: ".format(i + 1, rank_k), accuracy)

        train_result = hist.history
        train_result["test_topk_accuracies"] = accuracy
        slice_results[i + 1] = train_result
        top_rank_k_accuracies.append(accuracy[-1])

    print("Top{0} accuracies for all CVs: {1}".format(rank_k, top_rank_k_accuracies))
    print("Average top{0} accuracy: {1}".format(rank_k, sum(top_rank_k_accuracies) / rank_k))
    return slice_results

def topk_accuracy(prediction, y_test, classes, rank_k=10):
    accuracy = []
    sortedIndices = []
    pred_classes = []
    for ll in prediction:
        sortedIndices.append(
            sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True)
        )
    for k in range(1, rank_k + 1):
        id = 0
        trueNum = 0
        for sortedInd in sortedIndices:
            pred_classes.append(classes[sortedInd[:k]])
            if np.argmax(y_test[id]) in sortedInd[:k]:
                trueNum += 1
            id += 1
        accuracy.append((float(trueNum) / len(prediction)) * 100)

    return accuracy


gc_result_dict = run_deepbug_with_cv(
    dataset_name="jira", min_train_samples_per_class=20, num_cv=10
)

