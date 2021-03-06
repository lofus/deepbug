import numpy as np
from dataset import deepbug_chronological_cv
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

    # CNN channels (log stack layers)
    cnn_channels = 5
    print("Loading dataset ...")

    slices = deepbug_chronological_cv(
        dataset_name, min_train_samples_per_class, num_cv, merged_wordvec_model
    )

    slice_results = {}
    top_rank_k_accuracies = []
    for i, (CNN_train, RNN_train, y_train, CNN_test, RNN_test, y_test, classes) in enumerate(slices):

        #create the model
        model = deepbug_model(
            (max_sentence_len, embed_size_word2vec, cnn_channels), (max_sentence_len, embed_size_word2vec), len(classes)
        )

        # Train the deep learning model and test using the classifier
        early_stopping = EarlyStopping(monitor="val_loss", patience=8)
        plot_losses = PlotLosses()
        checkpt = ModelCheckpoint('deepbug.model.h5', monitor='val_acc', save_best_only=True, save_weights_only=True)

        print("model summary:")
        model.summary()

        hist = model.fit(
            [CNN_train, RNN_train],
            y_train,
            batch_size=batch_size,
            epochs=300,
            callbacks=[early_stopping, plot_losses]
        )

        prediction = model.predict([CNN_test, RNN_test])
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

