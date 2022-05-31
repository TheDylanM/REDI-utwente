# https://pythonlang.dev/repo/jacobgil-pytorch-grad-cam/

def main():
    classifiers = []
    for classifier in classifiers:
        # load/import pretrained classifier
        classifier = None


        # load dataset
        samples = []
        labels = []

        # define batches (also distinguish between training, validation, and testing)
        sample_batches = []
        label_batches = []

        # finetune classifier
        finetune_classifier(classifier, data)

        # dual training loop
            # train distractor
            train_distractor(distractor, classifier, data)
            new_imgs = generate_new_imgs(distractor, classifier, data)
            # retrain classifier
            retrain_classifier(classifier, new_imgs)

        
        # evaluation


    


if __name__ == "__main__":
    main()