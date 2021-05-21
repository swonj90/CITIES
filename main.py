import argparse
from cities import CITIES

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CITIES')
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--pretrained_model', type=str, default='bert')
    parser.add_argument('--embedding_key', type=str, default='bert.embedding.token.weight')
    args = parser.parse_args()
 
    cities = CITIES(args)
    # STEP 1
    # preprocess
    train_context, item_dict = cities.load_data()
    train_dataset, valid_dataset, item_embedding, pretrained_model = cities.preprocess(train_context, item_dict, is_training=True)

    # STEP 2
    # init cities model
    cities.load_model(item_embedding)

    # STEP 3
    # train
    cities.train(train_dataset, valid_dataset, item_embedding)

    # STEP 4
    # predict
    inference_dataset = cities.preprocess(train_context, item_dict, is_training=False)
    inferred_embedding = cities.predict(
        inference_dataset, item_embedding)

    # STEP 5
    # update model
    cities.update_model(pretrained_model, inferred_embedding)
