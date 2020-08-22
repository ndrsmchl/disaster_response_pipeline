import sys
from pipeline.functions import load_data, stratified_multilabel_train_test_split, make_pipline, \
     grid_search, evaluate_model, save_model
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
    if len(sys.argv) == 3:
        database_filepath, model_name = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        table_name = 'messages_categorized'
        X, Y, category_names = load_data(database_filepath, table_name)        
        X_train, X_test, Y_train, Y_test = stratified_multilabel_train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        forest = RandomForestClassifier(n_jobs=-1, random_state=0, verbose=10)
        MoC_forest = MultiOutputClassifier(forest)
        pipeline = make_pipline(forest)
        param_grid = {
            'features__text_pipeline__count_vect__ngram_range': [(1, 1)],# (1, 2)],
            #'features__text_pipeline__count_vect__max_df': [0.5, 1.0],
            #'features__text_pipeline__count_vect__max_features': [None, 5000],# 10000],
            #'features__text_pipeline__tfidf__use_idf': [True, False],
            #'clf__estimator__n_estimators': [200],
            'clf__n_estimators': [200],
            #'clf__max_features': ["auto", None],
            #'clf__min_samples_split': [2, 4],# 8],
            #'clf__estimator__class_weight': ["balanced_subsample"]
            'clf__class_weight': ["balanced_subsample"]
        }

        print('Searching best model...')
        best_model = grid_search(X_train, Y_train, pipeline, param_grid).best_estimator_
        
        print('Evaluating model...')
        evaluate_model(best_model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_name))
        save_model(best_model, model_name)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and a model name of the pickle file to the model to as thse second argument. \
              \n\nExample: python3 -m models.train_classifier.py data/DisasterResponse.db MyModel')

if __name__ == '__main__':
    main()