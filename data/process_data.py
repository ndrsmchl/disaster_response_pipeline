# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath, header=0).set_index("id")
    categories = pd.read_csv(categories_filepath, header=0).set_index("id")

    df = messages.join(categories, how="inner")

    return df

def clean_data(df):
    categories = df.categories.str.split(";", expand=True)
    col_names = categories.iloc[0, :].str.extract(r"([a-z\_]*)")[0].tolist()
    categories.columns = col_names

    for column in categories:
        # set each value to be the last character of the string; 
        # replace values "2" with "1" to make binary
        categories[column] = pd.to_numeric(categories[column].str[-1]).replace(2, 1)

    df = pd.concat([df.drop("categories", axis=1), categories], axis=1).drop_duplicates()

    return df


def save_data(df, database_filename, to_sqlite=True):
    if to_sqlite:
        engine = create_engine(f"sqlite:///{database_filename}")
        df.to_sql('messages_categorized', engine, index=False)
    
    df.to_csv("messages_categorized.csv")
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
              
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, True)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()