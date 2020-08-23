# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Loads the disaster messages with the respective categories from csv files and joins them.

    Parameters
    ----------
    messages_filepath : str
        Path to the csv file containing the messages. File must contain a column named 'id'.
    categories_filepath : str
        Path to the csv file containing the category information. File must contain a column named 'id'.

    Returns
    -------
    df : pandas DataFrame object
        Returns a dataframe with the messages and respective categories joined together.
    """
    messages = pd.read_csv(messages_filepath, header=0).set_index("id")
    categories = pd.read_csv(categories_filepath, header=0).set_index("id")

    df = messages.join(categories, how="inner")

    return df

def clean_data(df):
    """
    Splits the column 'categories' into one column per category  in the provided dataframe and extracts 
    the numeric value from the strings to obtain a dataframe containing 1 if a value if a category belongs
    to a message and 0 if not. 

    Parameters
    ----------
    df : pandas DataFrame object
        Dataframe containing a column 'categories' of type str with categories separated by ';'
    Returns
    -------
    cleaned_df : pandas DataFrame object
        Dataframe with one column per category and numeric value 0 or 1 indicating the affiliation of the 
        respective record to the category.
    """
    categories = df.categories.str.split(";", expand=True)
    col_names = categories.iloc[0, :].str.extract(r"([a-z\_]*)")[0].tolist()
    categories.columns = col_names

    for column in categories:
        # set each value to be the last character of the string; 
        # replace values "2" with "1" to make binary
        categories[column] = pd.to_numeric(categories[column].str[-1]).replace(2, 1)

    cleaned_df = pd.concat([df.drop("categories", axis=1), categories], axis=1).drop_duplicates()

    return cleaned_df


def save_data(df, database_filename):
    """
    Writes the provided dataframe to a sqlite database into a table 'messages_categorized'. If database 
    and table alreday exist they are replaced.

    Parameters
    ----------
    df : pandas DataFrame object
        Dataframe to write to an sqlite database.
    database_filename : str
        Name of the database to write to. If the specified database does not exist, it is created.

    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('messages_categorized', engine, index=False, if_exists="replace")
        

def main():
    """ Runs the data pre-processing."""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
              
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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