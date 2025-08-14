"""
Preprocessing of Data

Script Syntax:
> python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite destination db>

Script Execution:
> python process_data.py messages.csv categories.csv disaster_response_db.db

Arguments Description:
    1) Path to the messages CSV file
    2) Path to the categories CSV file
    3) Path to SQLite destination database
"""

# Import all the relevant libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

def load_data(messages_filepath, categories_filepath):
    """
    Load Data: Messages and Categories
    
    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> messages and categorie Data in a combined dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df 

def clean_data(df):
    """
    clean df function
    
    Arguments:
        df -> Combined data containing messages and categories
    Outputs:
        df -> Combined data containing messages and categories with clean categories
    """
    #Split `categories` into separate category columns.
    
    categories = df['categories'].str.split(pat=';', expand=True)

    # create a with the right names
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    
    #rename columns
    categories.columns = category_colnames

    ## Convert category values to numbers
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # Replace `categories` column in `df` with new category columns.
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()

    # fix value 2 in related field:
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    return df

def save_data(df, database_filename):
    """
    Save Data to Database
    
    Arguments:
        df -> Combined data containing cleaned up messages and categories
        database_filename -> Path to database
    """
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = os.path.basename(database_filename)
    table_name = table_name.replace(".db","") + "_table"
    df.to_sql(table_name, engine, if_exists='replace', index=False)


def main():
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