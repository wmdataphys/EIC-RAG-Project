import argparse
import sqlite3
import os

def create_db_and_table(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users(
            first_name TEXT,
            last_name TEXT,
            username TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_user_data(db_name, first_name, last_name, username):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users(first_name, last_name, username) VALUES(?,?,?)
    ''', (first_name, last_name, username))
    conn.commit()
    conn.close()

parser = argparse.ArgumentParser(description='User Information')
parser.add_argument('--first_name', required=True, help='First name of the user')
parser.add_argument('--last_name', required=True, help='Last name of the user')
parser.add_argument('--username', required=True, help='Username of the user')
parser.add_argument('--db_name', required=True, help='Database name')

args = parser.parse_args()

db_name = args.db_name
first_name = args.first_name
last_name = args.last_name
username = args.username

create_db_and_table(db_name)
insert_user_data(db_name, first_name, last_name, username)