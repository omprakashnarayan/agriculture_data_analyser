import mysql.connector
import nltk
from nltk.tokenize
import word_tokenize
from nltk.corpus
import stopwords

# Connect to the database
conn = mysql.connector.connect(
    host = "your_host",
    user = "your_username",
    password = "your_password",
    database = "your_database",
)
cursor = conn.cursor()

# Function to convert natural language queries into SQL queries
def convert_to_sql(query): #Tokenize the query
tokens = word_tokenize(query)

# Remove stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [token
    for token in tokens
    if token.lower() not in stop_words
]

# Extract relevant keywords from the query
keywords = []
for token in filtered_tokens:
    cursor.execute("SELECT keyword FROM keyword_mapping WHERE keyword=%s", (token, ))
result = cursor.fetchone()
if result:
    keywords.append(result[0])

# Generate SQL query
sql_query = "SELECT * FROM data_table WHERE "
for keyword in keywords:
    sql_query += keyword + " AND "

sql_query = sql_query[: -5]# Remove the extra "AND"
at the end

return sql_query


# Function to execute the SQL query and fetch data from the database
def execute_query(sql_query):
    cursor.execute(sql_query)
result = cursor.fetchall()
return result


# Function to insert data into the database
def insert_data(values):
    insert_query = (
        "INSERT INTO data_table (column1, column2, column3) VALUES (%s, %s, %s)"
    )
cursor.execute(insert_query, values)
conn.commit()
print("Data inserted successfully.")


# Function to update data in the database
def update_data(column_name, new_value, condition):
    update_query = f "UPDATE data_table SET {column_name} = %s WHERE {condition}"
cursor.execute(update_query, (new_value, ))
conn.commit()
print("Data updated successfully.")


# Function to delete data from the database
def delete_data(condition):
    delete_query = f "DELETE FROM data_table WHERE {condition}"
cursor.execute(delete_query)
conn.commit()
print("Data deleted successfully.")


# Main program loop
while True:
    user_input = input("Enter your query (type 'exit' to quit): ")
if user_input == "exit":
    break

# Check
if it is an insert, update, or delete operation
if user_input.startswith("insert"): #Example input: insert value1 value2 value3
values = user_input.split()[1: ]
insert_data(values)
elif user_input.startswith("update"): #Example input: update column_name new_value condition
input_parts = user_input.split()
column_name = input_parts[1]
new_value = input_parts[2]
condition = " ".join(input_parts[3: ])
update_data(column_name, new_value, condition)
elif user_input.startswith("delete"): #Example input: delete condition
condition = " ".join(user_input.split()[1: ])
delete_data(condition)
else :#Convert user input to SQL query
sql_query = convert_to_sql(user_input)

# Execute the SQL query and fetch data from the database
query_result = execute_query(sql_query)

# Display the query result
print("Query Result:")
for row in query_result:
    print(row)
print()
cursor.close()
conn.close()