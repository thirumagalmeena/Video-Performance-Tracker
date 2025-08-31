from src.database.db_handler import create_tables, get_connection

create_tables()
conn = get_connection()
print("Tables created successfully!")
