import sqlite3

# Connect to database
conn = sqlite3.connect('health_tracker.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    height REAL,
    weight REAL,
    fitness_level TEXT,
    goal TEXT,
    daily_calorie_goal INTEGER,
    location TEXT,
    time_availability TEXT,
    intensity TEXT
);
''')

cursor.execute('''
CREATE TABLE HealthConditions (
    condition_id INTEGER PRIMARY KEY,
    condition_name TEXT
);
''')

cursor.execute('''
CREATE TABLE UserHealthConditions (
    user_id INTEGER,
    condition_id INTEGER,
    condition_value INTEGER,
    FOREIGN KEY (user_id) REFERENCES Users(user_id),
    FOREIGN KEY (condition_id) REFERENCES HealthConditions(condition_id)
);
''')

conn.commit()
conn.close()
