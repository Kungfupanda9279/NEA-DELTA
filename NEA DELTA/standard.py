import numpy as np
from heapq import heappop, heappush

# Standardization (manual)
def standardize(data):
    # Convert to numpy array for easy manipulation
    data = np.array(data)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return (data - means) / stds

# Sample data: [age, weight, height, fitness_level (encoded), goal (encoded), health_conditions (binary)]
data = [
    [25, 70, 175, 1, 0, 1, 0],  # User 1
    [30, 80, 180, 0, 1, 0, 1],  # User 2
]

# Standardize the data
scaled_data = standardize(data)
print(scaled_data)


class KDTree:
    def __init__(self, data):
        self.data = data
        self.tree = self.build_kd_tree(data)

    def build_kd_tree(self, data, depth=0):
        if len(data) == 0:
            return None
        
        k = len(data[0])  # number of features
        axis = depth % k  # select axis based on depth

        # Convert to list and sort by the axis
        data = list(data)
        data.sort(key=lambda x: x[axis])
        
        median_index = len(data) // 2
        node = {
            'point': data[median_index],
            'left': self.build_kd_tree(data[:median_index], depth + 1),
            'right': self.build_kd_tree(data[median_index + 1:], depth + 1)
        }
        return node


    def search(self, point, k=3):
    # Use a heap to find the nearest k points
        heap = []
        self._search(self.tree, point, 0, k, heap)
    
    # Return only as many neighbors as available in the heap
        neighbors = []
        while heap:
            neighbors.append(heappop(heap)[1])
        return neighbors


    def _search(self, node, point, depth, k, heap):
        if node is None:
            return
        
        # Calculate distance
        distance = np.linalg.norm(np.array(point) - np.array(node['point']))
        
        if len(heap) < k:
            heappush(heap, (-distance, node['point']))
        elif -heap[0][0] > distance:
            heappop(heap)
            heappush(heap, (-distance, node['point']))

        # Determine whether to search left or right
        axis = depth % len(point)
        if point[axis] < node['point'][axis]:
            self._search(node['left'], point, depth + 1, k, heap)
            if len(heap) < k or abs(point[axis] - node['point'][axis]) < -heap[0][0]:
                self._search(node['right'], point, depth + 1, k, heap)
        else:
            self._search(node['right'], point, depth + 1, k, heap)
            if len(heap) < k or abs(point[axis] - node['point'][axis]) < -heap[0][0]:
                self._search(node['left'], point, depth + 1, k, heap)
class KNN:
    def __init__(self, k=3):
        self.k = k

    def recommend(self, user_data):
        # Use KD-Tree to search nearest neighbors
        tree = KDTree(scaled_data)
        neighbors = tree.search(user_data, k=self.k)

        # Simple exercise recommendation based on neighbors' goals
        exercise_suggestions = []
        for neighbor in neighbors:
            # Add simple exercise logic based on fitness level and goal
            if neighbor[3] == 1:  # If goal is weight loss (just an example)
                exercise_suggestions.append('Running or Cycling')
            else:
                exercise_suggestions.append('Strength Training')
        
        return exercise_suggestions

# Instantiate KNN and use it
knn = KNN(k=3)
exercise_recommendations = knn.recommend(scaled_data[0])  # Example: User 1
print(exercise_recommendations)


from flask import Flask, render_template, request, redirect
import sqlite3
import numpy as np

# Import or initialize your KNN logic (ensure this matches your existing code)
from standard import knn

app = Flask(__name__)

# Database connection
def get_db_connection():
    conn = sqlite3.connect('health_tracker.db')
    conn.row_factory = sqlite3.Row  # Allows fetching results as dictionaries
    return conn

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            # Fetch user inputs
            username = request.form['username']
            password = request.form['password']
            goals = request.form['goals']
            fitness_level = request.form['fitness_level']
            time_availability = int(request.form['time_availability'])
            intensity = request.form['intensity']
            location = request.form['location']
            age = int(request.form['age'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            daily_calorie_goal = int(request.form['daily_calorie_goal'])
            health_conditions = request.form.getlist('health_conditions')

            # Validate inputs
            if not username or not password:
                raise ValueError("Username and password are required!")
            
            # Insert into database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO Users (age, height, weight, fitness_level, goal, daily_calorie_goal, location, time_availability, intensity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (age, height, weight, fitness_level, goals, daily_calorie_goal, location, time_availability, intensity))
            
            # Get the user_id of the newly inserted user
            user_id = cursor.lastrowid  # Ensure this retrieves the ID
            
            # Insert health conditions
            for condition in health_conditions:
                cursor.execute('''
                    INSERT INTO UserHealthConditions (user_id, condition_id, condition_value)
                    VALUES (?, ?, 1)
                ''', (user_id, condition))
            
            # Commit changes
            conn.commit()
            conn.close()

            # Debug print (optional, for testing)
            print(f"Registered user with ID: {user_id}")

            # Redirect to recommendations page
            return redirect(f'/recommend?user_id={user_id}')
        
        except sqlite3.Error as db_error:
            return render_template('register.html', error=f"Database error: {db_error}")
        except ValueError as validation_error:
            return render_template('register.html', error=str(validation_error))
        except Exception as general_error:
            return render_template('register.html', error="An unexpected error occurred.")
    
    return render_template('register.html')


@app.route('/recommend')
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return "User ID not provided."

    # Fetch user details from the database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE user_id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close()

    if not user_data:
        return "No user data found for this ID."

    # Prepare data for the KNN algorithm
    user_data_list = [
        user_data['age'],
        user_data['weight'],
        user_data['height'],
        int(user_data['fitness_level']),  # Assuming fitness_level is encoded
        int(user_data['goal']),           # Assuming goal is encoded
    ]

    # Process recommendation logic
    recommendations = knn.recommend(user_data_list)  # Ensure your `knn` implementation matches
    return render_template('recommend.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)

