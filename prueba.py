import pickle

# Load the name_dict.pkl file
with open('name_dict.pkl', 'rb') as file:
    name_dict = pickle.load(file)

# Print the names

print(name_dict[75])
