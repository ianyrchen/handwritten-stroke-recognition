import pickle

def modify_string(s):
    if not s:
        return s  # Handle empty string case
    
    result = []
    i = 0
    while i < len(s):
        # Start of a potential sequence
        char = s[i]
        sequence_start = i
        
        # Find the length of the sequence of identical characters
        while i < len(s) and s[i] == char:
            i += 1
        sequence_length = i - sequence_start
        
        if sequence_length > 1:
            # Modify every other character in the sequence to be a dollar sign
            for j in range(sequence_length):
                if j % 2 == 0:
                    result.append(char)
                else:
                    result.append('$')
        else:
            # Single character sequence, add the character itself
            result.append(char)
            
    return ''.join(result)
# Main loop to process the pickle file
def main():
    input_filename = 'y_char_data.pkl'
    output_filename = 'ymod.pkl'
    
    # Load data from pickle file
    with open(input_filename, 'rb') as file:
        y_data = pickle.load(file)
    print(y_data[0])
    # Ensure y_data is a list of strings
    if not isinstance(y_data, list) or not all(isinstance(item, str) for item in y_data):
        raise ValueError(f"Unexpected data format in {input_filename}")
    
    # Process each string in the list
    ymod = [modify_string(string) for string in y_data]
    
    # Save modified list to new pickle file
    with open(output_filename, 'wb') as file:
        pickle.dump(ymod, file)
    
    print(f"Processed data has been saved to {output_filename}")

# Run the main function
if __name__ == "__main__":
    main()
