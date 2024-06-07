import os
import argparse


def rename_files_in_directory(directory_path, start_sequence=1, pattern="", num_digits=1):
    # Get a list of all files in the directory
    files = os.listdir(directory_path)
    files.sort()  # Optional: sort the files if needed

    # Loop through all files and rename them
    valid_extensions = {".jpg"}
    sequence_number = start_sequence

    for filename in files:
        # Check if the file has a valid extension
        filename_no_extension = os.path.splitext(filename)[0]
        file_extension = os.path.splitext(filename)[1]
        if file_extension in valid_extensions:
            # Format the sequence number with leading zeros
            formatted_sequence_number = f"{sequence_number:0{num_digits}d}"
            # Construct the new file name by replacing the placeholder with the formatted sequence number
            new_filename = pattern.replace("{seq}", formatted_sequence_number)
            # Create the full path for the old and new file names
            old_file_path = os.path.join(directory_path, filename)
            new_file_path = os.path.join(directory_path, new_filename + ".JPG")
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{old_file_path}' to '{new_file_path}'")

            #json
            old_json_path = os.path.join(directory_path, filename_no_extension + ".json")
            if os.path.exists(old_json_path):
                new_json_path = os.path.join(directory_path, new_filename + ".json")
                os.rename(old_json_path, new_json_path)
                print(f"Renamed '{old_json_path}' to '{new_json_path}'")

            # Increment the sequence number
            sequence_number += 1


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Rename .jpg and .jpeg files in a directory to a sequence number with static text.")
    parser.add_argument('--directory_path', type=str, default="/Users/nikornlansa/Downloads/indy_dataset",
                        help="The path to the directory containing the files to rename.")
    parser.add_argument('--start_sequence', type=int, default=1417,
                        help="The starting sequence number (default is 1).")
    parser.add_argument('--pattern', type=str, default="IMG_01_01-{seq}",
                        help="The pattern for the new filenames, with {seq} as the placeholder for the sequence number (default is 'file{seq}.jpg').")
    parser.add_argument('--num_digits', type=int, default=5,
                        help="The number of digits for the sequence number, including leading zeros (default is 1).")

    # Parse the arguments
    args = parser.parse_args()

    # Call the rename function with the provided arguments
    rename_files_in_directory(args.directory_path, args.start_sequence, args.pattern, args.num_digits)


if __name__ == "__main__":
    main()

