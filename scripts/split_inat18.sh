# Change to the desired directory
cd ./data/inat18 || exit # Exit if directory doesn't exist

# Ensure the target directory exists
mkdir -p ./val

# Extract file names, create directories in ./val, and move files while preserving structure
jq -r '.images[].file_name' val2018.json | while read -r file; do
  # Replace 'train_val2018' with '2018' in the file path
  src_file=$(echo "$file" | sed 's#^train_val2018#2018#')

  # Replace
  # Create the necessary directory structure under ./val
  mkdir -p ./val/$(dirname "$src_file")

  # Move the file to the corresponding directory in ./val with the modified structure
  mv "$src_file" "./val/$src_file"
done

# create train directory
mkdir -p ./train

# move train files (i.e., what's left in 2018)
mv "./2018" "./train/2018"