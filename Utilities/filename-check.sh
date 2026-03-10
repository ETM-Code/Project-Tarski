#!/usr/bin/env bash
# Recursively checks for invalid Windows file names, skipping the current directory (.)

base_dir="${1:-.}"

# Regular expression for invalid characters in Windows filenames
invalid_chars='[\\/:*?"<>|]'

# Reserved Windows names (case-insensitive)
reserved_names='^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$'

echo "Checking for invalid Windows filenames in: $base_dir"
echo

# Find all files and directories, excluding the base directory itself
find "$base_dir" \( -type f -o -type d \) ! -path "$base_dir" | while read -r path; do
  filename="$(basename "$path")"

  # Skip empty names just in case
  [[ -z "$filename" ]] && continue

  # Check for invalid characters
  if [[ "$filename" =~ $invalid_chars ]]; then
    echo "❌ Invalid chars: $path"
    continue
  fi

  # Check for reserved names
  if [[ "$filename" =~ $reserved_names ]]; then
    echo "❌ Reserved name: $path"
    continue
  fi

  # Check for trailing space or period
  if [[ "$filename" =~ [\ .]$ ]]; then
    echo "❌ Ends with space or period: $path"
    continue
  fi
done

echo
echo "✅ Done checking."