import os
from pathlib import Path

def get_validated_path(prompt_message: str, default_path: str) -> str:
    """Prompts the user for a path, validates it, and offers to create it."""
    while True:
        # Ask the user for the path, showing the default
        user_input = input(f"{prompt_message} [Default: {default_path}]: ").strip()
        
        # If the user just presses Enter, use the default
        path_str = user_input if user_input else default_path
        
        # Use pathlib for robust path handling
        path = Path(path_str).resolve() # .resolve() makes it an absolute path
        
        if path.exists():
            if path.is_dir():
                print(f"✅ Path found: {path}")
                return str(path)
            else:
                print(f"❌ Error: Path exists but is a file, not a directory. Please try again.")
        else:
            # If the path doesn't exist, ask to create it
            create_choice = input(f"🤔 Path does not exist. Create it? (y/n): ").lower()
            if create_choice == 'y':
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"✅ Directory created: {path}")
                    return str(path)
                except OSError as e:
                    print(f"❌ Error creating directory: {e}. Please check permissions and try again.")
            else:
                print("Creation skipped. Please enter a different path.")

def main():
    """Main function to run the setup wizard."""
    print("--- 🚀 Welcome to the Application Setup Wizard! ---")
    print("This will configure the paths for your models and data.\n")
    
    # Get the model directory path
    models_path = get_validated_path(
        prompt_message="Enter the path to your 'models' directory",
        default_path="./models" # Assumes models are in a top-level 'models' dir
    )
    
    print("-" * 20)
    
    # Get the data directory path
    data_path = get_validated_path(
        prompt_message="Enter the path to your 'data' directory",
        default_path="./backend/data" # Matches your existing structure
    )
    
    # Create the .env file
    env_content = f"MODEL_DIR={models_path}\nDATA_DIR={data_path}\n"
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("\n🎉 Success! Configuration saved to a new '.env' file.")
        print("You can now run the application locally or with Docker.")
    except IOError as e:
        print(f"\n❌ Critical Error: Could not write to .env file: {e}")

if __name__ == "__main__":
    main()