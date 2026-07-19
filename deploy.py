from huggingface_hub import HfApi
import sys

SPACE_ID = "Shivaraj22/Photomax"

def main():
    print(f"🚀 Preparing to upload to Hugging Face Space: {SPACE_ID}...\n")
    print("To upload, you need a Hugging Face Access Token with 'Write' permissions.")
    print("Get one here: https://huggingface.co/settings/tokens")
    token = input("Paste your Access Token here and press Enter: ").strip()
    
    if not token:
        print("❌ Token cannot be empty. Exiting.")
        return

    try:
        api = HfApi(token=token)
        
        print("\nUploading files... (This might take a minute)")
        api.upload_folder(
            folder_path=".",
            repo_id=SPACE_ID,
            repo_type="space",
            token=token,
            ignore_patterns=["output/*", "__pycache__/*", "*.pyc", ".git/*", "temp/*", "deploy.py"]
        )
        print("\n✅ Successfully uploaded all files!")
        print(f"🌐 View your live app here: https://huggingface.co/spaces/{SPACE_ID}")
        
    except Exception as e:
        print("\n❌ Upload failed!")
        print("-" * 40)
        print("Error details:", str(e))
        print("-" * 40)
        print("\nMake sure your token is correct and has 'Write' permissions!")

if __name__ == "__main__":
    main()
