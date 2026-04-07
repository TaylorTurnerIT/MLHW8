import zipfile

def analyze_docx_images(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as docx_zip:
            # Images in a .docx are stored in the 'word/media/' directory
            image_files = [f for f in docx_zip.namelist() if f.startswith('word/media/')]
            
            print(f"Total images found: {len(image_files)}\n")
            
            for img in image_files:
                info = docx_zip.getinfo(img)
                # The name is the part after 'word/media/'
                image_name = img.split('/')[-1] 
                
                print(f"Name: {image_name}")
                print(f"Internal Location: {img}")
                print(f"Size: {info.file_size / 1024:.2f} KB")
                print("-" * 30)
                
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except zipfile.BadZipFile:
        print(f"Error: The file '{file_path}' is not a valid .docx (ZIP) file.")

# Just replace with your actual file path
analyze_docx_images('Assignment 8.docx')