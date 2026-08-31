def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data