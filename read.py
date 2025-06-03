import os

def list_files_in_directory(directory_path):
    """Повертає список імен файлів у вказаній директорії."""
    try:
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        return files
    except FileNotFoundError:
        print(f"Директорію '{directory_path}' не знайдено.")
        return []
    except Exception as e:
        print(f"Сталася помилка: {e}")
        return []

# Приклад використання
if __name__ == "__main__":
    path = "data/cvrplib"  # Замінити на свою директорію
    file_list = list_files_in_directory(path)
    print("Файли в директорії:")
    for filename in file_list:
        print("–", filename)
