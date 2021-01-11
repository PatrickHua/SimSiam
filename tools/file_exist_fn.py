import os



def file_exist_check(file_dir):
    
    if os.path.isdir(file_dir):
        for i in range(2, 1000):
            if not os.path.isdir(file_dir + f'({i})'):
                file_dir += f'({i})'
                break
    return file_dir





