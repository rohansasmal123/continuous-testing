import os

def create_dir(root_dir):
    working_dir=os.path.join(root_dir, 'caa')
    if (not os.path.exists(working_dir)):
        os.makedirs(working_dir)
        os.makedirs(os.path.join(working_dir, 'rp'))
        os.makedirs(os.path.join(working_dir, 'rp','model'))
        os.makedirs(os.path.join(working_dir, 'rp','rfd'))
        os.makedirs(os.path.join(working_dir, 'rp','monitor'))
        os.makedirs(os.path.join(working_dir, 'litm'))
        os.makedirs(os.path.join(working_dir, 'litm','analysis'))
        os.makedirs(os.path.join(working_dir, 'litm','monitor'))
    else:
        print("Directory Exists")


if __name__ == "__main__":
    root_dir='C:\\directory_creation_testing\\'
    create_dir(root_dir)