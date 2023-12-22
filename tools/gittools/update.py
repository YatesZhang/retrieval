import os
import shutil
import argparse
try:
    import rich
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-i", "https://mirrors.aliyun.com/pypi/simple/"])
from rich import print

def copy_files(source, destination):
    for item in os.listdir(source):
        item_path = os.path.join(source, item)
        if item.startswith('.') or 'git' in item:
            print('skip [yellow]{item}[/yellow]'.format(item=item_path))
            continue
        if os.path.isfile(item_path):
            print('copy [green]{item}[/green] to {dest}'.format(item=item_path, dest=destination))
            shutil.copy2(item_path, destination)
        elif os.path.isdir(item_path):
            new_destination = os.path.join(destination, item)
            os.makedirs(new_destination, exist_ok=True)
            print('dive into [blue]{item}[/blue]'.format(item=item_path))
            copy_files(item_path, new_destination)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--destination', type=str, required=True)
    args = parser.parse_args()
    """ 
        e.g.
            args.destination = $ROOT/baidu/zhongce-aidata-algorithm/retrieval
    """
    args.destination = os.path.join(args.destination, "flamingo")
    if not os.path.exists(args.destination):
        print('create [red]{dest}[/red]'.format(dest=args.destination))
        os.makedirs(args.destination, exist_ok=True)
        # os.wait(1)
    copy_files(args.source, args.destination)
    if not os.path.exists(args.source):
        raise FileExistsError('source not exists')
    print('done')

if __name__ == '__main__':
    main()