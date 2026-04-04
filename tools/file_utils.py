import json
import os
import random
import shutil
import string


def walk_files(filedir):
    file_list = []
    for root, _, files in os.walk(filedir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def random_string(num):
    return ''.join(random.sample(string.ascii_letters + string.digits, num))


def is_image(path):
    return os.path.splitext(path)[1].lower() in ['.jpg', '.png', '.jpeg', '.bmp']


def is_video(path):
    return os.path.splitext(path)[1].lower() in ['.mp4', '.flv', '.avi', '.mov', '.mkv', '.wmv', '.rmvb', '.mts']


def filter_images(paths):
    return [path for path in paths if is_image(path)]


def filter_videos(paths):
    return [path for path in paths if is_video(path)]


def filter_dirs(paths):
    return [path for path in paths if os.path.isdir(path)]


def write_log(path, log, isprint=False):
    with open(path, 'a+', encoding='utf-8') as file:
        file.write(log + '\n')
    if isprint:
        print(log)


def save_json(path, data_dict):
    json_str = json.dumps(data_dict)
    with open(path, 'w+', encoding='utf-8') as file:
        file.write(json_str)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        txt_data = file.read()
    return json.loads(txt_data)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def ensure_parent_dir(path):
    ensure_dir(os.path.dirname(path))


def second2stamp(s):
    h = int(s / 3600)
    s = int(s % 3600)
    m = int(s / 60)
    s = int(s % 60)
    return "%02d:%02d:%02d" % (h, m, s)


def stamp2second(stamp):
    substamps = stamp.split(':')
    return int(substamps[0]) * 3600 + int(substamps[1]) * 60 + int(substamps[2])


def counttime(start_time, current_time, now_num, all_num):
    used_time = int(current_time - start_time)
    all_time = int(used_time / now_num * all_num)
    return second2stamp(used_time) + '/' + second2stamp(all_time)


def get_bar(percent, num=25):
    bar = '['
    for i in range(num):
        if i < round(percent / (100 / num)):
            bar += '#'
        else:
            bar += '-'
    bar += ']'
    return bar + ' ' + "%.2f" % percent + '%'


def copyfile(src, dst):
    try:
        shutil.copyfile(src, dst)
    except Exception as exc:
        print(exc)


def opt2str(opt):
    message = ''
    message += '---------------------- Options --------------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<35}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    return message


Traversal = walk_files
randomstr = random_string
is_img = is_image
is_imgs = filter_images
is_videos = filter_videos
is_dirs = filter_dirs
writelog = write_log
savejson = save_json
loadjson = load_json
makedirs = ensure_dir
