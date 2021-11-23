import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True)
save_folder = '/model/output/table'

def test(img_path):
    # img_path = '/dataset/0002514084.jpg'
    img = cv2.imread(img_path)
    result = table_engine(img)
    save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

    for line in result:
        line.pop('img')
        print(line)

    from PIL import Image

    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf' # PaddleOCR下提供字体包
    image = Image.open(img_path).convert('RGB')
    im_show = draw_structure_result(image, result, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save('/model/res_' + os.path.basename(img_path))

file_type_list = ['jpg','png']

def tests(img_dir):
    files = os.listdir(img_dir)
    for fi in files:
        fi_d = os.path.join(img_dir, fi)
        if os.path.isdir(fi_d):
            print(os.path.join(img_dir, fi_d))
            tests(fi_d)
        else:
            file_type = fi.split('.')[-1]
            if (file_type in file_type_list):
                print(os.path.join(img_dir, fi_d))  # 递归遍历/root目录下所有文件
                test(os.path.join(img_dir, fi_d))

if __name__ == "__main__":
    tests('/dataset')