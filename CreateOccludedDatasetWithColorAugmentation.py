import numpy as np
import BboxTools.BboxTools as bbt
import os
import scipy.io
from PIL import Image
import cv2
import time
import pandas as pd
import argparse
from WBAugmenter import WBEmulator as wbAug
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="WB color augmenter")
    parser_augment = parser.add_argument
    parser_augment("--input_image_dir", default='./test/',
                    help="Training image directory")
    parser_augment("--output_image_dir", default='./OccludedPASCAL3D/',
                    help="augmented image directory")
    parser_augment("--out_number", type=int, default=5,
                    help="Number of color-augmented images for each input image")
    return parser.parse_args()

args = parse_args()

occ_libs_dir = './occluded_lib/occluder_libs_test_%s.npz'
occ_libs_name = ['large', 'medium', 'small']

path_save = args.output_image_dir
path_to_original_pascal3dp = args.input_image_dir

#categories = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train', 'boat', 'bottle', 'chair', 'diningtable',
#              'sofa', 'tvmonitor']
categories = ['test']

save_anno_path = path_save + '/annotations_grouped'
save_img_path = path_save + '/images'
save_list_path = path_save + '/lists'

source_list_path = path_to_original_pascal3dp + '/Image_sets/%s_imagenet_val.txt'
source_image_path = path_to_original_pascal3dp + '/Images/%s_imagenet'
source_anno_path = path_to_original_pascal3dp + '/Annotations/%s_imagenet'
source_mask_path = path_to_original_pascal3dp + '/obj_mask/%s'

if not os.path.exists(path_to_original_pascal3dp + 'obj_mask'):
    os.system('mv ./obj_mask ' + path_to_original_pascal3dp)

# 0: only start randomly, 1: only start in box, 2: using both mode
l_s_thr = 150000
occluding_modes_l = ['s', 'm', 'l', 'l','lm', 'll',   'lll']
start_mode_l =       [1,   2,   2,   2,  0,    2,       0]
occluding_modes_s = ['m', 'l', 'l','lm',  'mm',  ]
start_mode_s =       [2,   2,   2,  2,     2,    ]

#occluding_rate = [(0.3, 0.1), (0.5, 0.1), (0.7, 0.1), (0.3, 0.3), (0.5, 0.3), (0.7, 0.3), (0.3, 0.5), (0.5, 0.5),
#                  (0.7, 0.5), ]

occluding_rate = [(0.3, 0.5), ]

folder_name_list = ['%sFGL%d_BGL%d' % ('%s', j, i) for i in range(1, 2) for j in range(1, 2)]

allowed_var = 0.1

start_off_box_l = [md for md, sd in zip(occluding_modes_l, start_mode_l) if sd == 0 or sd == 2]
start_in_box_l = [md for md, sd in zip(occluding_modes_l, start_mode_l) if sd == 1 or sd == 2]
start_off_box_s = [md for md, sd in zip(occluding_modes_s, start_mode_s) if sd == 0 or sd == 2]
start_in_box_s = [md for md, sd in zip(occluding_modes_s, start_mode_s) if sd == 1 or sd == 2]

limited_trying_times = 40


def mix_masks(masks, boxes):
    back = np.zeros(boxes[0].boundary, dtype=bool)
    for box, mask in zip(boxes, masks):
        box.assign(back, np.logical_or(mask, box.apply(back)))
    return back


def mix_imgs(masks, boxes, imgs):
    back_im = np.zeros(tuple(boxes[0].boundary) + (3,), dtype=np.uint8)

    for box, mask, img in zip(boxes, masks, imgs):
        mask = mask.reshape(mask.shape + (1,))
        box.assign(back_im, mask * img + (1 - mask) * box.apply(back_im))
    return back_im


def merge_occ_image(masks, occluder_map, image):
    masks = np.expand_dims(masks, axis=2)
    return masks * occluder_map + (1 - masks) * image


def check_occ_ratio(mask_map, object_annotation_box):
    in_box_size = 0
    for in_box_ in object_annotation_box:
        in_box_size += in_box_.size
    out_box_size = mask_map.size - in_box_size

    in_box_value = 0
    for in_box_ in object_annotation_box:
        in_box_value += np.sum(in_box_.apply(mask_map))
    out_box_value = np.sum(mask_map) - in_box_value

    return in_box_value / in_box_size, out_box_value / out_box_size


def check_occ_ratio_seg(mask_map, mask_obj):
    mask_obj = mask_obj > 10
    mask_map = mask_map > 0.5
    in_box_size = np.sum(mask_obj)
    in_box_value = np.sum(np.logical_and(mask_obj, mask_map))

    out_box_size = np.sum(np.logical_not(mask_obj))
    out_box_value = np.sum(np.logical_and(np.logical_not(mask_obj), mask_map))

    return in_box_value / in_box_size, out_box_value / out_box_size


def process_inbox(shape, center, boundary):
    tem_box = bbt.box_by_shape(shape, center, boundary)
    tem_box_ = bbt.box_by_shape(shape, center)
    return tem_box_.box_in_box(tem_box)


def apply_n_occluder(occluder_boxes, img_shape, in_box, boundary_constraint=25, overlap_constraint=-5):
    box_list = []
    processing_list = [None for _ in range(len(occluder_boxes))]
    for i in range(len(occluder_boxes)):
        flag_ = False
        x = 0
        y = 0
        ti_ = 0
        if in_box and i == 0:
            while not flag_:
                flag_ = True
                x = np.random.randint(boundary_constraint, img_shape[0] - boundary_constraint, dtype=int)
                y = np.random.randint(boundary_constraint, img_shape[1] - boundary_constraint, dtype=int)

                flag__ = False
                
                for iter_box in in_box:
                    if iter_box.include((x, y)):
                        flag__ = True

                if not flag__:
                    flag_ = False
        else:
            while not flag_ and ti_ < 40:
                ti_ += 1
                flag_ = True
                x = np.random.randint(boundary_constraint, img_shape[0] - boundary_constraint, dtype=int)
                y = np.random.randint(boundary_constraint, img_shape[1] - boundary_constraint, dtype=int)
                for exist_box in box_list:
                    if exist_box.pad(overlap_constraint).include((x, y)):
                        flag_ = False

        center = (x, y)
        occluder_box = occluder_boxes[i]
        this_box = bbt.box_by_shape(occluder_box.shape, center, image_boundary=img_shape)
        box_list.append(this_box)

        if not occluder_box.size == this_box.size:
            processing_list[i] = process_inbox(occluder_box.shape, center, img_shape)

    return box_list, processing_list


def get_occ(required_type, occ_libs):
    out_boxes = []
    out_masks = []
    out_images = []
    out_cates = []
    t_flag = ''
    for t in required_type:
        if t == 'l':
            t_flag = t
            this_lib = occ_libs['large']
        elif t == 's':
            t_flag = t
            this_lib = occ_libs['small']
        else:
            t_flag = 'm'
            this_lib = occ_libs['medium']

        # randomly choose an occluded object from 1231 coco objects
        idx = np.random.randint(0, this_lib['masks'].shape[0], dtype=int)
        # out_cates.append(findcategory(idx))
        out_cates.append(t_flag + str(idx))
        out_boxes.append(this_lib['boxes'][idx])
        out_masks.append(this_lib['masks'][idx])
        out_images.append(this_lib['images'][idx])

    return out_boxes, out_masks, out_images, out_cates


def generate_one_img(img, box_anno, occ_libs, seg_anno):
    img_size = img.shape[0] * img.shape[1]
    if img_size > l_s_thr:
        using_start_off_box = start_off_box_l
        using_start_in_box = start_in_box_l
    else:
        using_start_off_box = start_off_box_s
        using_start_in_box = start_in_box_s

    tried_times = 0
    fully_filled = False
    filled_level = np.zeros(len(occluding_rate), dtype=bool)
    filled_score = np.zeros(len(occluding_rate), dtype=bool)

    using_box = np.zeros(len(occluding_rate), dtype=object)
    using_mask = np.zeros(len(occluding_rate), dtype=object)
    using_occluder = np.zeros(len(occluding_rate), dtype=object)
    using_cate = np.zeros(len(occluding_rate), dtype=object)

    while tried_times < limited_trying_times and not fully_filled:
        tried_times += 1

        boxes = []
        masks = []
        occluders = []
        ratios = []
        cates = []

        for working_mode in using_start_in_box:
            t_boxes, t_masks, t_images, t_cates = get_occ(working_mode, occ_libs)
            t_boxes, t_process = apply_n_occluder(t_boxes, img_shape=img.shape[0:2], in_box=box_anno)

            for i, proc in enumerate(t_process):
                if proc:
                    # print()
                    # print(proc)
                    # print(t_masks[i].shape)
                    # print(t_boxes[i])
                    t_masks[i] = proc.apply(t_masks[i])
                    t_images[i] = proc.apply(t_images[i])

            mask_map = mix_masks(t_masks, t_boxes)
            occluder_map = mix_imgs(t_masks, t_boxes, t_images)

            # ratios.append(check_occ_ratio(mask_map, box_anno))
            ratios.append(check_occ_ratio_seg(mask_map, seg_anno))
            masks.append(mask_map)
            boxes.append(t_boxes)
            occluders.append(occluder_map)
            cates.append(t_cates)

        for working_mode in using_start_off_box:
            t_boxes, t_masks, t_images, t_cates = get_occ(working_mode, occ_libs)
            t_boxes, t_process = apply_n_occluder(t_boxes, img_shape=img.shape[0:2], in_box=None)

            for i, proc in enumerate(t_process):
                if proc:
                    t_masks[i] = proc.apply(t_masks[i])
                    t_images[i] = proc.apply(t_images[i])

            mask_map = mix_masks(t_masks, t_boxes)
            occluder_map = mix_imgs(t_masks, t_boxes, t_images)

            ratios.append(check_occ_ratio(mask_map, box_anno))
            masks.append(mask_map)
            boxes.append(t_boxes)
            occluders.append(occluder_map)
            cates.append(t_cates)

        ratios_np = np.array(ratios)
        ratios_base = np.array(occluding_rate)

        # n * 2 - 9 * 2 -> n * 1 * 2 - 1 * 9 * 2 -> n * 9 * 2 -> all(2) -> any(n) -> 9
        legal_assign = np.any(
            np.all(np.abs(np.expand_dims(ratios_np, axis=1) - np.expand_dims(ratios_base, axis=0)) < allowed_var,
                   axis=2), axis=0)

        # n * 2 - 9 * 2 -> n * 1 * 2 - 1 * 9 * 2 -> n * 9 * 2 -> sum(2) -> argmin(n) -> 9
        dist_assign = np.argmin(
            np.sum(np.abs(np.expand_dims(ratios_np, axis=1) - np.expand_dims(ratios_base, axis=0)) + 10 * (np.abs(np.expand_dims(ratios_np, axis=1) - np.expand_dims(ratios_base, axis=0)) >= allowed_var), axis=2), axis=0)
        dist_score = np.min(
            np.sum(np.abs(np.expand_dims(ratios_np, axis=1) - np.expand_dims(ratios_base, axis=0)) + 10 * (np.abs(np.expand_dims(ratios_np, axis=1) - np.expand_dims(ratios_base, axis=0)) >= allowed_var), axis=2), axis=0)

        for i in range(len(occluding_rate)):  # 9
            if legal_assign[i]:
                if (not filled_level[i]) or dist_score[i] < filled_score[i]:
                    filled_level[i] = legal_assign[i]  # False -> True
                    filled_score[i] = dist_score[i]

                    idx_ = dist_assign[i]

                    using_box[i] = boxes[idx_]
                    using_mask[i] = masks[idx_]
                    using_occluder[i] = occluders[idx_]
                    using_cate[i] = cates[idx_]

        fully_filled = np.all(filled_level)

    image_out = np.zeros(len(occluding_rate), dtype=object)
    for i in range(len(occluding_rate)):  # 9
        if filled_level[i]:
            image_out[i] = (merge_occ_image(using_mask[i], using_occluder[i], img.copy()))
    return filled_level, image_out, using_mask, using_box, using_cate


def load_one_annotation(anno_path,imgs_shape):
    bbox_ = []
    with open(anno_path) as jnt:
        lines = jnt.readlines()
        for line in lines:
            sl = line.split(' ')
            #print(line)
            #print(int(sl[0]),float(sl[1]),float(sl[2]),float(sl[3]),float(sl[4][:-1]))
            width = imgs_shape[1] * float(sl[3])
            height = imgs_shape[0] * float(sl[4][:-1])
            lx = imgs_shape[1] * float(sl[1]) - 0.5 * width
            ly = imgs_shape[0] * float(sl[2]) - 0.5 * height
            #print(int(sl[0]),':',lx,ly,lx + width,ly+height)
            bbox_.append(np.array([lx,ly,lx + width,ly+height]))
    return bbox_


def generate_dataset(cate, file_list, img_dir, anno_dir, mask_dir, save_img_dir, save_list_dir, save_anno_dir, occ_lib_dir,
                     occ_lib_names, record_file):
    occ_libs = {}
    annotations = [{'source': [], 'mask': [], 'box': [], 'occluder_box': [], 'occluder_mask': [], 'occluder_category': []} for _ in range(len(occluding_rate))]
    img_list_ = ['' for _ in range(len(occluding_rate))]

    for k in occ_lib_names:
        occ_libs[k] = dict(np.load(occ_lib_dir % k, allow_pickle=True))
        # occ_libs[k] = dict(np.load('tem_lib.npz', allow_pickle=True))
        occ_libs[k]['boxes'] = bbt.from_numpy(occ_libs[k]['boxes'])

    save_img_dir_list = [os.path.join(save_img_dir, folder_name % cate) for folder_name in folder_name_list]
    for folder_name in save_img_dir_list:
        os.makedirs(folder_name, exist_ok=True)
    os.makedirs(save_list_dir, exist_ok=True)
    os.makedirs(save_anno_dir, exist_ok=True)

    for file_name in file_list:
        #print(file_name)
        #try:
            # open images in PASCAL3D+
        box = []
        img = np.array(Image.open(os.path.join(img_dir, file_name + '.jpg')))
        mask = np.array(Image.open(os.path.join(mask_dir, file_name + '.JPEG')))
        anno = load_one_annotation(os.path.join(anno_dir, file_name + '.txt'),img.shape)
            
        # different height and width for images in PASCAL3D+
        if not mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        for anno_file in anno:
            box.append(bbt.from_numpy(anno_file, image_boundary=img.shape[0:2], sorts=('x0', 'y0', 'x1', 'y1')))
        filled_, images_, masks_, boxes_, cates_ = generate_one_img(img, box, occ_libs, mask)
        # try:

        #except:
            #print('Unknown Expectations at %s' % file_name)
            #record_file.write('Unknown Expectations at %s\n' % file_name)
            #continue

        if not np.all(filled_):
            record_file.write('Unfill %s: ' % file_name)

        for i in range(filled_.size):
            if filled_[i]:
                #print(i)
                Image.fromarray(images_[i].astype(np.uint8)).save(os.path.join(save_img_dir_list[i], file_name + '.JPEG'))
                annotations[i]['source'].append(os.path.join(img_dir, file_name + '.jpg'))
                annotations[i]['occluder_mask'].append(masks_[i])
                annotations[i]['mask'].append(mask)
                annotations[i]['box'].append(bbt.list_box_to_numpy(box, save_image_boundary=True).ravel())
                annotations[i]['occluder_box'].append(bbt.list_box_to_numpy(boxes_[i], save_image_boundary=True))
                annotations[i]['occluder_category'].append(cates_[i])

                img_list_[i] += file_name + '.JPEG' + '\n'
            else:
                record_file.write(' %d' % i)

        if not np.all(filled_):
            record_file.write('\n')

    for name_, anno_ in zip(folder_name_list, annotations):
        np.savez(os.path.join(save_anno_dir, (name_ % cate) + '.npz'), **anno_)

    for name_, list_ in zip(folder_name_list, img_list_):
        with open(os.path.join(save_list_dir, (name_ % cate) + '.txt'), 'w') as file:
            file.write(list_)

    return

def processSyntheticOcclusion():
    for cate in categories:
        print('Start cate: ', cate)
        tem = open('generating_record_%s_1030.txt' % cate, 'w')
        file_list_ = open(source_list_path % cate).readlines()
        file_list_ = [tem.strip('\n') for tem in file_list_]
        source_image_path_ = source_image_path % cate
        source_anno_path_ = source_anno_path % cate
        source_mask_path_ = source_mask_path % cate
        #print((cate, file_list_, source_image_path_, source_anno_path_, source_mask_path_, save_img_path, save_list_path,
        #                 save_anno_path, occ_libs_dir, occ_libs_name, tem))
        generate_dataset(cate, file_list_, source_image_path_, source_anno_path_, source_mask_path_, save_img_path, save_list_path,
                         save_anno_path, occ_libs_dir, occ_libs_name, tem)
        tem.close()

def unzipingAnnotations():
    source_anno_path = path_save + '/annotations_grouped'
    save_anno_path = path_save + '/annotations'
    for cate in categories:
        for name_ in folder_name_list:
            print('Start cate: ' + name_ % cate)
            this_folder = os.path.join(save_anno_path, name_) % cate
            os.makedirs(this_folder, exist_ok=True)
            data = np.load(os.path.join(source_anno_path, (name_ % cate) + '.npz'), allow_pickle=True)

            source_list = data['source']
            mask_list = data['mask']
            box_list = data['box']
            occluder_box_list = data['occluder_box']
            occluder_mask = data['occluder_mask']
            occluder_category = data['occluder_category']

            for i in range(data['mask'].size):
                this_name = source_list[i].split('/')[-1].split('.')[0]
                np.savez(os.path.join(this_folder, this_name + '.npz'), source=source_list[i], mask=mask_list[i], box=box_list[i], occluder_mask=occluder_mask[i], occluder_box=occluder_box_list[i], occluder_category=occluder_category[i], category=cate, occluder_level=name_.strip('%s')[1])
  

def generalizingTrainingAnnotations():
    for count,cate in enumerate(categories):
        for i in folder_name_list:
            folder_name = i % cate
            print('Start cate: ' + folder_name)
            listPath = args.output_image_dir + '/lists/' + folder_name + '.txt'
            imgDir = args.output_image_dir + '/images/' + folder_name + '/'
            os.makedirs(imgDir.replace('images','labels'), exist_ok=True)
            imgList = open(listPath)
            lines = imgList.readlines()
            for line in lines:
                imgPath = imgDir + line.split('.')[-2] + '.JPEG'
                annoPath = imgPath.replace('images','annotations').replace('JPEG','npz')
                jsonPath = imgPath.replace('images','labels').replace('JPEG','txt')
                #print(jsonPath)
                imgJson = open(jsonPath,'a')
                anno = np.load(annoPath)
                oc_cate_list = anno['occluder_category']
                oc_box_list = anno['occluder_box']
                for index in range(len(oc_cate_list)):
                    oc_cate = oc_cate_list[index]
                    oc_box = oc_box_list[index]
                    dictPath = './dictionary/' + oc_cate[0] + '.csv'
                    idx = int(oc_cate[1:])
                    oc_label = pd.read_csv(dictPath).loc[idx:idx,('label')]
                    oc_label = oc_label.values[0]
                    y1,y2,x1,x2,height,width = oc_box
                    y_cr = ((y1+y2)/2)/height
                    x_cr = ((x1+x2)/2)/width
                    w = abs(x2-x1)/width
                    h = abs(y2-y1)/height
                    #print(x_cr,y_cr,w,h)
                    imgJson.write(str(oc_label) + ' ' + '%.6g' % x_cr  + ' ' + '%.6g' % y_cr + ' ' + '%.6g' % w + ' ' + '%.6g' % h + ' \n' )
                imgJson.close()
            imgList.close()

def chosenColorAugmentation():
    wbColorAug = wbAug.WBEmulator()  # create an instance of the WB emulator
    for count,cate in enumerate(categories):
        for i in folder_name_list:
            folder_name = i % cate
            print('Start cate: ' + folder_name)    
            cateDir = args.output_image_dir + '/colorAugmentedImages/' + folder_name + '/'
            imgDir = args.output_image_dir + '/images/' + folder_name + '/'   
            jsonDir = cateDir.replace('colorAugmentedImages','colorAugmentedlabels') 
            labelDir = args.output_image_dir + '/labels/' + folder_name + '/'
            os.makedirs(cateDir, exist_ok=True)
            os.makedirs(jsonDir, exist_ok=True)
            wbColorAug.batch_processing(imgDir, cateDir,
                                        args.out_number, 1)
            
            for root, dirs, imgs in os.walk(cateDir): 
                for img in imgs:
                    imgidx = img.split('_')[0]
                    imgname = img.split('.')[0]
                    try:
                        srcp = labelDir + imgidx + '.txt'
                        dstp = jsonDir + imgname + '.txt'
                        shutil.copy(srcp, dstp)
                    except Exception:
                        print(srcp + '--x-->' + dstp)


if __name__ == '__main__':
    
    print('processing synthetic occlusion ..')
    processSyntheticOcclusion()

    print('unziping annotations ..')
    unzipingAnnotations()
      
    print('generalizing darknet style annotations .. ')
    generalizingTrainingAnnotations()

    print('processing color augmentation')
    chosenColorAugmentation()





