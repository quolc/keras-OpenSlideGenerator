import random
import math
import os
import copy

import numpy as np
import cv2
import tripy
import pyclipper
from openslide import OpenSlide

from skimage.color import rgb2hed, hed2rgb

import keras


class SimpleOpenSlideGenerator(object):

    def __init__(self, color_matching=None):
        self.use_color_matching = False
        if color_matching is not None:
            self.match_color_prepare(cv2.imread(color_matching) / 255.0)
            self.use_color_matching = True

    def fetch_patches_from_slide(self, wsi_filepath, count, src_size=512, patch_size=512, tissue_threshold=0.8,
                                 blur=0, he_augmentation=False):
        slide = OpenSlide(wsi_filepath)

        # load image with lower resolution
        desirable_long_edge = 2000

        level_downsample = 0
        for i, (w, h) in enumerate(slide.level_dimensions):
            if abs(max(w, h) - desirable_long_edge) < \
                    abs(max(slide.level_dimensions[level_downsample]) - desirable_long_edge):
                level_downsample = i
        magnification = slide.level_downsamples[level_downsample]
        image_downsample = slide.read_region((0, 0), level_downsample,
                                             (slide.level_dimensions[level_downsample]))

        # Otsu binarization
        
        # set transparent region to white
        alpha = np.asarray(image_downsample, dtype=np.uint8)[:,:,3]
        
        src = np.average(np.asarray(image_downsample, dtype=np.uint8)[:,:,:3], axis=2)
        src[alpha == 0] = 255
        src = 255 - cv2.convertScaleAbs(src)
        th, binarized = cv2.threshold(src, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # dilation
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binarized, kernel, iterations=1)

        ret_images = []
        src_downsampled_size = 512 / magnification
        padding = int(src_size / 2 ** 0.5 / magnification) + 10
        for i in range(count):
            while True:
                cx = random.randint(padding, slide.level_dimensions[level_downsample][0] - padding)
                cy = random.randint(padding, slide.level_dimensions[level_downsample][1] - padding)
                angle = random.random() * 2 * math.pi

                # crop
                crop_size = int(src_downsampled_size * 2 ** 0.5 * max(abs(math.cos(angle)), abs(math.sin(angle))))
                cropped = dilated[int(cy - crop_size / 2):int(cy + crop_size / 2),
                          int(cx - crop_size / 2):int(cx + crop_size / 2)]
                mat = cv2.getRotationMatrix2D((crop_size / 2, crop_size / 2),
                                              45 + 360 * angle / (2 * math.pi), 1)
                rotated = cv2.warpAffine(cropped, mat, (crop_size, crop_size))

                result = rotated[
                         int(crop_size / 2 - src_downsampled_size / 2):int(crop_size / 2 + src_downsampled_size / 2), \
                         int(crop_size / 2 - src_downsampled_size / 2):int(crop_size / 2 + src_downsampled_size / 2)]
                if np.average(result) / 255 > tissue_threshold:
                    break

            # transform to raw scale
            cx = cx * magnification
            cy = cy * magnification

            # real cropping
            crop_size = int(src_size * 2 ** 0.5 * max(abs(math.cos(angle)), abs(math.sin(angle))))
            cropped = np.asarray(slide.read_region((int(cx - crop_size / 2), int(cy - crop_size / 2)),
                                                   0, (crop_size, crop_size)), dtype=np.float32)[:, :, :3]
            mat = cv2.getRotationMatrix2D((crop_size / 2, crop_size / 2),
                                          45 + 360 * angle / (2 * math.pi), 1)
            rotated = cv2.warpAffine(cropped, mat, (crop_size, crop_size))

            result = rotated[int(crop_size / 2 - src_size / 2):int(crop_size / 2 + src_size / 2), \
                     int(crop_size / 2 - src_size / 2):int(crop_size / 2 + src_size / 2)]
            result = cv2.resize(result, (patch_size, patch_size)).transpose((2, 0, 1)) / 255

            # color matching
            if self.use_color_matching:
                result = self.match_color(result.transpose(1, 2, 0)).transpose(2, 0, 1)

            # blurring effect
            if blur > 0:
                blur_size = random.randint(1, blur)
                result = cv2.blur(result.transpose(1, 2, 0), (blur_size, blur_size)).transpose((2, 0, 1))

            if he_augmentation:
                hed = rgb2hed(np.clip(result.transpose(1, 2, 0), -1.0, 1.0))
                ah = 0.95 + random.random() * 0.1
                bh = -0.05 + random.random() * 0.1
                ae = 0.95 + random.random() * 0.1
                be = -0.05 + random.random() * 0.1
                hed[:, :, 0] = ah * hed[:, :, 0] + bh
                hed[:, :, 1] = ae * hed[:, :, 1] + be
                result = hed2rgb(hed).transpose(2, 0, 1)

            ret_images.append(np.clip(result, 1e-7, 1.0-1e-7))

        return ret_images

    # Neural color transfer by github.com/htoyryla, and github.com/ProGamerGov
    # https://github.com/ProGamerGov/Neural-Tools
    # https://github.com/leongatys/NeuralImageSynthesis/blob/master/ExampleNotebooks/ScaleControl.ipynb
    def match_color_prepare(self, source_img, eps=1e-5):
        self.mu_s = source_img.mean(0).mean(0)
        s = source_img - self.mu_s
        s = s.transpose(2, 0, 1).reshape(3, -1)
        self.Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])
        self.chol_s = np.linalg.cholesky(self.Cs)

        eva_s, eve_s = np.linalg.eigh(self.Cs)
        self.Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)

    def match_color(self, target_img, mode='pca', eps=1e-5):
        '''
        Matches the colour distribution of the target image to that of the source image
        using a linear transform.
        Images are expected to be of form (w,h,c) and float in [0,1].
        Modes are chol, pca or sym for different choices of basis.
        '''
        mu_t = target_img.mean(0).mean(0)
        t = target_img - mu_t
        t = t.transpose(2, 0, 1).reshape(3, -1)
        Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
        if mode == 'chol':
            chol_t = np.linalg.cholesky(Ct)
            ts = self.chol_s.dot(np.linalg.inv(chol_t)).dot(t)
        if mode == 'pca':
            eva_t, eve_t = np.linalg.eigh(Ct)
            Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
            ts = self.Qs.dot(np.linalg.inv(Qt)).dot(t)
        if mode == 'sym':
            eva_t, eve_t = np.linalg.eigh(Ct)
            Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
            Qt_Cs_Qt = Qt.dot(self.Cs).dot(Qt)
            eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
            QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
            ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
        matched_img = ts.reshape(*target_img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
        matched_img += self.mu_s
        matched_img[matched_img > 1] = 1
        matched_img[matched_img < 0] = 0
        return matched_img.astype(np.float32)

class OpenSlideGenerator(object):
    fetch_modes = ['area', 'slide', 'label', 'label-slide']

    def __init__(self, path, root, src_size, patch_size, fetch_mode='area', label_to_use=0,
                 rotation=True, flip=False, blur=0, he_augmentation=False, scale_augmentation=False,
                 color_matching=None,
                 dump_patch=None, verbose=1):
        self.path = path
        self.root = root
        self.src_size = src_size
        self.patch_size = patch_size
        self.fetch_mode = fetch_mode
        self.label_to_use = label_to_use
        if self.fetch_mode not in OpenSlideGenerator.fetch_modes:
            raise Exception('invalid fetch_mode %r' % self.fetch_mode)
        self.rotation = rotation
        self.flip = flip
        self.blur = blur
        self.he_augmentation = he_augmentation
        self.scale_augmentation = scale_augmentation
        self.dump_patch = dump_patch
        self.verbose = verbose

        self.use_color_matching = False
        if color_matching is not None:
            self.match_color_prepare(cv2.imread(color_matching) / 255.0)
            self.use_color_matching = True

        self.slide_names = []
        self.labels = []          # labels[LABEL_CATEGORY][LABEL]
        self.label_of_region = []
        self.structure = []
        self.shifted_structure = []
        self.triangulation = []
        self.regions_of_label = [] # dict()
        self.regions_of_label_slide = [] # dict()

        self.src_sizes = []

        self.total_weight = 0
        self.slide_weights = []     # total weight of a slide
        self.label_weights = []     # total weight of a label
        self.label_slide_weights = [] # total weight of regions of certain label in a slide.

        self.weights = []          # overall weight
        self.weights_in_slide = [] # weight in a slide
        self.weights_in_label = [] # weight in the same label
        self.weights_in_label_slide = [] # weight in the same label and slide

        self.total_area = 0
        self.slide_areas = []  # total area of a slide
        self.label_areas = []  # total area of a label

        self.total_triangles = 0
        self.slide_triangles = []     # total triangle number for each slide
        self.label_triangles = []     # total triangle number for each label
        self.label_slide_triangles = [] # total triangule number for each label-slide pair

        self.serialized_index = []             # serialized_index[ID] -> (SLIDE_ID, REGION_ID, TRIANGLE_ID)
        self.serialized_index_slide = []       # serialized_index_slide[SLIDE_ID][ID] -> (REGION_ID, TRIANGLE_ID)
        self.serialized_index_label = []       # serialized_index_label[label][ID] -> (SLIDE_ID, REGION_ID, TRIANGLE_ID)
        self.serialized_index_label_slide = [] # *[label][SLIDE_ID][ID] -> (REGION_ID, TRIANGLE_ID)

        # variables for Walker's alias method
        self.a_area = []
        self.p_area = []
        self.a_slide = []
        self.p_slide = []
        self.a_label = []
        self.p_label = []
        self.a_label_slide = []
        self.p_label_slide = []

        # OpenSlide objects
        self.slides = []

        # log
        self.fetch_count = [] # region-wise

        # states for parsing input text file
        # 0: waiting for new file entry
        # 1: waiting for region header or svs entry
        # 2: reading a region
        state = 0
        left_points = 0
        label_buffer = [] # label_buffer[SLIDE_ID][REGION_ID][LABEL_CATEGORY]
        slide_id = -1
        region_id = -1
        with open(path) as f:
            for line in map(lambda l: l.split("#")[0].strip(), f.readlines()):
                if len(line) == 0:
                    continue
                is_svs_line = (line[0] == "@")
                if is_svs_line:
                    line = line[1:]
                else:
                    try:
                        items = list(map(int, line.split()))
                    except Exception:
                        raise Exception('invalid dataset file format!')
                if state == 0:
                    if not is_svs_line:
                        raise Exception('invalid dataset file format!')

                    slide_id += 1
                    region_id = 0
                    svs_name = line.split()[0]
                    if len(line.split()) > 1 and line.split()[1].isdigit:
                        svs_src_size = int(line.split()[1])
                    else:
                        svs_src_size = self.src_size

                    self.slide_names.append(svs_name)
                    self.src_sizes.append(svs_src_size)
                    self.structure.append([])
                    label_buffer.append([])
                    state = 1
                elif state == 1:
                    if is_svs_line: # new file
                        slide_id += 1
                        region_id = 0
                        svs_name = line.split()[0]
                        if len(line.split()) > 1 and line.split()[1].isdigit:
                            svs_src_size = int(line.split()[1])
                        else:
                            svs_src_size = self.src_size # default src_size

                        self.slide_names.append(svs_name)
                        self.src_sizes.append(svs_src_size)
                        self.structure.append([])
                        label_buffer.append([])
                        state = 1
                    else: # region header
                        label_buffer[slide_id].append([])
                        for label_cat, label in enumerate(items[:-1]):
                            label_buffer[slide_id][region_id].append(label)
                            # handling newly found label category
                            if len(self.labels) < label_cat+1:
                                self.labels.append([])
                                self.regions_of_label.append(dict())
                                self.regions_of_label_slide.append(dict())
                                self.a_label.append(dict())
                                self.p_label.append(dict())
                                self.a_label_slide.append(dict())
                                self.p_label_slide.append(dict())
                                self.label_areas.append(dict())
                                self.label_weights.append(dict())
                                self.label_slide_weights.append(dict())
                                self.label_triangles.append(dict())
                                self.label_slide_triangles.append(dict())
                                self.serialized_index_label.append(dict())
                                self.serialized_index_label_slide.append(dict())
                            # handling newly found label
                            if label not in self.labels[label_cat]:
                                self.labels[label_cat].append(label)
                                self.regions_of_label[label_cat][label] = []
                                self.a_label[label_cat][label] = []
                                self.p_label[label_cat][label] = []
                                self.a_label_slide[label_cat][label] = []
                                self.p_label_slide[label_cat][label] = []
                                self.label_areas[label_cat][label] = 0
                                self.label_weights[label_cat][label] = 0
                                self.label_slide_weights[label_cat][label] = []
                                self.label_triangles[label_cat][label] = 0
                                self.label_slide_triangles[label_cat][label] = []
                                self.serialized_index_label[label_cat][label] = []
                                self.serialized_index_label_slide[label_cat][label] = []
                            self.regions_of_label[label_cat][label].append((slide_id, region_id))
                        self.structure[slide_id].append([])
                        left_points = items[-1]
                        if items[-1] < 3:
                            raise Exception('regions should consist of more than 3 points!')
                        state = 2
                elif state == 2:
                    if is_svs_line or len(items) != 2:
                        raise Exception('invalid dataset file format!')
                    self.structure[-1][-1].append((items[0], items[1]))
                    left_points -= 1
                    if left_points == 0:
                        state = 1
                        region_id += 1
        if state != 1: # dataset file should end with a completed region entry
            raise Exception('invalid dataset file format!')

        # set label_of_region
        for label_cat in range(len(self.labels)):
            self.label_of_region.append([])
            for slide_id, label_of_regions in enumerate(label_buffer):
                self.label_of_region[label_cat].append([])
                for region_id, label_of_categories in enumerate(label_of_regions):
                    if label_cat < len(label_of_categories):
                        self.label_of_region[label_cat][slide_id].append(label_of_categories[label_cat])
                    else:
                        self.label_of_region[label_cat][slide_id].append(-1)

        # calculate regions_of_label_slide
        for label_cat in range(len(self.labels)):
            for label in self.labels[label_cat]:
                self.regions_of_label_slide[label_cat][label] = []
                for i in range(len(self.structure)):
                    self.regions_of_label_slide[label_cat][label].append([])

        # prepare shifted (offset) structure
        self.shifted_structure = copy.deepcopy(self.structure)
        for i in range(len(self.shifted_structure)):
            for j in range(len(self.shifted_structure[i])):
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(self.shifted_structure[i][j], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                # offsetting
                shifted_region = pco.Execute(-self.src_sizes[i] / 2)
                # shifted_region = pco.Execute(0)
                if len(shifted_region) == 0:
                    self.shifted_structure[i][j] = [] # collapsed to a point
                else:
                    self.shifted_structure[i][j] = shifted_region[0]
                    for label_cat in range(len(self.labels)):
                        label = self.label_of_region[label_cat][i][j]
                        if label != -1:
                            self.regions_of_label_slide[label_cat][label][i].append(j)

        # load slides
        for name in self.slide_names:
            try:
                self.slides.append(OpenSlide(os.path.join(self.root, name)))
            except Exception as exc:
                raise Exception('an error has occurred while reading slide "{}"'.format(name))

        for label_cat in range(len(self.labels)):
            self.weights_in_label.append([])
            self.weights_in_label_slide.append([])

        # region triangulation
        total_region_count = 0
        for i in range(len(self.shifted_structure)):
            self.triangulation.append([])
            self.weights.append([])
            self.weights_in_slide.append([])
            for label_cat in range(len(self.labels)):
                self.weights_in_label[label_cat].append([])
                self.weights_in_label_slide[label_cat].append([])
            self.serialized_index_slide.append([])
            self.a_slide.append([])
            self.p_slide.append([])
            self.slide_weights.append(0)
            self.slide_triangles.append(0)
            w, h = self.slides[i].dimensions # slide width/height

            for label_cat in range(len(self.labels)):
                for label in self.labels[label_cat]:
                    self.a_label_slide[label_cat][label].append([])
                    self.p_label_slide[label_cat][label].append([])
                    self.serialized_index_label_slide[label_cat][label].append([])
                    self.label_slide_weights[label_cat][label].append(0)
                    self.label_slide_triangles[label_cat][label].append(0)

            for j in range(len(self.shifted_structure[i])):
                region = self.shifted_structure[i][j]
                total_region_count += 1

                # triangulation
                self.triangulation[-1].append(tripy.earclip(region))

                for x, y in region:
                    if w < x or h < y:
                        raise Exception('invalid polygon vertex position (%d, %d) in %s!' % (x, y, self.slide_names[i]))

                # triangle area calculation
                self.weights[i].append([])
                self.weights_in_slide[i].append([])
                self.slide_triangles[i] += len(self.triangulation[i][j])

                for label_cat in range(len(self.labels)):
                    self.weights_in_label[label_cat][i].append([])
                    self.weights_in_label_slide[label_cat][i].append([])
                    label = self.label_of_region[label_cat][i][j]
                    if label != -1:
                        self.label_triangles[label_cat][label] += len(self.triangulation[i][j])
                        self.label_slide_triangles[label_cat][label][i] += len(self.triangulation[i][j])

                for (x1, y1), (x2, y2), (x3, y3) in self.triangulation[i][j]:
                    a = x2 - x1
                    b = y2 - y1
                    c = x3 - x1
                    d = y3 - y1
                    area = abs(a*d - b*c)/2
                    weight = area / (self.src_sizes[i]**2)
                    self.weights[i][j].append(weight)
                    self.weights_in_slide[i][j].append(weight)
                    self.total_weight += weight
                    self.slide_weights[i] += weight
                    for label_cat in range(len(self.labels)):
                        self.weights_in_label[label_cat][i][j].append(weight)
                        self.weights_in_label_slide[label_cat][i][j].append(weight)
                        label = self.label_of_region[label_cat][i][j]
                        if label != -1:
                            self.label_weights[label_cat][label] += weight
                            self.label_slide_weights[label_cat][label][i] += weight

        # calculate raw slide size
        for i in range(len(self.structure)):
            self.slide_areas.append(0)

            for j in range(len(self.structure[i])):
                region = self.structure[i][j]
                triangles = tripy.earclip(region)
                for (x1, y1), (x2, y2), (x3, y3) in triangles:
                    a = x2 - x1
                    b = y2 - y1
                    c = x3 - x1
                    d = y3 - y1
                    area = abs(a*d - b*c)/2
                    self.total_area += area
                    self.slide_areas[-1] += area
                    for label_cat in range(len(self.labels)):
                        label = self.label_of_region[label_cat][i][j]
                        if label != -1:
                            self.label_areas[label_cat][label] += area

        # calculate the set of triangle weights for each fetch_mode
        for i in range(len(self.weights)): # svs
            for j in range(len(self.weights[i])): # region
                for k in range(len(self.weights[i][j])): # triangle
                    self.weights[i][j][k] /= self.total_weight
                    self.weights_in_slide[i][j][k] /= self.slide_weights[i]
                    self.serialized_index.append((i, j, k))
                    self.serialized_index_slide[i].append((j, k))
                    for label_cat in range(len(self.labels)):
                        label = self.label_of_region[label_cat][i][j]
                        if label != -1:
                            self.weights_in_label[label_cat][i][j][k] /= self.label_weights[label_cat][label]
                            if self.label_slide_weights[label_cat][label][i] > 0:
                                self.weights_in_label_slide[label_cat][i][j][k] /= self.label_slide_weights[label_cat][label][i]
                            self.serialized_index_label[label_cat][label].append((i, j, k))
                            self.serialized_index_label_slide[label_cat][label][i].append((j, k))
                    self.total_triangles += 1

        # Walker's alias method for weighted sampling of triangles
        def walker_precomputation(probs):
            EPS = 1e-10

            # normalization
            prob_sum = 0
            for prob in probs:
                prob_sum += prob
            prob_sum *= (1 + EPS)
            for i in range(len(probs)):
                probs[i] /= prob_sum

            a = [-1] * len(probs)
            p = [0] * len(probs)
            fixed = 0
            while fixed < len(probs):
                # block assignment of small items
                for i in range(len(probs)):
                    if p[i] == 0 and probs[i] * len(probs) <= (1.0 + EPS):
                        p[i] = probs[i] * len(probs)
                        probs[i] = 0
                        fixed += 1
                # packing of large items
                for i in range(len(probs)):
                    if probs[i] * len(probs) > 1.0:
                        for j in range(len(probs)):
                            if p[j] != 0 and a[j] == -1:
                                a[j] = i
                                probs[i] -= (1.0 - p[j]) / len(probs)
                            if probs[i] * len(probs) <= (1.0 + EPS):
                                break

            # fill -1 a
            for i in range(len(probs)):
                if a[i] == -1:
                    a[i] = i
            return a, p

        # pre-computation for 'area' mode - all triangles are treated in single array
        probs = []
        for i in range(len(self.weights)): # svs
            for j in range(len(self.weights[i])): # region
                for k in range(len(self.weights[i][j])): # triangle
                    probs.append(self.weights[i][j][k])
        self.a_area, self.p_area = walker_precomputation(probs)

        # pre-computaiton for 'slide' mode
        for i in range(len(self.weights)): # svs
            probs = []
            for j in range(len(self.weights[i])): # region
                for k in range(len(self.weights[i][j])): # triangle
                    probs.append(self.weights_in_slide[i][j][k])

            self.a_slide[i], self.p_slide[i] = walker_precomputation(probs)

        # pre-computation for 'label' mode
        for label_cat in range(len(self.labels)):
            for label in self.labels[label_cat]:
                probs = []
                for slide_id, region_id in self.regions_of_label[label_cat][label]:
                    for tri_id in range(len(self.weights_in_label[label_cat][slide_id][region_id])):
                        probs.append(self.weights_in_label[label_cat][slide_id][region_id][tri_id])

                self.a_label[label_cat][label], self.p_label[label_cat][label] = walker_precomputation(probs)

        # pre-computation for 'label-slide' mode
        for label_cat in range(len(self.labels)):
            for label in self.labels[label_cat]:
                for slide_id in range(len(self.weights)):
                    probs = []
                    for region_id in self.regions_of_label_slide[label_cat][label][slide_id]:
                        for tri_id in range(len(self.weights_in_label_slide[label_cat][slide_id][region_id])):
                            probs.append(self.weights_in_label_slide[label_cat][slide_id][region_id][tri_id])
                    self.a_label_slide[label_cat][label][slide_id], self.p_label_slide[label_cat][label][slide_id] = walker_precomputation(probs)

        if self.verbose > 0:
            print('loaded {} slide(s).'.format(len(self.shifted_structure)))
            for i in range(len(self.shifted_structure)):
                print('[{}] {}'.format(i, self.slide_names[i]))
                print('- {} regions'.format(len(self.shifted_structure[i])))
                print('- {} px2'.format(self.slide_areas[i]))
                print('- patch scale:', self.src_sizes[i])
                weight_sum = 0
                for region in self.weights[i]:
                    for w_triangle in region:
                        weight_sum += w_triangle
                print('- fetch probability (area mode):', weight_sum)
            print('there are total {} regions.'.format(total_region_count, int(self.total_area)))

        self.patch_per_epoch = 0
        for i in range(len(self.src_sizes)):
            self.patch_per_epoch += self.slide_areas[i] / (self.src_sizes[i] ** 2)
            self.patch_per_epoch = int(self.patch_per_epoch)
        if self.verbose > 0:
            print('patches per epoch is set to {}.'.format(self.patch_per_epoch))
            print()

        self.reset_fetch_count()

    def reset_fetch_count(self):
        self.fetch_count = []
        for slide in self.structure:
            self.fetch_count.append([])
            for _ in slide:
                self.fetch_count[-1].append(0)
        self.total_fetch_count = 0
        self.total_loop_count = 0

    def __len__(self):
        return self.patch_per_epoch

    # get random triangle index from all triangles in the dataset.
    def _get_random_index_all(self):
        q = random.random() * self.total_triangles
        i = int(q)
        if q - i < self.p_area[i]:
            return self.serialized_index[i]
        else:
            return self.serialized_index[self.a_area[i]]

    # get random triangle index from a specific slide.
    def _get_random_index_slide(self, slide_id):
        q = random.random() * self.slide_triangles[slide_id]
        i = int(q)
        if q - i < self.p_slide[slide_id][i]:
            return self.serialized_index_slide[slide_id][i]
        else:
            return self.serialized_index_slide[slide_id][self.a_slide[slide_id][i]]

    # get random triangle index which has a specific label.
    def _get_random_index_label(self, label):
        q = random.random() * self.label_triangles[self.label_to_use][label]
        i = int(q)
        if q - i < self.p_label[self.label_to_use][label][i]:
            return self.serialized_index_label[self.label_to_use][label][i]
        else:
            return self.serialized_index_label[self.label_to_use][label][self.a_label[self.label_to_use][label][i]]

    # get random triangle index which has specific a label in a slide.
    def _get_random_index_label_slide(self, label, slide):
        q = random.random() * self.label_slide_triangles[self.label_to_use][label][slide]
        i = int(q)
        if q - i < self.p_label_slide[self.label_to_use][label][slide][i]:
            return self.serialized_index_label_slide[self.label_to_use][label][slide][i]
        else:
            return self.serialized_index_label_slide[self.label_to_use][label][slide][self.a_label_slide[self.label_to_use][label][slide][i]]

    # winding-number algorithm
    def point_in_region(self, slide_id, region_id, cx, cy):
        def is_left(p0, p1, p2):
            return ((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))

        poly = self.structure[slide_id][region_id] # point-in-region problem for corner: should be judged for raw structure
        winding_number = 0
        for i in range(len(poly)):
            if poly[i][1] <= cy:
                if poly[(i+1)%len(poly)][1] > cy:
                    if is_left(poly[i], poly[(i+1)%len(poly)], (cx, cy)) > 0:
                        winding_number += 1
            else:
                if poly[(i+1)%len(poly)][1] <= cy:
                    if is_left(poly[i], poly[(i+1)%len(poly)], (cx, cy)) < 0:
                        winding_number -= 1
        return winding_number != 0

    def get_example(self, i):
        loop_count = 0
        while True:
            # select a triangle by the current fetch-mode
            if self.fetch_mode == 'area':
                slide_id, region_id, tri_id = self._get_random_index_all()
            elif self.fetch_mode == 'slide':
                if loop_count % 100 == 0: # prevent bias
                    slide_id = random.randint(0, len(self.structure) - 1)
                region_id, tri_id = self._get_random_index_slide(slide_id)
            elif self.fetch_mode == 'label':
                if loop_count % 100 == 0: # prevent bias
                    label = random.choice(self.labels[self.label_to_use])
                slide_id, region_id, tri_id = self._get_random_index_label(label)
            elif self.fetch_mode == 'label-slide':
                if loop_count % 100 == 0: # prevent bias
                    label = random.choice(self.labels[self.label_to_use])
                    while True:
                        slide_id = random.randint(0, len(self.structure) - 1)
                        if len(self.regions_of_label_slide[self.label_to_use][label][slide_id]) > 0:
                            break
                region_id, tri_id = self._get_random_index_label_slide(label, slide_id)
            loop_count += 1

            # select a point within the triangle as the center position of rectangle
            a1 = random.random()
            a2 = random.random()
            if a1 + a2 > 1.0:
                a1, a2 = 1.0 - a1, 1.0 - a2
            posx = (1 - a1 - a2) * self.triangulation[slide_id][region_id][tri_id][0][0] + \
                   a1 * self.triangulation[slide_id][region_id][tri_id][1][0] + \
                   a2 * self.triangulation[slide_id][region_id][tri_id][2][0]
            posy = (1 - a1 - a2) * self.triangulation[slide_id][region_id][tri_id][0][1] + \
                   a1 * self.triangulation[slide_id][region_id][tri_id][1][1] + \
                   a2 * self.triangulation[slide_id][region_id][tri_id][2][1]

            src_size = self.src_sizes[slide_id]
            if self.scale_augmentation:
                src_size *= 0.8 + random.random() * 0.4

            if self.rotation:
                angle = random.random() * math.pi * 2
            else:
                angle = -math.pi/4
            angles = [angle, angle + math.pi/2, angle + math.pi, angle + math.pi/2*3]
            discard = False
            corners = []
            for theta in angles:
                cx = posx + src_size / math.sqrt(2) * math.cos(theta)
                cy = posy + src_size / math.sqrt(2) * math.sin(theta)
                corners.append((cx, cy))
                if not self.point_in_region(slide_id, region_id, cx, cy):
                    discard = True
                    break
            if not discard:
                break

        self.fetch_count[slide_id][region_id] += 1
        self.total_fetch_count += 1
        self.total_loop_count += loop_count

        # cropping with rotation
        crop_size = int(src_size * 2**0.5 * max(abs(math.cos(angle)),
                                                     abs(math.sin(angle))))
        cropped = np.asarray(self.slides[slide_id].read_region(
                                                    (int(posx - crop_size/2),
                                                    int(posy - crop_size/2)),
                                                    0, (crop_size, crop_size)), dtype=np.float32)[:,:,:3]
        mat = cv2.getRotationMatrix2D((crop_size/2, crop_size/2),
                                          45 + 360 * angle/(2*math.pi), 1)
        rotated = cv2.warpAffine(cropped, mat, (crop_size, crop_size))

        result = rotated[int(crop_size/2-src_size/2):int(crop_size/2+src_size/2),\
                         int(crop_size/2-src_size/2):int(crop_size/2+src_size/2)]
        result = cv2.resize(result, (self.patch_size, self.patch_size)).transpose((2,0,1))

        if self.flip and random.randint(0, 1):
            result = result[:, :, ::-1]
        result *= (1.0 / 255.0)

        # color matching
        if self.use_color_matching:
            result = self.match_color(result.transpose(1,2,0)).transpose(2,0,1)

        # blurring effect
        if self.blur > 0:
            blur_size = random.randint(1, self.blur)
            result = cv2.blur(result.transpose(1,2,0), (blur_size, blur_size)).transpose((2,0,1))

        if self.he_augmentation:
            hed = rgb2hed(np.clip(result.transpose(1,2,0), -1.0, 1.0))
            ah = 0.95 + random.random() * 0.1
            bh = -0.05 + random.random() * 0.1
            ae = 0.95 + random.random() * 0.1
            be = -0.05 + random.random() * 0.1
            hed[:,:,0] = ah * hed[:,:,0] + bh
            hed[:,:,1] = ae * hed[:,:,1] + be
            result = hed2rgb(hed).transpose(2,0,1)
            result = np.clip(result, 0, 1.0).astype(np.float32)

        # debug
        if self.dump_patch is not None:
            from PIL import Image
            im = Image.fromarray(np.uint8(result.transpose((1,2,0))*255))
            im.save('./%s/%d_%d-%d-%d.png' % (self.dump_patch, self.label_of_region[self.label_to_use][slide_id][region_id], slide_id, region_id, i))

        return result, self.label_of_region[self.label_to_use][slide_id][region_id], (slide_id, region_id, posx, posy)

    def get_examples_of_slide_label(self, slide_id, label, count):
        if len(self.regions_of_label_slide[self.label_to_use][label][slide_id]) == 0:
            return []

        results = []
        for _ in range(count):
            loop_count = 0
            while True:
                region_id, tri_id = self._get_random_index_label_slide(label, slide_id)
                loop_count += 1

                # select a point within the triangle as the center position of rectangle
                a1 = random.random()
                a2 = random.random()
                if a1 + a2 > 1.0:
                    a1, a2 = 1.0 - a1, 1.0 - a2
                posx = (1 - a1 - a2) * self.triangulation[slide_id][region_id][tri_id][0][0] + \
                       a1 * self.triangulation[slide_id][region_id][tri_id][1][0] + \
                       a2 * self.triangulation[slide_id][region_id][tri_id][2][0]
                posy = (1 - a1 - a2) * self.triangulation[slide_id][region_id][tri_id][0][1] + \
                       a1 * self.triangulation[slide_id][region_id][tri_id][1][1] + \
                       a2 * self.triangulation[slide_id][region_id][tri_id][2][1]

                src_size = self.src_sizes[slide_id]
                if self.scale_augmentation:
                    src_size *= 0.8 + random.random() * 0.4

                if self.rotation:
                    angle = random.random() * math.pi * 2
                else:
                    angle = -math.pi/4
                angles = [angle, angle + math.pi/2, angle + math.pi, angle + math.pi/2*3]
                discard = False
                corners = []
                for theta in angles:
                    cx = posx + src_size / math.sqrt(2) * math.cos(theta)
                    cy = posy + src_size / math.sqrt(2) * math.sin(theta)
                    corners.append((cx, cy))
                    if not self.point_in_region(slide_id, region_id, cx, cy):
                        discard = True
                        break
                if not discard:
                    break

            # cropping with rotation
            crop_size = int(src_size * 2**0.5 * max(abs(math.cos(angle)),
                                                         abs(math.sin(angle))))
            cropped = np.asarray(self.slides[slide_id].read_region(
                                                        (int(posx - crop_size/2),
                                                        int(posy - crop_size/2)),
                                                        0, (crop_size, crop_size)), dtype=np.float32)[:,:,:3]
            mat = cv2.getRotationMatrix2D((crop_size/2, crop_size/2),
                                              45 + 360 * angle/(2*math.pi), 1)
            rotated = cv2.warpAffine(cropped, mat, (crop_size, crop_size))

            result = rotated[int(crop_size/2-src_size/2):int(crop_size/2+src_size/2),\
                             int(crop_size/2-src_size/2):int(crop_size/2+src_size/2)]
            result = cv2.resize(result, (self.patch_size, self.patch_size)).transpose((2,0,1))

            if self.flip and random.randint(0, 1):
                result = result[:, :, ::-1]
            result *= (1.0 / 255.0)

            # color matching
            if self.use_color_matching:
                result = self.match_color(result.transpose(1,2,0)).transpose(2,0,1)

            # blurring effect
            if self.blur > 0:
                blur_size = random.randint(1, self.blur)
                result = cv2.blur(result.transpose(1,2,0), (blur_size, blur_size)).transpose((2,0,1))

            if self.he_augmentation:
                hed = rgb2hed(np.clip(result.transpose(1,2,0), -1.0, 1.0))
                ah = 0.95 + random.random() * 0.1
                bh = -0.05 + random.random() * 0.1
                ae = 0.95 + random.random() * 0.1
                be = -0.05 + random.random() * 0.1
                hed[:,:,0] = ah * hed[:,:,0] + bh
                hed[:,:,1] = ae * hed[:,:,1] + be
                result = hed2rgb(hed).transpose(2,0,1)
                result = np.clip(result, 0, 1.0).astype(np.float32)

            results.append(result)
        return results

    def shape(self):
        return (self.patch_size, self.patch_size, 3)

    def flow(self, batch_size=32, preprocess_input=None):
        while True:
            images = []
            labels = []
            for i in range(batch_size):
                image, label, _ = self.get_example(i)
                images.append(image.transpose((1, 2, 0)))
                labels.append(keras.utils.to_categorical(self.labels[self.label_to_use].index(label), len(self.labels[self.label_to_use])))
            images = np.asarray(images, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.float32)
            if preprocess_input is not None:
                yield preprocess_input(images), labels
            else:
                yield images, labels

    # Neural color transfer by github.com/htoyryla, and github.com/ProGamerGov
    # https://github.com/ProGamerGov/Neural-Tools
    # https://github.com/leongatys/NeuralImageSynthesis/blob/master/ExampleNotebooks/ScaleControl.ipynb
    def match_color_prepare(self, source_img, eps=1e-5):
        self.mu_s = source_img.mean(0).mean(0)
        s = source_img - self.mu_s
        s = s.transpose(2, 0, 1).reshape(3, -1)
        self.Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])
        self.chol_s = np.linalg.cholesky(self.Cs)

        eva_s, eve_s = np.linalg.eigh(self.Cs)
        self.Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)

    def match_color(self, target_img, mode='pca', eps=1e-5):
        '''
        Matches the colour distribution of the target image to that of the source image
        using a linear transform.
        Images are expected to be of form (w,h,c) and float in [0,1].
        Modes are chol, pca or sym for different choices of basis.
        '''
        mu_t = target_img.mean(0).mean(0)
        t = target_img - mu_t
        t = t.transpose(2, 0, 1).reshape(3, -1)
        Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
        if mode == 'chol':
            chol_t = np.linalg.cholesky(Ct)
            ts = self.chol_s.dot(np.linalg.inv(chol_t)).dot(t)
        if mode == 'pca':
            eva_t, eve_t = np.linalg.eigh(Ct)
            Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
            ts = self.Qs.dot(np.linalg.inv(Qt)).dot(t)
        if mode == 'sym':
            eva_t, eve_t = np.linalg.eigh(Ct)
            Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
            Qt_Cs_Qt = Qt.dot(self.Cs).dot(Qt)
            eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
            QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
            ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
        matched_img = ts.reshape(*target_img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
        matched_img += self.mu_s
        matched_img[matched_img > 1] = 1
        matched_img[matched_img < 0] = 0
        return matched_img
