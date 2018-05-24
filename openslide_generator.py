import random
import math
import os

import numpy as np
import cv2
import tripy
from openslide import OpenSlide

from skimage.color import rgb2hed, hed2rgb

import keras


class OpenSlideGenerator(object):
    fetch_modes = ['area', 'slide', 'label']

    def __init__(self, path, root, src_size, patch_size, fetch_mode='area',
                 rotation=True, flip=False, blur=0, he_augmentation=False, scale_augmentation=False,
                 dump_patch=None, verbose=1):
        self.path = path
        self.root = root
        self.src_size = src_size
        self.patch_size = patch_size
        self.fetch_mode = fetch_mode
        if not self.fetch_mode in OpenSlideGenerator.fetch_modes:
            raise Exception('invalid fetch_mode %r' % self.fetch_mode)
        self.rotation = rotation
        self.flip = flip
        self.blur = blur
        self.he_augmentation = he_augmentation
        self.scale_augmentation = scale_augmentation
        self.dump_patch = dump_patch
        self.verbose = verbose

        self.slide_names = []
        self.label_of_region = []
        self.structure = []
        self.triangulation = []
        self.regions_of_label = dict()

        self.src_sizes = []

        self.total_area = 0
        self.slide_areas = []     # total area of a slide
        self.label_areas = dict() # total area of a label
        self.total_weight = 0
        self.slide_weights = []   # total weight of a slide
        self.label_weights = dict()

        self.total_weight = 0
        self.weights = []          # overall weight
        self.weights_in_slide = [] # weight in a slide
        self.weights_in_label = [] # weight in the same label

        self.total_triangles = 0
        self.slide_triangles = []     # total triangle number for each slide
        self.label_triangles = dict() # total triangle number for each label

        self.serialized_index = []           # serialized_index[ID] -> (SLIDE_ID, REGION_ID, TRIANGLE_ID)
        self.serialized_index_slide = []     # serialized_index_slide[SLIDE_ID][ID] -> (REGION_ID, TRIANGLE_ID)
        self.serialized_index_label = dict() # serialized_index_label[label][ID] -> (SLIDE_ID, REGION_ID, TRIANGLE_ID)

        # variables for Walker's alias method
        self.a_area = []
        self.p_area = []
        self.a_slide = []
        self.p_slide = []
        self.a_label = dict()
        self.p_label = dict()

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
        with open(path) as f:
            for line in map(lambda l: l.split("#")[0].strip(), f.readlines()):
                if len(line) == 0:
                    continue
                is_svs_line = (line[0] == "@")
                if is_svs_line:
                    line = line[1:]
                else:
                    try:
                        x, y = map(int, line.split())
                    except Exception:
                        raise Exception('invalid dataset file format!')
                if state == 0:
                    if not is_svs_line:
                        raise Exception('invalid dataset file format!')

                    svs_name = line.split()[0]
                    if len(line.split()) > 1 and line.split()[1].isdigit:
                        svs_src_size = int(line.split()[1])
                    else:
                        svs_src_size = self.src_size

                    self.slide_names.append(svs_name)
                    self.src_sizes.append(svs_src_size)
                    self.label_of_region.append([])
                    self.structure.append([])
                    state = 1
                elif state == 1:
                    if is_svs_line: # new file
                        svs_name = line.split()[0]
                        if len(line.split()) > 1 and line.split()[1].isdigit:
                            svs_src_size = int(line.split()[1])
                        else:
                            svs_src_size = self.src_size # default src_size

                        self.slide_names.append(svs_name)
                        self.src_sizes.append(svs_src_size)
                        self.label_of_region.append([])
                        self.structure.append([])
                        state = 1
                    else: # region header
                        self.label_of_region[-1].append(x)
                        # handling newly found label
                        if x not in self.regions_of_label.keys():
                            self.regions_of_label[x] = []
                            self.a_label[x] = []
                            self.p_label[x] = []
                            self.label_areas[x] = 0
                            self.label_weights[x] = 0
                            self.label_triangles[x] = 0
                            self.serialized_index_label[x] = []
                        self.structure[-1].append([])
                        self.regions_of_label[x].append((len(self.structure) - 1, len(self.structure[-1]) - 1))
                        left_points = y
                        if y < 3:
                            raise Exception('regions should consist of more than 3 points!')
                        state = 2
                elif state == 2:
                    if is_svs_line or left_points <= 0:
                        raise Exception('invalid dataset file format!')
                    self.structure[-1][-1].append((x, y))
                    left_points -= 1
                    if left_points == 0:
                        state = 1
        if state != 1: # dataset file should end with a completed region entry
            raise Exception('invalid dataset file format!')

        # load slides
        for name in self.slide_names:
            try:
                self.slides.append(OpenSlide(os.path.join(self.root, name)))
            except Exception as exc:
                raise Exception('an error has occurred while reading slide "{}"'.format(name))

        # region triangulation
        total_region_count = 0
        for i in range(len(self.structure)):
            self.triangulation.append([])
            self.weights.append([])
            self.weights_in_slide.append([])
            self.weights_in_label.append([])
            self.serialized_index_slide.append([])
            self.a_slide.append([])
            self.p_slide.append([])
            self.slide_areas.append(0)
            self.slide_weights.append(0)
            self.slide_triangles.append(0)
            w, h = self.slides[i].dimensions # slide width/height

            for j in range(len(self.structure[i])):
                region = self.structure[i][j]
                total_region_count += 1

                # triangulation
                self.triangulation[-1].append(tripy.earclip(region))

                for x, y in region:
                    if w < x or h < y:
                        raise Exception('invalid polygon vertex position (%d, %d) in %s!' % (x, y, self.slide_names[i]))

                # triangle area calculation
                self.weights[-1].append([])
                self.weights_in_slide[-1].append([])
                self.weights_in_label[-1].append([])
                self.slide_triangles[-1] += len(self.triangulation[-1][-1])
                label = self.label_of_region[i][j]
                self.label_triangles[label] += len(self.triangulation[-1][-1])
                for (x1, y1), (x2, y2), (x3, y3) in self.triangulation[-1][-1]:
                    a = x2 - x1
                    b = y2 - y1
                    c = x3 - x1
                    d = y3 - y1
                    area = abs(a*d - b*c)/2
                    weight = area / (self.src_sizes[i]**2)
                    self.weights[-1][-1].append(weight)
                    self.weights_in_slide[-1][-1].append(weight)
                    self.weights_in_label[-1][-1].append(weight)
                    self.total_area += area
                    self.total_weight += weight
                    self.slide_areas[-1] += area
                    self.slide_weights[-1] += weight
                    self.label_areas[label] += area
                    self.label_weights[label] += weight

        # calculate the set of triangle weights for each fetch_mode
        for i in range(len(self.weights)): # svs
            for j in range(len(self.weights[i])): # region
                for k in range(len(self.weights[i][j])): # triangle
                    self.weights[i][j][k] /= self.total_weight
                    self.weights_in_slide[i][j][k] /= self.slide_weights[i]
                    label = self.label_of_region[i][j]
                    self.weights_in_label[i][j][k] /= self.label_weights[label]
                    self.serialized_index.append((i, j, k))
                    self.serialized_index_slide[i].append((j, k))
                    self.serialized_index_label[label].append((i, j, k))
                    self.total_triangles += 1

        # Walker's alias method for weighted sampling of triangles
        def walker_precomputation(probs):
            EPS = 1e-10
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
                            if probs[i] * len(probs) <= 1.0:
                                break
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
        for label in self.regions_of_label.keys():
            probs = []
            for slide_id, region_id in self.regions_of_label[label]:
                for tri_id in range(len(self.weights_in_label[slide_id][region_id])):
                    probs.append(self.weights_in_label[slide_id][region_id][tri_id])

            self.a_label[label], self.p_label[label] = walker_precomputation(probs)

        if self.verbose > 0:
            print('loaded {} slide(s).'.format(len(self.structure)))
            for i in range(len(self.structure)):
                print('[{}] {}'.format(i, self.slide_names[i]))
                print('- {} regions'.format(len(self.structure[i])))
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
        q = random.random() * self.label_triangles[label]
        i = int(q)
        if q - i < self.p_label[label][i]:
            return self.serialized_index_label[label][i]
        else:
            return self.serialized_index_label[label][self.a_label[label][i]]

    # winding-number algorithm
    def point_in_region(self, slide_id, region_id, cx, cy):
        def is_left(p0, p1, p2):
            return ((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))

        poly = self.structure[slide_id][region_id]
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
        first_loop = True
        while True:
            # select a triangle by the current fetch-mode
            if self.fetch_mode == 'area':
                slide_id, region_id, tri_id = self._get_random_index_all()
            elif self.fetch_mode == 'slide':
                if first_loop: # prevent bias
                    slide_id = random.randint(0, len(self.structure) - 1)
                region_id, tri_id = self._get_random_index_slide(slide_id)
            elif self.fetch_mode == 'label':
                if first_loop: # prevent bias
                    label = random.choice(list(self.label_weights.keys()))
                slide_id, region_id, tri_id = self._get_random_index_label(label)

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
                print(src_size)

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
            first_loop = False

        self.fetch_count[slide_id][region_id] += 1

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
            im.save('./%s/%d_%d-%d-%d.png' % (self.dump_patch, self.label_of_region[slide_id][region_id], slide_id, region_id, i))

        return result, self.label_of_region[slide_id][region_id], (slide_id, region_id, posx, posy)

    def labels(self):
        return self.regions_of_label.keys()

    def shape(self):
        return (self.patch_size, self.patch_size, 3)

    def flow(self, batch_size=32):
        while True:
            images = []
            labels = []
            for i in range(batch_size):
                image, label, _ = self.get_example(i)
                images.append(image.transpose((1, 2, 0)))
                labels.append(keras.utils.to_categorical(list(self.labels()).index(label), len(self.labels())))
            images = np.asarray(images, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.float32)
            yield images, labels
