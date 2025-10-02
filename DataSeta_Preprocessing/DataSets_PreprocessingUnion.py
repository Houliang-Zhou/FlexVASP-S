from torch.utils.data import Dataset
import numpy as np
import re
import torch
import os


class DeepVaspS(Dataset):

    def __init__(self, data_dir, labels_set,leave_out_index,k,transform=None,Train=True,num_sample=1000):
        # data_path = data_dir+"data.npy"
        # labels_path = data_dir+"labels.npy"
        # if os.path.exists(data_path):
        #     self.data = np.load(data_path)
        #     self.labels = np.load(labels_path)
        # else:
        # self.data, self.labels = DeepVaspS.sample_numpy_load(data_dir, labels_set,num_sample)
        self.data, self.labels = DeepVaspS.sample_numpy_load_union(data_dir, labels_set,num_sample,k)
            # np.save(data_path, self.data)
            # np.save(labels_path, self.labels)

        # self.data, self.labels = create_empty()
        self.data = torch.from_numpy(self.data).type(torch.FloatTensor)
        self.labels = torch.from_numpy(self.labels).long()
        if Train:
            for i in range(num_sample):
                self.data = self.data[torch.arange(self.data.size(0)) != leave_out_index*num_sample]
                self.labels = self.labels[torch.arange(self.labels.size(0)) != leave_out_index*num_sample]
        else:
            self.data = self.data[leave_out_index * num_sample:(leave_out_index + 1) * num_sample]
            self.labels = self.labels[leave_out_index * num_sample:(leave_out_index + 1) * num_sample]

    def __getitem__(self, idx):
        item = self.data[idx]
        item_label = self.labels[idx]
        return item, item_label

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        #for imbalancedsampler
        return self.labels

    @staticmethod
    def sample_numpy_load(data_dir, labels_set, num_sample):
        """
        load .cnn format data and generate images include all data from same superfamily
        input are all .cnn files from one superfamily
        output are numpy array images(array size is (num_images, 1, x_dim, y_dim, z_dim)). labels (array size is (num_images))
        :param data_dir:
        :param labels_set: e.g. ENOLASE_LABELS_SET = [['1mdr', '2ox4'], ['2pgw'], ['1iyx', '3otr', '1te6', '1ebh']
        :return: images,labels
        """

        init_filename = '{}{}-CNN/{}_{}-025.SURF-clean.SURF.cnn'.format(data_dir, labels_set[0][0],
                                                   labels_set[0][0], '1')

        x_dim, y_dim, z_dim = DeepVaspS.voxel_parser(init_filename)[1:4]
        flatten_protein_list = [protein_family for sublist in labels_set for protein_family in sublist]
        num_images = len(flatten_protein_list) * num_sample
        images = np.empty((num_images, 1, x_dim, y_dim, z_dim))
        labels = np.empty((num_images))

        image_num_index = 0
        for subfamily in labels_set:
            for protein in subfamily:
                for img_num in range(1, num_sample+1):
                    #1BKH_1-025.SURF-clean.SURF.cnn
                    file_name = '{}{}-CNN/{}_{}-025.SURF-clean.SURF.cnn'.format(data_dir, protein,
                                                           protein, str(img_num))
                    labels[image_num_index] = labels_set.index(subfamily)
                    images[image_num_index] = DeepVaspS.voxel_parser(file_name)[0]
                    image_num_index += 1

        # images dim = n_sample * n_voxel
        return images, labels

    @staticmethod
    def sample_numpy_load_union(data_dir, labels_set, num_sample,k):
        """
        load .cnn format data and generate images include all data from same superfamily
        input are all .cnn files from one superfamily
        output are numpy array images(array size is (num_images, 1, x_dim, y_dim, z_dim)). labels (array size is (num_images))
        :param data_dir:
        :param labels_set: e.g. ENOLASE_LABELS_SET = [['1mdr', '2ox4'], ['2pgw'], ['1iyx', '3otr', '1te6', '1ebh']
        :return: images,labels
        """

        init_filename = '{}{}-CNN/{}_{}-025.SURF-clean.SURF.cnn'.format(data_dir, labels_set[0][0],
                                                   labels_set[0][0], '1')

        x_dim, y_dim, z_dim = DeepVaspS.voxel_parser(init_filename)[1:4]
        flatten_protein_list = [protein_family for sublist in labels_set for protein_family in sublist]
        num_images = len(flatten_protein_list) * num_sample
        images = np.empty((num_images, 1, x_dim, y_dim, z_dim))
        labels = np.empty((num_images))

        image_num_index = 0
        for subfamily in labels_set:
            for protein in subfamily:
                # for img_num in range(1, num_sample+1):
                for img_num in range(0, num_sample):
                    #1BKH_1-025.SURF-clean.SURF.cnn
                    # file_name = '{}{}-CNN/{}_{}-025.SURF-clean.SURF.cnn'.format(data_dir, protein,
                    #                                        protein, str(img_num))
                    file_name = '{}{}_union_{}/{}-union-{}.npy'.format(data_dir, protein,k,protein,str(img_num))
                    labels[image_num_index] = labels_set.index(subfamily)
                    # images[image_num_index] = DeepVaspS.voxel_parser(file_name)[0]
                    images[image_num_index] = np.load(file_name)
                    image_num_index += 1

        # images dim = n_sample * n_voxel
        return images, labels


    @staticmethod
    def voxel_parser(filename: str) -> (np.ndarray, int, int, int, list, list, list):
        """
        Read the data in .cnn protein image files into a 3D-array, output size of 3D-array is (1, x_dim, y_dim, z_dim)
        :return: 3D-array
        """
        with open(filename, encoding='gbk', errors='ignore') as voxel_file:
            lines = voxel_file.readlines()
            # Reads dimensions from\
            #
            # line 18
            match = re.search(r'BOUNDS xyz dim: \[([0-9]+) ([0-9]+) ([0-9]+)]', lines[17])
            match_x_bounds = re.search(r'BOUNDS xneg/xpos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]', lines[19])
            match_y_bounds = re.search(r'BOUNDS yneg/ypos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]', lines[20])
            match_z_bounds = re.search(r'BOUNDS zneg/zpos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]', lines[21])
            x_dim = int(match.group(1))
            y_dim = int(match.group(2))
            z_dim = int(match.group(3))
            data = np.zeros((x_dim * y_dim * z_dim))

            # Reads voxel data starting at line 25
            for i in range(24, len(lines)):
                line_num, val = lines[i].split()
                line_num = int(line_num)
                val = float(val)
                data[line_num] = val
            data = data.reshape((1, x_dim, y_dim, z_dim))

            return data, x_dim, y_dim, z_dim#, match_x_bounds.groups(), match_y_bounds.groups(), match_z_bounds.groups()












