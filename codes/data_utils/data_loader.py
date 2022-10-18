"""
DataLoader class definition.
Melanie Bernhardt - ETH Zurich

Loads the measurements, L matrix from disk.
Defines validation set, training set.
Defines batch iterator for training and validation.
Supports median, mean, CNN inpainting.
Supports loading of ideal time delays.
Supports now test set mode without the ground truth images. 
Can be used for loading data from up to three different dataset.

See readme.md for a detail explanation.
"""
import numpy as np
from codes.data_utils.data_utils import prepare_input_sos
import os
import scipy.io as sio
import hdf5storage
import sys

class DataGenerator:
    def __init__(self, filename, n_val,
                 msz, na, noise_rate,
                 usm_rate, random_mask_type,
                 n_batch, n_iterations, meas='syn',
                 use_med_filter=False,
                 use_mean_filter=False, 
                 type='train',
                 imz=None,
                 mix=False,
                 filename_mix=None,
                 fixed_rate=False, #Orig is false!!
                 mix_type='syn',
                 mix_triple=False,
                 filename_mix_triple=None,
                 mix_type_triple='ideal_time',
                 p_mix=1,
                 p_triple=1):
        # set the seed to ensure repeatable experiments.
        # in particular ensures the validation does not change.
        np.random.seed(seed=333)
        self.mix_triple = mix_triple
        self.fixed_rate = fixed_rate
        self.type = type
        self.n_batch = n_batch
        self.n_iterations = n_iterations
        print(os.getenv("DATA_PATH"))
        print(filename)
        # Load the data
        try:  # for v7 file
            mat = sio.loadmat(os.path.join(os.getenv("DATA_PATH"), filename))
        except NotImplementedError:  # for v7.3
            mat = hdf5storage.loadmat(os.path.join(
                os.getenv("DATA_PATH"), filename))
        if meas == 'syn':
             d_clean= mat['measmnts']
             self.d_clean=d_clean
             # print(self.d_clean.shape, 'shape pre concat agfszdgsrgfbx')
             # if self.d_clean.shape[1] == 1:
             #     for i in range(20):
             #        self.d_clean = np.concatenate((d_clean, self.d_clean), axis=1)
             # print(self.d_clean.shape, 'shape post concat agfszdgsrgfbx')
        elif meas == 'ideal_time':
            self.d_clean = mat['timedelays']
        self.n_obs = self.d_clean.shape[1]
        if type=='test':
            self.n_val = self.n_obs
        else:
            self.n_val = n_val
        self.d_val = self.d_clean[:, :self.n_val]
        self.d_train = self.d_clean[:, self.n_val:]
        self.n_train = self.d_train.shape[1]
        if type == 'train' or type == 'val':
            imgs = mat['imgs_gt']
            self.imgs = np.transpose(imgs, [2, 1, 0])
        if self.type == 'train' or self.type == 'val':
            self.img_sz = [self.imgs.shape[1], self.imgs.shape[2]]
            self.img_val = self.imgs[:self.n_val, :, :]
            self.img_train = self.imgs[self.n_val:, :, :]
        else:
            self.img_sz = imz
        # Masks and Matrices
        try:
            self.Linv = mat['Linv']
        except KeyError:
            self.Linv = None
        self.Lnrm = mat['L']
        self.fixed_mask = mat['maskFixed']
        self.fixed_mask[np.isnan(self.fixed_mask)] = 0
        self.Lnrm = np.diag(self.fixed_mask.ravel()) @ self.Lnrm #@ here means where self.Lnrm is true or 1
        self.s1 = mat['L_fact']
        del mat  # for memory issue
        self.aa = np.sum(self.Lnrm, axis=1, keepdims=True)
        self.n_reads = self.Lnrm.shape[0]
        self.msz = msz
        self.na = na
        self.noise_rate = noise_rate
        self.usm_rate = usm_rate
        self.random_mask_type = random_mask_type
        self.use_med_filter = use_med_filter
        self.use_mean_filter = use_mean_filter
        self.d_val_in, self.d_val_gt, self.d_val_mask = prepare_input_sos(
            self.d_val,
            msz=self.msz,
            na=self.na,
            noise_rate=self.noise_rate,
            usm_rate=self.usm_rate,
            fixedMask=self.fixed_mask,
            random_mask_type=self.random_mask_type,
            use_med_filter=self.use_med_filter,
            use_mean_filter=self.use_mean_filter)
        self.mix = mix
        if self.mix:
            try:  # for v7 file
                mat = sio.loadmat(os.path.join(
                    os.getenv("DATA_PATH"), filename_mix))
            except NotImplementedError:  # for v7.3
                mat = hdf5storage.loadmat(os.path.join(
                    os.getenv("DATA_PATH"), filename_mix))
            if mix_type == 'syn':
                self.d_clean_mix = mat['measmnts']
            elif mix_type == 'ideal_time':
                self.d_clean_mix = mat['timedelays']
            else:
                raise NotImplementedError
            self.n_train_mix = self.d_clean_mix.shape[1]
            self.img_train_mix = mat['imgs_gt']
            self.img_train_mix = np.transpose(self.img_train_mix, [2, 1, 0])
            self.n_batch_mix = p_mix
            self.n_batch_main = self.n_batch - p_mix
            print(self.n_batch_main)
        if self.mix_triple:
            try:  # for v7 file
                mat = sio.loadmat(os.path.join(
                    os.getenv("DATA_PATH"), filename_mix_triple))
            except NotImplementedError:  # for v7.3
                mat = hdf5storage.loadmat(os.path.join(
                    os.getenv("DATA_PATH"), filename_mix_triple))
            if mix_type_triple == 'syn':
                self.d_clean_mix_triple = mat['measmnts']
            elif mix_type_triple == 'ideal_time':
                self.d_clean_mix_triple = mat['timedelays']
            else:
                raise NotImplementedError
            self.n_train_mix_triple = self.d_clean_mix_triple.shape[1]
            self.img_train_mix_triple = mat['imgs_gt']
            self.img_train_mix_triple = np.transpose(self.img_train_mix_triple, [2, 1, 0])
            self.n_batch_mix_triple = p_triple
            print(self.n_batch_mix_triple)
            self.n_batch_main = self.n_batch - self.n_batch_mix - self.n_batch_mix_triple
            print(self.n_batch_main)

    def getBatchIterator(self, val=False):
        """
        Generates a batch iterator for the dataset for a fixed number of iterations (batches).
        Args:
            type: 'train' for training set batching
                   'val' for validation batching
                   'test' for testing (one single batch)
        Example:
            data = DataGenerator(config)
            training_batches = data.batch_iterator('train')
            val_batches = data.batch_iterator('val')
        """
        if val:
            # no shuffle needed
            # always keep the same validation data.
            # has to be init in init because multiple call
            # to val_iterator
            n = self.d_val_in.shape[1]
            print(n)
            num_batch_per_epoch = n//self.n_batch
            for batch_num in range(num_batch_per_epoch):
                start_index = batch_num * self.n_batch
                end_index = min((batch_num + 1) * self.n_batch, n)
                batch_d_in = self.d_val_in[:, start_index:end_index]
                batch_d_gt = self.d_val_gt[:, start_index:end_index]
                batch_mask = self.d_val_mask[:, start_index:end_index]
                batch_img = self.img_val[start_index:end_index, :, :]
                yield batch_d_in, batch_d_gt, batch_mask, batch_img
        elif self.type == 'test':
            n = self.d_val_in.shape[1]
            num_batch_per_epoch = n//self.n_batch
            for batch_num in range(num_batch_per_epoch):
                start_index = batch_num * self.n_batch
                end_index = min((batch_num + 1) * self.n_batch, n)
                batch_d_in = self.d_val_in[:, start_index:end_index]
                batch_d_gt = self.d_val_gt[:, start_index:end_index]
                batch_mask = self.d_val_mask[:, start_index:end_index]
                yield batch_d_in, batch_d_gt, batch_mask
        elif self.type == 'train':
            # Shuffle the data at each epoch
            if self.mix or self.mix_triple:
                num_batch_per_epoch = self.d_train.shape[1]//self.n_batch_main
            else:
                n = self.d_train.shape[1]
                num_batch_per_epoch = n//self.n_batch
            n_epochs = self.n_iterations//num_batch_per_epoch+1
            for _ in range(n_epochs):
                shuffle_indices = np.random.permutation(
                    np.arange(self.d_train.shape[1]))
                shuffled_imgs = self.img_train[shuffle_indices, :, :]
                for batch_num in range(num_batch_per_epoch):
                    if self.mix or self.mix_triple:
                        start_index = batch_num * self.n_batch_main
                        end_index = min(
                            (batch_num + 1) * self.n_batch_main, self.d_train.shape[1])
                    else:
                        start_index = batch_num * self.n_batch
                        end_index = min((batch_num + 1) * self.n_batch, n)
                    index_batch = shuffle_indices[start_index:end_index]
                    batch_train = self.d_train[:, index_batch]
                    if self.fixed_rate:
                        current_usm_rate = self.usm_rate
                    else:
                        current_usm_rate = np.random.uniform(0.1, self.usm_rate)
                    # resample the undersampling mask batch (to have a different usm rate)
                    batch_d_in, batch_d_gt, batch_mask = prepare_input_sos(
                        batch_train,
                        msz=self.msz,
                        na=self.na,
                        noise_rate=self.noise_rate,
                        usm_rate=current_usm_rate,
                        fixedMask=self.fixed_mask,
                        random_mask_type=self.random_mask_type,
                        use_med_filter=self.use_med_filter,
                        use_mean_filter=self.use_mean_filter
                    )
                    batch_img = shuffled_imgs[start_index:end_index, :, :]
                    if self.mix:
                        ind_batch_train_mix = np.random.choice(
                            np.arange(self.n_train_mix), size=self.n_batch_mix, replace=False)
                        batch_train_mix = self.d_clean_mix[:, ind_batch_train_mix]
                        batch_d_in_mix, batch_d_gt_mix, batch_mask_mix = prepare_input_sos(
                            batch_train_mix,
                            msz=self.msz,
                            na=self.na,
                            noise_rate=self.noise_rate,
                            usm_rate=current_usm_rate,
                            fixedMask=self.fixed_mask,
                            random_mask_type='test',
                            use_med_filter=self.use_med_filter,
                            use_mean_filter=self.use_mean_filter
                        )
                        batch_img_mix = self.img_train_mix[ind_batch_train_mix, :, :]
                        print(batch_mask.shape)
                        print(batch_mask_mix.shape)
                        print(batch_d_gt.shape)
                        print(batch_d_gt_mix.shape)
                        batch_img = np.concatenate(
                            (batch_img, batch_img_mix), axis=0)
                        batch_d_gt = np.concatenate(
                            (batch_d_gt, batch_d_gt_mix), axis=1)
                        batch_mask = np.concatenate(
                            (batch_mask, batch_mask_mix), axis=1)
                        batch_d_in = np.concatenate(
                            (batch_d_in, batch_d_in_mix), axis=1)
                    if self.mix_triple:
                        ind_batch_train_mix_triple = np.random.choice(
                            np.arange(self.n_train_mix_triple), size=self.n_batch_mix_triple, replace=False)
                        batch_train_mix_triple = self.d_clean_mix_triple[:,ind_batch_train_mix_triple]
                        batch_d_in_mix_triple, batch_d_gt_mix_triple, batch_mask_mix_triple = prepare_input_sos(
                            batch_train_mix_triple,
                            msz=self.msz,
                            na=self.na,
                            noise_rate=self.noise_rate,
                            usm_rate=current_usm_rate,
                            fixedMask=self.fixed_mask,
                            random_mask_type='test',
                            use_med_filter=self.use_med_filter,
                            use_mean_filter=self.use_mean_filter
                        )
                        batch_img_mix_triple = self.img_train_mix_triple[ind_batch_train_mix_triple, :, :]
                        print(batch_mask.shape)
                        print(batch_mask_mix_triple.shape)
                        print(batch_d_gt.shape)
                        print(batch_d_gt_mix_triple.shape)
                        batch_img = np.concatenate(
                            (batch_img, batch_img_mix_triple), axis=0)
                        batch_d_gt = np.concatenate(
                            (batch_d_gt, batch_d_gt_mix_triple), axis=1)
                        batch_mask = np.concatenate(
                            (batch_mask, batch_mask_mix_triple), axis=1)
                        batch_d_in = np.concatenate(
                            (batch_d_in, batch_d_in_mix_triple), axis=1)
                    yield batch_d_in, batch_d_gt, batch_mask, batch_img
