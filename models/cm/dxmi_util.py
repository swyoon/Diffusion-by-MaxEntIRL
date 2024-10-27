"""
functions added for GCD project
"""
from models.cm.image_datasets import ImageDataset, _list_image_files_recursively
import blobfile as bf
import torch
import os


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    cachefile=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    if cachefile is None or not os.path.exists(cachefile):
        print(f'No cache file found at {cachefile}. Creating new cache...')
        all_files = _list_image_files_recursively(data_dir)
        print(f'Found {len(all_files)} image files in {data_dir}')
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
            print(f'Extracted {len(sorted_classes)} unique classes from filenames')
        else:
            classes = None
            print('No class conditioning requested')

        cache_path = '.' + data_dir.replace('/', '_')
        torch.save({'all_files': all_files, 'classes': classes}, cache_path, _use_new_zipfile_serialization=False)
        print(f'Cache saved to {cache_path}')
    else:
        cached = torch.load(cachefile)
        all_files, classes = cached['all_files'], cached['classes']
        print(f'loaded from cache: {cachefile}')
    print(f'found {len(all_files)} files')

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    return dataset


def infinite_loader(dataloader):
    while True:
        for data, target in iter(dataloader):
            yield data, target


