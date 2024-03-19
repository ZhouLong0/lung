from src.fewshot.datasets.loader import DatasetFolder
from src.fewshot.datasets.transform import with_augment, without_augment
from torch.utils.data import DataLoader

def get_dataloader(sets, args, sampler=None, shuffle=True, pin_memory=False):
    if sampler is not None:
        loader = DataLoader(sets, batch_sampler=sampler,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    else:
        loader = DataLoader(sets, batch_size=args.batch_size_loader, shuffle=shuffle,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    return loader

def get_dataset(split, args, basic_normalisation=False, grayscale=False, out_name=False):
    """
    Generates a dataset based on the specified parameters.

    Args:
        split (str): The split of the dataset to generate.
        args (Namespace): The command-line arguments.
        basic_normalisation (bool, optional): Whether to apply basic normalisation. Defaults to False.
        grayscale (bool, optional): Whether to convert the images to grayscale. Defaults to False.
        out_name (bool, optional): Whether to include the output name. Defaults to False.

    Returns:
        sets: The generated dataset.
    """
    print(args)
    transform = without_augment(args.transform_size, enlarge=args.enlarge, basic_normalisation=basic_normalisation, grayscale=grayscale)
    # transform = with_augment(1152)
    sets = DatasetFolder(args.root + '/' + args.dataset_path, 
                         args.split_dir, 
                         split, 
                         transform, 
                         args.patch_size ,
                         out_name, 
                         args.sampling, 
                         args.trainset_name, 
                         args.support_hospital)
    print("sets", sets)
    return sets