import os

def get_classifier(version: str):
    from pathlib import Path
    from train import PatchClassifier_pl
    from Classifiers import PatchClassifier
    import torch

    path_to_chkpt = Path(os.environ["INSTANSEG_OUTPUT_PATH"]) / f"patch_classifier/{version}/checkpoints"
    chkpt = os.listdir(path_to_chkpt)[0]
    chkpt_path = path_to_chkpt / chkpt

    model = PatchClassifier_pl.load_from_checkpoint(
                                checkpoint_path=chkpt_path,
                                strict = True)

    # output_path = Path(path_to_chkpt).parents[0]/"model_weights.pth"

    # torch.save({'model_state_dict': model.path_classifier.state_dict(),}, output_path)

    # classifier = PatchClassifier(num_classes = model.num_classes,
    #                         patch_size = model.patch_size,
    #                         dim_in = int(model.dim_in),
    #                         from_file = output_path,
    #                         )

    return model



def export_panoptic_bioimageio(version,name, device = "cpu", 
                                return_model = False,
                                predict_embeddings = False, 
                                is_multiplexed = False,
                                instanseg_str = "brightfield_nuclei"):
    from instanseg.utils.utils import download_model
    import tifffile
    import sys
    import torch
    from Classifiers import PatchClassifier, Predictor
    from instanseg.utils.augmentations import Augmentations
    from pathlib import Path
    import shutil

    classifier = get_classifier(version)
    instanseg = download_model(instanseg_str).to(device)

    if not is_multiplexed:
        path = Path(os.environ["INSTANSEG_EXAMPLE_IMAGE_PATH"])/ "HE_example.tif"


    else:
        path = Path(os.environ["INSTANSEG_EXAMPLE_IMAGE_PATH"])/ "LuCa1.tif"


    input_data = tifffile.imread(path)
    Augmenter = Augmentations()
    input_tensor, _ = Augmenter.to_tensor(input_data, normalize=False)
    original_shape = input_tensor.shape[-2:]
    input_tensor, labels = Augmenter.torch_rescale(input_tensor, current_pixel_size=0.5, requested_pixel_size=0.5, crop =False, modality="Brightfield")
    input_tensor = input_tensor / 255.0
    input_tensor = input_tensor[:,:,:].to(device)
    
    predictor= Predictor(instanseg,classifier.eval(), 
                            predict_embeddings = predict_embeddings,
                            predict_multiplexed = is_multiplexed)
    x,y = predictor.forward(input_tensor[None])

    print(x.shape,y.shape)
    print(torch.unique(y.argmax(1), return_counts=True))

    original_name = Path(os.environ["BIOIMAGEIO_PATH"]) / instanseg_str
    new_name = str(original_name) + f"_{name}"

    if not os.path.exists(new_name):
        shutil.copytree(original_name, new_name)
    
    torch.jit.script(predictor).save(new_name + "/instanseg.pt")

    if return_model:
        return predictor, input_tensor

