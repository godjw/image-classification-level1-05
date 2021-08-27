from torch.utils.data import DataLoader

import torchvision.transforms as T

from tqdm import tqdm

from data_utils import *
#forbranch
def save_submission(model, transforms, helper, device):
    eval_metadata = helper.get_metadata()['eval_metadata']
    image_paths = helper.get_paths_and_labels()['eval_img_paths']
    dataset = EvalDataset(image_paths, transforms)
    loader = DataLoader(dataset, shuffle=False)

    model.eval()

    all_predictions = []
    for images in tqdm(loader, colour='BLUE'):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())
    eval_metadata['ans'] = all_predictions
    eval_metadata.to_csv((export_dir := os.path.join(helper.config.trial_name, 'submission.csv')), index=False)

    print(f'Successfully exported to {export_dir}')
