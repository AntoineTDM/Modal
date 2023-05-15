import torch
import wandb
import hydra
from tqdm import tqdm


@hydra.main(config_path="configs", config_name="config")
def train(cfg):
    logger = wandb.init(project="challenge", name="run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    debugging = hydra.utils.instantiate(cfg.debugging)

    # train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = debugging.test_dataloader()
    print("initialization OK")

    for epoch in tqdm(range(cfg.epochs)):
        model.train()
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(datamodule.augmented_dataloader()):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(model(images), labels)
            logger.log({"loss": loss.detach().cpu().numpy()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        print("train_acc", epoch_acc)
        logger.log(
            {
                "epoch": epoch,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )
        # if epoch > -1:  # We need to wait for the model to be trained a bit
        #     model.eval()
        #     print("calling unlabelled data loader")
        #     for i, unlabelled_images in enumerate(datamodule.unlabelled_dataloader()):
        #         print("beginning predictions")
        #         unlabelled_images = unlabelled_images.to(device)
        #         probabilities = torch.nn.functional.softmax(model(unlabelled_images), dim=1)
        #         predicted_labels = torch.max(probabilities, 1)
        #         confidence_threshold = cfg.threshold
        #         print("beginning threshold")
        #         confident_mask = probabilities.max(dim=1)[0] > confidence_threshold
        #         confident_images = unlabelled_images[confident_mask]
        #         confident_labels = predicted_labels[confident_mask]
        #         print("finished threshold, there are", confident_images.shape[0], "that passed")
        #         if confident_images.shape[0] > 0:
        #             # Add confident images to training set
        #             print("addinf...")
        #             datamodule.augmented_dataset += torch.utils.data.TensorDataset(
        #                 confident_images, confident_labels)
        #             print("done adding")

        # # This gives us a prediction score over the 150 original images, good for estimating the best threshold
        # epoch_num_correct = 0
        # num_samples = 0
        # confidence = 0
        # for i, batch in enumerate(test_loader):
        #     images, labels = batch
        #     images = images.to(device)
        #     labels = labels.to(device)
        #     preds = torch.nn.functional.softmax(model(images), dim=1)
        #     epoch_num_correct += ((preds.argmax(1) == labels).sum().detach().cpu().numpy())
        #     thresholded_preds = torch.where(preds > cfg.threshold, torch.ones_like(preds), torch.zeros_like(preds))
        #     confidence += ((thresholded_preds.sum(dim=1) > 0).sum().item())
        #     num_samples += len(images)
        # confidence /= num_samples
        # epoch_acc = epoch_num_correct / num_samples
        # print("test_acc", epoch_acc)
        # logger.log(
        #     {
        #         "epoch": epoch,
        #         "test_acc": epoch_acc,
        #         "threshold confidence": confidence
        #     }
        # )

        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(val_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        print("val_acc", epoch_acc)
        logger.log(
            {
                "epoch": epoch,
                "val_loss_epoch": epoch_loss,
                "val_acc": epoch_acc,
            }
        )
    torch.save(model.state_dict(), cfg.checkpoint_path)


if __name__ == "__main__":
    train()
