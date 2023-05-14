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
    custom = hydra.utils.instantiate(cfg.custom)

    train_loader = datamodule.train_dataloader()
    augmented_loader = custom.augmented_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = debugging.test_dataloader()

    print("initialization OK")
    print("train_loader batch number:", len(train_loader))
    print("augmented_loader batch number:", len(augmented_loader))

    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(augmented_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
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

        # This gives us a prediction score over the 150 original images, good for the threshold
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        confidence = 0
        for i, batch in enumerate(test_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            preds = torch.nn.functional.softmax(preds, dim=1)
            loss = loss_fn(preds, labels)
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            # apply a threshold of 0.5 to the tensor
            thresholded_preds = torch.where(preds > cfg.threshold, torch.ones_like(preds), torch.zeros_like(preds))
            # print(thresholded_preds)
            confidence += ((thresholded_preds.sum(dim=1) > 0).sum().item())
            num_samples += len(images)
            # print(preds.argmax(1))
        epoch_loss /= num_samples
        confidence /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        print("test_acc", epoch_acc)
        print("confidence", confidence)
        logger.log(
            {
                "epoch": epoch,
                "test_loss_epoch": epoch_loss,
                "test_acc": epoch_acc,
                "threshold confidence": confidence
            }
        )



        # This gives us a prediction score over the 150 original images
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        confidence = 0
        for i, batch in enumerate(test_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            preds = torch.nn.functional.softmax(preds, dim=1)
            loss = loss_fn(preds, labels)
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            # apply a threshold of 0.5 to the tensor
            thresholded_preds = torch.where(preds > cfg.threshold, torch.ones_like(preds),torch.zeros_like(preds))
            # print(thresholded_preds)
            confidence += ((thresholded_preds.sum(dim=1) > 0).sum().item())
            num_samples += len(images)
            # print(preds.argmax(1))
        epoch_loss /= num_samples
        confidence /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        print("test_acc", epoch_acc)
        print("confidence",confidence)
        logger.log(
            {
                "epoch": epoch,
                "test_loss_epoch": epoch_loss,
                "test_acc": epoch_acc,
                "threshold confidence": confidence
            }
        )

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
