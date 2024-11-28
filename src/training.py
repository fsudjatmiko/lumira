import torch
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from helper_logger import DataLogger
from model_base import SimpleCNN, BasicMobileNet
from helper_tester import ModelTesterMetrics
from dataset import SimpleTorchDataset
from torchvision import transforms

SEED = 424242
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.use_deterministic_algorithms(True)

device = torch.device("mps")
total_epochs = 100  # Increase the number of epochs
batch_size = 32  # Increase the batch size

if __name__ == "__main__":

    print("| Pytorch Model Training !")
    
    print("| Total Epoch :", total_epochs)
    print("| Batch Size  :", batch_size)
    print("| Device      :", device)

    logger = DataLogger("ComputerPartsClassification")
    metrics = ModelTesterMetrics()

    metrics.loss = torch.nn.BCEWithLogitsLoss()
    metrics.activation = torch.nn.Softmax(1)

    model = SimpleCNN(14).to(device)  # 14 classes for computer parts
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Increase learning rate

    training_augmentation = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]

    validation_dataset = SimpleTorchDataset('./splitted/val')
    training_dataset = SimpleTorchDataset('./splitted/train', training_augmentation)
    testing_dataset = SimpleTorchDataset('./splitted/test')

    validation_datasetloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    training_datasetloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    testing_datasetloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)

    # Training Evaluation Loop
    for current_epoch in range(total_epochs):
        print("Epoch :", current_epoch)
        
        # Training Loop
        model.train()  # set the model to train
        metrics.reset() # reset the metrics

        for (image, label) in tqdm(training_datasetloader, desc="Training :"):

            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = metrics.compute(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        training_mean_loss = metrics.average_loss()
        training_mean_accuracy = metrics.average_accuracy()

        # Evaluation Loop
        model.eval()  # set the model to evaluation
        metrics.reset()  # reset the metrics

        for (image, label) in tqdm(validation_datasetloader, desc="Testing  :"):
            
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            metrics.compute(output, label)

        evaluation_mean_loss = metrics.average_loss()
        evaluation_mean_accuracy = metrics.average_accuracy()

        logger.append(
            current_epoch,
            training_mean_loss,
            training_mean_accuracy,
            evaluation_mean_loss,
            evaluation_mean_accuracy
        )

        if logger.current_epoch_is_best:
            print("> Latest Best Epoch :", logger.best_accuracy())
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
            state_dictonary = {
                "model_state": model_state,
                "optimizer_state": optimizer_state
            }
            torch.save(
                state_dictonary, 
                logger.get_filepath("best_checkpoint.pth")
            )

        logger.save()
        print("")
    
    print("| Training Complete, Loading Best Checkpoint")
    
    # Load Model State
    state_dictonary = torch.load(
        logger.get_filepath("best_checkpoint.pth"), 
        map_location=device
    )
    model.load_state_dict(state_dictonary['model_state'])
    model = model.to(device)
    
    # Testing System 
    model.eval()  # set the model to evaluation
    metrics.reset()  # reset the metrics

    for (image, label) in tqdm(testing_datasetloader):
        
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        metrics.compute(output, label)

    testing_mean_loss = metrics.average_loss()
    testing_mean_accuracy = metrics.average_accuracy()

    print("")
    logger.write_text(f"# Final Testing Loss     : {testing_mean_loss}")
    logger.write_text(f"# Final Testing Accuracy : {testing_mean_accuracy}")
    logger.write_text(f"# Report :")
    logger.write_text(metrics.report())
    logger.write_text(f"# Confusion Matrix :")
    logger.write_text(metrics.confusion())
    print("")
