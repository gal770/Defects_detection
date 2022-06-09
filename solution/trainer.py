import os
import json
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader

from common import OUTPUT_DIR, CHECKPOINT_DIR

"""Trains, validates and tests model on a given dataset.
    Prints average scores for each prediction"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class LoggingParameters:
    """Data class holding parameters for logging."""
    model_name: str
    dataset_name: str
    optimizer_name: str
    optimizer_params: dict


class Trainer:
    """Abstract model trainer on a binary classification task."""

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 lr_scheduler: torch.optim,
                 criterion,
                 batch_size: int,
                 train_dataset: Dataset,
                 validation_dataset: Dataset,
                 test_dataset: Dataset):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.epoch = 0
        self.lr_scheduler = lr_scheduler

    def train_one_epoch(self) -> (float, float, float, float):
        """Train the model for a single epoch on the training dataset.
        Returns:
            (avg_loss, accuracy, detection rate, precision): tuple containing the average loss,
            overall accuracy, detection rate and precision across all dataset samples.
        """
        self.model.train()
        total_loss = 0.0
        avg_loss = 0.0
        accuracy = 0.0
        detection_rate = 0.0
        precision = 0.0
        nof_samples = 0.0

        correct_labeled_samples = 0.0
        nof_defect_samples = 0.0
        correct_defect_labeled = 0.0
        nof_defect_labeled = 0.0

        train_dataloader = DataLoader(self.train_dataset,
                                      self.batch_size,
                                      shuffle=True)
        print_every = int(len(train_dataloader) / 50)
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            self.optimizer.zero_grad()
            """Forward pass"""
            pred = self.model(inputs)
            """Compute loss and backward pass"""
            loss = self.criterion(pred, targets)
            loss.backward()
            self.optimizer.step()

            """ Computing total and avg loss"""
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            """ Computing overall accuracy, detection rate and precision"""
            correct_labeled_samples += (pred.argmax(1) == targets).type(torch.float).sum().item()
            nof_samples += self.batch_size
            accuracy = correct_labeled_samples / nof_samples

            ones = torch.tensor(np.ones(targets.size(), dtype=int)).to(device)
            nof_defect_samples += (targets == ones).type(torch.float).sum().item()
            for i in range(list(targets.size())[0]):
                if targets[i] == 1:
                    if pred.argmax(1)[i] == 1:
                        correct_defect_labeled += 1

            if nof_defect_samples > 0:
                detection_rate = correct_defect_labeled / nof_defect_samples

            nof_defect_labeled += (pred.argmax(1) == ones).type(torch.float).sum().item()
            if nof_defect_labeled > 0:
                precision = correct_defect_labeled / nof_defect_labeled

            if batch_idx % print_every == 0 or \
                    batch_idx == len(train_dataloader) - 1:
                print(f'Epoch [{self.epoch:03d}] | Loss: {avg_loss:.4f} | '
                      f'Overall Accuracy: {accuracy:.5f}[%]  '
                      f'({correct_labeled_samples}/{nof_samples}) '
                      f'Detection rate: {detection_rate:.5f}[%]  '
                      f'({correct_defect_labeled}/{nof_defect_samples}) '
                      f'Precision: {precision:.5f}[%]  '
                      f'({correct_defect_labeled}/{nof_defect_labeled}) '
                      )

        return avg_loss, accuracy, detection_rate, precision

    def evaluate_model_on_dataloader(
            self, dataset: torch.utils.data.Dataset) -> (float, float, float, float):
        """Evaluate model on validation and test datasets.

        Args:
            dataset: the dataset to evaluate the model on.
        Returns:
            (avg_loss, accuracy, detection rate, precision): tuple containing the average loss,
            overall accuracy, detection rate and precision across all dataset samples.
        Prints avg model scores with regards to correct and wrong predictions.
        """
        self.model.eval()
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=True)
        print_every = max(int(len(dataloader) / 50), 1)

        total_loss = 0
        avg_loss = 0
        accuracy = 0
        detection_rate = 0.0
        precision = 0.0

        nof_samples = 0
        correct_labeled_samples = 0
        nof_defect_samples = 0.0
        correct_defect_labeled = 0.0
        nof_defect_labeled = 0.0

        correct_defect_total_d_score = 0
        wrong_defect_total_d_score = 0
        correct_clean_total_c_score = 0
        wrong_clean_total_c_score = 0
        correct_defect_total_c_score = 0
        wrong_defect_total_c_score = 0
        correct_clean_total_d_score = 0
        wrong_clean_total_d_score = 0

        correct_defect_avg_d_score = 0
        wrong_defect_avg_d_score = 0
        correct_clean_avg_c_score = 0
        wrong_clean_avg_c_score = 0
        correct_defect_avg_c_score = 0
        wrong_defect_avg_c_score = 0
        correct_clean_avg_d_score = 0
        wrong_clean_avg_d_score = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            with torch.no_grad():
                inputs = inputs.to(device)
                targets = targets.to(device)

                """Compute forward pass and loss under torch.no_grad()"""
                pred = self.model(inputs)
                loss = self.criterion(pred, targets)
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)

                """ Computing overall accuracy, detection rate and precision"""
                nof_samples += self.batch_size
                correct_labeled_samples += (pred.argmax(1) == targets).type(torch.float).sum().item()
                accuracy = correct_labeled_samples / nof_samples

                ones = torch.tensor(np.ones(targets.size(), dtype=int)).to(device)
                nof_defect_samples += (targets == ones).type(torch.float).sum().item()
                nof_defect_labeled += (pred.argmax(1) == ones).type(torch.float).sum().item()

                for i in range(list(targets.size())[0]):
                    if targets[i] == 1:
                        if pred.argmax(1)[i] == 1:
                            correct_defect_labeled += 1

                if nof_defect_samples > 0:
                    detection_rate = correct_defect_labeled / nof_defect_samples
                if nof_defect_labeled > 0:
                    precision = correct_defect_labeled / nof_defect_labeled

                """Computing avg scores"""
                for i in range(int(targets.size()[0])):
                    if targets[i] == 1:
                        if pred[i, 1] > pred[i, 0]:  # Correct defective prediction
                            correct_defect_total_d_score += pred[i, 1]
                            correct_defect_total_c_score += pred[i, 0]
                        if pred[i, 1] < pred[i, 0]:  # Wrong defective prediction
                            wrong_clean_total_d_score += pred[i, 1]
                            wrong_clean_total_c_score += pred[i, 0]

                    if targets[i] == 0:
                        if pred[i, 0] > pred[i, 1]:  # Correct clean prediction
                            correct_clean_total_d_score += pred[i, 1]
                            correct_clean_total_c_score += pred[i, 0]
                        if pred[i, 1] > pred[i, 0]:  # Wrong clean prediction
                            wrong_defect_total_d_score += pred[i, 1]
                            wrong_defect_total_c_score += pred[i, 0]


                if batch_idx % print_every == 0 or batch_idx == len(dataloader) - 1:
                    print(f'Epoch [{self.epoch:03d}] | Loss: {avg_loss:.4f} | '
                          f'Acc: {accuracy:.5f}[%]  '
                          f'({correct_labeled_samples}/{nof_samples}) '
                          f'Detection Rate: {detection_rate:.5f}[%]  '
                          f'({correct_defect_labeled}/{nof_defect_samples}) '
                          f'Precision: {precision:.5f}[%]  '
                          f'({correct_defect_labeled}/{nof_defect_labeled}) '
                          )


        nof_clean_labeled = nof_samples - nof_defect_labeled
        correct_clean_labeled = correct_labeled_samples - correct_defect_labeled
        if correct_defect_labeled > 0:
            correct_defect_avg_d_score = correct_defect_total_d_score / correct_defect_labeled
            correct_defect_avg_c_score = correct_defect_total_c_score / correct_defect_labeled
        if correct_clean_labeled > 0:
            correct_clean_avg_d_score = correct_clean_total_d_score / correct_clean_labeled
            correct_clean_avg_c_score = correct_clean_total_c_score / correct_clean_labeled

        wrong_defect_labeled = nof_defect_labeled - correct_defect_labeled
        wrong_clean_labeled = nof_clean_labeled - correct_clean_labeled

        if wrong_defect_labeled > 0:
            wrong_defect_avg_d_score = wrong_defect_total_d_score / wrong_defect_labeled
            wrong_defect_avg_c_score = wrong_defect_total_c_score / wrong_defect_labeled
        if wrong_clean_labeled > 0:
            wrong_clean_avg_d_score = wrong_clean_total_d_score / wrong_clean_labeled
            wrong_clean_avg_c_score = wrong_clean_total_c_score / wrong_clean_labeled

        print(f'avg defective score for true defected: {correct_defect_avg_d_score:.3f}')
        print(f'avg clean score for true defected: {correct_defect_avg_c_score:.3f}')
        print(f'avg defective score for false defected: {wrong_defect_avg_d_score:.3f}')
        print(f'avg clean score for false defected: {wrong_defect_avg_c_score:.3f}')
        print(f'avg defective score for true clean: {correct_clean_avg_d_score:.3f}')
        print(f'avg clean score for true clean: {correct_clean_avg_c_score:.3f}')
        print(f'avg defective score for false clean: {wrong_clean_avg_d_score:.3f}')
        print(f'avg clean score for false clean: {wrong_clean_avg_c_score:.3f}')

        return avg_loss, accuracy, detection_rate, precision

    def validate(self):
        """Evaluate the model performance."""
        return self.evaluate_model_on_dataloader(self.validation_dataset)

    def test(self):
        """Test the model performance."""
        return self.evaluate_model_on_dataloader(self.test_dataset)

    @staticmethod
    def write_output(logging_parameters: LoggingParameters, data: dict):
        """Write logs to json.

        Args:
            logging_parameters: LoggingParameters. Some parameters to log.
            data: dict. Holding a dictionary to dump to the output json.
        """
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        output_filename = f"{logging_parameters.dataset_name}_" \
                          f"{logging_parameters.model_name}_" \
                          f"{logging_parameters.optimizer_name}.json"
        output_filepath = os.path.join(os.getcwd(), OUTPUT_DIR, output_filename)

        print(f"Writing output to {output_filepath}")
        # Load output file
        if os.path.exists(output_filepath):
            # pylint: disable=C0103
            with open(output_filepath, 'r', encoding='utf-8') as f:
                all_output_data = json.load(f)
        else:
            all_output_data = []

        # Add new data and write to file
        all_output_data.append(data)
        # pylint: disable=C0103
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, indent=4)


    def run(self, epochs, logging_parameters: LoggingParameters):
        """Train, evaluate and test model on dataset, finally log results."""
        output_data = {
            "model": logging_parameters.model_name,
            "dataset": logging_parameters.dataset_name,
            "optimizer": {
                "name": logging_parameters.optimizer_name,
                "params": logging_parameters.optimizer_params,
            },
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "train_detect_rate": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_detect_rate": [],
            "test_loss": [],
            "test_acc": [],
            "test_precision": [],
            "test_detect_rate": [],
        }
        model_filename = f"{logging_parameters.dataset_name}_" \
                         f"{logging_parameters.model_name}_" \
                         f"{logging_parameters.optimizer_name}.pt"
        checkpoint_filename = os.path.join(CHECKPOINT_DIR, model_filename)
        best_val_loss = 10000
        for self.epoch in range(1, epochs + 1):
            print(f'Epoch {self.epoch}/{epochs}')

            train_loss, train_acc, train_detect_rate, train_precision = self.train_one_epoch()
            val_loss, val_acc, val_detect_rate, val_precision = self.validate()
            test_loss, test_acc, test_detect_rate, test_precision = self.test()

            output_data["train_loss"].append(train_loss)
            output_data["train_acc"].append(train_acc)
            output_data["train_detect_rate"].append(train_detect_rate)
            output_data["train_precision"].append(train_precision)
            output_data["val_loss"].append(val_loss)
            output_data["val_acc"].append(val_acc)
            output_data["val_detect_rate"].append(val_detect_rate)
            output_data["val_precision"].append(val_precision)
            output_data["test_loss"].append(test_loss)
            output_data["test_acc"].append(test_acc)
            output_data["test_detect_rate"].append(test_detect_rate)
            output_data["test_precision"].append(test_precision)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save checkpoint
                print(f'Saving checkpoint {checkpoint_filename}')
                torch.save(self.model.state_dict(), os.path.join(os.getcwd(), checkpoint_filename))

        # Writing to output file
        self.write_output(logging_parameters, output_data)
