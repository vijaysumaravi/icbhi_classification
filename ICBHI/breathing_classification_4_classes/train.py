#!/usr/bin/env python3
"""Recipe for training a breathing classification system from audio data only using the ICBHI dataset.
The system classifies audio samples into normal and abnormal breathing with a simple CNN model.

To run this recipe, do the following:

> python train.py hparams/train_utterance.yaml --data_folder /path/to/ICBHI

Authors
 * Vijay Ravi 2025
 Based on the IEMOCAP recipe.
"""

import csv
import os
import sys
from enum import Enum, auto
import torch
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
from tqdm.contrib import tqdm
import torchaudio

import speechbrain as sb
from compute_eval_metrics import compute_metrics, plot_training_curves


class Breathing_classification_Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + breathing classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig_padded

        #padding to increase sequence length to chunk_len in hparams

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats, lens)
        outputs = self.modules.classifier(embeddings)

        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        _, lens = batch.sig_padded
        breathingID, _ = batch.utterance_label_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        loss = self.hparams.compute_cost(predictions, breathingID, lens)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, breathingID, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def output_predictions_test_set(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and create output file (id, predictions, true values).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )

            save_file = os.path.join(
                self.hparams.output_folder, "predictions.csv"
            )
            with open(save_file, "w", newline="", encoding="utf-8") as csvfile:
                outwriter = csv.writer(csvfile, delimiter=",")
                outwriter.writerow(["id", "prediction", "true_value"])

        self.on_evaluate_start(max_key=max_key, min_key=min_key)  # done before
        self.modules.eval()
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1

                seg_ids = batch.id
                true_vals = batch.utterance_label_encoded.data.squeeze(dim=1).tolist()
                output = self.compute_forward(batch, stage=Stage.TEST)
                predictions = (
                    torch.argmax(output, dim=-1).squeeze(dim=1).tolist()
                )

                with open(
                    save_file, "a", newline="", encoding="utf-8"
                ) as csvfile:
                    outwriter = csv.writer(csvfile, delimiter=",")
                    for seg_id, prediction, true_val in zip(
                        seg_ids, predictions, true_vals
                    ):
                        outwriter.writerow([seg_id, prediction, true_val])

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break
        self.step = 0


class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "start_end_time")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start_end_time):
        """Load the signal and extract a single segment based on start/end timestamps.
        
        Arguments
        ---------
        wav : str
            Path to the audio file.
        start_end_time : tuple
            Tuple of (start_time, end_time) in seconds.
        """
        full_sig = sb.dataio.dataio.read_audio(wav)

        # Get sampling rate from the signal info
        info = torchaudio.info(wav)
        sample_rate = info.sample_rate
        
        # Convert timestamps to sample indices
        start_time, end_time = start_end_time
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Extract the segment
        sig = full_sig[start_sample:end_sample]

        return sig

    # repeat the audio signal for padding to increase sequence length to chunk_len in hparams
    @sb.utils.data_pipeline.takes("sig")
    @sb.utils.data_pipeline.provides("sig_padded")
    def pad_audio_signal(sig):
        # Calculate target length in samples (seconds * sample_rate)
        chunk_len = int(hparams["audio_processing_options"]["chunk_windows_seconds"] * hparams["sample_rate"])
        # Calculate how many repetitions are needed
        num_repeats = (chunk_len + len(sig) - 1) // len(sig)
        # Repeat the signal
        sig_repeated = torch.cat([sig] * num_repeats, dim=0)
        # Truncate to exactly chunk_len
        sig_padded = sig_repeated[:chunk_len]
        return sig_padded

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.ignore_len()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("utterance_label")
    @sb.utils.data_pipeline.provides("utterance_label", "utterance_label_encoded")
    def label_pipeline(utterance_label):
        yield utterance_label
        utterance_label_encoded = label_encoder.encode_label_torch(utterance_label)
        yield utterance_label_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, pad_audio_signal, label_pipeline],
            output_keys=["id", "sig_padded", "utterance_label_encoded"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="utterance_label",
    )

    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from prep_icbi import prep_icbi

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prep_icbi,
            kwargs={
                "audio_dir": hparams["audio_dir"],
                "annotation_dir": hparams["annotation_dir"],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
                "audio_processing_options": hparams["audio_processing_options"],
                "split_type": hparams["split_type"],
                "split_variable": hparams["split_variable"],
                "split_ratio": hparams["split_ratio"],
                "seed": hparams["seed"],
            },
        )

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)
    print(f"Current GPU: {torch.cuda.current_device()}")
    # Initialize the Brain object to prepare for mask training.
    breathing_classification_brain = Breathing_classification_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    breathing_classification_brain.fit(
        epoch_counter=breathing_classification_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = breathing_classification_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )

    # Create output file with predictions
    breathing_classification_brain.output_predictions_test_set(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )

    # Compute and display evaluation metrics
    predictions_file = os.path.join(hparams["output_folder"], "predictions.csv")
    label_encoder_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    # print("\nComputing evaluation metrics:")
    compute_metrics(predictions_file, label_encoder_file, output_folder=hparams["output_folder"])

    # Plot training curves
    log_file = os.path.join(hparams["output_folder"], "train_log.txt")
    plot_training_curves(log_file, hparams["output_folder"])
