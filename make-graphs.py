import json
import matplotlib.pyplot as plt
import numpy as np
results_dir = "./fine-tuning-results"

with open(f"{results_dir}/no-augment-eval.json", 'r') as f:
    no_aug_eval = json.load(f)
with open(f"{results_dir}/no-augment-loss.json", 'r') as f:
    no_aug_train = json.load(f)
with open(f"{results_dir}/with-augment-eval.json", 'r') as f:
    with_aug_eval = json.load(f)
with open(f"{results_dir}/with-augment-loss.json", 'r') as f:
    with_aug_train = json.load(f)

def aggregate(data, key):
    res = []
    for dico in data:
        res.append(dico[key])
    return res

no_augment_losses = aggregate(no_aug_train, "loss")
no_augment_losses_epochs = aggregate(no_aug_train, "epoch")
with_augment_losses = aggregate(with_aug_train, "loss")
with_augment_losses_epochs = aggregate(with_aug_train, "epoch")

plt.figure(figsize=(12, 8))
plt.ylim((0.75, 2))
plt.plot(no_augment_losses_epochs, no_augment_losses, color='red', label='Loss (no augmentations in train or test)')
plt.plot(with_augment_losses_epochs, with_augment_losses, color='green', label='Loss (with augmentations in train and test)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss during training')
plt.legend()
plt.savefig(f"{results_dir}/losses.png")

no_augment_accuracy = aggregate(no_aug_eval, "eval_accuracy")
no_augment_precision = aggregate(no_aug_eval, "eval_precision")
no_augment_recall = aggregate(no_aug_eval, "eval_recall")
no_augment_metrics_epochs = aggregate(no_aug_eval, "epoch")
with_augment_accuracy = aggregate(with_aug_eval, "eval_accuracy")
with_augment_precision = aggregate(with_aug_eval, "eval_precision")
with_augment_recall = aggregate(with_aug_eval, "eval_recall")
with_augment_metrics_epochs = aggregate(with_aug_eval, "epoch")
plt.figure(figsize=(12, 8))
plt.plot(no_augment_metrics_epochs, no_augment_accuracy, ls='-', color='red', label='Accuracy (no augmentations in train or test)')
plt.plot(no_augment_metrics_epochs, no_augment_precision, ls='--', color='red', label='Precision (no augmentations in train or test)')
plt.plot(no_augment_metrics_epochs, no_augment_recall, ls=':', color='red', label='Recall (no augmentations in train or test)')
plt.plot(with_augment_metrics_epochs, with_augment_accuracy, ls='-', color='green', label='Accuracy (with augmentations in train and test)')
plt.plot(with_augment_metrics_epochs, with_augment_precision, ls='--', color='green', label='Precision (with augmentations in train and test)')
plt.plot(with_augment_metrics_epochs, with_augment_recall, ls=':', color='green', label='Recall (with augmentations in train and test)')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Metrics at eval phase')
plt.legend()
plt.savefig(f"{results_dir}/metrics.png")

no_augment_perplexity = np.exp(np.array(aggregate(no_aug_eval, "eval_loss")))
with_augment_perplexity = np.exp(np.array(aggregate(with_aug_eval, "eval_loss")))
plt.figure(figsize=(12, 8))
plt.plot(no_augment_metrics_epochs, no_augment_perplexity, ls='-', color='red', label='Perplexity (no augmentations in train or test)')
plt.plot(with_augment_metrics_epochs, with_augment_perplexity, ls='-', color='green', label='Perplexity (with augmentations in train and test)')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity at eval phase')
plt.savefig(f"{results_dir}/perplexity.png")