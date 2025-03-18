import torch
import os
import numpy as np
import random
import pandas as pd
import datetime

from torch.utils.data import TensorDataset,random_split
from torch.utils.data import DataLoader, TensorDataset

from models.VCNet import *
from utils.config_vcnet import *
from utils.metrics import *
from utils.data_aug import *
from utils.dataprocess import *
from utils.explainer_base import *
from utils.explainer_interaction import *
from utils.tools import *

from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

import logging

# # 初始化日志设置
# logging.basicConfig(
#     filename="predictions.log",  # 日志文件名称
#     level=logging.INFO,          # 设置日志级别
#     format="%(asctime)s - %(message)s"  # 日志格式
# )
class Trainer:
    def __init__(self, model, loaders, optimizer, scheduler, lossfn, params):
        self.gpu_id = params.device
        self.model = model.to(self.gpu_id)
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lossfn = lossfn
        self.params = params
        self.epochs_run = 0
        self.current_epoch = 0
        if not os.path.exists(params.savePath):  # 如果savePath不存在就生成一个
            os.makedirs(params.savePath, exist_ok=True)
        self.snapshot_path = f"%s{self.params.model_name}_cf{self.params.fold}_model.pt" % params.savePath
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)

    def _load_snapshot(self, snapshot_path):
        print('Load model now')
        # loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=self.gpu_id)
        self.model.load_state_dict(snapshot["model"])
        self.epochs_run = snapshot["epochs"]

    def _save_snapshot(self, epochs, bestMtc=None):
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc, 'model': self.model.state_dict()}
        torch.save(stateDict, self.snapshot_path)
        print(f"Epoch {epochs} | Training snapshot saved at {self.snapshot_path}")

    def compute_cf_loss(self, pred_cf, x_cf, x_original, target_cf):#, 
                    #proximity_weight=0.1, validity_weight=5.0, diversity_weight=0.05, sparsity_weight=0.1):
        """
        Compute counterfactual loss with balanced validity, proximity, diversity, and sparsity terms.
        """
         # Initial weights
        proximity_weight = 0.04
        validity_weight = 10.0
        diversity_weight = 0.03
        sparsity_weight = 0.05
        
        # Dynamic weight adjustment
        # epoch_scaler = min(self.current_epoch / self.params.num_epochs, 1.0)
        # validity_weight = 10.0 * (1 + epoch_scaler)
        # proximity_weight = 0.04 * (1 - epoch_scaler * 0.5)
        # diversity_weight = 0.03 * (1 + epoch_scaler * 0.5)
        # sparsity_weight = 0.05 * (1 - epoch_scaler * 0.5)
        
        if pred_cf.shape != target_cf.shape:
            if pred_cf.dim() == 1 and target_cf.dim() == 2:
                target_cf = target_cf.squeeze(1)  # 调整 target_cf 形状
            elif pred_cf.dim() == 2 and target_cf.dim() == 1:
                pred_cf = pred_cf.squeeze(1)  # 调整 pred_cf 形状
                    
        # Validity Loss: Ensure counterfactuals flip predictions
        validity_loss = self.lossfn(pred_cf, target_cf)

        # Proximity Loss: Penalize large changes from original inputs (Manhattan Distance)
        # proximity_loss = torch.mean(torch.abs(x_cf - x_original))
        
        # Proximity Loss: Scaled Manhattan distance
        proximity_loss = torch.norm(x_cf - x_original, p=1) / x_cf.size(1)

        # Diversity Loss: Encourage variance among counterfactuals
        diversity_loss = torch.clamp(-torch.mean(torch.cdist(x_cf, x_cf)), min=-5.0) / x_cf.size(0)

        # # Sparsity Loss: Penalize changes in too many features
        sparsity_loss = torch.sum(torch.abs(x_cf - x_original) > 0.1).float() / x_cf.size(0)
        #The overall performance on the test data are Acc:0.744, F1-score:0.488, ROC:0.772, Sensitivity:0.769, Specificity:0.739, Matthews Coef:0.392!!!
        
        # Sparsity Loss with weighted penalties
        # feature_importance = torch.abs(self.model.encoder[0].weight).mean(dim=0)
        # sparsity_loss = torch.sum((torch.abs(x_cf - x_original) > 0.1) * feature_importance) / x_cf.size(0)
        #The overall performance on the test data are Acc:0.768, F1-score:0.486, ROC:0.791, Sensitivity:0.692, Specificity:0.783, Matthews Coef:0.381!!!

    
        # Total Counterfactual Loss
        cf_loss = (
            validity_weight * validity_loss +
            proximity_weight * proximity_loss +
            diversity_weight * diversity_loss +
            sparsity_weight * sparsity_loss
        )
        # print(f'cf_loss: {cf_loss}, validity_loss: {validity_loss}, proximity_loss: {proximity_loss}, diversity_loss: {diversity_loss}, sparsity_loss: {sparsity_loss}')
        return cf_loss

    def train(self):
        train_dataloader, val_dataloader, test_dataloader = self.loaders
        best_record = {'train_loss': 0, 'train_acc': 0, 'train_f': 0, 'train_roc': 0, 'train_sen': 0,'train_spe': 0,'train_mcc': 0, 'valid_loss': 0,  'valid_acc': 0, 'valid_f': 0, 'valid_roc': 0, 'valid_sen': 0,  'valid_spe': 0, 'valid_mcc': 0}
        nobetter, best_f1 = 0, 0.0
        
        for epoch in range(self.epochs_run, self.params.num_epochs):
            self.current_epoch = epoch
            train_loss, train_acc, train_f, train_roc, train_sen, train_spe, train_mcc = self.train_epoch(train_dataloader, self.model, self.lossfn, self.optimizer, self.gpu_id, self.params.pred_thres)
            test_loss, test_acc, test_f, test_roc, test_sen, test_spe, test_mcc = self.test_epoch(val_dataloader, self.model, self.lossfn, self.gpu_id, self.params.pred_thres)
            self.scheduler.step(test_loss)
            print(
                ">>>Epoch:{} of Train Loss:{:.3f}, Valid Loss:{:.3f}\n"
                "Train Acc:{:.3f}, Train F1-score:{:.3f}, Train ROC:{:.3f}, Train Sensitivity:{:.3f}, Train Specificity:{:.3f}, Train Matthews Coef:{:.3f};\n"
                "Valid Acc:{:.3f}, Valid F1-score:{:.3f}, Valid ROC:{:.3f}, Valid Sensitivity:{:.3f}, Valid Specificity:{:.3f}, Valid Matthews Coef:{:.3f}!!!\n".format(
                    epoch, train_loss, test_loss,
                    train_acc, train_f, train_roc, train_sen, train_spe, train_mcc,
                    test_acc, test_f, test_roc, test_sen, test_spe, test_mcc))
            if best_f1 < test_f:
                nobetter = 0
                best_f1 = test_f
                best_record['train_loss'] = train_loss
                best_record['test_loss'] = test_loss
                best_record['train_acc'] = train_acc
                best_record['test_acc'] = test_acc
                best_record['train_f'] = train_f
                best_record['test_f'] = test_f
                best_record['train_roc'] = train_roc
                best_record['test_roc'] = test_roc
                best_record['train_sen'] = train_sen
                best_record['test_sen'] = test_sen
                best_record['train_spe'] = train_spe
                best_record['test_spe'] = test_spe
                best_record['train_mcc'] = train_mcc
                best_record['test_mcc'] = test_mcc
                print(f'>Bingo!!! Get a better Model with valid f1: {best_f1:.3f}!!!')
                self._save_snapshot(epoch, best_f1)
            else:
                nobetter += 1
                if nobetter >= self.params.earlyStop:
                    print(f'Valid f1 has not improved for more than {self.params.earlyStop} steps in epoch {epoch}, stop training.')
                    break
        print("Finally,the model's Valid Acc:{:.3f}, Valid F1-score:{:.3f}, Valid ROC:{:.3f}, Valid Sensitivity:{:.3f}, Valid Specificity:{:.3f}, Valid Matthews Coef:{:.3f}!!!\n\n\n".format(
                best_record['test_acc'], best_record['test_f'],  best_record['test_roc'], best_record['test_sen'], best_record['test_spe'], best_record['test_mcc']))            
        self.test(test_dataloader, self.model, self.gpu_id, cls_threshold=self.params.pred_thres, cf_threshold=self.params.pred_cf_thres)
        
    def train_epoch(self, train_dataloader, model, loss_fn, optimizer, gpu_id, threshold=0.5):
        train_loss, train_acc, train_f, train_roc, train_sen, train_spe, train_mcc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        model.train()
        pred_list, prob_list, label_list = [], [], []
        epoch_loss = 0.0
        num_batches = len(train_dataloader)
        for i, (x_batch, y_batch) in enumerate(train_dataloader):
            x_batch, y_batch = x_batch.cuda(gpu_id), y_batch.cuda(gpu_id)
            optimizer.zero_grad()
            # Factual and Counterfactual Predictions
            pred, x_cf = model(x_batch, target_class=1 - y_batch)
            
            if pred.shape != y_batch.shape:
                if pred.dim() == 1 and y_batch.dim() == 2:
                    y_batch = y_batch.squeeze(1)  # 调整 target_cf 形状
                elif pred.dim() == 2 and y_batch.dim() == 1:
                    pred = pred.squeeze(1)  # 调整 pred_cf 形状
            cls_loss = loss_fn(pred, y_batch)
            # Counterfactual Loss
            pred_cf, _ = model(x_cf, target_class=None)  # Predict on counterfactual samples
            cf_loss = self.compute_cf_loss(
                pred_cf=pred_cf.squeeze(1),
                x_cf=x_cf,
                x_original=x_batch,
                target_cf=1 - y_batch
            )
            # Total Loss
            loss = cls_loss + cf_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            pred_label = (pred > threshold).float()
            prob_list.extend(pred.detach().cpu().numpy())
            pred_list.extend(pred_label.detach().cpu().numpy())
            label_list.extend(y_batch.cpu().numpy())
                
        train_loss = epoch_loss/num_batches
        train_acc, train_f, train_roc, train_sen, train_spe, train_mcc = accuracy(label_list, pred_list), f1(label_list, pred_list), auc(label_list, prob_list), sensitivity(label_list, pred_list), specificity(label_list, pred_list), mcc(label_list, pred_list)
        return train_loss, train_acc, train_f, train_roc, train_sen, train_spe, train_mcc
    
    def test_epoch(self, test_dataloader, model, loss_fn, gpu_id, threshold=0.5):
        test_loss, test_acc, test_f, test_roc, test_sen, test_spe, test_mcc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        model.eval()
        pred_list, prob_list, label_list = [], [], []
        epoch_loss = 0.0
        num_batches = len(test_dataloader)
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                x_batch, y_batch = x_batch.cuda(gpu_id), y_batch.cuda(gpu_id)
                # Factual and Counterfactual Predictions
                pred, x_cf = model(x_batch, target_class=1 - y_batch)
                
                if pred.shape != y_batch.shape:
                    if pred.dim() == 1 and y_batch.dim() == 2:
                        y_batch = y_batch.squeeze(1)  # 调整 target_cf 形状
                    elif pred.dim() == 2 and y_batch.dim() == 1:
                        pred = pred.squeeze(1)  # 调整 pred_cf 形状
                cls_loss = loss_fn(pred, y_batch)
                # Counterfactual Loss
                pred_cf, _ = model(x_cf, target_class=None)  # Predict on counterfactual samples
                cf_loss = self.compute_cf_loss(
                    pred_cf=pred_cf,
                    x_cf=x_cf,
                    x_original=x_batch,
                    target_cf=1 - y_batch
                )
                # Total Loss
                loss = cls_loss + cf_loss
                epoch_loss += loss.item()
                pred_label = (pred > threshold).float()
                prob_list.extend(pred.detach().cpu().numpy())
                pred_list.extend(pred_label.detach().cpu().numpy())
                label_list.extend(y_batch.cpu().numpy())
                # if i == num_batches-1:
                #     logging.info(f"probability: {prob_list}, gt_label: {label_list}")
        test_loss = epoch_loss/num_batches
        test_acc, test_f, test_roc, test_sen, test_spe, test_mcc = accuracy(label_list, pred_list), f1(label_list, pred_list), auc(label_list, prob_list), sensitivity(label_list, pred_list), specificity(label_list, pred_list), mcc(label_list, pred_list)
        return test_loss, test_acc, test_f, test_roc, test_sen, test_spe, test_mcc
    
    def evaluate_counterfactual_performance(self, X, X_cf, cf_pred_list, y, sparsity_threshold=0.1):
        """
        Evaluate counterfactual performance based on validity, proximity, diversity, and sparsity.

        Args:
            X: Original inputs (tensor or compatible iterable).
            X_cf: Counterfactual inputs (tensor or compatible iterable).
            cf_pred_list: List or tensor of counterfactual predictions.
            y: Original labels (list or tensor).
            sparsity_threshold: Threshold for significant feature change (default: 0.1).

        Returns:
            validity: Fraction of counterfactuals that flip predictions.
            proximity: Average distance from original inputs (mean squared difference).
            diversity: Variance among counterfactuals (mean pairwise distance).
            sparsity: Average number of features with significant changes.
        """
        # Ensure cf_pred_list is a tensor
        cf_pred_list = torch.tensor(np.array(cf_pred_list)) if isinstance(cf_pred_list, list) else cf_pred_list
        X = torch.stack(X) if isinstance(X, list) else X
        X_cf = torch.stack(X_cf) if isinstance(X_cf, list) else X_cf
        y = torch.tensor(np.array(y)) if isinstance(y, list) else y

        # Validity: Fraction of counterfactuals that flip predictions
        validity = (cf_pred_list == (1 - y)).float().mean().item()

        # Proximity: Average distance from original inputs
        proximity = torch.mean((X_cf - X) ** 2).item()

        # Diversity: Variance among counterfactuals (pairwise distance)
        diversity = torch.mean(torch.cdist(X_cf, X_cf)).item()

        # Sparsity: Average number of features with significant changes
        feature_changes = torch.abs(X_cf - X) > sparsity_threshold
        sparsity = torch.mean(feature_changes.float().sum(dim=1)).item()

        return validity, proximity, diversity, sparsity

        
        
    def test(self, eval_dataloader, model, gpu_id, cls_threshold=0.5, cf_threshold=0.5):
        model.eval()
        pred_list, prob_list, label_list = [], [], [] 
        X, X_cf, y, cf_pred_list = [], [], [], []
        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(eval_dataloader):
                x_batch, y_batch = x_batch.cuda(gpu_id), y_batch.cuda(gpu_id)
                # Factual and Counterfactual Predictions
                # print('x_batch valid', x_batch.shape)
                pred, x_cf = model(x_batch, target_class=1 - y_batch)
                # Counterfactual Loss
                pred_cf, _ = model(x_cf, target_class=None)  # Predict on counterfactual samples
                pred_label = (pred > cls_threshold).float()
                prob_list.extend(pred.detach().cpu().numpy())
                pred_list.extend(pred_label.detach().cpu().numpy())
                label_list.extend(y_batch.cpu().numpy())
                cf_pred_label = (pred_cf > cf_threshold).float()
                cf_pred_list.extend(cf_pred_label.detach().cpu().numpy())
                X.extend(x_batch.detach().cpu())
                X_cf.extend(x_cf.detach().cpu())
                y.extend(y_batch.detach().cpu())
        
        valid_acc, valid_f, valid_roc, valid_sen, valid_spe, valid_mcc = accuracy(label_list, pred_list), f1(label_list, pred_list), auc(label_list, prob_list), sensitivity(label_list, pred_list), specificity(label_list, pred_list), mcc(label_list, pred_list)
        validity, proximity, diversity, sparsity = self.evaluate_counterfactual_performance(X, X_cf, cf_pred_list, y)
        print("The overall performance on the test data are Acc:{:.3f}, F1-score:{:.3f}, ROC:{:.3f}, Sensitivity:{:.3f}, Specificity:{:.3f}, Matthews Coef:{:.3f}!!!".format(valid_acc, valid_f, valid_roc, valid_sen, valid_spe, valid_mcc))
        print("Validity:{:.3f}, Proximity:{:.3f}, Diversity:{:.3f}, Sparsity:{:.3f}".format(validity, proximity, diversity, sparsity))

def random_upsample_dataset(X, y, pos_neg_ratio=1.0, random_state=42):
    """
    Balances the dataset by oversampling the minority class to achieve the desired positive-to-negative ratio.

    Parameters:
    - X (pd.DataFrame): Feature matrix
    - y (pd.Series): Target labels
    - pos_neg_ratio (float): Desired ratio of positive to negative samples (e.g., 0.5 for 1:2).

    Returns:
    - X_balanced (pd.DataFrame): Balanced feature matrix
    - y_balanced (pd.Series): Balanced target labels
    """
    # Calculate the desired minority class size based on the ratio
    class_counts = y.value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    original_minority_size = class_counts[minority_class]
    desired_minority_size = int(class_counts[majority_class] * pos_neg_ratio)

    # Ensure the desired size is not less than the original minority class size
    if desired_minority_size < original_minority_size:
        print(f"Warning: desired_minority_size ({desired_minority_size}) is less than the original size "
              f"({original_minority_size}). Adjusting to original size.")
        desired_minority_size = original_minority_size

    # Define sampling strategy
    sampling_strategy = {majority_class: class_counts[majority_class], minority_class: desired_minority_size}

    # Apply RandomOverSampler with the specified sampling strategy
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_balanced, y_balanced = ros.fit_resample(X, y)

    return X_balanced, y_balanced


def similarity_upsample_dataset(X, y, pos_neg_ratio=1.0, method="dissimilar"):
    """
    Upsample minority class samples based on similarity or dissimilarity to majority class samples.
    
    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target variable.
        pos_neg_ratio (float): Desired positive-to-negative sample ratio.
        method (str): Upsampling method, "similar" or "dissimilar".

    Returns:
        X_upsampled (pd.DataFrame): Upsampled feature matrix.
        y_upsampled (pd.Series): Upsampled target variable.
    """
    # Ensure X and y are DataFrames/Series for compatibility
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    # Separate majority and minority class samples
    class_counts = y.value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    X_majority = X[y == majority_class]
    X_minority = X[y == minority_class]

    # Compute pairwise similarity (e.g., cosine similarity)
    similarity_matrix = cosine_similarity(X_minority, X_majority)

    # Define selection strategy
    total_minority_samples = int(len(X_majority) * pos_neg_ratio)
    if method == "dissimilar":
        # Select the least similar minority samples
        dissimilarity_scores = similarity_matrix.min(axis=1)
        selected_indices = np.argsort(dissimilarity_scores)[:len(X_minority)]
    elif method == "similar":
        # Select the most similar minority samples
        similarity_scores = similarity_matrix.max(axis=1)
        selected_indices = np.argsort(similarity_scores)[-len(X_minority):]
    else:
        raise ValueError("Invalid method. Choose 'similar' or 'dissimilar'.")

    # Select the minority samples
    X_minority_selected = X_minority.iloc[selected_indices]
    y_minority_selected = y[y == minority_class].iloc[selected_indices]

    # Duplicate selected minority samples to achieve the desired ratio
    repeat_factor = int(np.ceil(total_minority_samples / len(X_minority_selected)))
    X_minority_upsampled = pd.concat([X_minority_selected] * repeat_factor, ignore_index=True)[:total_minority_samples]
    y_minority_upsampled = pd.concat([y_minority_selected] * repeat_factor, ignore_index=True)[:total_minority_samples]

    # Combine with original majority samples to achieve the exact ratio
    X_upsampled = pd.concat([X_majority, X_minority_upsampled], ignore_index=True)
    y_upsampled = pd.concat([pd.Series([majority_class] * len(X_majority)), pd.Series([minority_class] * total_minority_samples)], ignore_index=True)

    return X_upsampled, y_upsampled


def cluster_upsample_dataset(X, y, max_clusters=10, pos_neg_ratio=1.0, random_state=42):
    """
    Upsample the minority class using clustering and resampling.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target variable.
        max_clusters (int): Maximum number of clusters to consider for the minority class.
        pos_neg_ratio (float): Desired ratio of positive to negative samples.

    Returns:
        X_upsampled (pd.DataFrame): Upsampled feature matrix.
        y_upsampled (pd.Series): Upsampled target variable.
    """

    def find_optimal_clusters(X, max_clusters):
        if X.shape[0] < 2:
            raise ValueError("Insufficient samples for clustering. At least two samples are required.")
        silhouette_scores = []
        for k in range(2, min(max_clusters, X.shape[0]) + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            kmeans.fit(X)
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
        optimal_k = 2 + silhouette_scores.index(max(silhouette_scores))
        return optimal_k

    # Ensure input is compatible
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    # Separate majority and minority class samples
    class_counts = y.value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    X_majority = X[y == majority_class]
    X_minority = X[y == minority_class]

    # Determine optimal number of clusters
    num_clusters = find_optimal_clusters(X_minority, max_clusters)

    # Cluster minority samples
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_minority)

    # Calculate total number of minority samples needed
    desired_minority_size = int(len(X_majority) * pos_neg_ratio)

    # Upsample each cluster proportionally
    X_upsampled = X.copy()
    y_upsampled = y.copy()
    for cluster in range(num_clusters):
        cluster_samples = X_minority[np.array(cluster_labels) == cluster]
        if len(cluster_samples) == 0:
            continue
        n_samples_to_add = int(
            desired_minority_size / num_clusters
        )  # Distribute samples across clusters
        cluster_upsampled = resample(cluster_samples, replace=True, n_samples=n_samples_to_add, random_state=random_state)
        X_upsampled = pd.concat([X_upsampled, cluster_upsampled], ignore_index=True)
        y_upsampled = pd.concat([y_upsampled, pd.Series([minority_class] * len(cluster_upsampled))], ignore_index=True)

    # Ensure the final minority size matches the desired size
    if len(y_upsampled[y_upsampled == minority_class]) > desired_minority_size:
        excess = len(y_upsampled[y_upsampled == minority_class]) - desired_minority_size
        X_upsampled = X_upsampled.drop(X_upsampled[y_upsampled == minority_class].iloc[:excess].index)
        y_upsampled = y_upsampled.drop(y_upsampled[y_upsampled == minority_class].iloc[:excess].index)

    return X_upsampled, y_upsampled

def hybrid_upsample_dataset(X, y, pos_neg_ratio=1.0, random_weight=0.5, dissimilar_weight=0.5, random_state=42):
    """
    Hybrid upsampling method combining Random and Dissimilar strategies.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target variable.
        pos_neg_ratio (float): Desired ratio of positive to negative samples.
        random_weight (float): Proportion of samples from Random Upsampling.
        dissimilar_weight (float): Proportion of samples from Dissimilar Upsampling.

    Returns:
        X_upsampled (pd.DataFrame): Upsampled feature matrix.
        y_upsampled (pd.Series): Upsampled target variable.
    """
    assert random_weight + dissimilar_weight == 1.0, "Weights must sum to 1.0"

    # Ensure inputs are DataFrames/Series
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    # Separate majority and minority class samples
    class_counts = y.value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    X_majority = X[y == majority_class]
    X_minority = X[y == minority_class]

    # Desired total minority samples
    desired_minority_size = int(len(X_majority) * pos_neg_ratio)

    # Step 1: Random Upsampling
    random_sample_size = int(desired_minority_size * random_weight)
    X_random_upsampled = resample(X_minority, replace=True, n_samples=random_sample_size, random_state=random_state)
    y_random_upsampled = pd.Series([minority_class] * random_sample_size)

    # Step 2: Dissimilar Upsampling
    remaining_samples = desired_minority_size - random_sample_size
    similarity_matrix = cosine_similarity(X_minority, X_majority)
    dissimilarity_scores = similarity_matrix.min(axis=1)
    dissimilar_indices = np.argsort(dissimilarity_scores)[:remaining_samples]
    X_dissimilar_upsampled = X_minority.iloc[dissimilar_indices]
    y_dissimilar_upsampled = pd.Series([minority_class] * len(dissimilar_indices))

    # Combine random and dissimilar upsampling results
    X_combined = pd.concat([X_majority, X_random_upsampled, X_dissimilar_upsampled], ignore_index=True)
    y_combined = pd.concat([pd.Series([majority_class] * len(X_majority)), y_random_upsampled, y_dissimilar_upsampled], ignore_index=True)

    return X_combined, y_combined


def get_dataset(file_path, params, train_ratio=0.7, test_ratio=0.2):
    """
    Reads the dataset, balances the training data, and splits it into training and testing datasets.

    Parameters:
    - file_path (str): Path to the dataset file.
    - params (dict): Parameters for processing.
    - train_ratio (float): Ratio of data to use for training.

    Returns:
    - train_dataset (TensorDataset): Training dataset (balanced).
    - test_dataset (TensorDataset): Testing dataset (original).
    - columns (Index): Feature column names.
    - input_dim (int): Number of input features.
    """
    upsample_type = params.upsample_type
    if upsample_type == 'hybrid':
        mro = params.mixed_ratio
        rw, dw = mro.split('-')
        random_weight=float('0.'+rw)
        dissimilar_weight=float('0.'+dw)
    pos_neg_ratio = params.pos_neg_ratio
    data = pd.read_csv(file_path)
    # Split into training and testing data based on train_ratio
    train_size = int(train_ratio * len(data))
    test_size = int(test_ratio * len(data))
    valid_size = len(data) - train_size - test_size
    
     # Shuffle and split data
    shuffled_data = data.sample(frac=1, random_state=params.seed).reset_index(drop=True)
    train_data = shuffled_data.iloc[:train_size]
    valid_data = shuffled_data.iloc[train_size:train_size + valid_size]
    test_data = shuffled_data.iloc[train_size + valid_size:]

    # Separate features and labels
    X_train, y_train = train_data.drop(columns=['label']), train_data['label']
    X_valid, y_valid = valid_data.drop(columns=['label']), valid_data['label']
    X_test, y_test = test_data.drop(columns=['label']), test_data['label']

    # Balance the training and valid data
    if upsample_type == 'random':
        X_train_blance, y_train_blance = random_upsample_dataset(X_train, y_train, pos_neg_ratio=pos_neg_ratio,random_state=params.seed)
        X_valid_blance, y_valid_blance = random_upsample_dataset(X_valid, y_valid, pos_neg_ratio=pos_neg_ratio,random_state=params.seed)
    elif upsample_type == 'dissimilar':
        X_train_blance, y_train_blance = similarity_upsample_dataset(X_train, y_train, pos_neg_ratio=pos_neg_ratio, method="dissimilar")
        X_valid_blance, y_valid_blance = similarity_upsample_dataset(X_valid, y_valid, pos_neg_ratio=pos_neg_ratio, method="dissimilar")
    elif upsample_type == 'similar':
        X_train_blance, y_train_blance = similarity_upsample_dataset(X_train, y_train, pos_neg_ratio=pos_neg_ratio, method="similar")
        X_valid_blance, y_valid_blance = similarity_upsample_dataset(X_valid, y_valid, pos_neg_ratio=pos_neg_ratio, method="similar")
    elif upsample_type == 'cluster':
        X_train_blance, y_train_blance = cluster_upsample_dataset(X_train, y_train, max_clusters=10, pos_neg_ratio=pos_neg_ratio, random_state=params.seed)
        X_valid_blance, y_valid_blance = cluster_upsample_dataset(X_valid, y_valid, max_clusters=10, pos_neg_ratio=pos_neg_ratio, random_state=params.seed)
    elif upsample_type == 'hybrid':
        X_train_blance, y_train_blance = hybrid_upsample_dataset(X_train, y_train, random_weight=random_weight, dissimilar_weight=dissimilar_weight, pos_neg_ratio=pos_neg_ratio, random_state=params.seed)
        X_valid_blance, y_valid_blance = hybrid_upsample_dataset(X_valid, y_valid, random_weight=random_weight, dissimilar_weight=dissimilar_weight, pos_neg_ratio=pos_neg_ratio, random_state=params.seed)
    else:##default no balance
        X_train_blance, y_train_blance = X_train, y_train
        X_valid_blance, y_valid_blance = X_valid, y_valid
        
    # Convert training and valid data to tensors
    X_train_tensor = torch.tensor(X_train_blance.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_blance.values, dtype=torch.float32).unsqueeze(1)
    X_valid_tensor = torch.tensor(X_valid_blance.values, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid_blance.values, dtype=torch.float32).unsqueeze(1)

    # Convert original testing data to tensors
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Create TensorDataset objects for training and testing data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, valid_dataset, test_dataset, X_train.columns, X_train.shape[1]

def get_dataloader(file_path, params, train_ratio=0.7, test_ratio=0.2):
    train_dataset, valid_dataset, test_dataset, columns, input_dim = get_dataset(file_path, params, train_ratio=train_ratio, test_ratio=test_ratio)                   
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=params.batchSize, shuffle=True, drop_last=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=params.batchSize, shuffle=False, drop_last=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=params.batchSize, shuffle=False, drop_last=False, num_workers=1)

    # Print data stats
    print(f"Training data: {len(train_dataset)}, Validing data: {len(valid_dataset)}, Testing data: {len(test_dataset)}")
    check_lbl('Training', train_loader)
    check_lbl('Validing', valid_loader)
    check_lbl('Testing', test_loader)

    return input_dim, columns, [train_loader, valid_loader, test_loader]

def check_lbl(name, dataloader):
    zero, one = 0, 0
    for _, (_, labels) in enumerate(dataloader):
        one += torch.sum(labels == 1).item()
        zero += torch.sum(labels == 0).item()
    print(f'In {name}, there are {one} 1s, and {zero} 0s.')

def load_model_objs(input_dim, params):
    encoder_hid_dim, encoder_out_dim, pred_hid_dim, pred_out_dim, decoder_hid_dim, drop = \
    params.encoder_hid_dim, params.encoder_out_dim, params.pred_hid_dim, params.pred_out_dim, params.decoder_hid_dim, params.drop
    device = params.device
    model = VCNet(input_dim, encoder_hid_dim, encoder_out_dim, pred_hid_dim, pred_out_dim, decoder_hid_dim, drop)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    lossfn = nn.BCELoss()
    # lossfn = FocalLoss(alpha=0.25, gamma=2.0)
    return model, optimizer, scheduler, lossfn

def explain_prepare(train_dataloader, test_dataloader, model, gpu_id):
    X_train_list, y_train_list = [], []
    for _, (x_batch, y_batch) in enumerate(train_dataloader):
        X_train_list.extend(x_batch)
        y_train_list.extend(y_batch)
    X_train_numpy = torch.stack(X_train_list).numpy()
    y_train_numpy = torch.tensor(y_train_list).numpy()
    
    X_test_list, y_test_list, X_test_cf_list, test_pred_list = [], [], [], []
    model.eval()
    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(test_dataloader):
            x_batch, y_batch = x_batch.cuda(gpu_id), y_batch.cuda(gpu_id)
            pred, X_cf = model(x_batch, target_class=1 - y_batch)
            X_test_list.extend(x_batch.detach().cpu())
            y_test_list.extend(y_batch.detach().cpu())
            test_pred_list.extend(pred.detach().cpu())
            X_test_cf_list.extend(X_cf.detach().cpu())
          
    X_test_numpy = torch.stack(X_test_list).numpy()
    y_test_numpy = torch.stack(y_test_list).numpy()
    X_test_cf_numpy = torch.stack(X_test_cf_list).numpy()
    test_pred_numpy = torch.tensor(np.array(test_pred_list)).numpy()
    return X_train_numpy, X_test_numpy, y_test_numpy, test_pred_numpy, X_test_cf_numpy
    
def explainer_func(train_dataloader, test_dataloader, model, params, feature_names, instance_id=0):
    X_train, X_test, y_test, pred_test, X_test_cf = explain_prepare(train_dataloader, test_dataloader, model, params.device)
    explainer = Explainer(model, params.model_name, params.explain_mode, X_train, X_test, y_test, pred_test, X_test_cf, feature_names, instance_id)
    explainer.explain()
    interexplainer = InterExplainer(model, params, X_train, X_test, pred_test, X_test_cf, feature_names)
    interexplainer.interaction_explain()

def main():
    print("Running Python File:", os.path.basename(__file__))
    params = config()
    setup_seed(params.seed)    
    starttime = datetime.datetime.now()
    data_time1 = datetime.datetime.now()
    input_dim, columns, loaders = get_dataloader(params.datapath, params)
    data_time2 = datetime.datetime.now()
    print(f'Data Loading Time is {(data_time2 - data_time1).seconds}s. ')
    data_name = params.datapath[params.datapath.rindex('_')+1:params.datapath.index('.')]
    print(f'Model name: {params.model_name}, data name: {data_name}, upsampling: {params.upsample_type}, batch: {params.batchSize}, epoch:{params.num_epochs}, lr: {params.lr}, input_dim: {input_dim}, encoder_hid_dim: {params.encoder_hid_dim}, encoder_out_dim: {params.encoder_out_dim}, pred_hid_dim: {params.pred_hid_dim}, pred_out_dim: {params.pred_out_dim}, decoder_hid_dim: {params.decoder_hid_dim}, drop: {params.drop}, explain mode: {params.explain_mode}')
    train_time1 = datetime.datetime.now()
    model, optimizer, scheduler, lossfn = load_model_objs(input_dim, params)
    trainer = Trainer(model, loaders, optimizer, scheduler, lossfn, params)
    trainer.train()
    train_time2= datetime.datetime.now()
    print(f'Train time is {(train_time2 - train_time1).seconds}s. ')
    endtime = datetime.datetime.now()
    print('Start to interpret the predictions')
    explainer_func(loaders[0], loaders[2], model, params, columns, instance_id=0)
    print(f'Total running time of all codes is {(endtime - starttime).seconds}s. ')

    
if __name__ == '__main__':
    main()
    
# The overall performance on the test data are Acc:0.768, F1-score:0.537, ROC:0.818, Sensitivity:0.846, Specificity:0.754, Matthews Coef:0.462!!!
# Validity:1.000, Proximity:0.223, Diversity:0.221, Sparsity:8.146
# Train time is 81s. 
# Start to interpret the predictions
# SHAP explaination    
# The global explanation:
# The factual explanation of the positive instance 50 with the below features [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0.
#  0. 0.] and label: [1.]:
# The counterfactual explanation of the positive instance 50:
# The factual explanation of the negative instance 15 with the below features [1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
#  1. 0.] and label: [0.]:
# The counterfactual explanation of the negative instance 15:
# Feature interaction analysis for all pairs...
# Feature interaction analysis for pair: race_black and Acute respiratory failure with hypoxia
# Total running time of all codes is 85s. 