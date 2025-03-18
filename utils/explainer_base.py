import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from utils.LLM_Feat_Interpret import *

def random_select_indices(input_list):
    """
    Randomly select one index of `1` and one index of `0` from the input list.
    
    Args:
        input_list (list): A list of integers (0s and 1s).
    
    Returns:
        tuple: A tuple containing the index of a randomly selected `1` and the index of a randomly selected `0`.
    """
    # Get indices of 1s and 0s
    ones_indices = [i for i, val in enumerate(input_list) if val == 1]
    zeros_indices = [i for i, val in enumerate(input_list) if val == 0]

    # Ensure both 1s and 0s exist
    if not ones_indices or not zeros_indices:
        raise ValueError("The input list must contain at least one 1 and one 0.")
    
    # Randomly select one index from each list
    selected_one_index = random.choice(ones_indices)
    selected_zero_index = random.choice(zeros_indices)
    
    return selected_one_index, selected_zero_index

class Explainer:    
    def __init__(self, model, model_name, explain_mode, X_train, X_test, y_test, pred_test, X_test_cf, feature_names, attn_weights):
        self.model = model
        self.model_name = model_name
        self.explain_mode = explain_mode
        # Initialize SHAP explainer with the model and training data
        self.explainer = shap.Explainer(lambda X: self.model_callable(X, model), X_train)
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_cf = X_test_cf
        self.pred_test = pred_test
        self.feature_names = feature_names
        self.attn_weights = attn_weights

    def model_callable(self, X, model):
        # Convert input to PyTorch tensor
        model_input = torch.tensor(X, dtype=torch.float32)

        # Ensure input is on the same device as the model
        model_input = model_input.to(next(model.parameters()).device)

        # Pass input through the model
        with torch.no_grad():
            predictions, _ = model(model_input)  # Assuming the model returns a tuple

        # Convert predictions to NumPy
        return predictions.cpu().detach().numpy()
        
    def shap_explainer(self):
        # Compute SHAP values for the entire test set
        feature_contributions, shap_values = self.get_shap_values(self.explainer, self.X_test, self.feature_names)
        ####Global Interpretation: Summary Plot###
        # Global SHAP summary plot
        print('The global explanation:')
        self.global_explainer(shap_values, self.X_test, self.feature_names)
        # ###########VCNetAttn############
        # if self.model_name == 'VCAttNet' and self.attn_weights is not None:
        #     self.plot_attn_weights(inter_type='global')
        # ###########VCNetAttn############
        
        #Random select one pos and one neg for explanation
        pos_idx, neg_idx = random_select_indices(self.y_test)
        ###Instance-Level Waterfall Plot###
        # Plot SHAP waterfall plot for a positive instance
        print(f'The factual explanation of the positive instance {pos_idx} with the below features {self.X_test[pos_idx]} and label: {self.y_test[pos_idx]}:')
        self.plot_waterfall(shap_values, self.feature_names, instance_index=pos_idx, instan_type='pos', exp_type='ori')
        # Get SHAP values for the counterfactual
        print(f'The counterfactual explanation of the positive instance {pos_idx}:')
        cf_feature_contributions, cf_shap_values = self.get_shap_values(self.explainer, self.X_test_cf, self.feature_names)
        self.plot_waterfall(cf_shap_values, self.feature_names, instance_index=pos_idx, instan_type='pos', exp_type='cf')
        
        # Plot SHAP waterfall plot for a negative instance
        print(f'The factual explanation of the negative instance {neg_idx} with the below features {self.X_test[neg_idx]} and label: {self.y_test[neg_idx]}:')
        self.plot_waterfall(shap_values, self.feature_names, instance_index=neg_idx, instan_type='neg', exp_type='ori')
        # Get SHAP values for the counterfactual
        print(f'The counterfactual explanation of the negative instance {neg_idx}:')
        cf_feature_contributions, cf_shap_values = self.get_shap_values(self.explainer, self.X_test_cf, self.feature_names)
        self.plot_waterfall(cf_shap_values, self.feature_names, instance_index=neg_idx, instan_type='neg', exp_type='cf')
        
        # print(f'The feature interaction for all pairs')
        # self.feature_interaction_all(self.X_test, self.feature_names, topk=10)
        # fea_idx1 = random.randint(0, len(self.feature_names)-1)
        # fea_idx2 = random.randint(0, len(self.feature_names)-1)
        # fea1, fea2 = self.feature_names[fea_idx1], self.feature_names[fea_idx2]
        # print(f'The feature interaction for single pairs: {fea1} and {fea2}')
        # self.feature_interaction_pair(self.X_test, fea1, fea2)
        # ###########VCNetAttn Shift############
        # if self.model_name == 'VCAttNet' and self.attn_weights is not None:
        #     self.weights_shift(shap_values, cf_shap_values)
        # ###########VCNetAttn Shift############
        return feature_contributions, cf_feature_contributions
        
    def global_explainer(self, shap_values, X_test, feature_names, top_n = 10):
        # Generate the SHAP summary plot
        # print(f"SHAP values shape: {shap_values.values.shape}, type: {type(X_test)}")
        # Step 1: Filter out features ending with '_NaN'
        filtered_feature_indices = [i for i, name in enumerate(feature_names) if not name.endswith('_NaN')]
        filtered_feature_names = [feature_names[i] for i in filtered_feature_indices]
        # print('filtered_feature_names', filtered_feature_names)
        X_test_new = X_test[:, filtered_feature_indices]
        shap_values_new = shap_values.values[:, filtered_feature_indices]

        # Ensure top_n is within the range of available features
        if top_n > shap_values_new.shape[1]:
            raise ValueError(f"Requested top_n ({top_n}) exceeds the number of features ({shap_values_new.shape[1]}).")

        # Calculate mean absolute SHAP values for each feature
        mean_shap_values = np.abs(shap_values_new).mean(axis=0)

        # Get indices of the top N features
        top_indices = np.argsort(mean_shap_values)[-top_n:][::-1]

        # Get the names of the top features
        top_features = [filtered_feature_names[i] for i in top_indices]

        # Filter SHAP values and X_test for the top N features
        filtered_shap_values = shap_values_new[:, top_indices]
        filtered_X_test = X_test[:, top_indices]  # Use NumPy slicing for feature selection

        # Plot the SHAP summary plot for the top N features
        shap.summary_plot(filtered_shap_values, filtered_X_test, feature_names=top_features, show=False)
        
        # shap.summary_plot(shap_values.values, X_test, feature_names, show=False)
        # Save the figure
        plt.savefig(f"./figures/{self.model_name}_global_explanation.png", bbox_inches="tight")  # Save as PNG
        # Close the plot to free memory
        plt.close()
        
    def feature_interaction_pair(self, X_test, fea1, fea2):
        # Compute SHAP interaction values for the test data
        shap_interaction_values = self.deep_explainer.shap_interaction_values(X_test)
        shap.dependence_plot((feature_1, feature_2), shap_interaction_values, X_test)
        
        # Step 3: Visualize SHAP interaction values for a single prediction
        # (Example: interaction effects for the first prediction in the dataset)
        shap.summary_plot(shap_interaction_values[0], X_test, max_display=10)

        # Step 4: Visualize a specific feature interaction
        # Specify two features to visualize their interaction
        shap.dependence_plot((fea1, fea2), shap_interaction_values[0], X)

        # Step 5: Aggregate and summarize SHAP interaction values
        # Interaction importance plot (global interaction effects)
        shap.summary_plot(shap_interaction_values, X_test)
        plt.savefig(f"./figures/{self.model_name}_feature_interaction_pair_explanation.png", bbox_inches="tight")  # Save as PNG
        plt.close()
    
    def feature_interaction_all(self, X_test, feature_names, topk=10):
        # Compute SHAP interaction values for the test data
        shap_interaction_values = self.deep_explainer.shap_interaction_values(X_test)
        # Initialize a DataFrame to store interaction effects
        interaction_df = pd.DataFrame(columns=["Feature 1", "Feature 2", "Interaction Value"])
        # Iterate over all feature pairs
        num_features = len(feature_names)
        for i in range(num_features):
            for j in range(i, num_features):  # Pair combinations
                mean_interaction_value = np.mean(np.abs(shap_interaction_values[i][:, j]))
                interaction_df = interaction_df.append({
                    "Feature 1": feature_names[i],
                    "Feature 2": feature_names[j],
                    "Interaction Value": mean_interaction_value
                }, ignore_index=True)

        # Sort interactions by absolute interaction value
        interaction_df = interaction_df.sort_values(by="Interaction Value", ascending=False).reset_index(drop=True)
        # Bar plot for top 10 interactions
        top_interactions = interaction_df.head(topk)
        plt.figure(figsize=(10, 6))
        plt.barh(
            y=[f"{row['Feature 1']} & {row['Feature 2']}" for _, row in top_interactions.iterrows()],
            width=top_interactions["Interaction Value"],
            color="skyblue"
        )
        plt.xlabel("Mean Absolute SHAP Interaction Value")
        plt.ylabel("Feature Pairs")
        plt.title("Top 10 Feature Interactions")
        plt.gca().invert_yaxis()
        plt.savefig(f"./figures/{self.model_name}_feature_interaction_all_explanation.png", bbox_inches="tight")  # Save as PNG
        plt.show()
        plt.close()

    # Function to get SHAP values
    def get_shap_values(self, explainer, data, feature_names):
        """
        Compute SHAP values for a specific instance.
        """
        shap_values = explainer(data)
        feature_contributions = dict(zip(
            feature_names,
            shap_values.values[0]
        ))
        return feature_contributions, shap_values
    
    def LLM_explainer(self, feature_contributions, prediction, cf_feature_contributions):
        llm_output = explain_with_llm(feature_contributions, prediction, counterfactual)
        print('LLM explanation:', llm_output)

    def plot_waterfall(self, shap_values, feature_names, instance_index=0, base_value=0.0, instan_type='pos', exp_type='ori', top_k=10):
        # Step 1: Filter out features ending with '_NaN'
        valid_indices = [i for i, name in enumerate(feature_names) if not name.endswith('_NaN')]
        valid_feature_names = [feature_names[i] for i in valid_indices]
        # print('shap_values', shap_values.shape)
        valid_shap_values = shap_values.values[:, valid_indices]
        
        # Step 2: Get SHAP values for the specific instance
        shap_value = valid_shap_values[instance_index]
        
        # Step 3: Calculate absolute SHAP values for ranking
        abs_shap_values = np.abs(shap_value)

        # Step 4: Get indices of the Top K features
        top_k_indices = np.argsort(abs_shap_values)[-top_k:][::-1]

        # Step 5: Filter SHAP values and feature names for Top K features
        top_k_shap_values = shap_value[top_k_indices]
        top_k_feature_names = [valid_feature_names[i] for i in top_k_indices]

        # Step 6: Create a SHAP Explanation object for Top K features
        shap_exp = shap.Explanation(
            values=top_k_shap_values,
            base_values=base_value,
            data=np.zeros(len(top_k_shap_values)),  # Example input data for the waterfall plot
            feature_names=top_k_feature_names
        )

        # Step 7: Generate the waterfall plot
        shap.waterfall_plot(shap_exp)

        # Save the plot
        plt.savefig(f"./figures/{self.model_name}_{instan_type}_instance_{instance_index}_{exp_type}_top{top_k}_explanation.png", bbox_inches="tight")
        plt.show()
        plt.close()
    
        # shap_value = shap_values[instance_index]
        # # print('shap_values', shap_values, 'shap_value', shap_value)
        # # Create a SHAP Explanation object
        # shap_exp = shap.Explanation(values=shap_value, base_values=base_value, data=np.zeros(len(shap_value)), feature_names=feature_names)
        # # Generate the waterfall plot
        # shap.waterfall_plot(shap_exp)
        # plt.savefig(f"./figures/{self.model_name}_instance_{instance_index}_{exp_type}_explanation.png", bbox_inches="tight")  # Save as PNG
        # plt.show()
        # plt.close()
    
    def explain(self):
        if self.explain_mode == 'shap':
            print('SHAP explaination')
            self.feature_contributions, self.cf_feature_contributions = self.shap_explainer()
        elif self.explain_mode == 'mix':
            print('SHAP and LLM explaination')
            self.feature_contributions, self.cf_feature_contributions = self.shap_explainer()
            self.LLM_explainer(self.feature_contributions, self.pred_test, self.cf_feature_contributions)
            


# feature_names = [col for col in X_train_df.columns if col != "T2D"]    